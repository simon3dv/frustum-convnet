from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from configs.config import cfg
from datasets.dataset_info import KITTICategory

from models.model_util import get_box3d_corners_helper
from models.model_util import huber_loss

from models.common import Conv1d, Conv2d, DeConv1d, init_params
from models.common import softmax_focal_loss_ignore, get_accuracy

from ops.query_depth_point.query_depth_point import QueryDepthPoint
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from models.box_transform import size_decode, size_encode, center_decode, center_encode, angle_decode, angle_encode

from models.pspnet import PSPNet

NUM_SIZE_CLUSTER = len(KITTICategory.CLASSES)
MEAN_SIZE_ARRAY = KITTICategory.MEAN_SIZE_ARRAY


# single scale PointNet module
class PointNetModule(nn.Module):
    def __init__(self, Infea, mlp, dist, nsample, use_xyz=True, use_feature=True):
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.mlp = mlp

        if Infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, nsample)

        if self.use_xyz:
            self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        else:
            self.conv1 = Conv2d(Infea, mlp[0], 1)

        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)
        self.joint_conv1 = Conv2d(mlp[1]*2+mlp[0]*2,mlp[2],1)

        init_params([self.conv1[0], self.conv2[0], self.conv3[0], self.joint_conv1[0]], 'kaiming_normal')
        init_params([self.conv1[1], self.conv2[1], self.conv3[1], self.joint_conv1[1]], 1)

    def forward(self, pc, feat, new_pc=None,
                img1=None, img2=None, P=None, query_v1=None):
        batch_size = pc.size(0)

        npoint = new_pc.shape[2]
        k = self.nsample

        indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample

        assert indices.data.max() < pc.shape[2] and indices.data.min() >= 0

        indices_rgb = torch.gather(query_v1, 1, indices.view(batch_size, npoint * k)) \
            .view(batch_size, npoint, k)

        assert indices_rgb.data.max() < img1.shape[2]*img1.shape[3]
        assert indices_rgb.data.min() >= 0

        grouped_pc = None
        grouped_feature = None

        if self.use_xyz:
            grouped_pc = torch.gather(
                pc, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, 3, -1)
            ).view(batch_size, 3, npoint, k)

            grouped_pc = grouped_pc - new_pc.unsqueeze(3)

        if self.use_feature:
            grouped_feature = torch.gather(
                feat, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1)
            ).view(batch_size, feat.size(1), npoint, k)

            # grouped_feature = torch.cat([new_feat.unsqueeze(3), grouped_feature], -1)

        img1 = img1.view(batch_size,self.mlp[0],-1)
        grouped_rgb1 = torch.gather(
            img1, 2,
            indices_rgb.view(batch_size, 1, npoint * k).expand(-1, self.mlp[0], -1)
        ).view(batch_size, self.mlp[0], npoint, k)

        grouped_rgb2 = None
        if img2 is not None:
            img2 = img2.view(batch_size,self.mlp[1],-1)
            grouped_rgb2 = torch.gather(
                img2, 2,
                indices_rgb.view(batch_size, 1, npoint * k).expand(-1, self.mlp[1], -1)
            ).view(batch_size, self.mlp[1], npoint, k)

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()

        grouped_feature = self.conv1(grouped_feature)
        # mlp[0]+mlp[0]:
        fusion_feature_1 = torch.cat([grouped_feature, grouped_rgb1], 1)

        grouped_feature = self.conv2(grouped_feature)

        # mlp[1]+mlp[1]:
        fusion_feature_2 = torch.cat([grouped_feature, grouped_rgb2], 1)

        grouped_feature = self.conv3(grouped_feature)

        fusion_feature = torch.cat([fusion_feature_1, fusion_feature_2], 1)

        fusion_feature = self.joint_conv1(fusion_feature)

        # output, _ = torch.max(grouped_feature, -1)

        valid = (num > 0).view(batch_size, 1, -1, 1)
        grouped_feature = torch.cat([grouped_feature, fusion_feature], 1)
        grouped_feature = grouped_feature * valid.float()

        return grouped_feature


# multi-scale PointNet module
class PointNetFeat(nn.Module):
    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()

        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        assert len(u) == 4
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 32, use_xyz=True, use_feature=True)

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 64, use_xyz=True, use_feature=True)

        self.pointnet3 = PointNetModule(
            input_channel - 3, [128, 128, 256], u[2], 64, use_xyz=True, use_feature=True)

        self.pointnet4 = PointNetModule(
            input_channel - 3, [256, 256, 512], u[3], 128, use_xyz=True, use_feature=True)

        self.econv1 = Conv2d(32, 64, 1)
        self.econv2 = Conv2d(64, 64, 1)
        self.econv3 = Conv2d(64, 128, 1)
        self.econv4 = Conv2d(128, 128, 1)
        self.econv5 = Conv2d(128, 256, 1)
        self.econv6 = Conv2d(256, 256, 1)

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None,
                img=None, P=None, query_v1=None):
        pc = point_cloud
        pc1 = sample_pc[0]
        pc2 = sample_pc[1]
        pc3 = sample_pc[2]
        pc4 = sample_pc[3]

        img1 = self.econv1(img)
        img2 = self.econv2(img1)
        img3 = self.econv3(img2)
        img4 = self.econv4(img3)
        img5 = self.econv5(img4)
        img6 = self.econv6(img5)

        feat1 = self.pointnet1(pc, feat, pc1,  img1, img2, P, query_v1,)
        feat1, _ = torch.max(feat1, -1)

        feat2 = self.pointnet2(pc, feat, pc2,  img1, img2, P, query_v1,)
        feat2, _ = torch.max(feat2, -1)

        feat3 = self.pointnet3(pc, feat, pc3,  img3, img4, P, query_v1,)
        feat3, _ = torch.max(feat3, -1)

        feat4 = self.pointnet4(pc, feat, pc4,  img5, img6, P, query_v1,)
        feat4, _ = torch.max(feat4, -1)

        if one_hot_vec is not None:
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat1.shape[-1])
            feat1 = torch.cat([feat1, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat2.shape[-1])
            feat2 = torch.cat([feat2, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat3.shape[-1])
            feat3 = torch.cat([feat3, one_hot], 1)

            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat4.shape[-1])
            feat4 = torch.cat([feat4, one_hot], 1)

        return feat1, feat2, feat3, feat4


# FCN
class ConvFeatNet(nn.Module):
    def __init__(self, i_c=256, num_vec=3):
        super(ConvFeatNet, self).__init__()

        self.block1_conv1 = Conv1d(i_c + num_vec, 256, 3, 1, 1)

        self.block2_conv1 = Conv1d(256, 256, 3, 2, 1)
        self.block2_conv2 = Conv1d(256, 256, 3, 1, 1)
        self.block2_merge = Conv1d(256 + 256 + num_vec, 256, 1, 1)

        self.block3_conv1 = Conv1d(256, 640, 3, 2, 1)
        self.block3_conv2 = Conv1d(640, 640, 3, 1, 1)
        self.block3_merge = Conv1d(640 + 512 + num_vec, 640, 1, 1)

        self.block4_conv1 = Conv1d(640, 1280, 3, 2, 1)
        self.block4_conv2 = Conv1d(1280, 1280, 3, 1, 1)
        self.block4_merge = Conv1d(1280 + 1024 + num_vec, 1280, 1, 1)

        self.block2_deconv = DeConv1d(256, 640, 1, 1, 0)
        self.block3_deconv = DeConv1d(640, 640, 2, 2, 0)
        self.block4_deconv = DeConv1d(1280, 640, 4, 4, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):

        x = self.block1_conv1(x1)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = torch.cat([x, x2], 1)
        x = self.block2_merge(x)
        xx1 = x

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = torch.cat([x, x3], 1)
        x = self.block3_merge(x)
        xx2 = x

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = torch.cat([x, x4], 1)
        x = self.block4_merge(x)
        xx3 = x

        xx1 = self.block2_deconv(xx1)
        xx2 = self.block3_deconv(xx2)
        xx3 = self.block4_deconv(xx3)

        x = torch.cat([xx1, xx2[:, :, :xx1.shape[-1]], xx3[:, :, :xx1.shape[-1]]], 1)

        return x


# the whole pipeline
class PointNetDet(nn.Module):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__()

        self.feat_net = PointNetFeat(input_channel, 0)
        self.conv_net = ConvFeatNet()

        self.num_classes = num_classes

        num_bins = cfg.DATA.NUM_HEADING_BIN
        self.num_bins = num_bins

        output_size = 3 + num_bins * 2 + NUM_SIZE_CLUSTER * 4

        self.reg_out = nn.Conv1d(1920, output_size, 1)
        self.cls_out = nn.Conv1d(1920, 2, 1)
        self.relu = nn.ReLU(True)

        nn.init.kaiming_uniform_(self.cls_out.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.reg_out.weight, mode='fan_in')

        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

        self.cnn = ModifiedResnet()

    def _slice_output(self, output):

        batch_size = output.shape[0]

        num_bins = self.num_bins

        center = output[:, 0:3].contiguous()

        heading_scores = output[:, 3:3 + num_bins].contiguous()

        heading_res_norm = output[:, 3 + num_bins:3 + num_bins * 2].contiguous()

        size_scores = output[:, 3 + num_bins * 2:3 + num_bins * 2 + NUM_SIZE_CLUSTER].contiguous()

        size_res_norm = output[:, 3 + num_bins * 2 + NUM_SIZE_CLUSTER:].contiguous()
        size_res_norm = size_res_norm.view(batch_size, NUM_SIZE_CLUSTER, 3)

        return center, heading_scores, heading_res_norm, size_scores, size_res_norm

    def get_center_loss(self, pred_offsets, gt_offsets):

        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)

        return center_loss

    def get_heading_loss(self, heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label):

        heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)

        # b, NUM_HEADING_BIN -> b, 1
        heading_res_norm_select = torch.gather(heading_res_norm, 1, heading_class_label.view(-1, 1))

        heading_res_norm_loss = huber_loss(
            heading_res_norm_select.squeeze(1) - heading_res_norm_label, delta=1.0)

        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(self, size_scores, size_res_norm, size_class_label, size_res_label_norm):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)

        # b, NUM_SIZE_CLUSTER, 3 -> b, 1, 3
        size_res_norm_select = torch.gather(size_res_norm, 1,
                                            size_class_label.view(batch_size, 1, 1).expand(
                                                batch_size, 1, 3))

        size_norm_dist = torch.norm(
            size_res_label_norm - size_res_norm_select.squeeze(1), 2, dim=-1)

        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)

        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):

        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds

        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)

        corners_3d_pred = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

        # N, 8, 3
        corners_dist = torch.min(
            torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1).mean(-1),
            torch.norm(corners_3d_pred - corners_3d_gt_flip, 2, dim=-1).mean(-1))
        # corners_dist = torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1)
        corners_loss = huber_loss(corners_dist, delta=1.0)

        return corners_loss, corners_3d_gt

    def forward(self,
                data_dicts):

        image = data_dicts.get('image')
        out_image = self.cnn(image)
        P = data_dicts.get('P')
        query_v1 = data_dicts.get('query_v1')

        point_cloud = data_dicts.get('point_cloud')
        one_hot_vec = data_dicts.get('one_hot')
        cls_label = data_dicts.get('label')
        size_class_label = data_dicts.get('size_class')
        center_label = data_dicts.get('box3d_center')
        heading_label = data_dicts.get('box3d_heading')
        size_label = data_dicts.get('box3d_size')

        center_ref1 = data_dicts.get('center_ref1')
        center_ref2 = data_dicts.get('center_ref2')
        center_ref3 = data_dicts.get('center_ref3')
        center_ref4 = data_dicts.get('center_ref4')

        batch_size = point_cloud.shape[0]

        object_point_cloud_xyz = point_cloud[:, :3, :].contiguous()
        if point_cloud.shape[1] > 3:
            object_point_cloud_i = point_cloud[:, [3], :].contiguous()
        else:
            object_point_cloud_i = None

        mean_size_array = torch.from_numpy(MEAN_SIZE_ARRAY).type_as(point_cloud)

        feat1, feat2, feat3, feat4 = self.feat_net(
            object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4],
            object_point_cloud_i,
            one_hot_vec,
            out_image,
            P,
            query_v1
        )

        x = self.conv_net(feat1, feat2, feat3, feat4)

        cls_scores = self.cls_out(x)
        outputs = self.reg_out(x)

        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        # b, c, n -> b, n, c
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)

        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)

        cls_probs = F.softmax(cls_scores, -1)

        if center_label is None:
            assert not self.training, 'Please provide labels for training.'

            det_outputs = self._slice_output(outputs)

            center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

            # decode
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)

            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            center_preds = center_boxnet + center_ref2

            heading_preds = angle_decode(heading_res_norm, heading_pred_label)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            # corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

            cls_probs = cls_probs.view(batch_size, -1, 2)
            center_preds = center_preds.view(batch_size, -1, 3)

            size_preds = size_preds.view(batch_size, -1, 3)
            heading_preds = heading_preds.view(batch_size, -1)

            outputs = (cls_probs, center_preds, heading_preds, size_preds)
            return outputs

        fg_idx = (cls_label.view(-1) == 1).nonzero().view(-1)

        assert fg_idx.numel() != 0

        outputs = outputs[fg_idx, :]
        center_ref2 = center_ref2[fg_idx]

        det_outputs = self._slice_output(outputs)

        center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

        heading_probs = F.softmax(heading_scores, -1)
        size_probs = F.softmax(size_scores, -1)

        # cls_loss = F.cross_entropy(cls_scores, mask_label, ignore_index=-1)
        cls_loss = softmax_focal_loss_ignore(cls_probs, cls_label.view(-1), ignore_idx=-1)

        # prepare label
        center_label = center_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        heading_label = heading_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]
        size_label = size_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        size_class_label = size_class_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]

        # encode regression targets
        center_gt_offsets = center_encode(center_label, center_ref2)
        heading_class_label, heading_res_norm_label = angle_encode(heading_label)
        size_res_label_norm = size_encode(size_label, mean_size_array, size_class_label)

        # loss calculation

        # center_loss
        center_loss = self.get_center_loss(center_boxnet, center_gt_offsets)


        # heading loss
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(
            heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label)

        # size loss
        size_class_loss, size_res_norm_loss = self.get_size_loss(
            size_scores, size_res_norm, size_class_label, size_res_label_norm)

        # corner loss regulation
        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label)
        size = size_decode(size_res_norm, mean_size_array, size_class_label)

        corners_loss, corner_gts = self.get_corner_loss(
            (center_preds, heading, size),
            (center_label, heading_label, size_label)
        )

        BOX_LOSS_WEIGHT = cfg.LOSS.BOX_LOSS_WEIGHT
        CORNER_LOSS_WEIGHT = cfg.LOSS.CORNER_LOSS_WEIGHT
        HEAD_REG_WEIGHT = cfg.LOSS.HEAD_REG_WEIGHT
        SIZE_REG_WEIGHT = cfg.LOSS.SIZE_REG_WEIGHT

        # Weighted sum of all losses
        loss = cls_loss + \
            BOX_LOSS_WEIGHT * (center_loss +
                               heading_class_loss + size_class_loss +
                               HEAD_REG_WEIGHT * heading_res_norm_loss +
                               SIZE_REG_WEIGHT * size_res_norm_loss +
                               CORNER_LOSS_WEIGHT * corners_loss)

        # some metrics to monitor training status

        with torch.no_grad():

            # accuracy
            cls_prec = get_accuracy(cls_probs, cls_label.view(-1))
            heading_prec = get_accuracy(heading_probs, heading_class_label.view(-1))
            size_prec = get_accuracy(size_probs, size_class_label.view(-1))

            # iou metrics
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            heading_preds = angle_decode(heading_res_norm, heading_pred_label)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(), corner_gts.detach().cpu().numpy())

            iou2ds, iou3ds = overlap[:, 0], overlap[:, 1]
            iou2d_mean = iou2ds.mean()
            iou3d_mean = iou3ds.mean()
            iou3d_gt_mean = (iou3ds >= cfg.IOU_THRESH).mean()
            iou2d_mean = torch.tensor(iou2d_mean).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3d_mean).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor(iou3d_gt_mean).type_as(cls_prec)

        losses = {
            'total_loss': loss,
            'cls_loss': cls_loss,
            'center_loss': center_loss,
            'head_cls_loss': heading_class_loss,
            'head_res_loss': heading_res_norm_loss,
            'size_cls_loss': size_class_loss,
            'size_res_loss': size_res_norm_loss,
            'corners_loss': corners_loss
        }

        metrics = {
            'cls_acc': cls_prec,
            'head_acc': heading_prec,
            'size_acc': size_prec,
            'IoU_2D': iou2d_mean,
            'IoU_3D': iou3d_mean,
            'IoU_' + str(cfg.IOU_THRESH): iou3d_gt_mean
        }

        return losses, metrics

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    from datasets.provider_fusion import ProviderDataset
    dataset = ProviderDataset(npoints=1024, split='val',
        random_flip=True, random_shift=True, one_hot=True,
        overwritten_data_path='kitti/data/pickle_data/frustum_caronly_wimage_val.pickle',
        gen_image=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = PointNetDet(3, num_vec=0, num_classes=2).cuda()
    t = 0
    for batch, data_dicts in enumerate(dataloader):
        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
        # dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'one_hot',
        # 'ref_label', 'center_ref1', 'center_ref2', 'center_ref3', 'center_ref4',
        # 'size_class', 'box3d_size', 'box3d_heading', 'image', 'P', 'query_v1'])
        tic = time.perf_counter()
        losses, metrics= model(data_dicts_var)
        tic2 = time.perf_counter()
        t += (tic2-tic)
        print("Time:%.2fms"%(t))
        print()
        for key,value in losses.items():
            print(key,value)
        print()
        for key,value in metrics.items():
            print(key,value)
    print("Avr Time:%.2fms"%(t/len(dataset)))