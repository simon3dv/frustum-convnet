import os
import sys
import numpy as np
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
from nuscenes2kitti_object import nuscenes2kitti_object
import ipdb
from PIL import Image
def pto_depth_map(velo_points,
                  H=32, W=256, C=5, dtheta=np.radians(1.33), dphi=np.radians(90. / 256.0)):
    """
    Ref:https://github.com/Durant35/SqueezeSeg/blob/master/src/nodes/segment_node.py
    Project velodyne points into front view depth map.
    :param velo_points: velodyne points in shape [:,4]
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param C: the channel size of depth map
        3 cartesian coordinates (x; y; z),
        an intensity measurement and
        range r = sqrt(x^2 + y^2 + z^2)
    :param dtheta: the delta theta of H, in radian
    :param dphi: the delta phi of W, in radian
    :return: `depth_map`: the projected depth map of shape[H,W,C]
    """

    x, y, z, i = velo_points[:, 1], -velo_points[:, 0], velo_points[:, 2], velo_points[:, 3]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W-1

    # print(np.min(phi_))
    # print(np.max(phi_))
    #
    # print z
    # print np.radians(2.)
    # print np.arcsin(z/d)
    theta = np.radians(2.) - np.arcsin(z / d)
    # print theta
    theta_ = (theta / dtheta).astype(int)
    # print theta_
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H-1
    # print theta,phi,theta_.shape,phi_.shape
    # print(np.min((phi/dphi)),np.max((phi/dphi)))
    # np.savetxt('./dump/'+'phi'+"dump.txt",(phi_).astype(np.float32), fmt="%f")
    # np.savetxt('./dump/'+'phi_'+"dump.txt",(phi/dphi).astype(np.float32), fmt="%f")
    # print(np.min(theta_))
    # print(np.max(theta_))

    depth_map = np.zeros((H, W, C))
    # 5 channels according to paper
    if C == 5:
        depth_map[theta_, phi_, 0] = x
        depth_map[theta_, phi_, 1] = y
        depth_map[theta_, phi_, 2] = z
        depth_map[theta_, phi_, 3] = i
        depth_map[theta_, phi_, 4] = d
    else:
        depth_map[theta_, phi_, 0] = i
    return depth_map

def keep_32(velo_points,
                  H=64, W=512, C=5, dtheta=np.radians(1.33), dphi=np.radians(90. / 512.0), odd=False,scale=1):
    x, y, z= velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W-1
    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H-1

    if odd:
        keep_v = np.mod(theta_,2)==1
    else:
        keep_v = np.mod(theta_,2)==0

    if scale == 1:
        keep = keep_v
    else:
        keep_h = np.mod(phi_,round(1/scale))==0
        keep = np.logical_and(keep_v, keep_h)

    lidar_keep = velo_points[keep,:]

    return lidar_keep, keep
def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

if __name__=='__main__':
    data_idx = 10

    dataset = nuscenes2kitti_object(os.path.join(ROOT_DIR, 'data/nuScenes2KITTI'))
    lidar = dataset.get_lidar(data_idx)
    lidar = pto_depth_map(lidar)
    depth_map = Image.fromarray(
        (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))

    depth_map.show()