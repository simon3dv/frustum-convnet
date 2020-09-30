# Deprecated

Results:
```
k2k(2020.3.2):
iou3d_0.7:0.873(50)
Car AP@0.70, 0.70, 0.70:
bbox AP:99.97, 99.95, 99.96
bev AP:97.36, 89.50, 89.08
3d AP:87.60, 85.52, 78.27

k2k_densefusion_plus
iou3d_0.7:0.884(50)
Car AP@0.70, 0.70, 0.70:
bbox AP:99.95, 99.94, 99.95
bev AP:90.32, 89.62, 89.30
3d AP:88.83, 86.44, 79.03
```
Other experiment results: https://www.cnblogs.com/simingfan/p/12375617.html

What's new in simon3dv/frustum-convnet compared to zhixinwang/frustum-convnet:
```
    feature fusion experiment: see 
        models/det_*.py like models/det_densefusion.py
        rgb feature extractor:models/pspnet.py, models/extractors.py

        cfgs/fusion/*
    domain adaptation experiment:
        nuscenes2kitti: same as https://github.com/simon3dv/frustum_pointnets_pytorch
        cfgs/da/*
```       

2020.9.30
