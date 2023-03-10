[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cagroup3d-class-aware-grouping-for-3d-object/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=cagroup3d-class-aware-grouping-for-3d-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cagroup3d-class-aware-grouping-for-3d-object/3d-object-detection-on-sun-rgbd-val)](https://paperswithcode.com/sota/3d-object-detection-on-sun-rgbd-val?p=cagroup3d-class-aware-grouping-for-3d-object)

# CAGroup3D

This repo is the official implementation of the paper:
#### CAGroup3D: Class-Aware Grouping for 3D Object Detection on Point Clouds
[PaperLink](https://arxiv.org/abs/2210.04264)

<img src="CAGroup3D.jpg">

## NEWS
- Official implementation based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) is released.
- 🔥 CAGroup3D is accepted at NeurIPS 2022.

## Introduction
This paper presents a novel two-stage fully sparse convolutional 3D object detection framework, named CAGroup3D. The proposed method first generates some high-quality 3D proposals by leveraging the class-aware local group strategy on the object surface voxels with the same semantic predictions, which considers semantic consistency and diverse locality abandoned in previous bottom-up approaches. Then, to recover the features of missed voxels due to incorrect voxel-wise segmentation, we build a fully sparse convolutional RoI pooling module to directly aggregate fine-grained spatial information from backbone for further proposal refinement.
## Requirements
The code is tested on the following environment:

- Unbuntu 18.04
- Python 3.7
- Pytorch 1.10
- CUDA 11.1
## Installation

- Clone this repo and install the `pcdet` library
```bash
git clone https://github.com/Haiyang-W/CAGroup3D.git
# install spconv
pip install spconv-cu113
cd CAGroup3D/
python setup.py develop
```

- Compile additional CUDA ops
```bash
# rotate iou ops
cd CAGroup3D/pcdet/ops/rotated_iou/cuda_op
python setup.py install
# knn ops
cd ../../knn
python setup.py develop
```

- Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
```bash
apt-get install -y python3-dev libopenblas-dev
pip install ninja==1.10.2.3
pip install \
  -U git+https://github.com/NVIDIA/MinkowskiEngine@v0.5.4 \
  --install-option="--blas=openblas" \
  --install-option="--force_cuda" \
  -v \
  --no-deps
```

## Data preparation
We haven't achieved compatibility with the generated data of OpenPCDet yet and use the same data format as [MMdeteciton3d](https://github.com/open-mmlab/mmdetection3d) for now. We will try to implement indoor data pre-processing based on OpenPCDet as soon as possible.
- follow  [MMdetection3D](https://github.com/open-mmlab/mmdetection3d) to create data (ScanNetV2, SunRGBD),  we also provide processed data in [here](https://drive.google.com/drive/folders/1sKvq4WBSEb4CWMdCTN6lCHLXnn3NwUv_).


- remember to modify the `DATA_PATH` in **tools/cfgs/dataset_configs/scannet_dataset.yaml**, **sunrgbd_dataset.yaml** or link the generated data as follows:
```shell
ln -s ${mmdet3d_scannet_dir} ./CAGroup3D/data/scannet
ln -s ${mmdet3d_sunrgbd_dir} ./CAGroup3D/data/sunrgbd
``` 

## Get started
### ScanNetV2
- Training, `num_gpus` x `batch_size` can be set to 2x8 or 4x4.
```bash
cd tools/
CUDA_VISIBLE_DEVICES={} ./scripts/dist_train.sh {num_gpus} --cfg_file cfgs/scannet_models/CAGroup3D.yaml --ckpt_save_interval 1 --extra_tag {your name} --fix_random_seed
```

- Testing
```bash
cd tools/
python test.py --cfg_file cfgs/scannet_models/CAGroup3D.yaml --ckpt {your pth}
# dist test is also supported
CUDA_VISIBLE_DEVICES={} ./scripts/dist_test.sh {num_gpus} --cfg_file cfgs/scannet_models/CAGroup3D.yaml --ckpt {your pth}
```
### Sun RGB-D
- Training, `num_gpus` x `batch_size` can be set to 2x8 or 4x4.
```bash
cd tools/
CUDA_VISIBLE_DEVICES={} ./scripts/dist_train.sh {num_gpus} --cfg_file cfgs/sunrgbd_models/CAGroup3D.yaml --ckpt_save_interval 1 --extra_tag {your name} --fix_random_seed
```

- Testing
```bash
cd tools/
python test.py --cfg_file cfgs/sunrgbd_models/CAGroup3D.yaml --ckpt {your pth}
# dist test is also supported
CUDA_VISIBLE_DEVICES={} ./scripts/dist_test.sh {num_gpus} --cfg_file cfgs/sunrgbd_models/CAGroup3D.yaml --ckpt {your pth}
```

### Main Results
All models are trained with 4 3090 GPUs and the pretrained models will be released soon.

|   Dataset | mAP@0.25 | mAP0.50 | Pretrain Model (will soon) |
|----------|----------:|:-------:|:-------:|
| [ScanNet](tools/cfgs/scannet_models/CAGroup3D.yaml) | 75.1(74.5)  |	61.3(60.3) | [model](https://github.com/Haiyang-W/CAGroup3D) |
| [Sun RGB-D](tools/cfgs/sunrgbd_models/CAGroup3D.yaml) | 66.8(66.4)   |	50.2(49.5) | [model](https://github.com/Haiyang-W/CAGroup3D) |

## Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{
wang2022cagroupd,
title={{CAG}roup3D: Class-Aware Grouping for 3D Object Detection on Point Clouds},
author={Haiyang Wang and Lihe Ding and Shaocong Dong and Shaoshuai Shi and Aoxue Li and Jianan Li and Zhenguo Li and Liwei Wang},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=nLKkHwYP4Au}
}
```

## TODO

- [x] Implement CAGroup3D on OpenPCDet
- [ ] clean up and release the code of MMdetection3D version CAGroup3D
- [ ] add score refinement in the RoI-Conv Module

## Acknowledgments
This project is based on the following codebases.
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [FCAF3D](https://github.com/SamsungLabs/fcaf3d)
* [RBGNet](https://github.com/Haiyang-W/RBGNet)
