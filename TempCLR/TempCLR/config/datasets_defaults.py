# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Tuple
from dataclasses import dataclass
from omegaconf import OmegaConf
from TempCLR.utils.typing import StringTuple, FloatTuple


############################## DATASETS ##############################


@dataclass
class Transforms:
    flip_prob: float = 0.0
    max_size: float = 1080
    downsample_dist: str = 'categorical'
    downsample_factor_min: float = 1.0
    downsample_factor_max: float = 1.0
    downsample_cat_factors: Tuple[float] = (1.0,)
    center_jitter_factor: float = 0.0
    center_jitter_dist: str = 'uniform'
    crop_size: int = 256
    scale_factor_min: float = 1.0
    scale_factor_max: float = 1.0
    scale_factor: float = 0.0
    scale_dist: str = 'uniform'
    noise_scale: float = 0.0
    rotation_factor: float = 0.0
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)
    brightness: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    contrast: float = 0.0
    extreme_crop_prob: float = 0.0
    torso_upper_body_prob: float = 0.5
    motion_blur_prob: float = 0.0
    motion_blur_kernel_size_min: int = 3
    motion_blur_kernel_size_max: int = 21
    sobel_kernel_size: int = 3
    sobel_prob: float = 0.2
    color_drop_prob: float = 0.0
    color_jitter_prob: float = 0.0


@dataclass
class NumWorkers:
    train: int = 8
    val: int = 2
    test: int = 2


@dataclass
class Splits:
    train: StringTuple = tuple()
    val: StringTuple = tuple()
    test: StringTuple = tuple()


@dataclass
class Dataset:
    data_folder: str = 'data/'
    metrics: StringTuple = ('mpjpe14',)


@dataclass
class DatasetWithKeypoints(Dataset):
    binarization = True
    body_thresh: float = 0.05
    hand_thresh: float = 0.2
    head_thresh: float = 0.3
    keyp_folder: str = 'keypoints'
    keyp_format: str = 'openpose25_v1'
    use_face_contour: bool = True


@dataclass
class SequenceDataset(Dataset):
    interpenetration_threshold: float = 1.5
    pos_to_sample: int = 1
    window_size: int = 10
    neg_to_sample: int = 1


@dataclass
class ParameterOptions:
    return_params: bool = True
    return_shape: bool = False
    return_expression: bool = False
    return_full_pose: bool = False
    return_vertices: bool = False


@dataclass
class FreiHand(DatasetWithKeypoints, ParameterOptions):
    data_folder: str = 'data/freihand'
    mask_folder: str = 'data/freihand/masks'
    metrics: StringTuple = ('mpjpe', 'v2v')
    return_vertices: bool = True
    return_params: bool = True
    return_shape: bool = True
    file_format: str = 'npz'
    is_right: bool = True
    split_size: float = 0.8


@dataclass
class HO3D(DatasetWithKeypoints, ParameterOptions):
    data_folder: str = 'data/ho3d'
    metrics: Tuple[str] = ('mpjpe',)
    return_vertices: bool = True
    return_params: bool = True
    return_shape: bool = True
    file_format: str = 'json'
    split_size: float = 0.80
    is_right: bool = True
    split_by_frames: bool = False
    subsequences_length: int = 16
    subsequences_stride: int = 1


@dataclass
class DatasetConfig:
    batch_size: int = 1
    ratio_2d: float = 0.5
    use_packed: bool = True
    use_face_contour: bool = True
    vertex_flip_correspondences: str = ''
    transforms: Transforms = Transforms()
    splits: Splits = Splits()
    num_workers: NumWorkers = NumWorkers()


@dataclass
class HandConfig(DatasetConfig):
    splits: Splits = Splits(train=('freihand',))
    freihand: FreiHand = FreiHand()
    ho3d: HO3D = HO3D()


hand_conf = OmegaConf.structured(HandConfig)
