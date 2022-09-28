
import sys
# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import torchvision
from loguru import logger

from . import transforms as T


def build_transforms(transf_cfg, is_train: bool,
                     enable_augment: bool = True,
                     return_full_imgs: bool = False):
    if is_train and enable_augment:
        flip_prob = transf_cfg.get('flip_prob', 0)
        #  downsample_prob = transf_cfg.get('downsample_prob', 0)
        max_size = transf_cfg.get('max_size', -1)
        downsample_dist = transf_cfg.get('downsample_dist', 'categorical')
        downsample_cat_factors = transf_cfg.get(
            'downsample_cat_factors', (1.0,))
        downsample_factor_min = transf_cfg.get('downsample_factor_min', 1.0)
        downsample_factor_max = transf_cfg.get('downsample_factor_max', 1.0)
        scale_factor = transf_cfg.get('scale_factor', 0.0)
        scale_factor_min = transf_cfg.get('scale_factor_min', 0.0)
        scale_factor_max = transf_cfg.get('scale_factor_max', 0.0)
        scale_dist = transf_cfg.get('scale_dist', 'uniform')
        rotation_factor = transf_cfg.get('rotation_factor', 0.0)
        noise_scale = transf_cfg.get('noise_scale', 0.0)
        center_jitter_factor = transf_cfg.get('center_jitter_factor', 0.0)
        center_jitter_dist = transf_cfg.get('center_jitter_dist', 'normal')
        color_jitter_prob = transf_cfg.get('color_jitter_prob', 0.0)
        motion_blur_prob = transf_cfg.get('motion_blur_prob', 0.0)
        motion_blur_kernel_size_min = transf_cfg.get(
            'motion_blur_kernel_size_min', 3)
        motion_blur_kernel_size_max = transf_cfg.get(
            'motion_blur_kernel_size_max', 7)
        sobel_kernel_size = transf_cfg.get('sobel_kernel_size', 0)
        color_drop_prob = transf_cfg.get('color_drop_prob', 0.0)
        sobel_prob = transf_cfg.get('sobel_prob', 0.0)
    else:
        flip_prob = 0.0
        max_size = -1
        color_drop_prob = 0.0
        sobel_prob = 0.0
        #  downsample_prob = 0.0
        #  downsample_factor = 1.0
        downsample_dist = 'categorical'
        downsample_cat_factors = (1.0,)
        downsample_factor_min = 1.0
        downsample_factor_max = 1.0
        scale_factor = 0.0
        scale_factor_min = 1.0
        scale_factor_max = 1.0
        rotation_factor = 0.0
        noise_scale = 0.0
        center_jitter_factor = 0.0
        center_jitter_dist = transf_cfg.get('center_jitter_dist', 'normal')
        scale_dist = transf_cfg.get('scale_dist', 'uniform')
        extreme_crop_prob = 0.0
        torso_upper_body_prob = 0.0
        motion_blur_prob = 0.0
        motion_blur_kernel_size_min = transf_cfg.get(
            'motion_blur_kernel_size_min', 3)
        motion_blur_kernel_size_max = transf_cfg.get(
            'motion_blur_kernel_size_max', 7)
        color_jitter_prob = transf_cfg.get('color_jitter_prob', 0.0)
        sobel_kernel_size = transf_cfg.get('sobel_kernel_size', 0)

    normalize_transform = T.Normalize(
        transf_cfg.get('mean'), transf_cfg.get('std'))
    logger.debug('Normalize {}', normalize_transform)

    crop_size = transf_cfg.get('crop_size')
    crop = T.Crop(crop_size=crop_size, is_train=is_train,
                  scale_factor_max=scale_factor_max,
                  scale_factor_min=scale_factor_min,
                  scale_factor=scale_factor,
                  scale_dist=scale_dist,
                  return_full_imgs=return_full_imgs,
                  )
    pixel_noise = T.ChannelNoise(noise_scale=noise_scale)
    logger.debug('Crop {}', crop)

    downsample = T.SimulateLowRes(
        dist=downsample_dist,
        cat_factors=downsample_cat_factors,
        factor_min=downsample_factor_min,
        factor_max=downsample_factor_max)

    transform = T.Compose(
        [
            T.Resize(max_size),
            T.SobelFilter(sobel_kernel_size, sobel_prob),
            T.MotionBlur(motion_blur_prob,
                         kernel_size_min=motion_blur_kernel_size_min,
                         kernel_size_max=motion_blur_kernel_size_max,
                         ),
            T.BBoxCenterJitter(center_jitter_factor, dist=center_jitter_dist),
            T.ColorJitter(color_jitter_prob),
            T.ColorDrop(color_drop_prob),
            pixel_noise,
            T.RandomHorizontalFlip(flip_prob),
            T.RandomRotation(
                is_train=is_train, rotation_factor=rotation_factor),
            crop,
            downsample,
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
