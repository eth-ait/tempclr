# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import numpy as np

import torch

DIM_FLIP = np.array([1, -1, -1], dtype=np.float32)
DIM_FLIP_TENSOR = torch.tensor([1, -1, -1], dtype=torch.float32)


def flip_pose(pose_vector, pose_format='aa'):
    if pose_format == 'aa':
        if torch.is_tensor(pose_vector):
            dim_flip = DIM_FLIP_TENSOR
        else:
            dim_flip = DIM_FLIP
        return (pose_vector.reshape(-1, 3) * dim_flip).reshape(-1)
    elif pose_format == 'rot-mat':
        rot_mats = pose_vector.reshape(-1, 9).clone()

        rot_mats[:, [1, 2, 3, 6]] *= -1
        return rot_mats.view_as(pose_vector)
    else:
        raise ValueError(f'Unknown rotation format: {pose_format}')
