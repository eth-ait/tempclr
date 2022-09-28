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
import torch.nn as nn
import torch.nn.functional as F

#  from .camera_head import CameraHead
from .camera_projection import build_cam_proj


def build_camera_head(cfg, feat_dim):
    return CameraHead(cfg, feat_dim)
