# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Dict

import numpy as np
import torch.nn as nn
from loguru import logger

from .hand_heads import build_hand_head, HAND_HEAD_REGISTRY


def build_model(exp_cfg) -> Dict[str, nn.Module]:
    network_cfg = exp_cfg.get('network', {})
    net_type = network_cfg.get('type', 'TempCLR')

    logger.info(f'Going to build a: {net_type}')
    if net_type in HAND_HEAD_REGISTRY:
        network = build_hand_head(exp_cfg)
    else:
        raise ValueError(f'Unknown network type: {net_type}')

    return {
        'network': network
    }
