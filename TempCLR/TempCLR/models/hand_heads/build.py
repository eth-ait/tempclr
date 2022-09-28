# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from .hand_heads import HAND_HEAD_REGISTRY


def build(exp_cfg):
    network_cfg = exp_cfg.get('network', {})
    hand_cfg = exp_cfg.get('hand_model', {})
    batch_size = exp_cfg.datasets.hand.batch_size
    network_type = network_cfg.get('type', 'MANORegressor')
    if network_type == 'MANORegressor':
        loss_cfg = exp_cfg.get('losses', {}).get('hand', {})
        network_cfg = network_cfg.get('hand', {})
        encoder_cfg = exp_cfg.get('losses', {}).get('encoder', {})
        temporal_backbone_cfg = network_cfg.get('temporal_backbone', {})

    elif network_type == 'MANOGroupRegressor':
        loss_cfg = exp_cfg.get('losses', {}).get('hand', {})
        network_cfg = network_cfg.get('hand', {})
        encoder_cfg = exp_cfg.get('losses', {}).get('encoder', {})

    else:
        raise ValueError(f'Unknown network type: {network_type}')
    return HAND_HEAD_REGISTRY.get(network_type)(
        hand_cfg, network_cfg=network_cfg, loss_cfg=loss_cfg, encoder_cfg=encoder_cfg,
        temporal_backbone_cfg=temporal_backbone_cfg, batch_size=batch_size)
