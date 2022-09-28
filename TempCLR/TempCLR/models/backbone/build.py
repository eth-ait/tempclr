# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import torch

from .resnet import resnets
from .fpn import build_fpn_backbone
from .hrnet import build as build_hr_net


def build_backbone(backbone_cfg):
    backbone_type = backbone_cfg.get('type', 'resnet50')
    #  use_avgpool = cfg.get('network', {}).get('type') != 'attention'
    pretrained = backbone_cfg.get('pretrained', True)

    if 'fpn' in backbone_type:
        backbone = build_fpn_backbone(backbone_cfg, pretrained=pretrained)
        return backbone, backbone.get_output_dim()
    elif 'hrnet' in backbone_type:
        backbone = build_hr_net(
            backbone_cfg, pretrained=True)
        return backbone, backbone.get_output_dim()
    elif 'resnet' in backbone_type:
        resnet_cfg = backbone_cfg.get('resnet')
        backbone = resnets[backbone_type](
            pretrained=backbone_cfg.pretrained, freeze=backbone_cfg.freeze, **resnet_cfg)
        return backbone, backbone.get_output_dim()
    else:
        msg = 'Unknown backbone type: {}'.format(backbone_type)
        raise ValueError(msg)
