# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import sys
import os
import os.path as osp
import time
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

from loguru import logger

from TempCLR.data.structures import StructureList
from TempCLR.utils import Tensor, TensorList, IntList


class MANOLossModule(nn.Module):
    '''
    '''

    def __init__(self, loss_cfg):
        super(MANOLossModule, self).__init__()

        self.stages_to_penalize = loss_cfg.get('stages_to_penalize', [-1])
        logger.info(f'Stages to penalize: {self.stages_to_penalize}')

        self.loss_enabled = defaultdict(lambda: True)
        self.loss_activ_step = {}

        shape_loss_cfg = loss_cfg.get('shape', {})
        self.shape_weight = shape_loss_cfg.get('weight', 0.0)
        self.shape_loss = build_loss(**shape_loss_cfg)
        self.loss_activ_step['shape'] = shape_loss_cfg.enable

        temporal_loss_cfg = loss_cfg.get('temporal_consistency', {})
        self.temporal_loss = build_loss(type='temporal_consistency', loss_cfg=temporal_loss_cfg)
        self.temporal_loss_weights = (temporal_loss_cfg.pose_w, temporal_loss_cfg.shape_w)

        wrist_pose_cfg = loss_cfg.get('wrist_pose', {})
        wrist_pose_loss_type = wrist_pose_cfg.type
        self.wrist_pose_loss_type = wrist_pose_loss_type
        self.wrist_pose_loss = build_loss(**wrist_pose_cfg)
        self.wrist_pose_weight = wrist_pose_cfg.weight
        self.loss_activ_step['wrist_pose'] = wrist_pose_cfg.enable
        logger.debug(
            'Wrist pose weight, loss: {}, {}',
            wrist_pose_cfg.weight, self.wrist_pose_loss)

        hand_pose_cfg = loss_cfg.get('hand_pose', {})
        hand_pose_loss_type = loss_cfg.hand_pose.type
        self.hand_use_conf = hand_pose_cfg.get('use_conf_weight', False)

        self.hand_pose_weight = loss_cfg.hand_pose.weight
        if self.hand_pose_weight > 0:
            self.hand_pose_loss_type = hand_pose_loss_type
            self.hand_pose_loss = build_loss(**loss_cfg.hand_pose)
            self.loss_activ_step['hand_pose'] = loss_cfg.hand_pose.enable

    def is_active(self) -> bool:
        return any(self.loss_enabled.values())

    def toggle_losses(self, step) -> None:
        for key in self.loss_activ_step:
            self.loss_enabled[key] = step >= self.loss_activ_step[key]

    def extra_repr(self) -> str:
        msg = [
            f'Shape weight: {self.shape_weight}',
            f'Global pose weight: {self.wrist_pose_weight}',
            f'Hand pose weight: {self.hand_pose_weight}',
        ]
        return '\n'.join(msg)

    def single_loss_step(
            self,
            parameters: Dict[str, Tensor],
            gt_wrist_pose: Optional[Tensor] = None,
            gt_wrist_pose_idxs: Optional[Tensor] = None,
            gt_hand_pose: Optional[Tensor] = None,
            gt_hand_pose_idxs: Optional[Tensor] = None,
            gt_shape: Optional[Tensor] = None,
            gt_shape_idxs: Optional[Tensor] = None,
            device: torch.device = None,
    ) -> Dict[str, Tensor]:
        losses = defaultdict(
            lambda: torch.tensor(0, device=device, dtype=torch.float32))

        if (self.shape_weight > 0 and self.loss_enabled['betas'] and
                gt_shape is not None and len(gt_shape_idxs) > 0):
            losses['shape_loss'] = (
                    self.shape_loss(
                        parameters['betas'][gt_shape_idxs], gt_shape) *
                    self.shape_weight)

        if (self.wrist_pose_weight > 0 and
                self.loss_enabled['wrist_pose'] and
                gt_wrist_pose is not None and len(gt_wrist_pose_idxs) > 0):
            losses['wrist_pose_loss'] = (
                    self.wrist_pose_loss(
                        parameters['wrist_pose'][gt_wrist_pose_idxs],
                        gt_wrist_pose) *
                    self.wrist_pose_weight)

        if (self.hand_pose_weight > 0 and self.loss_enabled['hand_pose'] and
                len(gt_hand_pose_idxs) > 0 and gt_hand_pose is not None):
            losses['hand_pose_loss'] = (
                    self.hand_pose_loss(
                        parameters['hand_pose'][gt_hand_pose_idxs], gt_hand_pose) *
                    self.hand_pose_weight)

        if self.temporal_loss_weights[0] > 0 or self.temporal_loss_weights[1] > 0 and len(gt_hand_pose_idxs) > 0:
            losses['temporal_consistency'] = self.temporal_loss(
                global_orientation_input=parameters['wrist_pose'][gt_wrist_pose_idxs],
                pose_input=parameters['hand_pose'][gt_hand_pose_idxs],
                shape_input=parameters['betas'][gt_shape_idxs])

        return losses

    def forward(
            self,
            network_params: Dict[str, Dict[str, Tensor]],
            targets: StructureList,
            device: torch.device = None,
            gt_confs: Optional[Tensor] = None,
            keypoint_part_indices: Optional[Union[TensorList, IntList]] = None,
            **kwargs,
    ) -> Dict[str, Tensor]:
        ''' Forward pass for the MANO parameter loss module
        '''

        is_right = network_params.get('is_right', True)
        gt_betas, gt_wrist_pose, gt_hand_pose = [None] * 3
        # Get the GT pose of the right hand
        gt_hand_pose_idxs = torch.tensor(
            [ii for ii, t in enumerate(targets) if t.has_field('hand_pose')],
            dtype=torch.long, device=device)
        if len(gt_hand_pose_idxs) > 0:
            gt_hand_pose = torch.stack(
                [t.get_field('hand_pose').right_hand_pose if is_right else
                 t.get_field('hand_pose').left_hand_pose
                 for t in targets if t.has_field('hand_pose')])

        # Get the GT global pose
        gt_wrist_pose_idxs = torch.tensor(
            [ii for ii, t in enumerate(targets)
             if t.has_field('wrist_pose') or t.has_field('global_rot')
             ],
            device=device, dtype=torch.long)
        if len(gt_wrist_pose_idxs) > 0:
            gt_wrist_pose = []
            for t in targets:
                if t.has_field('wrist_pose'):
                    gt_wrist_pose.append(t.get_field('wrist_pose').global_rot)
                elif t.has_field('global_rot'):
                    gt_wrist_pose.append(t.get_field('global_rot').global_rot)
            gt_wrist_pose = torch.stack(gt_wrist_pose)

        # Get the GT shape
        gt_betas_idxs = torch.tensor(
            [ii for ii, t in enumerate(targets) if t.has_field('betas')],
            device=device, dtype=torch.long)
        if len(gt_betas_idxs) > 0:
            gt_betas = torch.stack(
                [t.get_field('betas').betas for t in targets
                 if t.has_field('betas')])

        # Get the number of stages
        num_stages = network_params.get('num_stages', 1)

        output_losses = {}
        for ii, curr_key in enumerate(self.stages_to_penalize):
            curr_params = network_params.get(curr_key, None)
            if curr_params is None:
                logger.warning(f'Network output for {curr_key} is None')
                continue

            curr_losses = self.single_loss_step(
                curr_params,
                gt_wrist_pose=gt_wrist_pose,
                gt_wrist_pose_idxs=gt_wrist_pose_idxs,
                gt_hand_pose=gt_hand_pose,
                gt_hand_pose_idxs=gt_hand_pose_idxs,
                gt_shape=gt_betas,
                gt_shape_idxs=gt_betas_idxs,
                device=device)
            for key in curr_losses:
                out_key = f'{curr_key}_{key}'
                output_losses[out_key] = curr_losses[key]

        return output_losses


class RegularizerModule(nn.Module):
    def __init__(self, loss_cfg, hand_pose_mean=None):
        super(RegularizerModule, self).__init__()
        self.stages_to_regularize = loss_cfg.get(
            'stages_to_penalize', [])

    def forward(
            self,
            network_params: Dict,
            **kwargs
    ) -> Dict[str, Tensor]:

        prior_losses = defaultdict(lambda: 0)
        if len(self.stages_to_regularize) < 1:
            return prior_losses

        for ii, curr_key in enumerate(self.stages_to_regularize):
            curr_params = network_params.get(curr_key, None)
            if curr_params is None:
                logger.warning(f'Network output for {curr_key} is None')
                continue

            curr_losses = self.single_regularization_step(curr_params)
            for key in curr_losses:
                prior_losses[f'{curr_key}_{key}'] = curr_losses[key]

        return prior_losses
