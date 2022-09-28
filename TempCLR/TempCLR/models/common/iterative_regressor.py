# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Dict, Tuple, Optional
import os.path as osp

import pickle

from loguru import logger
from collections import defaultdict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .pose_utils import PoseParameterization
from .networks import build_regressor, TemporalEncoder
from .rigid_alignment import RotationTranslationAlignment

from ..body_models import KeypointTensor
from ..backbone import build_backbone, make_projection_head
from ..camera import CameraParams, build_cam_proj

from TempCLR.data.structures import StructureList
from TempCLR.utils import (
    Tensor, BlendShapeDescription, StringList, AppearanceDescription)


class HMRLikeRegressor(nn.Module):
    def __init__(
            self,
            body_model_cfg: DictConfig,
            network_cfg: DictConfig,
            loss_cfg: DictConfig,
            encoder_cfg: DictConfig,
            temporal_backbone_cfg: DictConfig,
            batch_size: int,
            dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        super(HMRLikeRegressor, self).__init__()

        self.temporal_encoder = None
        self.projection_head = None

        self.batch_size = batch_size
        # Pose only the final stage of the model to save computation
        self.pose_last_stage = network_cfg.get('pose_last_stage', True)
        logger.info(f'Pose last stage: {self.pose_last_stage}')

        camera_cfg = network_cfg.get('camera', {})
        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']

        self.model = self._build_model(body_model_cfg)

        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        self.camera_scale_func = camera_data['scale_func']

        pose_space = self._build_pose_space(body_model_cfg)
        blendshape_space = self._build_blendshape_space(body_model_cfg)
        appearance_space = self._build_appearance_space(body_model_cfg)

        self.pose_space = pose_space
        self.blendshape_space = blendshape_space
        self.appearance_space = appearance_space

        camera_space = {'dim': camera_param_dim, 'mean': camera_mean}

        param_dict = dict(**pose_space, **blendshape_space, **appearance_space)
        param_dict['camera'] = camera_space

        mean_lst = []
        start = 0
        # Build the indices used to extract the individual parameters
        for name, desc in param_dict.items():
            buffer_name = f'{name}_idxs'
            indices = list(range(start, start + desc['dim']))
            #  logger.info(f'{buffer_name}: {indices}')
            indices = torch.tensor(indices, dtype=torch.long)
            self.register_buffer(buffer_name, indices)
            mean_lst.append(desc['mean'].view(-1))
            start += desc['dim']

            self.register_buffer(f'{name}_mean', desc['mean'])

        self.param_names = list(param_dict.keys())

        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()

        backbone_cfg = network_cfg.get('backbone', {})
        self.backbone, feat_dims = build_backbone(backbone_cfg)

        if backbone_cfg.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        self.feature_key = network_cfg.get('feature_key', 'avg_pooling')
        if self.feature_key not in feat_dims:
            self.feature_key = 'concat'

        feat_dim = feat_dims[self.feature_key]

        if temporal_backbone_cfg.active:
            self.temporal_encoder = TemporalEncoder(input_size=feat_dim, temporal_backbone_cfg=temporal_backbone_cfg)
            self.seq_len = temporal_backbone_cfg.seq_len
            if temporal_backbone_cfg.freeze:
                for param in self.temporal_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.temporal_encoder.parameters():
                    param.requires_grad = True

        self._feat_dim = feat_dim
        self._param_dim = param_dim
        self.register_buffer('param_mean', param_mean)

        self.regressor, num_stages = build_regressor(
            network_cfg, feat_dim, param_dim, param_mean=param_mean)
        logger.info(f'Regressor network: {self.regressor}')

        self._num_stages = num_stages

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def get_mean(self) -> Tensor:
        return self.param_mean

    @property
    def feat_dim(self) -> int:
        ''' Returns the dimension of the expected feature vector '''
        return self._feat_dim

    @property
    def num_stages(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        return self._num_stages

    @property
    def num_betas(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        return self.model.num_betas

    @property
    def num_expression_coeffs(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        if hasattr(self.model, 'num_expression_coeffs'):
            return self.model.num_expression_coeffs
        else:
            return 0

    def flat_params_to_dict(self, param_tensor: Tensor) -> Dict[str, Tensor]:
        ''' Convert a flat parameter tensor to a dictionary of parameters
        '''
        param_dict = {}
        for name in self.param_names:
            indices = getattr(self, f'{name}_idxs')
            param_dict[name] = torch.index_select(param_tensor, 1, indices)
            logger.debug(f'{name}: {param_dict[name].shape}')
        return param_dict

    def _build_pose_space(
            self, body_model_cfg
    ) -> Dict[str, PoseParameterization]:
        mean_pose_path = osp.expandvars(self.curr_model_cfg.mean_pose_path)
        self.mean_poses_dict = {}
        if osp.exists(mean_pose_path):
            logger.debug('Loading mean pose from: {} ', mean_pose_path)
            with open(mean_pose_path, 'rb') as f:
                self.mean_poses_dict = pickle.load(f)
        return {}

    def _build_appearance_space(
            self, body_model_cfg,
            dtype=torch.float32,
    ) -> Dict[str, AppearanceDescription]:
        return {}

    def _build_blendshape_space(
            self, body_model_cfg,
            dtype=torch.float32,
    ) -> Dict[str, BlendShapeDescription]:
        return {}

    def compute_features(
            self,
            images: Tensor,
            extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        ''' Computes features for the current input
        '''
        feat_dict = self.backbone(images)
        features = feat_dict[self.feature_key]

        if self.temporal_encoder:
            features = features.reshape(features.shape[0] // self.seq_len, self.seq_len, -1)
            features = self.temporal_encoder(features)
            features = features.reshape(-1, features.size(-1))

        if self.projection_head:
            return self.projection_head(features)

        return features

    def forward(
            self,
            images: Tensor,
            targets: StructureList = None,
            compute_losses: bool = True,
            cond: Optional[Tensor] = None,
            extra_features: Optional[Tensor] = None,
            **kwargs
    ):
        batch_size = len(images)
        device, dtype = images.device, images.dtype

        # Compute the features
        features = self.compute_features(images, extra_features=extra_features)

        regr_output = self.regressor(
            features, cond=cond, extra_features=extra_features)

        if torch.is_tensor(regr_output):
            parameters = [regr_output]
        elif isinstance(regr_output, (tuple, list)):
            parameters = regr_output[0]

        param_dicts = []
        # Iterate over the estimated parameters and decode them. For example,
        # rotation predictions need to be converted from whatever format is
        # predicted by the network to rotation matrices.
        for ii, params in enumerate(parameters):
            curr_params_dict = self.flat_params_to_dict(params)
            out_dict = {}
            for key, val in curr_params_dict.items():
                if hasattr(self, f'{key}_decoder'):
                    decoder = getattr(self, f'{key}_decoder')
                    out_dict[key] = decoder(val)
                    out_dict[f'raw_{key}'] = val.clone()
                else:
                    out_dict[key] = val
            param_dicts.append(out_dict)

        num_stages = len(param_dicts)

        if self.pose_last_stage:
            merged_params = param_dicts[-1]
        else:
            # If we want to pose all prediction stages to visualize the meshes,
            # then it is much faster to concatenate all parameters, pose and
            # split, instead of running the skinning function N times.
            merged_params = {}
            for key in param_dicts[0].keys():
                param = []
                for ii in range(num_stages):
                    if param_dicts[ii][key] is None:
                        continue
                    param.append(param_dicts[ii][key])
                merged_params[key] = torch.cat(param, dim=0)

        # Compute the body surface using the current estimation of the pose and
        # the shape
        model_output = self.model(
            get_skin=True, return_shaped=True, **merged_params)

        # Split the vertices, joints, etc. to stages
        out_params = defaultdict(lambda: dict())
        for key in model_output:
            if isinstance(model_output[key], (KeypointTensor,)):
                curr_val = model_output[key]
                out_list = torch.split(curr_val._t, batch_size, dim=0)
                if len(out_list) == num_stages:
                    for ii, value in enumerate(out_list):
                        out_params[f'stage_{ii:02d}'][key] = (
                            KeypointTensor.from_obj(value, curr_val)
                        )
                else:
                    # Else add only the last
                    out_key = f'stage_{num_stages - 1:02d}'
                    out_params[out_key][key] = KeypointTensor.from_obj(
                        out_list[-1], curr_val)
            elif torch.is_tensor(model_output[key]):
                curr_val = model_output[key]
                out_list = torch.split(curr_val, batch_size, dim=0)
                # If the number of outputs is equal to the number of stages
                # then store each stage
                if len(out_list) == num_stages:
                    for ii, value in enumerate(out_list):
                        out_params[f'stage_{ii:02d}'][key] = value
                else:
                    # Else add only the last
                    out_key = f'stage_{num_stages - 1:02d}'
                    out_params[out_key][key] = out_list[-1]

        # Extract the estimated camera parameters
        camera_params = param_dicts[-1]['camera']
        scale = camera_params[:, 0].view(-1, 1)
        translation = camera_params[:, 1:3]
        # Pass the predicted scale through exp() to make sure that the
        # scale values are always positive
        scale = self.camera_scale_func(scale)

        est_joints3d = out_params[f'stage_{num_stages - 1:02d}']['joints']

        # Project the joints on the image plane WeakPerspectiveCamera
        proj_joints = self.projection(
            est_joints3d, scale=scale, translation=translation)

        # Add the projected joints
        out_params['est_joints3d'] = est_joints3d
        out_params['proj_joints'] = proj_joints
        out_params['num_stages'] = num_stages
        out_params['features'] = features
        out_params['camera_parameters'] = CameraParams(
            translation=translation, scale=scale)

        stage_keys = []
        for n in range(num_stages):
            stage_key = f'stage_{n:02d}'
            stage_keys.append(stage_key)
            out_params[stage_key]['faces'] = model_output['faces']
            out_params[stage_key].update(param_dicts[n])

        out_params['stage_keys'] = stage_keys
        out_params[stage_keys[-1]]['proj_joints'] = proj_joints

        return out_params
