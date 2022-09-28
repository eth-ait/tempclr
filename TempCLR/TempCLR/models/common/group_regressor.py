# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import os.path as osp
from typing import Dict, Tuple, Optional
from loguru import logger
from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.nn as nn

from omegaconf import DictConfig

from ..body_models import KeypointTensor

from .networks import build_regressor

from ..backbone import build_backbone
from ..camera import CameraParams, build_cam_proj
from ..common.pose_utils import (
    PoseParameterization)

from TempCLR.data.structures import StructureList
from TempCLR.utils import (
    Tensor, BlendShapeDescription, StringList, AppearanceDescription,
    IntList)

__all__ = [
    'GroupRegressor',
]


class GroupRegressor(nn.Module):
    def __init__(
            self,
            body_model_cfg: DictConfig,
            network_cfg: DictConfig,
            loss_cfg: DictConfig,
            dtype: Optional[torch.dtype] = torch.float32
    ):
        super(GroupRegressor, self).__init__()

        self.predict_face = network_cfg.get('predict_face', False)
        logger.info(f'Predict face: {self.predict_face}')
        self.predict_hands = network_cfg.get('predict_hands', False)
        logger.info(f'Predict hands: {self.predict_hands}')

        joints_to_exclude_from_body = network_cfg.get(
            'joints_to_exclude', ('head', 'left_wrist', 'right_wrist'))
        logger.info(
            f'Joints to exclude from body pose: {joints_to_exclude_from_body}')

        # A list of string list. Each sub-list should contain the parameters
        # that will be grouped together. For example, for a MANO regressor the
        # groups might look like this:
        # - ['camera']
        # - ['lighting', 'texture']
        # - ['wrist_pose', 'hand_pose']
        groups = network_cfg.get('groups')
        # Pose only the final stage of the model to save computation
        self.pose_last_stage = network_cfg.get('pose_last_stage', True)
        logger.info(f'Pose last stage: {self.pose_last_stage}')

        camera_cfg = network_cfg.get('camera', {})
        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']

        self.model = self._build_model(body_model_cfg)

        joint_indices = []
        joint_names_to_index = {}
        for jname in joints_to_exclude_from_body:
            joint_indices.append(
                self.model.keypoint_names.index(jname) - 1)
            joint_names_to_index[jname] = joint_indices[-1]
        self.joint_names_to_index = joint_names_to_index
        self.joint_indices = joint_indices
        self.joints_to_exclude_from_body = joints_to_exclude_from_body

        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        self.camera_scale_func = camera_data['scale_func']

        pose_space = self._build_pose_space(
            body_model_cfg, joints_to_exclude=joints_to_exclude_from_body)

        blendshape_space = self._build_blendshape_space(body_model_cfg)
        appearance_space = self._build_appearance_space(body_model_cfg)

        self.pose_space = pose_space
        self.blendshape_space = blendshape_space
        self.appearance_space = appearance_space

        camera_space = {'dim': camera_param_dim, 'mean': camera_mean}

        param_dict = dict(**pose_space, **blendshape_space, **appearance_space)
        param_dict['camera'] = camera_space

        backbone_cfg = network_cfg.get('backbone', {})
        self.backbone, feat_dims = build_backbone(backbone_cfg)

        self.feature_key = network_cfg.get('feature_key', 'avg_pooling')
        feat_dim = feat_dims[self.feature_key]
        self._feat_dim = feat_dim
        #  self._param_dim = param_dim
        #  self.register_buffer('param_mean', param_mean)

        # Store the groups
        self.groups = groups

        # A list of dicts. Each element is a dictionary, whose keys are the
        # parameters names and the values are the indices on the flat parameter
        # vector
        param_indices = []

        # Find the indices we need to keep
        self.body_indices_to_keep = [
            ii - 1 for ii in np.arange(1, self.model.num_body_joints + 1)
            if (ii - 1) not in joint_indices]

        # Store the number of body joints
        self.num_body_joints = self.model.num_body_joints

        # The list of regressors
        self.regressors = nn.ModuleList()
        # Go through each group
        for group in groups:
            logger.info(group)
            start = 0
            curr_param_indices = {}
            mean_lst = []
            # For every parameter name in the group
            for param_name in group:
                if param_name in param_dict:
                    desc = param_dict[param_name]

                    # Get the mean pose parameter
                    mean = desc['mean']
                    if param_name == 'body_pose':
                        # If we are processing the body pose, then remove all
                        # joints we wish to exclude
                        mean = mean.reshape(-1, desc['ind_dim'])
                        mean = mean[self.body_indices_to_keep]

                    mean_lst.append(mean.view(-1))

                    # Store the indices used to extract the parameter from the
                    # flattened vector
                    indices = list(range(start, start + mean_lst[-1].numel()))
                    curr_param_indices[param_name] = indices
                    # Update the start value
                    start += mean_lst[-1].numel()

                elif param_name in joints_to_exclude_from_body:
                    desc = param_dict['body_pose']
                    mean = desc['mean'].reshape(-1, desc['ind_dim'])
                    mean_lst.append(
                        mean[joint_names_to_index[param_name]].view(-1))

                    # Store the indices used to extract the parameter from the
                    # flattened vector
                    indices = list(range(start, start + mean_lst[-1].numel()))
                    curr_param_indices[f'{param_name}_pose'] = indices
                    # Update the start value
                    start += mean_lst[-1].numel()
                else:
                    raise KeyError(
                        f'{param_name} not in param_dict or joints from body')

            # Save the parameter indices
            param_indices.append(curr_param_indices)
            #  Create the flattened mean
            param_mean = torch.cat(mean_lst).view(1, -1)
            # Get the dimension of the paramters
            param_dim = param_mean.numel()
            # Create the iterative regressor
            regressor, num_stages = build_regressor(
                network_cfg, feat_dim, param_dim, param_mean=param_mean)
            self.regressors.append(regressor)
            #  logger.info(f'Regressor network: {regressor}')

        self.param_indices = param_indices
        self._num_stages = num_stages

        # Build the losses
        self._build_losses(loss_cfg)

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

    def _build_pose_space(
            self,
            body_model_cfg: DictConfig,
            joints_to_exclude: Optional[StringList] = None
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

    def _build_parameter_losses(
            self, loss_cfg
    ):
        raise NotImplementedError

    def compute_features(
            self,
            images: Tensor,
            extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        ''' Computes features for the current input
        '''
        feat_dict = self.backbone(images)
        features = feat_dict[self.feature_key]
        return features

    def flat_params_to_dict(
            self,
            param_tensor: Tensor,
            param_indices: Dict[str, IntList],
    ) -> Dict[str, Tensor]:
        ''' Convert a flat parameter tensor to a dictionary of parameters
        '''
        device = param_tensor.device
        param_dict = {}
        for name, indices in param_indices.items():
            if not torch.is_tensor(indices):
                indices = torch.tensor(
                    indices, dtype=torch.long, device=device)
            param_dict[name] = torch.index_select(param_tensor, 1, indices)
            logger.debug(f'{name}: {param_dict[name].shape}')
        return param_dict

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

        num_stages = self.num_stages

        # Compute the features
        features = self.compute_features(images, extra_features=extra_features)

        param_dicts = [{} for n in range(self.num_stages)]
        # Iterate through all the regressors
        for ii, regressor in enumerate(self.regressors):
            # Get the dictionary of names to indices in the flattened tensor
            curr_param_indices = self.param_indices[ii]
            # Predict the parameters using the current features
            regr_output = regressor(features)
            if torch.is_tensor(regr_output):
                parameters = [regr_output]
            elif isinstance(regr_output, (tuple, list)):
                parameters = regr_output[0]

            # Go through all the stages
            for list_id, params in enumerate(parameters):
                # Convert the flat parameter tensor to a dictionary of tensors
                curr_params_dict = self.flat_params_to_dict(
                    params, curr_param_indices)
                out_dict = {}
                # Copy parameters. If it is a pose parameter, then decode it
                for key, val in curr_params_dict.items():
                    if hasattr(self, f'{key}_decoder'):
                        decoder = getattr(self, f'{key}_decoder')
                        out_dict[key] = decoder(val)
                        out_dict[f'raw_{key}'] = val.clone()
                    elif (key.replace('_pose', '')
                          in self.joints_to_exclude_from_body):
                        decoder = getattr(self, f'body_pose_decoder')
                        out_dict[key] = decoder(val)
                        out_dict[f'raw_{key}'] = val.clone()
                    else:
                        out_dict[key] = val
                # Add the estimated parameters to the corresponding dictionary
                param_dicts[list_id].update(out_dict)

        merged_param_dicts = []
        for n, param_dict in enumerate(param_dicts):
            merged_param_dicts.append({})
            # Go through the parameters
            for key, val in param_dict.items():
                # Ignore ra
                if 'raw' in key:
                    merged_param_dicts[-1][key] = val
                # Build the body pose from the predicted vector and the
                # excluded joints
                elif key == 'body_pose':
                    body_pose = torch.eye(
                        3, device=device, dtype=dtype).reshape(
                        1, 1, 3, 3).expand(
                        batch_size, self.num_body_joints, -1, -1)
                    body_pose = body_pose.contiguous()
                    body_pose[:, self.body_indices_to_keep] = val

                    # Add the joints excluded if they are predicted
                    for jname, jindex in self.joint_names_to_index.items():
                        pose_key = f'{jname}_pose'
                        if key in param_dict:
                            body_pose[:, jindex] = param_dict[
                                pose_key].reshape(batch_size, 3, 3)
                    merged_param_dicts[-1][key] = body_pose
                else:
                    merged_param_dicts[-1][key] = val

        # Compute the body surface using the current estimation of the pose and
        # the shape
        model_output = self.model(
            get_skin=True, return_shaped=True, **merged_param_dicts[-1])

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
        if 'camera' in param_dicts[-1]:
            camera_params = param_dicts[-1]['camera']
            scale = camera_params[:, 0].view(-1, 1)
            translation = camera_params[:, 1:3]
        else:
            scale = torch.full([batch_size, 1], 0.3742, dtype=dtype,
                               device=device)
            translation = torch.zeros([batch_size, 2], dtype=dtype,
                                      device=device)
        # Pass the predicted scale through exp() to make sure that the
        # scale values are always positive
        scale = self.camera_scale_func(scale)

        est_joints3d = out_params[f'stage_{num_stages - 1:02d}']['joints']
        # Project the joints on the image plane
        proj_joints = self.projection(
            est_joints3d, scale=scale, translation=translation)

        # Add the projected joints
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
            out_params[stage_key].update(merged_param_dicts[n])

        out_params['stage_keys'] = stage_keys
        out_params[stage_keys[-1]]['proj_joints'] = proj_joints

        return out_params
