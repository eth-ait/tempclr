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
from omegaconf import DictConfig

import numpy as np
import torch

from .registry import HAND_HEAD_REGISTRY
from .hand_loss_modules import (
    RegularizerModule as MANORegularizerModule)

from ..common.networks import build_network
from ..common.iterative_regressor import HMRLikeRegressor
from ..common.pose_utils import (
    build_pose_parameterization, PoseParameterization)

from ..body_models import (
    build_hand_model, build_hand_texture)

from ..rendering import SRenderY

from TempCLR.data.structures import StructureList
from TempCLR.utils import (
    Tensor, AppearanceDescription, BlendShapeDescription, StringList)

__all__ = [
    'MANORegressor',
]


@HAND_HEAD_REGISTRY.register()
class MANORegressor(HMRLikeRegressor):
    def __init__(
            self,
            hand_model_cfg: DictConfig,
            network_cfg: DictConfig,
            loss_cfg: DictConfig,
            encoder_cfg: DictConfig,
            temporal_backbone_cfg: DictConfig,
            is_contrastive_training=False,
            batch_size=32,
            img_size: int = 224,
            extra_feat_dim: int = 2048,
            feat_fusion_cfg: Optional[DictConfig] = None,
            dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        ''' MANO Regressor
        '''
        self.use_photometric = network_cfg.get('use_photometric', False)
        logger.info(f'Predict hand texture: {self.use_photometric}')

        self.use_hand_seg = network_cfg.get('use_hand_seg', False)
        logger.info(
            f'Use hand segmentation mask for supervision: {self.use_hand_seg}')

        super(MANORegressor, self).__init__(
            hand_model_cfg, network_cfg, loss_cfg, encoder_cfg, temporal_backbone_cfg,
            batch_size=batch_size, dtype=dtype)

        if self.use_photometric:
            assert 'renderer' in network_cfg, (
                'No renderer config in network config')
            renderer_cfg = network_cfg.get('renderer')

            # Build the differentiable renderer
            self._setup_renderer(renderer_cfg, img_size=img_size)

        if feat_fusion_cfg is None:
            feat_fusion_cfg = DictConfig(dict())
        self.use_feat_fusion = feat_fusion_cfg.get('active', False)
        if self.use_feat_fusion:
            logger.info(feat_fusion_cfg.pretty())
            feat_fusion_type = feat_fusion_cfg.get('type', 'weighted')
            self.feat_fusion_type = feat_fusion_type
            logger.info(f'Feature fusion type: {self.feat_fusion_type}')

            self.detach_fusion_input = feat_fusion_cfg.get(
                'detach_fusion_input', True)
            logger.info(f'Detach fusion input: {self.detach_fusion_input}')
            if feat_fusion_type == 'weighted':
                output_dim = 1
            else:
                raise ValueError(
                    f'Unknown feature fusion type: {feat_fusion_type}')
            network_cfg = feat_fusion_cfg.get('network', {})
            # Build the fusion network
            self.feat_fusion_net = build_network(
                self.feat_dim + extra_feat_dim, output_dim, network_cfg)

    def _setup_renderer(
            self,
            renderer_cfg: DictConfig,
            img_size: int = 224,
    ) -> None:
        ''' Build the differentiable renderer used in training
        '''
        if not self.use_photometric:
            return

        # Load the topology used for trainign
        topology_path = renderer_cfg.get(
            'topology_path', 'data/mano/hand_template.obj')
        uv_size = renderer_cfg.get('uv_size', 1024)
        # Build the renderer
        # TODO: Replace with build function
        self.renderer = SRenderY(
            img_size, obj_filename=topology_path, uv_size=uv_size)

        # Load displacement map
        displacement_path = renderer_cfg.get(
            'displacement_path', 'data/flame/displacements.npy')
        if osp.isfile(displacement_path):
            logger.info(f'Loading displacement map from: {displacement_path}')
            dis = np.load(displacement_path)
            uv_dis = torch.tensor(dis).to(dtype=torch.float32)
        else:
            logger.info('Creating zero displacement map')
            uv_dis = torch.zeros([512, 512], dtype=torch.float32)
        self.register_buffer('uv_dis', uv_dis)

    def _build_model(self, hand_model_cfg, dtype=torch.float32):
        self.hand_model_cfg = hand_model_cfg
        # Build the actual hand model
        model = build_hand_model(hand_model_cfg)
        self.model_type = model.name
        # The config of the model
        self.curr_model_cfg = hand_model_cfg.get(self.model_type, {})
        self.is_right = model.is_right
        logger.info(f'Hand model: {model}')
        return model

    def _build_appearance_space(self, hand_model_cfg):
        ''' Builds an appearance space for the head model
        '''
        # Call the parent method
        appearance_desc = super(MANORegressor, self)._build_appearance_space(
            hand_model_cfg)
        if not self.use_photometric:
            return appearance_desc

        logger.info('Building appearance space!')
        model_type = hand_model_cfg.get('type', 'mano')
        model_cfg = hand_model_cfg.get(model_type)

        self.manotex = build_hand_texture(hand_model_cfg)

        texture_cfg = model_cfg.get('texture', {})
        texture_dim = texture_cfg.get('dim', 50)
        self.texture_dim = texture_dim
        texture_mean = torch.zeros([texture_dim], dtype=torch.float32)
        # Head appearance is modeled by a texture space of FLAME
        texture_desc = AppearanceDescription(
            dim=self.texture_dim, mean=texture_mean)
        appearance_desc['texture'] = texture_desc

        lighting_cfg = model_cfg.get('lighting', {})
        lighting_dim = lighting_cfg.get('dim', 27)
        logger.info(f'Lighting dimension: {lighting_dim}')
        lighting_type = lighting_cfg.get('type', 'sh')
        logger.info(f'Lighting type: {lighting_type}')
        self.lighting_type = lighting_type

        lighting_mean = torch.zeros([lighting_dim], dtype=torch.float32)
        lighting_desc = AppearanceDescription(
            dim=lighting_dim, mean=lighting_mean)
        appearance_desc['lighting'] = lighting_desc
        # Create the description of the texture space
        return appearance_desc

    def _build_pose_space(
            self, hand_model_cfg
    ) -> Dict[str, PoseParameterization]:
        param_desc = super(MANORegressor, self)._build_pose_space(
            hand_model_cfg)
        is_right = self.model.is_right

        wrist_pose_desc = build_pose_parameterization(
            1, **self.curr_model_cfg.wrist_pose)
        self.wrist_pose_decoder = wrist_pose_desc.decoder

        if 'hand_pose' in self.curr_model_cfg:
            hand_pose_cfg = self.curr_model_cfg.get('hand_pose', {})
        else:
            hand_pose_cfg = (self.curr_model_cfg.get('right_hand_pose', {})
                             if is_right else
                             self.curr_model_cfg.get('left_hand_pose', {}))

        pca_basis = (self.model.right_hand_components if self.is_right
                     else self.model.left_hand_components)
        mean_pose = (self.mean_poses_dict.get('right_hand_pose', None)
                     if self.is_right else
                     self.mean_poses_dict.get('left_hand_pose', None))
        hand_pose_desc = build_pose_parameterization(
            num_angles=self.model.num_hand_joints,
            pca_basis=pca_basis, mean=mean_pose,
            **hand_pose_cfg)
        logger.debug('Hand pose decoder: {}', hand_pose_desc.decoder)
        param_desc['hand_pose'] = hand_pose_desc
        self.hand_pose_decoder = hand_pose_desc.decoder

        return {
            'wrist_pose': wrist_pose_desc,
            'hand_pose': hand_pose_desc,
        }

    def _build_blendshape_space(
            self, hand_model_cfg, dtype=torch.float32
    ) -> Dict[str, BlendShapeDescription]:
        blendshape_desc = super(MANORegressor, self)._build_blendshape_space(
            hand_model_cfg)
        num_betas = self.model.num_betas

        shape_mean_path = hand_model_cfg.get('shape_mean_path', '')
        shape_mean_path = osp.expandvars(self.curr_model_cfg.shape_mean_path)
        if osp.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=dtype)
        shape_desc = BlendShapeDescription(dim=num_betas, mean=shape_mean)
        blendshape_desc['betas'] = shape_desc
        return blendshape_desc

    def compute_features(
            self,
            images: Tensor,
            extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        '''
        '''
        feat_dict = self.backbone(images)

        if self.feature_key in feat_dict:
            features = feat_dict[self.feature_key]
        else:
            features = feat_dict['concat']

        if self.temporal_encoder:
            features = features.reshape(features.shape[0] // self.seq_len, self.seq_len, -1)
            features = self.temporal_encoder(features)
            features = features.reshape(-1, features.size(-1))

        if self.projection_head:
            features = self.projection_head(features)

        if not self.use_feat_fusion:
            return features
        assert extra_features is not None, (
            'Feature fusion is active, but extra features are None'
        )

        if self.feat_fusion_type == 'weighted':
            common_dim = min(len(features), len(extra_features))
            fusion_input = torch.cat(
                [features[:common_dim], extra_features[:common_dim]], dim=1)
            # Detach the features
            if self.detach_fusion_input:
                fusion_input = fusion_input.detach()
            fusion_weights = self.feat_fusion_net(fusion_input).view(-1, 1)
            # Return a weighted average for the features
            fused = (
                    fusion_weights * features[:common_dim] +
                    (1 - fusion_weights) *
                    extra_features[:common_dim])
            output = torch.cat([fused, features[common_dim:]], dim=0)
            return output
        else:
            raise NotImplementedError(
                f'Feature fusion "{self.feat_fusion_type}"'
                ' not implemented yet')

    def forward(
            self,
            images: Tensor,
            targets: Optional[StructureList] = None,
            compute_losses: bool = True,
            cond: Optional[Tensor] = None,
            extra_features: Optional[Tensor] = None,
            **kwargs
    ):

        out_params = super(MANORegressor, self).forward(
            images, targets, compute_losses=False, cond=cond,
            extra_features=extra_features,
            **kwargs)

        out_params['is_right'] = self.is_right

        stage_keys = out_params['stage_keys']
        final_stage_out = out_params[stage_keys[-1]]
        vertices = final_stage_out['vertices']
        camera = out_params['camera_parameters']

        if self.use_photometric:
            camera = out_params['camera_parameters']
            scale, translation = camera['scale'], camera['translation']
            # Project the final vertices
            proj_vertices = self.projection(
                vertices, scale=scale, translation=translation)
            # Concatenate the depth dimension
            proj_vertices = torch.cat([
                proj_vertices, vertices[:, :, [2]]], dim=-1)

            # Get the predicted lighting coefficients: B x Light dim
            lighting = final_stage_out['lighting']
            if self.lighting_type == 'sh':
                # B x Num SH x 3 (RGB)
                lighting = lighting.reshape(lighting.shape[0], -1, 3)
            # Get the predicted texture space coefficients: B x Texture Dim
            # Extract the texture code
            texture = final_stage_out['texture']
            # Forward it through the texture layer to get the albedo
            albedo = self.manotex(texture)

            # Render the estimated parameters
            ops = self.renderer(vertices, proj_vertices, albedo, lighting)
            out_params['albedo_images'] = ops['albedo_images'].detach()
            out_params['albedo'] = albedo.detach()

            # Use the alpha mask to keep only the valid pixels
            predicted_images = ops['images'] * ops['alpha_images']
            # Store the images so that we can print them in the summaries
            out_params['predicted_images'] = predicted_images.detach()
            out_params['normal_images'] = ops['normal_images'].detach()
            out_params['normals'] = ops['normals'].detach()
            out_params['transformed_normals'] = ops[
                'transformed_normals'].detach()

        out_params['vertices'] = vertices

        return out_params


@HAND_HEAD_REGISTRY.register()
class MANOGroupRegressor(HMRLikeRegressor):
    def __init__(
            self,
            hand_model_cfg: DictConfig,
            network_cfg: DictConfig,
            loss_cfg: DictConfig,
            img_size: int = 224,
            extra_feat_dim: int = 2048,
            feat_fusion_cfg: Optional[DictConfig] = None,
            dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        ''' MANO Regressor
        '''
        self.use_photometric = network_cfg.get('use_photometric', False)
        logger.info(f'Predict hand texture: {self.use_photometric}')

        super(MANOGroupRegressor, self).__init__(
            hand_model_cfg, network_cfg, loss_cfg, dtype=dtype)

        if self.use_photometric:
            assert 'renderer' in network_cfg, (
                'No renderer config in network config')
            renderer_cfg = network_cfg.get('renderer')

            # Build the differentiable renderer
            self._setup_renderer(renderer_cfg, img_size=img_size)

        if feat_fusion_cfg is None:
            feat_fusion_cfg = DictConfig(dict())
        self.use_feat_fusion = feat_fusion_cfg.get('active', False)
        if self.use_feat_fusion:
            logger.info(feat_fusion_cfg.pretty())
            feat_fusion_type = feat_fusion_cfg.get('type', 'weighted')
            self.feat_fusion_type = feat_fusion_type
            logger.info(f'Feature fusion type: {self.feat_fusion_type}')

            self.detach_fusion_input = feat_fusion_cfg.get(
                'detach_fusion_input', True)
            logger.info(f'Detach fusion input: {self.detach_fusion_input}')
            if feat_fusion_type == 'weighted':
                output_dim = 1
            else:
                raise ValueError(
                    f'Unknown feature fusion type: {feat_fusion_type}')
            network_cfg = feat_fusion_cfg.get('network', {})
            # Build the fusion network
            self.feat_fusion_net = build_network(
                self.feat_dim + extra_feat_dim, output_dim, network_cfg)

    def _setup_renderer(
            self,
            renderer_cfg: DictConfig,
            img_size: int = 224,
    ) -> None:
        ''' Build the differentiable renderer used in training
        '''
        if not self.use_photometric:
            return

        # Load the topology used for trainign
        topology_path = renderer_cfg.get(
            'topology_path', 'data/mano/hand_template.obj')
        uv_size = renderer_cfg.get('uv_size', 1024)
        # Build the renderer
        # TODO: Replace with build function
        self.renderer = SRenderY(
            img_size, obj_filename=topology_path, uv_size=uv_size)

        # Load displacement map
        displacement_path = renderer_cfg.get(
            'displacement_path', 'data/flame/displacements.npy')
        if osp.isfile(displacement_path):
            logger.info(f'Loading displacement map from: {displacement_path}')
            dis = np.load(displacement_path)
            uv_dis = torch.tensor(dis).to(dtype=torch.float32)
        else:
            logger.info('Creating zero displacement map')
            uv_dis = torch.zeros([512, 512], dtype=torch.float32)
        self.register_buffer('uv_dis', uv_dis)

    def _build_model(self, hand_model_cfg, dtype=torch.float32):
        self.hand_model_cfg = hand_model_cfg
        # Build the actual hand model
        model = build_hand_model(hand_model_cfg)
        self.model_type = model.name
        # The config of the model
        self.curr_model_cfg = hand_model_cfg.get(self.model_type, {})
        self.is_right = model.is_right
        logger.info(f'Hand model: {model}')
        return model

    def _build_appearance_space(self, hand_model_cfg):
        ''' Builds an appearance space for the head model
        '''
        # Call the parent method
        appearance_desc = super(
            MANOGroupRegressor, self)._build_appearance_space(
            hand_model_cfg)
        if not self.use_photometric:
            return appearance_desc

        logger.info('Building appearance space!')
        model_type = hand_model_cfg.get('type', 'mano')
        model_cfg = hand_model_cfg.get(model_type)

        self.manotex = build_hand_texture(hand_model_cfg)

        texture_cfg = model_cfg.get('texture', {})
        texture_dim = texture_cfg.get('dim', 50)
        self.texture_dim = texture_dim
        texture_mean = torch.zeros([texture_dim], dtype=torch.float32)
        # Head appearance is modeled by a texture space of FLAME
        texture_desc = AppearanceDescription(
            dim=self.texture_dim, mean=texture_mean)
        appearance_desc['texture'] = texture_desc

        lighting_cfg = model_cfg.get('lighting', {})
        lighting_dim = lighting_cfg.get('dim', 27)
        logger.info(f'Lighting dimension: {lighting_dim}')
        lighting_type = lighting_cfg.get('type', 'sh')
        logger.info(f'Lighting type: {lighting_type}')
        self.lighting_type = lighting_type

        lighting_mean = torch.zeros([lighting_dim], dtype=torch.float32)
        lighting_desc = AppearanceDescription(
            dim=lighting_dim, mean=lighting_mean)
        appearance_desc['lighting'] = lighting_desc
        # Create the description of the texture space
        return appearance_desc

    def _build_pose_space(
            self, hand_model_cfg
    ) -> Dict[str, PoseParameterization]:
        param_desc = super(MANOGroupRegressor, self)._build_pose_space(
            hand_model_cfg)
        is_right = self.model.is_right

        wrist_pose_desc = build_pose_parameterization(
            1, **self.curr_model_cfg.wrist_pose)
        self.wrist_pose_decoder = wrist_pose_desc.decoder

        if 'hand_pose' in self.curr_model_cfg:
            hand_pose_cfg = self.curr_model_cfg.get('hand_pose', {})
        else:
            hand_pose_cfg = (self.curr_model_cfg.get('right_hand_pose', {})
                             if is_right else
                             self.curr_model_cfg.get('left_hand_pose', {}))

        pca_basis = (self.model.right_hand_components if self.is_right
                     else self.model.left_hand_components)
        mean_pose = (self.mean_poses_dict.get('right_hand_pose', None)
                     if self.is_right else
                     self.mean_poses_dict.get('left_hand_pose', None))
        hand_pose_desc = build_pose_parameterization(
            num_angles=self.model.num_hand_joints,
            pca_basis=pca_basis, mean=mean_pose,
            **hand_pose_cfg)
        logger.debug('Hand pose decoder: {}', hand_pose_desc.decoder)
        param_desc['hand_pose'] = hand_pose_desc
        self.hand_pose_decoder = hand_pose_desc.decoder

        return {
            'wrist_pose': wrist_pose_desc,
            'hand_pose': hand_pose_desc,
        }

    def _build_blendshape_space(
            self, hand_model_cfg, dtype=torch.float32
    ) -> Dict[str, BlendShapeDescription]:
        blendshape_desc = super(
            MANOGroupRegressor, self)._build_blendshape_space(hand_model_cfg)
        num_betas = self.model.num_betas

        shape_mean_path = hand_model_cfg.get('shape_mean_path', '')
        shape_mean_path = osp.expandvars(self.curr_model_cfg.shape_mean_path)
        if osp.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=dtype)
        shape_desc = BlendShapeDescription(dim=num_betas, mean=shape_mean)
        blendshape_desc['betas'] = shape_desc
        return blendshape_desc

    def compute_features(
            self,
            images: Tensor,
            extra_features: Optional[Tensor] = None,
    ) -> Tensor:
        '''
        '''
        feat_dict = self.backbone(images)
        features = feat_dict[self.feature_key]
        if not self.use_feat_fusion:
            return features
        assert extra_features is not None, (
            'Feature fusion is active, but extra features are None'
        )

        if self.feat_fusion_type == 'weighted':
            common_dim = min(len(features), len(extra_features))
            fusion_input = torch.cat(
                [features[:common_dim], extra_features[:common_dim]], dim=1)
            # Detach the features
            if self.detach_fusion_input:
                fusion_input = fusion_input.detach()
            fusion_weights = self.feat_fusion_net(fusion_input).view(-1, 1)
            # Return a weighted average for the features
            fused = (
                    fusion_weights * features[:common_dim] +
                    (1 - fusion_weights) *
                    extra_features[:common_dim])
            output = torch.cat([fused, features[common_dim:]], dim=0)
            return output
        else:
            raise NotImplementedError(
                f'Feature fusion "{self.feat_fusion_type}"'
                ' not implemented yet')

    def forward(
            self,
            images: Tensor,
            targets: Optional[StructureList] = None,
            compute_losses: bool = False,
            cond: Optional[Tensor] = None,
            extra_features: Optional[Tensor] = None,
            **kwargs
    ):
        out_params = super(MANOGroupRegressor, self).forward(
            images, targets, compute_losses=False, cond=cond,
            extra_features=extra_features,
            **kwargs)

        out_params['is_right'] = self.is_right

        stage_keys = out_params['stage_keys']
        final_stage_out = out_params[stage_keys[-1]]
        vertices = final_stage_out['vertices']
        camera = out_params['camera_parameters']

        if self.use_photometric:
            camera = out_params['camera_parameters']
            scale, translation = camera['scale'], camera['translation']
            # Project the final vertices
            proj_vertices = self.projection(
                vertices, scale=scale, translation=translation)
            # Concatenate the depth dimension
            proj_vertices = torch.cat([
                proj_vertices, vertices[:, :, [2]]], dim=-1)

            # Get the predicted lighting coefficients: B x Light dim
            lighting = final_stage_out['lighting']
            if self.lighting_type == 'sh':
                # B x Num SH x 3 (RGB)
                lighting = lighting.reshape(lighting.shape[0], -1, 3)
            # Get the predicted texture space coefficients: B x Texture Dim
            # Extract the texture code
            texture = final_stage_out['texture']
            # Forward it through the texture layer to get the albedo
            albedo = self.manotex(texture)

            # Render the estimated parameters
            ops = self.renderer(vertices, proj_vertices, albedo, lighting)
            out_params['albedo_images'] = ops['albedo_images'].detach()
            out_params['albedo'] = albedo.detach()

            # Store the images so that we can print them in the summaries
            predicted_images = ops['images'] * ops['alpha_images']
            out_params['predicted_images'] = predicted_images.detach()
            out_params['normal_images'] = ops['normal_images'].detach()
            out_params['normals'] = ops['normals'].detach()
            out_params['transformed_normals'] = ops[
                'transformed_normals'].detach()

        return out_params
