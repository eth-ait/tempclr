# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Union
import os.path as osp
from omegaconf import DictConfig

from loguru import logger
from .hand_models import MANO, HTML


def build_hand_model(body_model_cfg: DictConfig) -> Union[MANO]:
    model_type = body_model_cfg.get('type', 'mano')
    model_folder = osp.expandvars(
        body_model_cfg.get('model_folder', 'data/models'))

    is_right = body_model_cfg.get('is_right', True)
    vertex_ids_path = body_model_cfg.get('vertex_ids_path', '')
    curr_model_cfg = body_model_cfg.get(model_type, {})
    logger.debug(f'Building {model_type.upper()} body model')
    if model_type.lower() == 'mano':
        model_key = 'mano'
        model = MANO
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')

    model_path = osp.join(model_folder, model_key)
    return model(model_folder=model_path,
                 vertex_ids_path=vertex_ids_path,
                 is_right=is_right,
                 **curr_model_cfg)


def build_hand_texture(body_model_cfg: DictConfig):
    ''' Factory function for the head model
    '''
    model_type = body_model_cfg.get('type', 'flame')
    model_folder = osp.expandvars(
        body_model_cfg.get('model_folder', 'data/models'))

    curr_model_cfg = body_model_cfg.get(model_type, {})

    texture_cfg = curr_model_cfg.get('texture', {})

    logger.debug(f'Building {model_type.upper()} body model')
    model_path = osp.join(model_folder, model_type)
    if model_type.lower() == 'mano':
        model_key = 'mano'
        model = HTML
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')
    return model(**texture_cfg)
