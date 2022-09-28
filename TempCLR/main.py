# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""

import OpenGL
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
if os.environ.get('TEST_NO_ACCELERATE'):
    OpenGL.USE_ACCELERATE = False

import sys
import os.path as osp
from tqdm import tqdm
import torch

from threadpoolctl import threadpool_limits
from loguru import logger

from TempCLR.utils.checkpointer import Checkpointer
from TempCLR.data import make_all_data_loaders
from TempCLR.models.build import build_model
from TempCLR.config import parse_args
from TempCLR.evaluation import build as build_evaluator

DEFAULT_FORMAT = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |'
                  ' <level>{level: <8}</level> |'
                  ' <cyan>{name}</cyan>:<cyan>{function}</cyan>:'
                  '<cyan>{line}</cyan> - <level>{message}</level>')

DIST_FORMAT = ('<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> |'
               ' <level>{{level: <8}}</level> |'
               ' <red><bold>Rank {rank: <3} </bold></red> |'
               ' <cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:'
               '<cyan>{{line}}</cyan> - <level>{{message}}</level>')


def main():
    exp_cfg = parse_args()

    device = torch.device(f'cuda')

    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    local_rank = 0
    distributed = False
    output_folder = osp.expandvars(exp_cfg.output_folder)
    save_images = exp_cfg.save_reproj_images
    logger_format = DEFAULT_FORMAT
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               format=logger_format,
               colorize=True)

    logger.info(f'Rank = {local_rank}: device = {device}')

    model_dict = build_model(exp_cfg)
    model = model_dict['network']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        logger.opt(ansi=True).info(
            f'<bold>{name} </bold>:'
            f' {str(param.requires_grad)}'
            f', {str(tuple(param.shape))}')

    # Copy the model to the correct device
    model = model.to(device=device)

    checkpoint_folder = osp.join(output_folder, exp_cfg.checkpoint_folder)
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpointer = Checkpointer(
        model, save_dir=checkpoint_folder, pretrained=exp_cfg.pretrained,
        distributed=distributed, rank=local_rank)

    code_folder = osp.join(output_folder, exp_cfg.code_folder)
    os.makedirs(code_folder, exist_ok=True)

    # Set the model to evaluation mode
    data_loaders = make_all_data_loaders(exp_cfg, split='test')

    arguments = {'iteration': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]

    model.eval()

    evaluator = build_evaluator(
        exp_cfg, rank=local_rank, distributed=distributed, save_imgs=save_images)

    with evaluator:
        evaluator.run(model, data_loaders, exp_cfg, device,
                      step=arguments['iteration'])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
