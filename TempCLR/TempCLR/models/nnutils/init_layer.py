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
import torch.nn as nn
import torch.nn.init as nninit

from loguru import logger


def init_weights(layer,
                 name='',
                 init_type='xavier', distr='uniform',
                 gain=1.0,
                 activ_type='leaky-relu', lrelu_slope=0.01, **kwargs):
    if len(name) < 1:
        name = str(layer)
    logger.info(
        'Initializing {} with {}_{}: gain={}', name, init_type, distr, gain)
    weights = layer.weight
    if init_type == 'xavier':
        if distr == 'uniform':
            nninit.xavier_uniform_(weights, gain=gain)
        elif distr == 'normal':
            nninit.xavier_normal_(weights, gain=gain)
        else:
            raise ValueError(
                'Unknown distribution "{}" for Xavier init'.format(distr))
    elif init_type == 'kaiming':

        activ_type = activ_type.replace('-', '_')
        if distr == 'uniform':
            nninit.kaiming_uniform_(weights, a=lrelu_slope,
                                    nonlinearity=activ_type)
        elif distr == 'normal':
            nninit.kaiming_normal_(weights, a=lrelu_slope,
                                   nonlinearity=activ_type)
        else:
            raise ValueError(
                'Unknown distribution "{}" for Kaiming init'.format(distr))
