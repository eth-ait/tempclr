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

from torchvision.models import resnet

class ProjectionHead(nn.Module):
    def __init__(self, projection_head_input_dim, projection_head_hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(
                projection_head_input_dim,
                projection_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(projection_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                projection_head_hidden_dim,
                output_dim,
                bias=False,
            ),
        )
    
    def forward(self, x):
        return self.head(x)
        
def make_conv_layer(input_dim, cfg):
    num_layers = cfg.get('num_layers')
    num_filters = cfg.num_filters

    expansion = resnet.Bottleneck.expansion

    layers = []
    for i in range(num_layers):
        downsample = nn.Conv2d(input_dim, num_filters, stride=1,
                               kernel_size=1, bias=False)

        layers.append(
            resnet.Bottleneck(input_dim, num_filters // expansion,
                              downsample=downsample)
        )
        input_dim = num_filters
    return nn.Sequential(*layers)


def make_subsample_layers(input_dim, cfg):
    num_filters = cfg.get('num_filters')
    strides = cfg.get('strides')
    kernel_sizes = cfg.get('kernel_sizes')

    param_desc = zip(num_filters, kernel_sizes, strides)
    layers = []
    for out_dim, kernel_size, stride in param_desc:
        layers.append(
            ConvNormActiv(
                input_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                **cfg,
            )
        )
        input_dim = out_dim
    return nn.Sequential(*layers), out_dim


def make_projection_head(projection_head_input_dim, projection_head_hidden_dim, output_dim) -> ProjectionHead:
    return ProjectionHead(projection_head_input_dim, projection_head_hidden_dim, output_dim)
