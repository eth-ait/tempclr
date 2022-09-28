# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import argparse
from omegaconf import OmegaConf
from .defaults import conf as default_conf

def parse_args():
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = 'PyTorch MANO Regressor'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfgs',
                        required=True, nargs='+',
                        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='The configuration of the Detector')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--num-gpus', dest='num_gpus',
                        default=1, type=int,
                        help='Number of gpus')
    parser.add_argument('--backend', dest='backend',
                        default='nccl', type=str,
                        choices=['nccl', 'gloo'],
                        help='Backend used for multi-gpu training')

    cmd_args = parser.parse_args()

    cfg = default_conf.copy()
    for exp_cfg in cmd_args.exp_cfgs:
        if exp_cfg:
            cfg.merge_with(OmegaConf.load(exp_cfg))
    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))

    cfg.network.use_sync_bn = (cfg.network.use_sync_bn and
                               cmd_args.num_gpus > 1)
    cfg.local_rank = cmd_args.local_rank
    cfg.num_gpus = cmd_args.num_gpus

    return cfg
