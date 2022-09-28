# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import time
import numpy as np
import torch

from loguru import logger


class Timer(object):
    def __init__(self, name='', sync=False, verbose=False):
        super(Timer, self).__init__()
        self.elapsed = []
        self.name = name
        self.sync = sync
        self.verbose = verbose

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def print(self):
        logger.info(f'[{self.name}]: {np.mean(self.elapsed):.3f}')

    def __exit__(self, type, value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        self.elapsed.append(elapsed)
        if self.verbose:
            logger.info(
                f'[{self.name}]: {elapsed:.3f}, {np.mean(self.elapsed):.3f}')
