# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from fvcore.common.registry import Registry


HAND_HEAD_REGISTRY = Registry('HAND_HEAD_REGISTRY')
HAND_HEAD_REGISTRY.__doc__ = """
Registry for the hand prediction heads, which predict a 3D head/face
from a single image.
"""
