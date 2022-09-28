# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Union, List

import numpy as np

import PIL.Image as pil_img


class ImageList:
    def __init__(self, images: List):
        assert isinstance(images, (list, tuple))

    def to_tensor(self):
        pass
