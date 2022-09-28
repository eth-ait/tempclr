# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import NewType, List, Union, Tuple

from .abstract_structure import AbstractStructure
from .keypoints import Keypoints2D, Keypoints3D

from .betas import Betas
from .global_rot import GlobalRot
from .body_pose import BodyPose
from .hand_pose import HandPose

from .vertices import Vertices
from .joints import Joints
from .bbox import BoundingBox

from .image_list import ImageList, ImageListPacked, to_image_list
from .points_2d import Points2D

StructureList = NewType('StructureList', List[AbstractStructure])
