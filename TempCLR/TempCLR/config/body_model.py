# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from dataclasses import dataclass
from omegaconf import OmegaConf
from .utils import Variable, Pose


@dataclass
class PCA:
    num_comps: int = 12
    flat_hand_mean: bool = False


@dataclass
class Clusters:
    fname: str = 'data/clusters'
    tau: float = 1.0


@dataclass
class PoseClusters(Pose):
    clusters: Clusters = Clusters()


@dataclass
class PoseWithPCA(Pose):
    pca: PCA = PCA()


@dataclass
class PoseWithPCAAndClusters(PoseWithPCA, PoseClusters):
    pass


@dataclass
class Shape(Variable):
    num: int = 10


@dataclass
class Expression(Variable):
    num: int = 10


@dataclass
class Texture(Variable):
    dim: int = 50
    path: str = 'data/flame/texture.npz'


@dataclass
class Lighting(Variable):
    dim: int = 27
    type: str = 'sh'


@dataclass
class AbstractBodyModel:
    extra_joint_path: str = ''
    v_template_path: str = ''
    mean_pose_path: str = ''
    shape_mean_path: str = ''
    use_compressed: bool = True
    gender = 'neutral'
    learn_joint_regressor: bool = False


@dataclass
class MANO(AbstractBodyModel):
    betas: Shape = Shape()
    wrist_pose: Pose = Pose()
    hand_pose: PoseWithPCAAndClusters = PoseWithPCAAndClusters()
    translation: Variable = Variable()
    texture: Texture = Texture()
    lighting: Lighting = Lighting()


@dataclass
class HandModel:
    type: str = 'mano'
    is_right: bool = True
    model_folder: str = 'models'
    vertex_ids_path: str = ''

    mano: MANO = MANO()


hand_conf = OmegaConf.structured(HandModel)
