# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Tuple
from dataclasses import dataclass


@dataclass
class FScores:
    hand: Tuple[float] = (5.0 / 1000, 15.0 / 1000)
    head: Tuple[float] = (5.0 / 1000, 15.0 / 1000)


@dataclass
class Variable:
    create: bool = True
    requires_grad: bool = True


@dataclass
class Pose(Variable):
    type: str = 'cont-rot-repr'


@dataclass
class Normalization:
    type: str = 'batch-norm'
    affine: bool = True
    elementwise_affine: bool = True


@dataclass
class LeakyRelu:
    negative_slope: float = 0.01


@dataclass
class Activation:
    type: str = 'relu'
    leaky_relu: LeakyRelu = LeakyRelu()


@dataclass
class RealNVP:
    num_flow_blocks: int = 2
    coupling_type: str = 'half-affine'
    normalization: Normalization = Normalization()
    activation: Activation = Activation()
    use_fc: bool = False
    mask_type: str = 'top'
    shuffle_mask: bool = True

    hidden_features: int = 256
    num_blocks_per_layer: int = 2
    use_volume_preserving: bool = False
    dropout_probability: float = 0.0
    batch_norm_within_layers: bool = False
    batch_norm_between_layers: bool = False
    activation: Activation = Activation()

    @dataclass
    class Coupling:
        hidden_dims: Tuple[int] = (256, 256)
        dropout: float = 0.0
        normalization: Normalization = Normalization()
        activation: Activation = Activation()

    coupling: Coupling = Coupling()


@dataclass
class NSF:
    num_flow_blocks: int = 2
    @dataclass
    class Transf:
        type: str = 'rq-coupling'
        hidden_features: int = 256
        num_transform_blocks: int = 2
        dropout: float = 0.0
        use_batch_norm: bool = False
        num_bins: int = 8
        tail_bound: int = 3
        apply_unconditional_transform: bool = True
    transf: Transf = Transf()

    @dataclass
    class Linear:
        type: str = 'lu'
    linear: Linear = Linear()


@dataclass
class MAF:
    hidden_features: int = 256
    num_layers: int = 5
    num_blocks_per_layer: int = 1
    use_residual_blocks: bool = False
    use_random_masks: bool = True
    use_random_permutations: bool = True
    dropout_probability: float = 0.0
    batch_norm_between_layers: bool = False
    batch_norm_within_layers: bool = False


@dataclass
class Autoregressive:
    type: str = 'real-nvp'
    input_dim: int = 21 * 6
    real_nvp: RealNVP = RealNVP()
    maf: MAF = MAF()
    nsf: NSF = NSF()
    ckpt: str = ''
