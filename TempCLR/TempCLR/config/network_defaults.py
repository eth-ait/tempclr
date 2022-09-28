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
from dataclasses import dataclass, make_dataclass, field
from omegaconf import OmegaConf

@dataclass
class HandConditioning:
    wrist_pose: bool = True
    finger_pose: bool = True
    shape: bool = True


@dataclass
class LeakyReLU:
    negative_slope: float = 0.01


@dataclass
class ELU:
    alpha: float = 1.0


@dataclass
class PReLU:
    num_parameters: int = 1
    init: float = 0.25


@dataclass
class Activation:
    type: str = 'relu'
    inplace: bool = True

    leaky_relu: LeakyReLU = LeakyReLU()
    prelu: PReLU = PReLU()
    elu: ELU = ELU()


@dataclass
class BatchNorm:
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True


@dataclass
class GroupNorm:
    num_groups: int = 32
    eps: float = 1e-05
    affine: bool = True


@dataclass
class LayerNorm:
    eps: float = 1e-05
    elementwise_affine: bool = True


@dataclass
class Normalization:
    type: str = 'batch-norm'
    batch_norm: BatchNorm = BatchNorm()
    layer_norm = LayerNorm = LayerNorm()
    group_norm: GroupNorm = GroupNorm()


@dataclass
class HeadConditioning:
    neck_pose: bool = True
    jaw_pose: bool = True
    shape: bool = True
    expression: bool = True


@dataclass
class WeakPerspective:
    regress_scale: bool = True
    regress_translation: bool = True
    mean_scale: float = 0.9


@dataclass
class Perspective:
    regress_translation: bool = False
    regress_rotation: bool = False
    regress_focal_length: bool = False
    focal_length: float = 5000.0


@dataclass
class Camera:
    type: str = 'weak-persp'
    pos_func: str = 'softplus'
    weak_persp: WeakPerspective = WeakPerspective()
    perspective: Perspective = Perspective()


@dataclass
class ResNet:
    replace_stride_with_dilation: Tuple[bool] = (False, False, False)


@dataclass
class HRNet:
    @dataclass
    class Stage:
        num_modules: int = 1
        num_branches: int = 1
        num_blocks: Tuple[int] = (4,)
        num_channels: Tuple[int] = (64,)
        block: str = 'BOTTLENECK'
        fuse_method: str = 'SUM'

    @dataclass
    class SubSample:
        num_layers: int = 3
        num_filters: Tuple[int] = (512,) * num_layers
        kernel_size: int = 7
        norm_type: str = 'bn'
        activ_type: str = 'relu'
        dim: int = 2
        kernel_sizes = [kernel_size] * len(num_filters)
        stride: int = 2
        strides: Tuple[int] = (stride,) * len(num_filters)
        padding: int = 1

    use_old_impl: bool = True
    pretrained_layers: Tuple[str] = ('*',)
    pretrained_path: str = (
        '$CLUSTER_HOME/network_weights/hrnet_v2/hrnetv2_w48_imagenet_pretrained.pth'
    )
    stage1: Stage = Stage()
    stage2: Stage = Stage(num_branches=2, num_blocks=(4, 4),
                          num_channels=(48, 96), block='BASIC')
    stage3: Stage = Stage(num_modules=4, num_branches=3,
                          num_blocks=(4, 4, 4),
                          num_channels=(48, 96, 192),
                          block='BASIC')
    stage4: Stage = Stage(num_modules=3, num_branches=4,
                          num_blocks=(4, 4, 4, 4,),
                          num_channels=(48, 96, 192, 384),
                          block='BASIC',
                          )


@dataclass
class Backbone:
    type: str = 'resnet50'
    pretrained: bool = False
    projection_head: bool = False
    freeze: bool = False
    resnet: ResNet = ResNet()
    hrnet: HRNet = HRNet()

@dataclass
class TemporalBackbone:
    active: bool = False
    seq_len: int = 8
    num_layers: int = 2
    add_linear: bool = True
    use_residual: bool = True
    bidirectional: bool = False
    hidden_size: int = 1024
    freeze: bool = False

@dataclass
class MLP:
    layers: Tuple[int] = (1024, 1024)
    activation: Activation = Activation()
    normalization: Normalization = Normalization()
    preactivated: bool = False
    dropout: float = 0.0
    init_type: str = 'xavier'
    gain: float = 0.01
    bias_init: float = 0.0


@dataclass
class FeatureFusion:
    active: bool = False
    fusion_type: str = 'weighted'

    @dataclass
    class Net:
        type: str = 'mlp'
        mlp: MLP = MLP()

    network: Net = Net()


@dataclass
class LSTM:
    bias: bool = True
    hidden_size: int = 1024


@dataclass
class GRU:
    bias: bool = True
    hidden_size: int = 1024


@dataclass
class RNN:
    type: str = 'lstm'
    layer_dims: Tuple[int] = (1024,)
    init_type: str = 'randn'
    learn_mean: bool = True
    dropout: float = 0.0
    lstm: LSTM = LSTM()
    gru: GRU = GRU()
    mlp: MLP = MLP(layers=tuple(), gain=1.0)


@dataclass
class HMRLike:
    type: str = 'mlp'
    feature_key: str = 'avg_pooling'
    append_params: bool = True
    num_stages: int = 3
    pose_last_stage: bool = True
    detach_mean: bool = False
    learn_mean: bool = False

    backbone: Backbone = Backbone(type='resnet50')
    camera: Camera = Camera()
    mlp: MLP = MLP()
    rnn: RNN = RNN()


@dataclass
class Hand(HMRLike):
    use_photometric: bool = False
    is_right: bool = True
    groups: Tuple[str] = (
        (
            'wrist_pose',
            'hand_pose',
            'camera',
            'betas',
        ),
    )

    @dataclass
    class Renderer:
        topology_path: str = 'data/mano/hand_template.obj'
        uv_size: int = 256
        displacement_path: str = 'data/mano/displacement.npy'
    renderer: Renderer = Renderer()

    use_hand_seg: bool = False
    temporal_backbone: TemporalBackbone = TemporalBackbone()


@dataclass
class Network:
    type: str = 'hand-iterative-model'
    use_sync_bn: bool = False

    hand_add_shape_noise: bool = False
    hand_shape_std: float = 0.0
    hand_shape_prob: float = 0.0

    # Hand noise parameters
    add_hand_pose_noise: bool = False
    hand_pose_std: float = 0.0
    num_hand_components: int = 3
    hand_noise_prob: float = 0.0

    hand_randomize_global_rot: bool = False
    hand_global_rot_max: float = 0.0
    hand_global_rot_min: float = 0.0
    hand_global_rot_noise_prob: float = 0.0

    hmr: HMRLike = HMRLike()
    hand: Hand = Hand()

conf = OmegaConf.structured(Network)
