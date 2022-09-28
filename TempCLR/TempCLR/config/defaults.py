# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Tuple, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf
from .network_defaults import conf as network_cfg, Network
from .datasets_defaults import (
    hand_conf as hand_data_conf, HandConfig,
)
from .body_model import (
    hand_conf, HandModel,
)
from ..utils import StringTuple


@dataclass
class MPJPE:
    alignments: Tuple[str] = ('root', 'procrustes', 'none')
    root_joints: Tuple[str] = tuple()


@dataclass
class Metrics:
    v2v: Tuple[str] = ('procrustes', 'root', 'none')
    mpjpe: MPJPE = MPJPE()
    mpjpe_2d: MPJPE = MPJPE()
    fscores_thresh: Optional[Tuple[float, float]] = (5.0 / 1000, 15.0 / 1000)


@dataclass
class Evaluation:
    hand: Metrics = Metrics(
        mpjpe=MPJPE(root_joints=('right_wrist',)),
        mpjpe_2d=MPJPE(alignments=('none',)),
        fscores_thresh=(5.0 / 1000, 15.0 / 1000)
    )


@dataclass
class Config:
    num_gpus: int = 1
    local_rank: int = 0
    use_cuda: bool = True
    is_training: bool = False
    logger_level: str = 'info'
    use_half_precision: bool = False
    output_folder: str = 'output'
    summary_folder: str = 'summaries'
    results_folder: str = 'results'
    code_folder: str = 'code'
    save_reproj_images: bool = False
    summary_steps: int = 100
    img_summary_steps: int = 100
    hd_img_summary_steps: int = 1000
    create_image_summaries: bool = True
    imgs_per_row: int = 2

    part_key: str = 'hand'
    experiment_tags: StringTuple = tuple()

    @dataclass
    class Degrees:
        hand: Tuple[float] = tuple()

    degrees: Degrees = Degrees()

    pretrained: str = ''

    checkpoint_folder: str = 'checkpoints'

    float_dtype: str = 'float32'
    hand_vertex_ids_path: str = ''

    network: Network = network_cfg
    hand_model: HandModel = hand_conf

    @dataclass
    class Datasets:
        use_equal_sampling: bool = True
        use_packed: bool = False
        hand: HandConfig = hand_data_conf

    datasets: Datasets = Datasets()

    evaluation: Evaluation = Evaluation()


conf = OmegaConf.structured(Config)
