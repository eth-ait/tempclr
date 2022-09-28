# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import List, Tuple, Union
import os.path as osp

from loguru import logger
import functools
import torch
import torch.utils.data as dutils
from . import datasets
from .structures import (StructureList,
                         ImageList, ImageListPacked)
from .transforms import build_transforms
from TempCLR.utils import Tensor, TensorList

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}


def make_data_sampler(dataset):
    return dutils.SequentialSampler(dataset)


def make_hand_dataset(name, dataset_cfg, transforms,
                      **kwargs):
    if name == 'freihand':
        obj = datasets.FreiHand
    elif name == 'ho3d':
        obj = datasets.HO3D
    else:
        raise ValueError(f'Unknown dataset: {name}')

    logger.info(f'Building dataset: {name}')
    args = dict(**dataset_cfg[name])
    args.update(kwargs)
    vertex_flip_correspondences = osp.expandvars(dataset_cfg.get(
        'vertex_flip_correspondences', ''))

    dset_obj = obj(transforms=transforms, hand_only=True,
                   vertex_flip_correspondences=vertex_flip_correspondences,
                   **args)

    logger.info(f'Created dataset: {dset_obj.name()}')
    return dset_obj


class MemoryPinning(object):
    def __init__(
            self,
            full_img_list: Union[ImageList, List[Tensor]],
            images: Tensor,
            targets: StructureList
    ):
        super(MemoryPinning, self).__init__()
        self.img_list = full_img_list
        self.images = images
        self.targets = targets

    def pin_memory(
            self
    ) -> Tuple[Union[ImageList, ImageListPacked, TensorList],
               Tensor, StructureList]:
        if self.img_list is not None:
            if isinstance(self.img_list, (ImageList, ImageListPacked)):
                self.img_list.pin_memory()
            elif isinstance(self.img_list, (list, tuple)):
                self.img_list = [x.pin_memory() for x in self.img_list]
        return (
            self.img_list,
            self.images.pin_memory(),
            self.targets,
        )


def collate_batch(
        batch,
        return_full_imgs=False,
        pin_memory=False
):
    if return_full_imgs:
        images, cropped_images, targets, _ = zip(*batch)
    else:
        _, cropped_images, targets, _ = zip(*batch)

    out_targets = []
    for t in targets:
        if t is None:
            continue
        if type(t) == list:
            out_targets += t
        else:
            out_targets.append(t)
    out_cropped_images = []
    for img in cropped_images:
        if img is None:
            continue
        if torch.is_tensor(img):
            if len(img.shape) < 4:
                img.unsqueeze_(dim=0)
            out_cropped_images.append(img)
        elif isinstance(img, (list, tuple)):
            for d in img:
                d.unsqueeze_(dim=0)
                out_cropped_images.append(d)

    if len(out_cropped_images) < 1:
        return None, None, None

    full_img_list = None
    if return_full_imgs:
        full_img_list = images

    out_cropped_images = torch.cat(out_cropped_images)

    if pin_memory:
        return MemoryPinning(
            full_img_list,
            out_cropped_images,
            out_targets
        )
    else:
        return full_img_list, out_cropped_images, out_targets


def make_data_loader(dataset, batch_size=32, num_workers=0,
                     is_train=True, sampler=None, collate_fn=None,
                     batch_sampler=None, pin_memory=False,
                     ):
    if batch_sampler is None:
        sampler = make_data_sampler(dataset)

    if batch_sampler is None:
        assert sampler is not None, (
            'Batch sampler and sampler can\'t be "None" at the same time')
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True and is_train,
            pin_memory=pin_memory,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            pin_memory=pin_memory,
        )
    return data_loader


def make_all_data_loaders(
        exp_cfg,
        split='test',
        return_hand_full_imgs=False,
        enable_augment=True,
        **kwargs
):
    dataset_cfg = exp_cfg.get('datasets', {})

    hand_dsets_cfg = dataset_cfg.get('hand', {})
    hand_dset_names = hand_dsets_cfg.get('splits', {})[split]
    hand_transfs_cfg = hand_dsets_cfg.get('transforms', {})
    hand_num_workers = hand_dsets_cfg.get(
        'num_workers', DEFAULT_NUM_WORKERS).get(split, 0)

    hand_transforms = build_transforms(
        hand_transfs_cfg, is_train=False,
        enable_augment=enable_augment,
        return_full_imgs=return_hand_full_imgs)

    if hand_transforms:
        logger.info(
            'Hand transformations: \n{}',
            '\n'.join(list(map(str, hand_transforms))))
    else:
        logger.info(
            'Fixed Hand Transformation per Sequence')

    hand_datasets = []
    for dataset_name in hand_dset_names:
        dset = make_hand_dataset(dataset_name, hand_dsets_cfg,
                                 transforms=hand_transforms,
                                 is_train=False, split=split, **kwargs)
        hand_datasets.append(dset)

    hand_batch_size = hand_dsets_cfg.get('batch_size')

    hand_collate_fn = functools.partial(
        collate_batch,
        return_full_imgs=return_hand_full_imgs)

    hand_data_loaders = []
    for hand_dataset in hand_datasets:
        hand_data_loaders.append(
            make_data_loader(hand_dataset, batch_size=hand_batch_size,
                             num_workers=hand_num_workers,
                             is_train=False,
                             batch_sampler=None,
                             collate_fn=hand_collate_fn,
                             ))

    return {
        'hand': hand_data_loaders
    }
