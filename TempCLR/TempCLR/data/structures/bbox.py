# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from __future__ import absolute_import
from __future__ import print_function
from copy import deepcopy

import numpy as np

import cv2

import torch
from loguru import logger

#  from ...utils.torch_utils import to_tensor
from ..utils import bbox_area, bbox_to_wh
from .abstract_structure import AbstractStructure
from ...utils.transf_utils import get_transform

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoundingBox(AbstractStructure):
    def __init__(self, bbox, size, flip_axis=0, transform=True, **kwargs):
        super(BoundingBox, self).__init__()
        self.bbox = bbox
        self.flip_axis = flip_axis
        self.size = size
        self.transform = transform

    def __repr__(self):
        msg = ', '.join(map(str, map(float, self.bbox)))
        return f'Bounding box: {msg}'

    def to_tensor(self, *args, **kwargs):
        if not torch.is_tensor(self.bbox):
            self.bbox = torch.from_numpy(self.bbox)

        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)

    def rotate(self, rot=0, *args, **kwargs):
        (h, w) = self.size[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        if self.transform:
            bbox = self.bbox.copy().reshape(4)
            xmin, ymin, xmax, ymax = bbox
            points = np.array(
                [[xmin, ymin],
                 [xmin, ymax],
                 [xmax, ymin],
                 [xmax, ymax]],
            )

            bbox = (np.dot(points, M[:2, :2].T) + M[:2, 2] + 1)
            xmin, ymin = np.amin(bbox, axis=0)
            xmax, ymax = np.amax(bbox, axis=0)

            new_bbox = np.array([xmin, ymin, xmax, ymax])
        else:
            new_bbox = self.bbox.copy().reshape(4)

        bbox_target = type(self)(
            new_bbox, size=(nH, nW, 3), transform=self.transform)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.rotate(rot=rot, *args, **kwargs)
            bbox_target.add_field(k, v)

        return bbox_target

    def crop(self, center, scale, rot=0, crop_size=224, *args, **kwargs):
        if self.transform:
            bbox = self.bbox.copy().reshape(4)
            xmin, ymin, xmax, ymax = bbox
            points = np.array(
                [[xmin, ymin],
                 [xmin, ymax],
                 [xmax, ymin],
                 [xmax, ymax]],
            )
            transf = get_transform(
                center, scale, (crop_size, crop_size), rot=rot)

            bbox = (np.dot(points, transf[:2, :2].T) + transf[:2, 2] + 1)
            xmin, ymin = np.amin(bbox, axis=0)
            xmax, ymax = np.amax(bbox, axis=0)

            new_bbox = np.array([xmin, ymin, xmax, ymax])
        else:
            new_bbox = self.bbox.copy().reshape(4)

        bbox_target = type(self)(new_bbox, size=(crop_size, crop_size),
                                 transform=self.transform)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(center=center, scale=scale,
                           crop_size=crop_size, rot=rot,
                           *args, **kwargs)
            bbox_target.add_field(k, v)

        return bbox_target

    def shift(self, vector, *args, **kwargs):
        if torch.is_tensor(self.bbox):
            bbox = self.bbox.clone().reshape(4)
        else:
            bbox = self.bbox.copy().reshape(4)

        xmin, ymin, xmax, ymax = bbox
        if torch.is_tensor(self.bbox):
            new_bbox = torch.tensor(
                [xmin + vector[0],
                 ymin + vector[1],
                 xmax + vector[0],
                 ymax + vector[1]])
        else:
            new_bbox = np.array(
                [xmin + vector[0],
                 ymin + vector[1],
                 xmax + vector[0],
                 ymax + vector[1]])

        bbox_target = type(self)(new_bbox, size=self.size,
                                 flip_axis=self.flip_axis,
                                 transform=self.transform)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.shift(vector, *args, **kwargs)
            bbox_target.add_field(k, v)

        return bbox_target

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig)
                       for s, s_orig in zip(size, self.size))
        ratio_h, ratio_w, _ = ratios
        bbox = self.bbox.copy().reshape(4)
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = xmin * ratio_w, xmax * ratio_w
        ymin, ymax = ymin * ratio_h, ymax * ratio_h

        new_bbox = np.array([xmin, ymin, xmax, ymax])
        bbox_target = type(self)(new_bbox, size=size,
                                 flip_axis=self.flip_axis,
                                 transform=self.transform)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(size=size, *args, **kwargs)
            bbox_target.add_field(k, v)

        return bbox_target

    def __len__(self):
        return 1

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        xmin, xmax = self.bbox.reshape(-1)[[0, 2]]
        #  logger.info(f'Before: {xmin}, {xmax}')
        W = self.size[1]
        new_xmin = W - xmax
        new_xmax = W - xmin
        new_ymin, new_ymax = self.bbox[[1, 3]]
        #  logger.info(f'After: {xmin}, {xmax}')

        if torch.is_tensor(self.bbox):
            flipped_bbox = torch.tensor(
                [new_xmin, new_ymin, new_xmax, new_ymax],
                dtype=self.bbox.dtype, device=self.bbox.device)
        else:
            flipped_bbox = np.array(
                [new_xmin, new_ymin, new_xmax, new_ymax],
                dtype=self.bbox.dtype)

        bbox_target = type(self)(flipped_bbox, self.size,
                                 transform=self.transform)
        #  logger.info(bbox_target)
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            bbox_target.add_field(k, v)

        bbox_target.add_field('is_flipped', True)
        return bbox_target

    def to(self, *args, **kwargs):
        bbox_tensor = self.bbox
        if not torch.is_tensor(self.bbox):
            bbox_tensor = torch.tensor(bbox_tensor)
        bbox_target = type(self)(bbox_tensor.to(*args, **kwargs), self.size,
                                 transform=self.transform)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            bbox_target.add_field(k, v)
        return bbox_target
