# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import os
import os.path as osp
import pickle
import time
import random
from collections import defaultdict

import torch
import torch.utils.data as dutils
import numpy as np
from itertools import compress
from loguru import logger
from scipy.spatial.transform import Rotation as R
from skimage.util.shape import view_as_windows
from ..structures import (Keypoints2D, Keypoints3D,
                          Betas, GlobalRot,
                          HandPose, BoundingBox, Vertices)
from ..utils import (
    create_flip_indices,
    keyps_to_bbox, bbox_to_center_scale,
    KEYPOINT_NAMES_DICT, KEYPOINT_PARTS,
)

from TempCLR.utils import read_img, binarize


class HO3D(dutils.Dataset):
    def __init__(self,
                 data_folder='data/ho3d',
                 split='train',
                 img_folder='rgb',
                 param_folder='meta',
                 model_type='mano',
                 dtype=torch.float32,
                 metrics=None,
                 transforms=None,
                 return_params=True,
                 return_vertices=True,
                 split_size=0.80,
                 split_by_frames=False,
                 dset_scale_factor=2.0,
                 subsequences_length=16,
                 subsequences_stride=1,
                 is_right=True,
                 vertex_flip_correspondences: str = '',
                 **kwargs):

        super(HO3D, self).__init__()
        split_by_frames = True

        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.dset_scale_factor = dset_scale_factor
        # Do we return a right hand?
        self.is_right = is_right
        self.split = split
        self.is_train = 'train' in split
        self.return_params = return_params

        key = ('train' if 'val' in split or 'train' in split else 'evaluation')
        self.data_folder = osp.expandvars(osp.expanduser(data_folder))

        self.subject_path = osp.join(self.data_folder, key)
        self.img_paths = []

        self.global_rot = []
        self.hand_pose = []
        self.shape = []
        self.hand_joints3d = []
        self.intrinsics = []
        self.translation = []
        self.bboxes = []
        self.full_pose = []

        subject_ids = sorted(os.listdir(self.subject_path))

        num_subjects = len(subject_ids)
        if key == 'train' and not split_by_frames:
            if 'train' in split:
                subject_ids = subject_ids[:int(num_subjects * split_size)]
            else:
                subject_ids = subject_ids[int(num_subjects * split_size):]
        elif key != 'train':
            subject_ids = ["SM1", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "AP10", "AP11", "AP12",
                           "AP13", "AP14"]
        start = time.perf_counter()

        global_id = 0
        self.idxes_per_seq = defaultdict(list)

        for subject_id in subject_ids:
            curr_subj_path = osp.join(
                self.subject_path, subject_id, img_folder)
            curr_frames = [
                osp.join(curr_subj_path, fname)
                for fname in sorted(os.listdir(curr_subj_path))
            ]

            mask = [True for _ in range(0, len(curr_frames))]
            if key == 'train' and split_by_frames:
                # Sample %split_size frames from each sequence
                idxs = [i for i in range(0, len(curr_frames))]
                img_idxes = random.sample(idxs, int(len(curr_frames) * split_size))
                mask = [i in img_idxes for i in range(0, len(curr_frames))]

                if 'train' not in split:
                    mask = [not i for i in mask]

            self.img_paths += list(compress(curr_frames, mask))
            param_path = osp.join(self.subject_path, subject_id, param_folder)

            curr_id = 0
            for fname in sorted(os.listdir(param_path)):
                if not fname.endswith(".pkl"):
                    continue

                take_sample = True
                if key == 'train':
                    take_sample = mask[curr_id]

                curr_id += 1
                if take_sample:
                    self.idxes_per_seq[subject_id].append(global_id)
                    global_id += 1
                    with open(osp.join(param_path, fname), 'rb') as f:
                        data = pickle.load(f)
                        if "test" not in split:
                            self.intrinsics.append(data['camMat'])
                            self.shape.append(data['handBeta'])
                            self.hand_joints3d.append(data['handJoints3D'])
                            pose = data['handPose'].reshape(-1, 3)
                            self.full_pose.append(pose)
                            self.translation.append(data['handTrans'].flatten())
                        else:
                            self.bboxes.append(data["handBoundingBox"])

        elapsed = time.perf_counter() - start
        logger.info(f'[{self.name()}] Loading parameters: {elapsed}')
        for path in self.img_paths:
            assert osp.exists(path)

        self.img_paths = np.stack(self.img_paths)
        if "test" not in split:
            self.shape = np.stack(self.shape)
            self.hand_joints3d = np.stack(self.hand_joints3d)
            self.intrinsics = np.stack(self.intrinsics)
            self.translation = np.stack(self.translation)
            self.full_pose = np.stack(self.full_pose)
        else:
            self.bboxes = np.stack(self.bboxes)

        self.num_items = len(self.img_paths)

        self.transforms = transforms
        self.dtype = dtype

        self.return_vertices = return_vertices

        self.source = 'ho3d'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source].copy()
        if not self.is_right:
            self.keypoint_names = [
                name.replace('right', 'left') for name in self.keypoint_names]
        self.flip_indices = np.arange(len(self.keypoint_names))

        self.subseq_start_end_indices = []
        for idx in self.idxes_per_seq.keys():
            indexes = self.idxes_per_seq[idx]
            if len(indexes) < subsequences_length:
                continue
            chunks = view_as_windows(np.array(indexes), (subsequences_length,), step=subsequences_stride)
            for interval in chunks[:, (0, -1)].tolist():
                start_finish = [idx for idx in range(interval[0], interval[1] + 1, 1)]
                self.subseq_start_end_indices.append(start_finish)

    def get_subseq_start_end_indices(self):
        return self.subseq_start_end_indices

    def __repr__(self):
        return f'HO3D( \n\t Split: {self.split}\n)'

    def name(self):
        return f'HO3D/{self.split}'

    def get_num_joints(self):
        return 21

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def get_elements_per_index(self):
        return 1

    @staticmethod
    def project_points(K, xyz):
        coord_change_mat = np.array(
            [[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        xyz = xyz.dot(coord_change_mat.T)

        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:], xyz

    def __getitem__(self, index):
        img_fn = self.img_paths[index]
        img = read_img(img_fn)

        bbox = np.stack(self.bboxes[index], axis=0).astype(np.float32)

        center, scale, bbox_size = bbox_to_center_scale(bbox, dset_scale_factor=2.0)

        target = BoundingBox(bbox, size=img.shape)
        target.add_field('bbox_size', bbox_size)
        target.add_field('center', center)
        target.add_field('orig_center', center)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('scale', scale)

        target.add_field('index', index)
        target.add_field('sequence_id', os.path.dirname(img_fn))

        if self.transforms is not None:
            force_flip = False
            if not self.is_right:
                force_flip = not self.is_right
            full_img, cropped_img, target = self.transforms(
                img, target, force_flip=force_flip)

        return full_img, cropped_img, target, index

    def get_elements_per_index(self):
        return 1
