# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import sys
import os
import os.path as osp
import pickle

import json
import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..structures import (Keypoints2D, Keypoints3D,
                          Betas, GlobalRot,
                          BoundingBox,
                          HandPose, Vertices)
from ..utils import KEYPOINT_NAMES_DICT

from TempCLR.utils import read_img, binarize

FOLDER_MAP_FNAME = 'folder_map.pkl'

IMG_SIZE = 224
REF_BOX_SIZE = 200


class FreiHand(dutils.Dataset):
    def __init__(
            self,
            data_folder='data/freihand',
            mask_folder='data/freihand/masks',
            hand_only=True,
            split='train',
            dtype=torch.float32,
            metrics=None,
            transforms=None,
            return_masks=True,
            return_params=True,
            return_vertices=True,
            return_shape=True,
            file_format='json',
            dset_scale_factor=2.0,
            split_size=0.8,
            is_right=True,
            vertex_flip_correspondences: str = '',
            **kwargs
    ):

        super(FreiHand, self).__init__()

        assert hand_only, 'FreiHand can only be used as a hand dataset'
        if metrics is None:
            metrics = []
        self.metrics = metrics

        self.is_right = is_right
        logger.info(f'Freihand: return right hand = {self.is_right}')
        self.bc, self.closest_faces = None, None
        if not self.is_right:
            vertex_flip_correspondences = osp.expandvars(
                vertex_flip_correspondences)
            err_msg = (
                    'Vertex flip correspondences path does not exist:' +
                    f' {vertex_flip_correspondences}'
            )
            assert osp.exists(vertex_flip_correspondences), err_msg
            logger.info(vertex_flip_correspondences)
            flip_data = np.load(vertex_flip_correspondences)
            self.bc = flip_data['bc']
            self.closest_faces = flip_data['closest_faces']

        self.split = split
        self.is_train = 'train' in split
        self.return_params = return_params
        self.return_vertices = return_vertices
        self.dset_scale_factor = dset_scale_factor

        self.mask_folder = osp.expandvars(mask_folder)
        self.return_masks = return_masks and osp.exists(self.mask_folder)
        logger.info(f'Return masks: {self.return_masks}')

        self.return_shape = return_shape
        key = ('training' if 'val' in split or 'train' in split else
               'evaluation')
        self.data_folder = osp.expandvars(osp.expanduser(data_folder))
        self.img_folder = osp.join(self.data_folder, key, 'rgb')
        self.transforms = transforms
        self.dtype = dtype

        intrinsics_path = osp.join(self.data_folder, f'{key}_K.json')
        param_path = osp.join(self.data_folder, f'{key}_mano.json')
        xyz_path = osp.join(self.data_folder, f'{key}_xyz.json')
        vertices_path = osp.join(self.data_folder, f'{key}_verts.json')

        start = time.perf_counter()

        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
        with open(param_path, 'r') as f:
            param = json.load(f)
        with open(xyz_path, 'r') as f:
            xyz = json.load(f)
        if self.return_vertices:
            with open(vertices_path, 'r') as f:
                vertices = json.load(f)

        elapsed = time.perf_counter() - start
        logger.info(f'Loading parameters: {elapsed}')

        mean_pose_path = os.path.join(os.environ.get("MEAN_POSE_PATH"), "all_means.pkl")
        mean_poses_dict = {}
        if osp.exists(mean_pose_path):
            logger.info('Loading mean pose from: {} ', mean_pose_path)
            with open(mean_pose_path, 'rb') as f:
                mean_poses_dict = pickle.load(f)

        num_green_bg = len(xyz)
        if self.split != 'test':
            #  num_items = len(xyz) * 4
            # For green background images
            train_idxs = np.arange(0, int(split_size * num_green_bg))
            val_idxs = np.arange(int(split_size * num_green_bg), num_green_bg)

            all_train_idxs = []
            all_val_idxs = []
            for idx in range(4):
                all_val_idxs.append(val_idxs + num_green_bg * idx)
                all_train_idxs.append(train_idxs + num_green_bg * idx)
            self.train_idxs = np.concatenate(all_train_idxs)
            self.val_idxs = np.concatenate(all_val_idxs)

        if split == 'train':
            self.img_idxs = self.train_idxs
            self.param_idxs = self.train_idxs % num_green_bg
            self.start = 0
        elif split == 'val':
            self.img_idxs = self.val_idxs
            self.param_idxs = self.val_idxs % num_green_bg
            #  self.start = len(self.train_idxs)
        elif 'test' in split:
            self.img_idxs = np.arange(len(intrinsics))
            self.param_idxs = np.arange(len(intrinsics))

        self.num_items = len(self.img_idxs)

        self.intrinsics = intrinsics

        xyz = np.asarray(xyz, dtype=np.float32)
        param = np.asarray(param, dtype=np.float32).reshape(len(xyz), -1)
        if self.return_vertices:
            vertices = np.asarray(vertices, dtype=np.float32)

        right_hand_mean = mean_poses_dict['right_hand_pose']['aa'].squeeze()
        self.poses = param[:, :48].reshape(num_green_bg, -1, 3)
        self.poses[:, 1:] += right_hand_mean[np.newaxis]
        self.betas = param[:, 48:58].copy()

        if self.return_vertices:
            self.vertices = vertices
        self.xyz = xyz

        folder_map_fname = osp.expandvars(
            osp.join(self.data_folder, split, FOLDER_MAP_FNAME))
        self.use_folder_split = osp.exists(folder_map_fname)
        if self.use_folder_split:
            self.img_folder = osp.join(self.data_folder, split)
            with open(folder_map_fname, 'rb') as f:
                data_dict = pickle.load(f)
            self.items_per_folder = max(data_dict.values())

        self.source = 'freihand-right' if self.is_right else 'freihand-left'
        self.keypoint_names = KEYPOINT_NAMES_DICT[self.source].copy()
        self.flip_indices = np.arange(len(self.keypoint_names))

    def get_elements_per_index(self):
        return 1

    def __repr__(self):
        return 'FreiHand( \n\t Split: {}\n)'.format(self.split)

    def name(self):
        return 'FreiHand/{}'.format(self.split)

    def __len__(self):
        return self.num_items

    def only_2d(self):
        return False

    def project_points(self, K, xyz):
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]

    def __getitem__(self, index):
        img_idx = self.img_idxs[index]
        param_idx = self.param_idxs[index]

        if self.use_folder_split:
            folder_idx = index // self.items_per_folder
            file_idx = index

        K = self.intrinsics[param_idx].copy()
        if isinstance(K, list):
            K = np.array(K)

        pose = self.poses[param_idx].copy()

        global_rot = pose[0].reshape(-1)
        right_hand_pose = pose[1:].reshape(-1)

        keypoints3d = self.xyz[param_idx].copy()
        keypoints2d = self.project_points(K, keypoints3d)

        keypoints3d -= keypoints3d[0]

        keypoints2d = np.concatenate(
            [keypoints2d, np.ones_like(keypoints2d[:, [-1]])], axis=-1
        )

        keypoints3d = np.concatenate(
            [keypoints3d, np.ones_like(keypoints2d[:, [-1]])], axis=-1
        )

        if self.use_folder_split:
            img_fn = osp.join(
                self.img_folder, f'folder_{folder_idx:010d}',
                f'{file_idx:010d}.jpg')
        else:
            img_fn = osp.join(self.img_folder, f'{img_idx:08d}.jpg')

        img = read_img(img_fn)

        target = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            dtype=self.dtype)

        # Store the keypoints in the original crop
        keypoints_hd = Keypoints2D(
            keypoints2d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            apply_crop=False,
            dtype=self.dtype)
        target.add_field('keypoints_hd', keypoints_hd)

        keyp3d_target = Keypoints3D(
            keypoints3d, img.shape,
            flip_indices=self.flip_indices,
            flip_axis=0,
            source=self.source,
            dtype=self.dtype)
        target.add_field('keypoints3d', keyp3d_target)
        target.add_field('intrinsics', K)

        if self.return_masks:
            mask_path = osp.join(self.mask_folder, f'{param_idx:08d}.jpg')
            mask = read_img(mask_path, dtype=np.float32)
            mask = (mask > 0.9).all(axis=-1).astype(np.float32)[:, :, None]

            target.add_field('mask', mask)

        target.add_field('bbox_size', IMG_SIZE)
        center = np.array([IMG_SIZE, IMG_SIZE], dtype=np.float32) * 0.5
        target.add_field('orig_center', np.asarray(img.shape[:-1]) * 0.5)
        target.add_field('center', center)
        scale = IMG_SIZE / REF_BOX_SIZE
        target.add_field('scale', scale)

        if self.return_params:
            global_rot_field = GlobalRot(global_rot=global_rot)
            target.add_field('global_rot', global_rot_field)
            hand_pose_field = HandPose(right_hand_pose=right_hand_pose,
                                       left_hand_pose=None)
            target.add_field('hand_pose', hand_pose_field)

        if hasattr(self, 'translation'):
            translation = self.translation[param_idx]
        else:
            translation = np.zeros([3], dtype=np.float32)
        target.add_field('translation', translation)

        if self.return_vertices:
            vertices = self.vertices[param_idx]
            hand_vertices_field = Vertices(
                vertices, flip_index=0,
                bc=self.bc,
                closest_faces=self.closest_faces,
            )
            target.add_field('vertices', hand_vertices_field)
        if self.return_shape:
            target.add_field('betas', Betas(self.betas[param_idx]))

        if self.transforms is not None:
            force_flip = False
            if not self.is_right:
                force_flip = not self.is_right
            full_img, cropped_image, target = self.transforms(
                img, target, force_flip=force_flip)

        target.add_field('name', self.name())

        return full_img, cropped_image, target, index
