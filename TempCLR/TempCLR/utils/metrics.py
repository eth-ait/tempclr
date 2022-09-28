# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Dict, Union, Optional, Tuple

import sys
import numpy as np

import open3d as o3d
import torch
import torch.nn.functional as F
from loguru import logger

from .np_utils import np2o3d_pcl
from .typing import Tensor, Array, IntList


def build_alignment(name: str, **kwargs):
    if name == 'procrustes':
        return ProcrustesAlignment()
    elif name == 'root':
        return RootAlignment(**kwargs)
    elif name == 'scale':
        return ScaleAlignment()
    elif name == 'no' or name == 'none':
        return NoAlignment()
    else:
        raise ValueError(f'Unknown alignment type: {name}')


def point_error(
    input_points: Union[Array, Tensor],
    target_points: Union[Array, Tensor]
) -> Array:
    ''' Calculate point error

    Parameters
    ----------
        input_points: numpy.array, BxPx3
            The estimated points
        target_points: numpy.array, BxPx3
            The ground truth points
    Returns
    -------
        numpy.array, BxJ
            The point error for each element in the batch
    '''
    if torch.is_tensor(input_points):
        input_points = input_points.detach().cpu().numpy()
    if torch.is_tensor(target_points):
        target_points = target_points.detach().cpu().numpy()

    return np.sqrt(np.power(input_points - target_points, 2).sum(axis=-1))


def mpjpe(
    input_joints: Union[Array, Tensor],
    target_joints: Union[Array, Tensor]
) -> Array:
    ''' Calculate mean per-joint point error

    Parameters
    ----------
        input_joints: numpy.array, Jx3
            The joints predicted by the model
        target_joints: numpy.array, Jx3
            The ground truth joints
    Returns
    -------
        numpy.array, BxJ
            The per joint point error for each element in the batch
    '''
    if torch.is_tensor(input_joints):
        input_joints = input_joints.detach().cpu().numpy()
    if torch.is_tensor(target_joints):
        target_joints = target_joints.detach().cpu().numpy()

    return np.sqrt(np.power(input_joints - target_joints, 2).sum(axis=-1))


def vertex_to_vertex_error(input_vertices, target_vertices):
    return np.sqrt(np.power(input_vertices - target_vertices, 2).sum(axis=-1))


class NoAlignment(object):
    def __init__(self):
        super(NoAlignment, self).__init__()

    def __repr__(self):
        return 'NoAlignment'

    @property
    def name(self):
        return 'none'

    def __call__(self, S1: Array, S2: Array) -> Tuple[Array, Array]:
        return S1, S2


class ProcrustesAlignment(object):
    def __init__(self):
        super(ProcrustesAlignment, self).__init__()

    def __repr__(self):
        return 'ProcrustesAlignment'

    @property
    def name(self):
        return 'procrustes'

    def __call__(self, S1: Array, S2: Array) -> Tuple[Array, Array]:
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrustes problem.
        '''
        if len(S1.shape) < 2:
            S1 = S1.reshape(1, *S1.shape)
            S2 = S2.reshape(1, *S2.shape)

        transposed = False
        if S1.shape[1] != 3 and S1.shape[1] != 3:
            S1 = np.transpose(S1, [0, 2, 1])
            S2 = np.transpose(S2, [0, 2, 1])
            transposed = True

        assert(S2.shape[1] == S1.shape[1])
        batch_size = len(S1)

        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        # 1. Remove mean.
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2, axis=(1, 2))

        # 3. The outer product of X1 and X2.
        K = X1 @ np.transpose(X2, [0, 2, 1])

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = np.transpose(Vh, [0, 2, 1])
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.tile(np.eye(3)[np.newaxis], [batch_size, 1, 1])
        Z[:, -1, -1] *= np.sign(np.linalg.det(U @ Vh))
        # Construct R.
        R = V @ (Z @ np.transpose(U, [0, 2, 1]))

        # 5. Recover scale.
        scale = np.einsum('bii->b', R @ K) / var1

        # 6. Recover translation.
        t = mu2.squeeze(-1) - scale[:, np.newaxis] * np.einsum(
            'bmn,bn->bm', R, mu1.squeeze(-1))

        # 7. Error:
        S1_hat = scale.reshape(-1, 1, 1) * (R @ S1) + t.reshape(
            batch_size, -1, 1)

        if transposed:
            S1 = np.transpose(S1, [0, 2, 1])
            S2 = np.ascontiguousarray(np.transpose(S2, [0, 2, 1]))
            S1_hat = np.ascontiguousarray(np.transpose(S1_hat, [0, 2, 1]))

        return S1_hat, S2


class ScaleAlignment(object):
    def __init__(self):
        super(ScaleAlignment, self).__init__()

    def __repr__(self):
        return 'ScaleAlignment'

    @property
    def name(self):
        return 'scale'

    def __call__(self, S1: Array, S2: Array) -> Tuple[Array, Array]:
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if len(S1.shape) < 2:
            S1 = S1.reshape(1, *S1.shape)
            S2 = S2.reshape(1, *S2.shape)

        batch_size = len(S1)
        if S1.shape[1] != 3 and S1.shape[1] != 3:
            S1 = np.transpose(S1, [0, 2, 1])
            S2 = np.transpose(S2, [0, 2, 1])
            transposed = True

        assert(S2.shape[1] == S1.shape[1])

        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        # 1. Remove mean.
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2, axis=(1, 2))
        var2 = np.sum(X2 ** 2, axis=(1, 2))

        # 5. Recover scale.
        scale = np.sqrt(var2 / var1)

        # 6. Recover translation.
        t = mu2 - scale * mu1

        # 7. Error:
        S1_hat = scale.reshape(-1, 1, 1) * S1 + t.reshape(batch_size, -1, 1)

        if transposed:
            S1_hat = np.transpose(S1_hat, [0, 2, 1])

        return S1_hat, S2


class RootAlignment(object):
    def __init__(self, root: Optional[IntList] = None, **kwargs) -> None:
        super(RootAlignment, self).__init__()
        if root is None:
            root = [0]
        self.root = root

    def set_root(self, new_root):
        self.root = new_root

    @property
    def name(self):
        return 'root'

    def __repr__(self):
        return f'RootAlignment: root = {self.root}'

    def align_by_root(self, joints: Array) -> Array:
        root_joint = joints[:, self.root, :].mean(axis=1, keepdims=True)
        return joints - root_joint

    def __call__(self, est: Array, gt: Array) -> Tuple[Array, Array]:
        gt_out = self.align_by_root(gt)
        est_out = self.align_by_root(est)
        return est_out, gt_out


def point_fscore(
        pred: torch.Tensor,
        gt: torch.Tensor,
        thresh: float) -> Dict[str, float]:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()

    pred_pcl = np2o3d_pcl(pred)
    gt_pcl = np2o3d_pcl(gt)

    gt_to_pred = np.asarray(gt_pcl.compute_point_cloud_distance(pred_pcl))
    pred_to_gt = np.asarray(pred_pcl.compute_point_cloud_distance(gt_pcl))

    recall = (pred_to_gt < thresh).sum() / len(pred_to_gt)
    precision = (gt_to_pred < thresh).sum() / len(gt_to_pred)
    if recall + precision > 0.0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0.0

    return {
        'fscore': fscore,
        'precision': precision,
        'recall': recall,
    }


class PointError(object):
    def __init__(
        self,
        alignment_object: Union[
            ProcrustesAlignment, RootAlignment, NoAlignment],
        name: str = '',
    ) -> None:
        super(PointError, self).__init__()
        self._alignment = alignment_object
        self._name = name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'PointError: Alignment = {self._alignment}'

    def set_root(self, new_root):
        if hasattr(self._alignment, 'set_root'):
            self._alignment.set_root(new_root)

    def set_alignment(self, alignment_object: Union[
            ProcrustesAlignment, RootAlignment, NoAlignment]) -> None:
        self._alignment = alignment_object

    def __call__(self, est_points, gt_points):
        aligned_est_points, aligned_gt_points = self._alignment(
            est_points, gt_points)

        return point_error(aligned_est_points, aligned_gt_points)
