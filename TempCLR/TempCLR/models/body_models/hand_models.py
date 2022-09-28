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
from typing import List, Tuple, Dict

import pickle

from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn

from .lbs import lbs, vertices2joints
from .utils import to_tensor, JointsFromVerticesSelector, KeypointTensor
from TempCLR.data.utils import (
    KEYPOINT_NAMES_DICT, KEYPOINT_CONNECTIONS,
    KEYPOINT_PARTS,
    PART_NAMES,
    kp_connections,
    get_part_idxs,
)
from TempCLR.utils import (
    to_np, Struct, Tensor, IntList, StringList, BlendShapeDescription, Timer)


class MANO(nn.Module):
    NAME = 'mano'

    ''' Implements the MANO module as a Pytorch layer

        For more details on the formulation see:
        @article{MANO:SIGGRAPHASIA:2017,
          title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
          author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
          journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
          volume = {36},
          number = {6},
          pages = {245:1--245:17},
          series = {245:1--245:17},
          publisher = {ACM},
          month = nov,
          year = {2017},
          url = {http://doi.acm.org/10.1145/3130800.3130883},
          month_numeric = {11}
        }
    '''

    def __init__(
            self,
            model_folder: str = 'models/mano',
            data_struct: Struct = None,
            dtype=torch.float32,
            betas: BlendShapeDescription = None,
            use_sparse: bool = False,
            is_right: bool = True,
            ext: str = 'pkl',
            extra_joint_path: str = '',
            learn_joint_regressor: bool = False,
            **kwargs,
    ) -> None:
        '''
            Keyword Arguments:
                -
        '''
        super(MANO, self).__init__()

        if data_struct is None:
            model_fn = (
                f'MANO_{("right" if is_right else "left").upper()}.{ext}')
            mano_path = os.path.join(model_folder, model_fn)
            if ext == 'npz':
                file_data = np.load(mano_path, allow_pickle=True)
            else:
                with open(mano_path, 'rb') as smpl_file:
                    file_data = pickle.load(smpl_file, encoding='latin1')
            data_struct = Struct(**file_data)

        self.dtype = dtype
        if betas is None:
            betas = {'num': 10}
        self._num_betas = betas.get('num', 10)

        self._is_right = is_right

        self.hand_components = data_struct.hands_components
        if self.is_right:
            self.right_hand_components = self.hand_components
        else:
            self.left_hand_components = self.hand_components

        if extra_joint_path:
            self.extra_joint_selector = JointsFromVerticesSelector(
                fname=extra_joint_path, **kwargs)

        self.faces = to_np(data_struct.f, dtype=np.int64)
        self.register_buffer('faces_tensor',
                             to_tensor(self.faces, dtype=torch.long))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype))

        # The shape components
        shapedirs = data_struct.shapedirs[:, :, :self.num_betas]
        # The shape components
        self.register_buffer('shapedirs',
                             to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(data_struct.weights), dtype=dtype))

        self.NUM_JOINTS = self.J_regressor.shape[0]

        self._keypoint_names = self.build_keypoint_names()

        # Store the flag for the learned joint regressor
        self.learn_joint_regressor = learn_joint_regressor
        if self.learn_joint_regressor:
            # Start from the regressor of the model
            joint_regressor = [self.J_regressor]
            # Add any extra joints
            if extra_joint_path:
                extra_joints_regressor = self.extra_joint_selector.as_tensor(
                    self.num_verts, self.faces_tensor)
                joint_regressor.append(extra_joints_regressor)
            joint_regressor = torch.cat(joint_regressor)

            # Store as a parameter
            joint_regressor = nn.Parameter(joint_regressor, requires_grad=True)
            self.register_parameter('joint_regressor', joint_regressor)

    def extra_repr(self) -> str:
        msg = [
            f'Arm: {("right" if self.is_right else "left").title()}',
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Number of shape coefficients: {self.num_betas}',
        ]
        return '\n'.join(msg)

    def build_keypoint_names(self) -> StringList:
        model_keypoint_names = KEYPOINT_NAMES_DICT[self.NAME.lower()]
        if hasattr(self, 'extra_joint_selector'):
            model_keypoint_names += (
                self.extra_joint_selector.extra_joint_names())
        for ii, keyp_name in enumerate(model_keypoint_names):
            if self.is_right:
                if 'right' not in keyp_name:
                    model_keypoint_names[ii] = f'right_{keyp_name}'
            else:
                if 'left' not in keyp_name:
                    model_keypoint_names[ii] = f'left_{keyp_name}'
        return model_keypoint_names

    @property
    def num_hand_joints(self):
        return self.NUM_JOINTS - 1

    @property
    def name(self):
        return self.NAME

    @property
    def keypoint_names(self) -> StringList:
        return self._keypoint_names

    @property
    def parts(self) -> Dict[str, Dict[str, IntList]]:
        if not hasattr(self, '_parts'):
            parts = get_part_idxs(self.keypoint_names, KEYPOINT_PARTS)
            self._parts = parts
        return self._parts

    @property
    def part_connections(self) -> List[Tuple[int, int]]:
        if not hasattr(self, '_part_connections'):
            _part_connections = {}
            for part_name in PART_NAMES:
                _part_connections[part_name] = kp_connections(
                    self.keypoint_names, KEYPOINT_CONNECTIONS,
                    part=part_name, keypoint_parts=KEYPOINT_PARTS)
            self._part_connections = _part_connections

        return self._part_connections

    @property
    def is_right(self) -> bool:
        return self._is_right

    @property
    def num_betas(self) -> int:
        ''' Returns the number of shape coefficients of the model '''
        return self._num_betas

    @property
    def num_verts(self) -> int:
        ''' Returns the number of vertices in the MANO mesh '''
        return self.v_template.shape[0]

    @property
    def num_faces(self) -> int:
        ''' Returns the number of triangles in the MANO mesh '''
        return self.faces.shape[0]

    @property
    def connections(self) -> List[Tuple[int, int]]:
        if not hasattr(self, '_connections'):
            connections = kp_connections(
                self.keypoint_names, KEYPOINT_CONNECTIONS)
            self._connections = connections

        return self._connections

    def forward(
            self, betas: Tensor = None,
            wrist_pose: Tensor = None,
            hand_pose: Tensor = None,
            return_shaped: bool = True,
            transl: Tensor = None, get_skin: bool = True,
            return_full_pose: bool = False,
            **kwargs):
        ''' Forward pass for the MANO model

            Parameters
            ----------
            get_skin: bool, optional
                Return the vertices and the joints of the model. (default=true)
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [betas, wrist_pose, transl, hand_pose, ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if wrist_pose is None:
            wrist_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if hand_pose is None:
            hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device)
        # Concate the pose tensors

        full_pose = torch.cat([wrist_pose, hand_pose], dim=1)
        # Apply linear blend skinning (LBS)
        lbs_output = lbs(betas, full_pose, self.v_template,
                         self.shapedirs, self.posedirs,
                         self.J_regressor, self.parents,
                         self.lbs_weights,
                         pose2rot=False, return_shaped=return_shaped)
        vertices = lbs_output['vertices']
        joints = lbs_output['joints']

        if self.learn_joint_regressor:
            final_joints = vertices2joints(self.joint_regressor, vertices)
        else:
            final_joint_set = [joints]
            if hasattr(self, 'extra_joint_selector'):
                # Add any extra joints that might be needed
                extra_joints = self.extra_joint_selector(
                    vertices, self.faces_tensor)
                final_joint_set.append(extra_joints)
            # Create the final joint set
            final_joints = torch.cat(final_joint_set, dim=1)

        # Apply the translation vector
        if transl is not None:
            final_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output_joints = KeypointTensor(
            final_joints,
            source=self.name,
            keypoint_names=self.keypoint_names,
            part_indices=self.parts,
            connections=self.connections,
            part_connections=self.part_connections,
        )
        output = defaultdict(
            lambda: None, joints=output_joints, faces=self.faces)

        if get_skin:
            output['vertices'] = vertices
        if return_full_pose:
            output['full_pose'] = full_pose
        if return_shaped:
            output['v_shaped'] = lbs_output['v_shaped']

        return output


class HTML(nn.Module):
    """
    current FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/albedoModel2020_FLAME_albedoPart.npz'
    ## adapted from BFM
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/FLAME_albedo_from_BFM.npz'
    """

    def __init__(
            self,
            dim: int = 50,
            path: str = 'data/mano/texture.pkl',
            #  uv_path: str = 'data/mano/uv.pkl',
            **kwargs,
    ) -> None:
        super(HTML, self).__init__()

        self._dim = dim
        path = osp.expandvars(path)
        # need to read the face_uvs and verts_uvs.
        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.path = path

        tex_mean = model['mean']  # the mean texture
        tex_basis = model['basis'][:, :self.dim]  # 101 PCA comps
        vec2texImg_index = model['index_map']  # the index map, from a compact vector to a 2D texture image

        self.tex_img1d = torch.zeros(1024 * 1024 * 3)

        self.register_buffer(
            'mean', torch.tensor(tex_mean, dtype=torch.float32))
        self.register_buffer(
            'basis', torch.tensor(tex_basis, dtype=torch.float32))
        self.register_buffer(
            'vec2texImg_index',
            torch.tensor(vec2texImg_index, dtype=torch.long))

        self.index_select_timer = Timer(name='Index select', sync=True)
        self.brackets_timer = Timer(name='Brackets', sync=True)

    @property
    def dim(self):
        return self._dim

    def extra_repr(self) -> str:
        msg = [
            f'Dimension: {self.dim}',
            f'Loaded from: {self.path}',
            f'mean: {self.mean.shape}',
            f'Basis: {self.basis.shape}',
        ]
        return '\n'.join(msg)

    def forward(
            self,
            code: Tensor,
    ) -> Tensor:
        ''' Forward pass of the HTML model
        '''
        batch_size = code.shape[0]
        dtype, device = code.dtype, code.device

        # Compute the offsets from the mean
        offsets = torch.einsum('bm,nm->bn', [code, self.basis])
        # Compute the current texture image
        tex_code = offsets + self.mean[None, :]

        # Convert it to a square image
        texture_1d = torch.zeros(
            [batch_size, 1024 * 1024 * 3], dtype=dtype, device=device)
        texture_1d.index_add_(1, self.vec2texImg_index, tex_code)
        # Resize to an image shape
        texture = texture_1d.reshape(batch_size, 3, 1024, 1024).transpose(
            2, 3).contiguous()

        return texture / 255
