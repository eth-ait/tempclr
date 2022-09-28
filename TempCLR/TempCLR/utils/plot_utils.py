# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Union, List, Tuple, Optional

import sys
import numpy as np
import torch

import trimesh
import pyrender
import matplotlib.cm as mpl_cm
from PIL import Image, ImageDraw
from loguru import logger
import cv2

from .typing import (
    Tensor, Array, IntPair, StringList, FloatList, FloatTuple)
#  from TempCLR.data.structures import StructureList, Keypoints2D

# royal blue : [70 / 255.0, 87 / 255.0, 125 / 230.0]
# light purple: [177 / 255.0, 150 / 255.0, 167 / 230.0]
COLORS = {
    'N': [1.0, 1.0, 0.9],
    'GT': [1.0, 1.0, 0.9],
    'pre_fusion': (81 / 255, 23 / 255, 186 / 255),
    'final': (0.4, 0.4, 0.8),
    'default': [70 / 255.0, 87 / 255.0, 125 / 230.0],
}
NUM_STAGES = 10
cmap = mpl_cm.get_cmap('tab10')
for stage in range(NUM_STAGES):
    COLORS[f'stage_{stage:02d}'] = cmap(stage / NUM_STAGES)


HAND_COLORS = np.array([[0.4, 0.4, 0.4],
                        [0.4, 0.4, 0.],
                        [0.6, 0.6, 0.],
                        [0.8, 0.8, 0.],
                        [0., 0.4, 0.2],
                        [0., 0.6, 0.3],
                        [0., 0.8, 0.4],
                        [0., 0.2, 0.4],
                        [0., 0.3, 0.6],
                        [0., 0.4, 0.8],
                        [0.4, 0., 0.4],
                        [0.6, 0., 0.6],
                        [0.7, 0., 0.8],
                        [0.4, 0., 0.],
                        [0.6, 0., 0.],
                        [0.8, 0., 0.],
                        [1., 0., 0.],
                        [1., 1., 0.],
                        [0., 1., 0.5],
                        [1., 0., 1.],
                        [0., 0.5, 1.]])
HAND_COLORS = HAND_COLORS[:, ::-1]
FINGER_NAMES = [
    f'{side}_{finger_name}'
    for side in ['left', 'right']
    for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']]

RIGHT_FINGER = [
    'right_wrist',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky']
LEFT_FINGER = [name.replace('right', 'left') for name in RIGHT_FINGER]


def blend_images(img1, img2, alpha=0.7):
    return img1 * alpha + (1 - alpha) * img2


def undo_img_normalization(
    img: Union[Tensor, Array],
    mean: Union[Tensor, Array],
    std: Union[Tensor, Array],
    dtype=np.float32
) -> Array:
    default_img = np.zeros(
        [3, img.shape[1], img.shape[2]], dtype=dtype)
    if img is None:
        return default_img
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy().squeeze()
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy().squeeze()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy().squeeze()
    return (img * std[:, np.newaxis, np.newaxis] +
            mean[:, np.newaxis, np.newaxis])


def keyp_target_to_image(
    img: Array,
    keypoints
) -> Array:
    ''' Converts a keypoint target to an image
    '''
    # Get the connections and names
    connections = keypoints.connections
    keypoint_names = keypoints.names

    # Extract the keypoints
    keypoints_array = keypoints.as_array(scale=True)
    valid = keypoints_array[:, -1] > 0
    return create_skel_img(img,
                           keypoints_array[:, :-1],
                           connections=connections,
                           names=keypoint_names,
                           valid=valid
                           )


def create_skel_img(
    img: Array,
    keypoints: Array,
    connections: List[IntPair],
    valid: Optional[Array] = None,
    names: Optional[StringList] = None,
) -> Array:
    ''' Draws a 2D skeleton on an image
    '''

    channels = img.shape[-1]
    kp_img = np.copy(img)*255
    
    # Check that the image is in HWC format. If not, then permute the axes.
    if not (channels == 3 or channels == 4):
        kp_img = Image.fromarray(np.uint8(np.transpose(kp_img, [1, 2, 0])))
    else:
        kp_img = Image.fromarray(kp_img)
    
    rgb_dict = get_keypoint_rgb(names)

    draw = ImageDraw.Draw(kp_img)

    if valid is None:
        valid = np.ones([keypoints.shape[0]])

    for idx, pair in enumerate(connections):
        if pair[0] > len(valid) or pair[1] > len(valid):
            continue
        if not valid[pair[0]] or not valid[pair[1]]:
            continue

        joint_name = names[pair[1]]
        color = rgb_dict[joint_name]
        parent_joint_name = names[pair[0]]
        
        start_pt = tuple(keypoints[pair[0], :2].astype(np.int32).tolist())
        end_pt = tuple(keypoints[pair[1], :2].astype(np.int32).tolist())

        draw.line([end_pt, start_pt], fill=rgb_dict[joint_name], width=3)
        draw.ellipse((end_pt[0] - 3, end_pt[1] - 3, end_pt[0] + 3, end_pt[1] + 3), fill=rgb_dict[joint_name])
        draw.ellipse((start_pt[0] - 3, start_pt[1] - 3, start_pt[0] + 3, start_pt[1] + 3), fill=rgb_dict[parent_joint_name])

    return np.asarray(kp_img) / 255


def get_keypoint_rgb(joints):
    rgb_dict= {}
    for joint_name in joints:
        if joint_name.endswith('thumb'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('index'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict


def create_bbox_img(
    img: Array,
    bounding_box: Array,
    color: Optional[Tuple[float]] = (0.0, 0.0, 0.0),
    linewidth: int = 2
) -> Array:
    bbox_img = img.copy()
    if torch.is_tensor(bounding_box):
        bounding_box = bounding_box.detach().cpu().numpy()
    xmin, ymin, xmax, ymax = bounding_box.reshape(4)
    xmin = int(round(xmin))
    xmax = int(round(xmax))
    ymin = int(round(ymin))
    ymax = int(round(ymax))

    cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), color, linewidth)
    return bbox_img


class Renderer(object):
    def __init__(self, near=0.1, far=20000,
                 width=224, height=224,
                 bg_color=(0.0, 0.0, 0.0, 0.0), ambient_light=None,
                 use_raymond_lighting=True,
                 light_color=None, light_intensity=3.0):
        if light_color is None:
            light_color = np.ones(3)

        self.near = near
        self.far = far

        self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                                   viewport_height=height,
                                                   point_size=1.0)

        if ambient_light is None:
            ambient_light = (0.1, 0.1, 0.1)

        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=ambient_light)

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                        aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 2])
        self.scene.add(pc, pose=camera_pose)

        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def __call__(self, vertices, faces, img=None,
                 img_size=224,
                 body_color=(1.0, 1.0, 1.0, 1.0),
                 **kwargs):

        centered_verts = vertices - np.mean(vertices, axis=0, keepdims=True)
        meshes = self.create_mesh(centered_verts, faces,
                                  vertex_color=body_color)

        for node in self.scene.get_nodes():
            if node.name == 'mesh':
                self.scene.remove_node(node)
        for mesh in meshes:
            self.scene.add(mesh, name='mesh')

        color, _ = self.renderer.render(self.scene)

        return color.astype(np.uint8)

    def create_mesh(self, vertices, faces,
                    vertex_color=(0.9, 0.9, 0.7, 1.0)):
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                      [1, 0, 0])
        tri_mesh.apply_transform(rot)

        meshes = []

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            baseColorFactor=vertex_color)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
        meshes.append(mesh)
        return meshes


class WeakPerspectiveCamera(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5

    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=pyrender.camera.DEFAULT_Z_FAR,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale
        P[1, 1] = self.scale
        P[0, 3] = self.translation[0] * self.scale
        P[1, 3] = -self.translation[1] * self.scale
        P[2, 2] = -1

        return P


class AbstractRenderer(object):
    def __init__(self, faces=None, img_size=224, use_raymond_lighting=True):
        super(AbstractRenderer, self).__init__()

        self.img_size = img_size
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_size,
            viewport_height=img_size,
            point_size=1.0)
        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.0, 0.0, 0.0))
        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(
        self,
        vertices: Array,
        faces: Array,
        color: FloatTuple = (0.3, 0.3, 0.3, 1.0),
        wireframe: bool = False,
        deg: float = 0,
        face_colors: Optional[Array] = None,
        vertex_colors: Optional[Array] = None,
    ) -> pyrender.Mesh:

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        if face_colors is not None:
            face_colors = np.mean(face_colors, axis=1)
        #  mesh = self.mesh_constructor(vertices, faces, process=False,
            #  face_colors=face_colors)

        curr_vertices = vertices.copy()
        mesh = self.mesh_constructor(
            curr_vertices, faces,
            process=False,
            face_colors=face_colors,
            vertex_colors=vertex_colors)
        if deg != 0:
            rot = self.transf(
                np.radians(deg), [0, 1, 0],
                point=np.mean(curr_vertices, axis=0))
            mesh.apply_transform(rot)
        
        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material, smooth=False)

    def update_mesh(self, vertices, faces, body_color=(1.0, 1.0, 1.0, 1.0),
                    deg=0, face_colors=None, vertex_colors=None,):
        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(
            vertices, faces, color=body_color, deg=deg,
            face_colors=face_colors, vertex_colors=vertex_colors)
        self.scene.add(body_mesh, name='body_mesh')


class SMPLifyXRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224):
        super(SMPLifyXRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, translation, rotation=None, focal_length=5000,
                      camera_center=None):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)
        if rotation is None:
            rotation = np.eye(3, dtype=translation.dtype)
        if camera_center is None:
            camera_center = np.array(
                [self.img_size, self.img_size], dtype=translation.dtype) * 0.5

        camera_transl = translation.copy()
        camera_transl[0] *= -1.0
        pc = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = camera_transl
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 camera_translation, bg_imgs=None,
                 body_color=(1.0, 1.0, 1.0),
                 upd_color=None,
                 **kwargs):
        if upd_color is None:
            upd_color = {}

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            self.update_camera(camera_translation[bidx])

            curr_col = upd_color.get(bidx, None)
            if curr_col is None:
                curr_col = body_color
            self.update_mesh(vertices[bidx], faces, body_color=curr_col)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)

            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                output_imgs.append(color[:-1])
            else:
                valid_mask = (color[3] > 0)[np.newaxis]

                output_img = (color[:-1] * valid_mask +
                              (1 - valid_mask) * bg_imgs[bidx])
                output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class OverlayRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224, tex_size=1):
        super(OverlayRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, scale, translation):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = WeakPerspectiveCamera(scale, translation,
                                   znear=1e-5,
                                   zfar=1000)
        camera_pose = np.eye(4)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 camera_scale, camera_translation, bg_imgs=None,
                 deg=0,
                 return_with_alpha=False,
                 body_color=None,
                 **kwargs):

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(camera_scale):
            camera_scale = camera_scale.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = COLORS['N']

            if bg_imgs is not None:
                _, H, W = bg_imgs[bidx].shape
                # Update the renderer's viewport
                self.renderer.viewport_height = H
                self.renderer.viewport_width = W

            self.update_camera(camera_scale[bidx], camera_translation[bidx])
            self.update_mesh(vertices[bidx], faces, body_color=body_color,
                             deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
            else:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class GTRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224):
        super(GTRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, intrinsics):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)
        pc = pyrender.IntrinsicsCamera(
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            zfar=1000)
        camera_pose = np.eye(4)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 intrinsics, bg_imgs=None, deg=0,
                 return_with_alpha=False,
                 **kwargs):
        ''' Returns a B3xHxW batch of mesh overlays
        ''' 
        
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        body_color = COLORS['GT']
        output_imgs = []
        for bidx in range(batch_size):
            if bg_imgs is not None:
                _, H, W = bg_imgs[bidx].shape
                # Update the renderer's viewport
                self.renderer.viewport_height = H
                self.renderer.viewport_width = W
            self.update_camera(intrinsics[bidx])
            self.update_mesh(vertices[bidx], faces, body_color=body_color,
                             deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
            else:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class HDRenderer(OverlayRenderer):
    def __init__(self, znear: float = 1.0e-2, zfar: float = 1000):
        super(HDRenderer, self).__init__()
        self.znear = znear
        self.zfar = zfar

    def update_camera(self, focal_length, translation, center):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=center[0],
            cy=center[1],
            znear=self.znear,
            zfar=self.zfar
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = translation.copy()
        camera_pose[0, 3] *= (-1)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self,
                 vertices: Tensor,
                 faces: Union[Tensor, Array],
                 focal_length: Union[Tensor, Array],
                 camera_translation: Union[Tensor, Array],
                 camera_center: Union[Tensor, Array],
                 bg_imgs: Array,
                 render_bg: bool = True,
                 deg: float = 0,
                 return_with_alpha: bool = False,
                 body_color: List[float] = None,
                 face_colors: Optional[Union[Tensor, Array]] = None,
                 vertex_colors: Optional[Union[Tensor, Array]] = None,
                 **kwargs):
        '''
            Parameters
            ----------
            vertices: BxVx3, torch.Tensor
                The torch Tensor that contains the current vertices to be drawn
            faces: Fx3, np.array
                The faces of the meshes to be drawn. Right now only support a
                batch of meshes with the same topology
            focal_length: B, torch.Tensor
                The focal length used by the perspective camera
            camera_translation: Bx3, torch.Tensor
                The translation of the camera estimated by the network
            camera_center: Bx2, torch.Tensor
                The center of the camera in pixels
            bg_imgs: np.ndarray
                Optional background images used for overlays
            render_bg: bool, optional
                Render on top of the background image
            deg: float, optional
                Degrees to rotate the mesh around itself. Used to render the
                same mesh from multiple viewpoints. Defaults to 0 degrees
            return_with_alpha: bool, optional
                Whether to return the rendered image with an alpha channel.
                Default value is False.
            body_color: list, optional
                The color used to render the image.
        '''
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        if torch.is_tensor(focal_length):
            focal_length = focal_length.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        if torch.is_tensor(camera_center):
            camera_center = camera_center.detach().cpu().numpy()
        if face_colors is not None and torch.is_tensor(face_colors):
            face_colors = face_colors.detach().cpu().numpy()
        if vertex_colors is not None and torch.is_tensor(vertex_colors):
            vertex_colors = vertex_colors.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = COLORS['N']

            _, H, W = bg_imgs[bidx].shape
            # Update the renderer's viewport
            self.renderer.viewport_height = H
            self.renderer.viewport_width = W

            self.update_camera(
                focal_length=focal_length[bidx],
                translation=camera_translation[bidx],
                center=camera_center[bidx],
            )
            face_color = None
            if face_colors is not None:
                face_color = face_colors[bidx]
            vertex_color = None
            if vertex_colors is not None:
                vertex_color = vertex_colors[bidx]
            self.update_mesh(
                vertices[bidx], faces, body_color=body_color, deg=deg,
                face_colors=face_color,
                vertex_colors=vertex_color,
            )

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if render_bg:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
            else:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
        return np.stack(output_imgs, axis=0)
