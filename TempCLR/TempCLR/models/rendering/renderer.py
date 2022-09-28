"""
Author: Yao Feng
Copyright (c) 2020, Yao Feng
All rights reserved.
"""

# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""

from typing import Dict, Optional
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from loguru import logger
from .utils import (
    dict2obj, face_vertices as face_vertices_to_triangles,
    generate_triangles, vertex_normals)

from TempCLR.utils import Tensor


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(
            self,
            image_size: int = 224,
            blur_radius: float = 0.0,
            faces_per_pixel: int = 1,
            bin_size: Optional = None,
            max_faces_per_bin: Optional = None,
            perspective_correct: bool = False,
    ) -> None:
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': blur_radius,
            'faces_per_pixel': faces_per_pixel,
            'bin_size': bin_size,
            'max_faces_per_bin': max_faces_per_bin,
            'perspective_correct': perspective_correct,
        }
        logger.info(f'Rasterization settings: {raster_settings}')
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        # Rotate 180 degrees around the z-axis
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]

        # Convert to Pytorch3D meshes
        meshes_screen = Meshes(
            verts=fixed_vertices.float(), faces=faces.long())

        raster_settings = self.raster_settings
        # Run rasterization
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        # pix_to_face(N,H,W,K), bary_coords(N,H,W,K,3),attribute: (N, nf, 3, D)
        # pixel_vals = interpolate_face_attributes(fragment, attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1]))
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(
            attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat(
            [pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):
    def __init__(
            self,
            image_size: int,
            obj_filename: os.PathLike,
            uv_size: int = 256
    ) -> None:
        '''
            Parameters
            ----------
                image_size: int
                    The desired size for the rendered image
                obj_file: os.PathLike
                    The filename for loading the topology to be rendererd
                uv_size: int
                    The size of the UV map to be rendered
        '''
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        obj_filename = osp.expandvars(obj_filename)
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        dense_triangles = generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(
            dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat(
            [uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices_to_triangles(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(
            1, faces.max() + 1, 1).float() / 255.
        face_colors = face_vertices_to_triangles(colors, faces)
        self.register_buffer('face_colors', face_colors)

        # SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([
            1 / np.sqrt(4 * pi),
            (2 * pi / 3) * np.sqrt(3 / (4 * pi)),
            (2 * pi / 3) * np.sqrt(3 / (4 * pi)),
            (2 * pi / 3) * np.sqrt(3 / (4 * pi)),
            (pi / 4) * 3 * np.sqrt(5 / (12 * pi)),
            (pi / 4) * 3 * np.sqrt(5 / (12 * pi)),
            (pi / 4) * 3 * np.sqrt(5 / (12 * pi)),
            (pi / 4) * (3 / 2) * np.sqrt(5 / (12 * pi)),
            (pi / 4) * (1 / 2) * np.sqrt(5 / (4 * pi))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def extra_repr(self) -> str:
        msg = [
            f'Image size: {self.image_size}',
            f'UV size: {self.uv_size}',
        ]
        return '\n'.join(msg)

    def forward(
            self,
            vertices: Tensor,
            transformed_vertices: Tensor,
            albedos: Tensor,
            lights: Optional[Tensor] = None,
            light_type: str = 'point',
            znear: float = 5,
            zfar: float = 100,
    ) -> Dict[str, Tensor]:
        '''
        -- Texture Rendering

        Parameters:
            vertices: [batch_size, V, 3]
                vertices in world space, for calculating normals, then shading
            transformed_vertices: [batch_size, V, 3], range:[-1,1]
            projected vertices, in image space, for rasterization
            albedos: [batch_size, 3, h, w]
                Albedo in uv map format
            lights:
                spherical harmonic: [N, 9(shcoeff), 3(rgb)]
                points/directional lighting: [N, n_lights, 6(xyzrgb)]
            light_type:
                point or directional
        '''
        batch_size = vertices.shape[0]
        # rasterizer near 0 far 100. move mesh so minz larger than 0
        min_z, _ = torch.min(
            transformed_vertices[:, :, 2], dim=1, keepdim=True)
        transformed_vertices[:, :, 2] = (
                                                transformed_vertices[:, :, 2] - min_z) + znear

        # attributes
        face_vertices = face_vertices_to_triangles(
            vertices, self.faces.expand(batch_size, -1, -1))
        normals = vertex_normals(
            vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = face_vertices_to_triangles(
            normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = vertex_normals(
            transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = face_vertices_to_triangles(
            transformed_normals, self.faces.expand(batch_size, -1, -1))

        attributes = torch.cat([
            self.face_uvcoords.expand(batch_size, -1, -1, -1),
            transformed_face_normals.detach(),
            face_vertices.detach(),
            face_normals], dim=-1)

        # rasterize
        rendering = self.rasterizer(
            transformed_vertices, self.faces.expand(batch_size, -1, -1),
            attributes)

        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(
                        vertice_images.permute(0, 2, 3, 1).reshape(
                            [batch_size, -1, 3]),
                        normal_images.permute(0, 2, 3, 1).reshape(
                            [batch_size, -1, 3]),
                        lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2],
                         albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
                else:
                    shading = self.add_directionlight(normal_images.permute(
                        0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2],
                         albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.0

        outputs = {
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images,
            'transformed_normals': transformed_normals,
        }

        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
        sh_coeff: [bz, 9, 3]

            Parameters
            ----------
                normal_images: torch.Tensor
                    A tensor containing the normal images. Its size should be
                    Bx3xHxW
                sh_coeff: torch.Tensor
                    The estimated spherical harmonic coefficients. Should be
                    Bx9x3
            Returns
            -------
                torch.Tensor
                    A tensor with size Bx3xHxW that corresponds the per-pixel
                    shading information
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2,
            3 * (N[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        # [bz, 9, 3, h, w]
        shading = torch.sum(
            sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
        vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(
            light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        # normals_dot_lights = torch.clamp(
        #  (normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (
                normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = (normals_dot_lights[:, :, :, None] *
                   light_intensities[:, :, None, :])
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
        normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(
            light_direction[:, :, None, :].expand(
                -1, -1, normals.shape[1], -1),
            dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp(
            (normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0., 1.)
        shading = (normals_dot_lights[:, :, :, None] *
                   light_intensities[:, :, None, :])
        return shading.mean(1)

    def render_shape(
            self,
            vertices: Tensor,
            transformed_vertices: Tensor,
            images: Optional[Tensor] = None,
            detail_normal_images: Optional[Tensor] = None,
            lights: Optional[Tensor] = None
    ) -> Tensor:
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor(
                [
                    [-1, 1, 1],
                    [1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [0, 0, 1]
                ]
            )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float() * 1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(
                vertices.device)
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = face_vertices_to_triangles(
            vertices, self.faces.expand(batch_size, -1, -1))
        normals = vertex_normals(
            vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = face_vertices_to_triangles(
            normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = vertex_normals(
            transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = face_vertices_to_triangles(
            transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_colors.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                               -1)
        # rasterize
        rendering = self.rasterizer(
            transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(
            0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape(
            [batch_size, albedo_images.shape[2], albedo_images.shape[3],
             3]).permute(0, 3, 1, 2)
        shaded_images = albedo_images * shading_images

        if images is None:
            images = torch.zeros_like(shaded_images).to(vertices.device)
        shape_images = (
                shaded_images * alpha_images + images * (1 - alpha_images))
        return shape_images

    def render_depth(
            self,
            transformed_vertices: Tensor
    ) -> Tensor:
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:, :, 2] = (
                transformed_vertices[:, :, 2] -
                transformed_vertices[:, :, 2].min())
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        # Attributes
        attributes = face_vertices_to_triangles(
            z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(
            transformed_vertices, self.faces.expand(batch_size, -1, -1),
            attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_normal(
            self,
            transformed_vertices: Tensor,
            normals: Tensor,
    ) -> Tensor:
        '''
        -- rendering normal
        '''
        batch_size = normals.shape[0]

        # Attributes
        attributes = face_vertices_to_triangles(
            normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(
            transformed_vertices, self.faces.expand(batch_size, -1, -1),
            attributes)

        #  alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(
            self,
            vertices: Tensor
    ) -> Tensor:
        '''
        project vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = face_vertices_to_triangles(
            vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices
