# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Dict, Optional, Tuple
import os
import os.path as osp
from PIL import Image
from copy import deepcopy
from collections import defaultdict, OrderedDict
import json
import numpy as np
import torch
from numpy import ndarray
from torchvision.utils import make_grid
from tqdm import tqdm
from loguru import logger
from .data.structures import StructureList, BoundingBox
from .data.utils import (
    KEYPOINT_NAMES_DICT,
    map_keypoints,
)
from .models.body_models import KeypointTensor

from .models import HAND_HEAD_REGISTRY
from .utils import (Tensor, Array,
                    FloatList,
                    PointError, build_alignment,
                    undo_img_normalization,
                    OverlayRenderer,
                    GTRenderer,
                    COLORS,
                    keyp_target_to_image, create_skel_img
                    )


def build(exp_cfg, distributed=False, rank=0, save_imgs=False):
    network_cfg = exp_cfg.get('network', {})
    net_type = network_cfg.get('type', 'TempCLR')
    if net_type in HAND_HEAD_REGISTRY:
        logger.info('Building evaluator ...')
        return PartEvaluator(exp_cfg, rank=rank, distributed=distributed, save_imgs=save_imgs)
    else:
        raise ValueError(f'Unknown network type: {net_type}')


class Evaluator(object):
    def __init__(self, exp_cfg, rank=0, distributed=False):
        super(Evaluator, self).__init__()
        self.std = None
        self.means = None
        self.rank = rank
        self.distributed = distributed

        self.imgs_per_row = exp_cfg.get('imgs_per_row', 3)
        self.exp_cfg = deepcopy(exp_cfg)
        self.output_folder = osp.expandvars(exp_cfg.output_folder)

        self.summary_folder = osp.join(
            self.output_folder, exp_cfg.summary_folder)
        os.makedirs(self.summary_folder, exist_ok=True)
        self.summary_steps = exp_cfg.summary_steps

        self.results_folder = osp.join(
            self.output_folder, exp_cfg.results_folder)
        self.batch_size = exp_cfg.datasets.hand.batch_size

        os.makedirs(self.results_folder, exist_ok=True)
        del exp_cfg

    @torch.no_grad()
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    @staticmethod
    def _compute_mpjpe(
            model_output,
            targets: StructureList,
            metric_align_dicts: Dict,
            mpjpe_root_joints_names: Optional[Array] = None,
    ) -> Dict[str, Array]:
        # keypoint annotations.
        gt_joints_3d_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field('keypoints3d')], dtype=np.int)
        output = {}
        # Get the number of valid instances
        num_instances = len(gt_joints_3d_indices)
        if num_instances < 1:
            return output
        # Get the data from the output of the model
        est_joints = model_output.get('joints', None)
        joints_from_model_np = est_joints.detach().cpu().numpy()

        for t in targets:
            if not t.has_field('keypoints3d'):
                continue
            gt_source = t.get_field('keypoints3d').source
            break

        # Get the indices that map the estimated joints to the order of the
        # ground truth joints
        target_names = KEYPOINT_NAMES_DICT.get(gt_source)
        target_indices, source_indices, target_dim = map_keypoints(
            target_dataset=gt_source,
            source_dataset=est_joints.source,
            names_dict=KEYPOINT_NAMES_DICT,
            source_names=est_joints.keypoint_names,
            target_names=target_names,
        )

        # Create the final array
        est_joints_np = np.zeros(
            [num_instances, target_dim, 3], dtype=np.float32)
        # Map the estimated joints to the order used by the ground truth joints
        for ii in gt_joints_3d_indices:
            est_joints_np[ii, target_indices] = joints_from_model_np[
                ii, source_indices]

        # Stack all 3D joint tensors
        gt_joints3d = np.stack(
            [t.get_field('keypoints3d').as_array()
             for t in targets if t.has_field('keypoints3d')])
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [
                    target_names.index(name)
                    for name in mpjpe_root_joints_names
                ]
                alignment.set_root(root_indices)
            metric_value = alignment(
                est_joints_np[gt_joints_3d_indices],
                gt_joints3d[:, :, :-1])
            name = f'{alignment_name}_mpjpe'
            output[name] = metric_value
        return output

    @staticmethod
    def _compute_mpjpe14(
            model_output,
            targets: StructureList,
            metric_align_dicts: Dict,
            J14_regressor: Array,
            **extra_args,
    ) -> Dict[str, Array]:
        output = {}
        gt_joints_3d_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field('joints14')], dtype=np.long)
        if len(gt_joints_3d_indices) < 1:
            return output
        # Stack all 3D joint tensors
        gt_joints3d = np.stack(
            [t.get_field('joints14').joints.detach().cpu().numpy()
             for t in targets if t.has_field('joints14')])

        # Get the data from the output of the model
        est_vertices = model_output.get('vertices', None)
        est_vertices_np = est_vertices.detach().cpu().numpy()
        est_joints_np = np.einsum(
            'jv,bvn->bjn', J14_regressor, est_vertices_np)
        for alignment_name, alignment in metric_align_dicts.items():
            metric_value = alignment(est_joints_np[gt_joints_3d_indices],
                                     gt_joints3d)
            name = f'{alignment_name}_mpjpe14'
            output[name] = metric_value
        return output

    @staticmethod
    def _compute_v2v(
            model_output,
            targets: StructureList,
            metric_align_dicts: Dict,
            **extra_args,
    ) -> Dict[str, Array]:
        ''' Computes the Vertex-to-Vertex error for the current input
        '''
        output = {}
        # Ground truth vertices
        gt_verts_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field('vertices')], dtype=np.int)
        if len(gt_verts_indices) < 1:
            return output

        # Stack all vertices
        gt_vertices = np.stack(
            [t.get_field('vertices').vertices.detach().cpu().numpy()
             for t in targets if t.has_field('vertices')])

        # Get the data from the output of the model
        est_vertices = model_output.get('vertices', None)
        est_vertices_np = est_vertices.detach().cpu().numpy()

        for alignment_name, alignment in metric_align_dicts.items():
            metric_value = alignment(est_vertices_np, gt_vertices)
            name = f'{alignment_name}_v2v'
            output[name] = metric_value
        return output

    @staticmethod
    def _compute_2d_mpjpe(model_output,
                          targets: StructureList,
                          metric_align_dicts: Dict,
                          mpjpe_root_joints_names: Optional[Array] = None):

        # keypoint annotations.
        gt_joints_2d_indices = np.array(
            [ii for ii, t in enumerate(targets)
             if t.has_field('keypoints_hd')], dtype=np.int)
        output = {}
        # Get the number of valid instances
        num_instances = len(gt_joints_2d_indices)
        if num_instances < 1:
            return output
        # Get the data from the output of the model
        est_joints = model_output.get('proj_joints', None)
        joints_from_model_np = est_joints.detach().cpu().numpy()

        for t in targets:
            if not t.has_field('keypoints_hd'):
                continue
            gt_source = t.get_field('keypoints_hd').source
            break

        # Get the indices that map the estimated joints to the order of the
        # ground truth joints
        target_names = KEYPOINT_NAMES_DICT.get(gt_source)
        target_indices, source_indices, target_dim = map_keypoints(
            target_dataset=gt_source,
            source_dataset=est_joints.source,
            names_dict=KEYPOINT_NAMES_DICT,
            source_names=est_joints.keypoint_names,
            target_names=target_names,
        )

        # Create the final array
        est_joints_np = np.zeros(
            [num_instances, target_dim, 2], dtype=np.float32)
        # Map the estimated joints to the order used by the ground truth joints
        for ii in gt_joints_2d_indices:
            est_joints_np[ii, target_indices] = joints_from_model_np[
                ii, source_indices]

        # Stack all 2D joint tensors
        gt_joints2d = np.stack(
            [t.get_field('keypoints_hd').as_array()
             for t in targets if t.has_field('keypoints_hd')])
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [
                    target_names.index(name)
                    for name in mpjpe_root_joints_names
                ]
                alignment.set_root(root_indices)
            metric_value = alignment(
                est_joints_np[gt_joints_2d_indices],
                gt_joints2d[:, :, :-1])
            name = f'{alignment_name}_mpjpe_2d'
            output[name] = metric_value
        return output

    def compute_metric(
            self,
            model_output,
            targets: StructureList,
            metrics: Dict,
            mpjpe_root_joints_names: Optional[Array] = None,
            **extra_args,
    ):
        output_metric_values = {}
        for metric_name, metric in metrics.items():
            if metric_name == 'mpjpe':
                curr_vals = self._compute_mpjpe(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'mpjpe14':
                curr_vals = self._compute_mpjpe14(
                    model_output, targets, metric, **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'v2v':
                curr_vals = self._compute_v2v(
                    model_output, targets, metric, **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'mpjpe_2d':
                curr_vals = self._compute_2d_mpjpe(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            else:
                raise ValueError(f'Unsupported metric: {metric_name}')
        return output_metric_values

    def _create_keypoint_images(
            self,
            images: ndarray,
            targets: StructureList,
            model_output: Dict,
            gt_available: bool = True
    ) -> Tuple[ndarray, ndarray]:
        gt_keyp_imgs = []
        est_keyp_imgs = []

        proj_joints = model_output.get('proj_joints', None)
        if proj_joints is not None:
            keypoint_names = proj_joints.keypoint_names
            est_connections = proj_joints._connections
            if torch.is_tensor(proj_joints):
                proj_joints = proj_joints.detach().cpu().numpy()
            elif isinstance(proj_joints, KeypointTensor, ):
                proj_joints = proj_joints._t.detach().cpu().numpy()

        # Scale the predicted keypoints to image coordinates
        crop_size = images.shape[-1]
        proj_joints = (proj_joints + 1) * 0.5 * crop_size

        for ii, (img, target) in enumerate(zip(images, targets)):
            if gt_available:
                gt_keyp_imgs.append(keyp_target_to_image(
                    img, target))

            if proj_joints is not None:
                est_keyp_imgs.append(create_skel_img(
                    img, proj_joints[ii],
                    names=keypoint_names,
                    connections=est_connections,
                ))

        if gt_available:
            gt_keyp_imgs = np.transpose(np.stack(gt_keyp_imgs), [0, 3, 1, 2])

        est_keyp_imgs = np.transpose(np.stack(est_keyp_imgs), [0, 3, 1, 2])
        return gt_keyp_imgs, est_keyp_imgs

    def render_mesh_overlay(self, bg_imgs,
                            vertices, faces,
                            camera_scale,
                            camera_translation, genders=None,
                            flip=False, renderer=None,
                            degrees=None,
                            body_color=None,
                            ):
        if degrees is None:
            degrees = []
        body_imgs = renderer(
            vertices,
            faces,
            camera_scale,
            camera_translation,
            bg_imgs=bg_imgs,
            genders=genders,
            body_color=body_color,
        )
        if flip:
            body_imgs = body_imgs[:, :, :, ::-1]

        out_imgs = [body_imgs]
        # Add the rendered meshes
        for deg in degrees:
            body_imgs = renderer(
                vertices, faces,
                camera_scale, camera_translation,
                bg_imgs=None,
                genders=genders,
                deg=deg,
                return_with_alpha=False,
                body_color=body_color,
            )
            out_imgs.append(body_imgs)
        return np.concatenate(out_imgs, axis=-1)

    def create_image_summaries(
            self,
            step: int,
            batch_num: int,
            dset_name: str,
            images: Tensor,
            targets: StructureList,
            model_output: Dict,
            degrees: Optional[FloatList] = None,
            renderer: Optional[OverlayRenderer] = None,
            gt_renderer: Optional[GTRenderer] = None,
            gt_available: bool = True,
            render_gt_meshes: bool = True,
            prefix: str = '',
            draw_keyps: bool = True,
    ) -> None:

        if degrees is None:
            degrees = []

        backgrounds = np.stack([np.zeros_like(undo_img_normalization(img, self.means, self.std)) for img in images])
        crop_size = images.shape[-1]
        images = np.stack([
            undo_img_normalization(img, self.means, self.std)
            for img in images])
        _, _, crop_size, _ = images.shape

        summary_imgs = OrderedDict()
        summary_imgs['rgb'] = images

        # Create the keypoint images
        if draw_keyps:
            gt_keyp_imgs, est_keyp_imgs = self._create_keypoint_images(
                images, targets, model_output=model_output, gt_available=gt_available)
            if gt_available:
                summary_imgs['gt_keypoint_images'] = gt_keyp_imgs
            summary_imgs['est_keypoint_images'] = est_keyp_imgs

        render_gt_meshes = (render_gt_meshes and
                            any([t.has_field('vertices') for t in targets]))
        stage_keys = model_output.get('stage_keys', [])
        last_stage = stage_keys[-1]
        if render_gt_meshes:
            gt_mesh_imgs = []
            faces = model_output[last_stage]['faces']
            for bidx, t in enumerate(targets):
                if (not t.has_field('vertices') or
                        not t.has_field('intrinsics')):
                    gt_mesh_imgs.append(np.zeros_like(images[bidx]))
                    continue

                curr_gt_vertices = t.get_field(
                    'vertices').vertices.detach().cpu().numpy().squeeze()
                intrinsics = t.get_field('intrinsics')

                mesh_img = gt_renderer(
                    curr_gt_vertices[np.newaxis], faces=faces,
                    intrinsics=intrinsics[np.newaxis],
                    bg_imgs=images[[bidx]])
                gt_mesh_imgs.append(mesh_img.squeeze())

            gt_mesh_imgs = np.stack(gt_mesh_imgs)
            B, C, H, W = gt_mesh_imgs.shape
            row_pad = (crop_size - H) // 2
            gt_mesh_imgs = np.pad(
                gt_mesh_imgs,
                [[0, 0], [0, 0], [row_pad, row_pad], [row_pad, row_pad]])
            summary_imgs['gt_meshes'] = gt_mesh_imgs

        camera_params = model_output.get('camera_parameters', None)
        scale = camera_params.scale
        translation = camera_params.translation

        for stage_key in stage_keys:
            if stage_key not in model_output:
                continue
            curr_stage_output = model_output.get(stage_key)
            vertices = curr_stage_output.get('vertices', None)
            if vertices is None:
                continue
            vertices = vertices.detach().cpu().numpy()

            faces = curr_stage_output['faces']

            body_color = COLORS['default']
            overlays = self.render_mesh_overlay(
                images,
                vertices, faces,
                scale, translation,
                degrees=degrees if stage_key == stage_keys[-1] else None,
                renderer=renderer,
                body_color=body_color,
            )
            summary_imgs[f'overlays_{stage_key}'] = overlays

            deg = [i * 45.0 for i in range(1, 8)]
            meshes_from_rand_angle_view = self.render_mesh_overlay(
                backgrounds,
                vertices, faces,
                scale, translation,
                degrees=deg,
                renderer=renderer,
                body_color=body_color
            )

            summary_imgs[f'meshes_from_rand_angle_{stage_key}'] = meshes_from_rand_angle_view

        predicted_images = model_output.get('predicted_images', None)
        if predicted_images is not None:
            predicted_images = predicted_images.detach().cpu().numpy()
            summary_imgs['predicted'] = predicted_images

        summary_imgs = np.concatenate(
            list(summary_imgs.values()), axis=3)

        img_grid = make_grid(
            torch.from_numpy(summary_imgs), nrow=self.imgs_per_row)

        img_tab_name = (f'{dset_name}/{prefix}/Images' if len(prefix) > 0 else
                        f'{dset_name}/Images_{step}_batch_{batch_num}')

        ndarr = img_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)

        if self.save_images:
            im.save(os.path.join(self.save_dir, f"batch_{batch_num}.png"))

        return

    def build_metric_utilities(self, exp_cfg, part_key):
        eval_cfg = exp_cfg.get('evaluation', {}).get(part_key, {})
        v2v_cfg = eval_cfg.get('v2v', {})

        mpjpe_cfg = eval_cfg.get('mpjpe', {})
        mpjpe_alignments = mpjpe_cfg.get('alignments', [])
        mpjpe_root_joints_names = mpjpe_cfg.get('root_joints', [])

        mpjpe_2d_cfg = eval_cfg.get('mpjpe_2d', {})
        mpjpe_2d_alignments = mpjpe_2d_cfg.get('alignments', [])
        mpjpe_2d_root_joints_names = mpjpe_2d_cfg.get('root_joints', [])

        model_name = exp_cfg.get(f'{part_key}_model', {}).get('type', 'smplx')
        keypoint_names = KEYPOINT_NAMES_DICT[model_name]

        if not hasattr(self, 'mpjpe_root_joints_names'):
            self.mpjpe_root_joints_names = {}
        self.mpjpe_root_joints_names[part_key] = mpjpe_root_joints_names

        mpjpe_root_joints = [
            keypoint_names.index(name)
            for name in mpjpe_root_joints_names]

        v2v = {
            name: PointError(build_alignment(name))
            for name in v2v_cfg
        }
        mpjpe = {
            name: PointError(build_alignment(name, root=mpjpe_root_joints))
            for name in mpjpe_alignments
        }

        mpjpe_2d = {
            name: PointError(build_alignment(name, root=mpjpe_root_joints))
            for name in mpjpe_2d_alignments
        }

        return {'mpjpe': mpjpe, 'v2v': v2v}


class PartEvaluator(Evaluator):
    VALID_PART_KEYS = ['hand']

    def __init__(self, exp_cfg, rank: int = 0, distributed: bool = False, save_imgs=False):
        super(PartEvaluator, self).__init__(
            exp_cfg, rank=rank, distributed=distributed)
        part_key = exp_cfg.get('part_key', 'body')
        assert part_key in self.VALID_PART_KEYS, (
            f'Part key must be one of {self.VALID_PART_KEYS}, got: {part_key}'
        )
        self.save_images = save_imgs
        self.part_key = part_key
        self.means = np.array(self.exp_cfg.datasets.get(
            part_key, {}).transforms.mean)
        self.std = np.array(self.exp_cfg.datasets.get(
            part_key, {}).transforms.std)
        self.save_dir = os.path.join(os.path.expandvars(exp_cfg.get("output_folder")), exp_cfg.get("summary_folder"))
        # Initialize part renderers
        self.degrees = exp_cfg.get('degrees', {}).get(part_key, tuple())
        crop_size = exp_cfg.get('datasets', {}).get(part_key, {}).get(
            'crop_size', 256)
        self.renderer = OverlayRenderer(img_size=crop_size)
        self.render_gt_meshes = exp_cfg.get('render_gt_meshes', True)
        if self.render_gt_meshes:
            self.gt_renderer = GTRenderer(img_size=crop_size)

        self.J14_regressor = None
        self.metrics = self.build_metric_utilities(exp_cfg, part_key=part_key)

    @torch.no_grad()
    def run(self, model, dataloaders, exp_cfg, device, step=0):
        if self.rank > 0:
            return
        model.eval()

        # Copy the model to avoid deadlocks and convert to float
        if self.distributed:
            eval_model = deepcopy(model.module).float()
        else:
            eval_model = deepcopy(model).float()
        eval_model.eval()
        assert not eval_model.training, 'Model is in training mode!'

        part_dataloaders = dataloaders.get(self.part_key)

        for dataloader in part_dataloaders:
            dset = dataloader.dataset

            dset_name = dset.name()

            if "freihand" in dset_name.lower():
                target_dset_name = "freihand-right"
            else:
                target_dset_name = "ho3d"

            metric_values = defaultdict(lambda: [])
            desc = f'Evaluating dataset: {dset_name}'
            logger.info(f'Starting evaluation for: {dset_name}')

            xyz_pred_list, verts_pred_list = list(), list()

            for ii, batch in enumerate(
                    tqdm(dataloader, desc=desc, dynamic_ncols=True)):
                _, images, targets = batch

                images_copy = deepcopy(images)
                del images

                # Transfer to the device
                images_copy = images_copy.to(device=device)
                targets_copy = deepcopy(targets)
                del targets
                targets_copy = [target.to(device) for target in targets_copy]

                model_output = eval_model(images_copy, targets_copy, device=device)
                verts = model_output['vertices'].detach().cpu().numpy()

                # Convert in OpenGL coordinate system
                recov_verts = verts
                if target_dset_name == 'ho3d':
                    recov_verts[:, :, 1:] = -recov_verts[:, :, 1:]
                    new_verts = recov_verts.tolist()
                else:
                    new_verts = recov_verts.tolist()

                verts_pred_list = verts_pred_list + new_verts

                est_joints3d = model_output['est_joints3d'].detach().cpu().numpy()

                target_names = KEYPOINT_NAMES_DICT.get(target_dset_name)
                target_indices, source_indices, target_dim = map_keypoints(
                    target_dataset=target_dset_name,
                    source_dataset=model_output['est_joints3d'].source,
                    names_dict=KEYPOINT_NAMES_DICT,
                    source_names=model_output['est_joints3d'].keypoint_names,
                    target_names=target_names,
                )

                # Create the final array
                est_joints_np = est_joints3d.copy()
                # Map the estimated joints to the order used by the ground truth joints
                for i in range(est_joints_np.shape[0]):
                    est_joints_np[i, target_indices] = est_joints3d[
                        i, source_indices]

                if target_dset_name == 'ho3d':
                    est_joints_np[:, :, 1:] = -est_joints_np[:, :, 1:]
                    new_joints = est_joints_np.tolist()
                else:
                    new_joints = est_joints_np.tolist()

                xyz_pred_list = xyz_pred_list + new_joints

                num_stages = model_output.get('num_stages', 1)
                stage_n_out = model_output.get(
                    f'stage_{num_stages - 1:02d}', {})

                gt_available = True
                if isinstance(targets_copy[0], BoundingBox):
                    gt_available = False

                if (ii == 0 and exp_cfg.get('create_image_summaries', True)) or self.save_images:
                    self.create_image_summaries(
                        step, ii, dset_name,
                        images_copy,
                        targets_copy,
                        model_output,
                        degrees=self.degrees,
                        renderer=self.renderer,
                        gt_renderer=self.gt_renderer,
                        gt_available=gt_available
                    )

                    images = np.stack(
                        [undo_img_normalization(img, self.means, self.std)
                         for img in images_copy]).transpose(0, 2, 3, 1)

                    image_list = []
                    for i in range(images.shape[0]):
                        im = Image.fromarray(np.uint8(images[i] * 255))
                        image_list.append(im)

                curr_metrics = self.compute_metric(
                    stage_n_out, targets_copy,
                    metrics=self.metrics,
                    J14_regressor=self.J14_regressor,
                    mpjpe_root_joints_names=self.mpjpe_root_joints_names[
                        self.part_key]
                )

                logging_metrics = defaultdict(lambda: [])
                for key, value in curr_metrics.items():
                    metric_values[key].append(value)
                    logging_metrics[key].append(value.tolist())

            for metric_name in metric_values:
                metric_array = np.concatenate(
                    metric_values[metric_name], axis=0)

                mean_metric_value = np.mean(metric_array) * 1000
                if metric_name == "none_mpjpe_2d":
                    logger.info('[{:06d}] {}, {}: {:.4f} (px)',
                                step, dset_name, metric_name,
                                mean_metric_value,
                                )
                else:
                    logger.info('[{:06d}] {}, {}: {:.4f} (mm)',
                                step, dset_name, metric_name,
                                mean_metric_value,
                                )

            self.dump(f"{osp.expandvars(exp_cfg.output_folder)}/pred.json", xyz_pred_list, verts_pred_list)

    @staticmethod
    def dump(pred_out_path, xyz_pred_list, verts_pred_list):
        """ Save predictions into a json file for official ho3dv2 evaluation. """

        # make sure its only lists
        def roundall(rows):
            return [[round(val, 5) for val in row] for row in rows]

        xyz_pred_list = [roundall(x) for x in xyz_pred_list]
        verts_pred_list = [roundall(x) for x in verts_pred_list]

        # save to a json
        with open(pred_out_path, "w") as fo:
            json.dump([xyz_pred_list, verts_pred_list], fo)
        logger.info('Dumped {:06d} joints and {:06d} verts predictions to {}', len(xyz_pred_list),
                    len(verts_pred_list), f"{pred_out_path}")

    @staticmethod
    def translate_encodings(encoding: torch.Tensor, translate_x: torch.Tensor, translate_y: torch.Tensor
                            ) -> torch.Tensor:
        """Translates the encodings along first two dimensions with linear scaling
        Args:
            encoding (Tensor): image encodings/projections from the network
            translate_x (Tensor): normalized jitter along x-axis of the input image
            translate_y (Tensor): normalized jitter along y-axis of the input image
        Returns:
            Tensor: Translated encodings based on scaled normalized jitter.
        """
        max_encodings = torch.max(encoding.detach(), dim=1).values
        min_encodings = torch.min(encoding.detach(), dim=1).values

        encoding[..., 0] += (
                translate_x * (max_encodings[:, 0] - min_encodings[:, 0])
        ).view((-1, 1))
        encoding[..., 1] += (
                translate_y * (max_encodings[:, 1] - min_encodings[:, 1])
        ).view((-1, 1))
        return encoding

    @staticmethod
    def compute_error_accel(joints_gt, joints_pred, vis=None):
        """
        Computes acceleration error:
            1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (Nx21x3).
            joints_pred (Nx21x3).
            vis (N).
        Returns:
            error_accel (N-2).
        """
        joints_pred = joints_pred.reshape(-1, 21, 3)
        joints_gt = joints_gt.reshape(-1, 21, 3)

        # (N-2)x21x3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)

        return np.mean(normed[new_vis], axis=1)

    def rotate_encoding(self, encoding: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Function to 2D rotate a batch of encodings by a batch of angles.
        The third dimension is n not changed.
        Args:
            encoding (Tensor): Encodings of shape (batch_size,m,3)
            angle ([type]): batch of angles (batch_size,)
        Returns:
            Tensor: Rotated batch of keypoints.
        """
        center_xyz = torch.mean(encoding.detach(), 1)
        rot_mat = self.get_rotation_2D_matrix(
            angle, center_xyz[:, 0], center_xyz[:, 1], scale=1.0
        )
        rot_mat = rot_mat.to(encoding.device)
        encoding[..., :2] = torch.bmm(
            torch.cat((encoding[..., :2], torch.ones_like(encoding[..., -1:])), dim=2),
            rot_mat,
        )
        return encoding

    @staticmethod
    def get_rotation_2D_matrix(angle: torch.Tensor, center_x: torch.Tensor, center_y: torch.Tensor, scale: torch.Tensor
                               ) -> torch.Tensor:
        """Generates 2D rotation matrix transpose. the matrix generated is for the whole batch.
        The implementation of 2D matrix is same as that in openCV.
        Args:
            angle (Tensor): 1D tensor of rotation angles for the batch
            center_x (Tensor): 1D tensor of x coord of center of the keypoints.
            center_y (Tensor): 1D tensor of x coord of center of the keypoints.
            scale (Tensor): Scale, set it to 1.0.
        Returns:
            Tensor: Returns a tensor of 2D rotation matrix for the batch.
        """
        # convert to radians
        angle = angle * np.pi / 180
        alpha = scale * torch.cos(angle)
        beta = scale * torch.sin(angle)
        rot_mat = torch.zeros((len(angle), 3, 2))
        rot_mat[:, :, 0] = torch.stack(
            [alpha, beta, (1 - alpha) * center_x - beta * center_y], dim=1
        )
        rot_mat[:, :, 1] = torch.stack(
            [-beta, alpha, (1 - alpha) * center_y + beta * center_x], dim=1
        )

        return rot_mat
