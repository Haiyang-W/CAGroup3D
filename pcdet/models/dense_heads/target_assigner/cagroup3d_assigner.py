from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from pcdet.models.model_utils.cagroup_utils import rotation_3d_in_axis

def volume(boxes):
    return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]

def find_points_in_boxes(points, gt_bboxes, expanded_volumes=None):
    n_points = len(points)
    n_boxes = len(gt_bboxes)
    if expanded_volumes is not None:
        volumes = expanded_volumes
    else:
        volumes = volume(gt_bboxes).to(points.device) # (n_box,)
        volumes = volumes.expand(n_points, n_boxes).contiguous()

    gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
    expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
    shift = torch.stack((
        expanded_points[..., 0] - gt_bboxes[..., 0],
        expanded_points[..., 1] - gt_bboxes[..., 1],
        expanded_points[..., 2] - gt_bboxes[..., 2]
    ), dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = gt_bboxes[..., :3] + shift
    dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
    dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
    dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
    dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
    bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

    inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle  (n_points, n_box)
    return inside_gt_bbox_mask # n_points, n_box


def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    return torch.sqrt(centerness_targets)

class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""

class CAGroup3DAssigner(BaseAssigner):
    def __init__(self, cfg):
        self.limit = cfg.LIMIT
        self.topk = cfg.TOPK
        self.n_scales = cfg.N_SCALES
        self.return_ins_label = cfg.get('RETURN_INS_LABEL', True)

    def assign(self, points_list, gt_bboxes_ori, gt_labels_ori):

        centerness_targets_all = []
        gt_bboxes_all = []
        labels_all = []
        class_num = len(points_list)
        for cls_id in range(class_num):
            float_max = 1e8
            points = points_list[cls_id]

            # below is based on FCOSHead._get_target_single
            n_points = len(points)
            assert n_points > 0, "empty points in class {}".format(cls_id)
            select_inds = torch.nonzero((gt_labels_ori == cls_id)).squeeze(1)
            if len(select_inds) == 0:
                labels = torch.zeros((len(points)), dtype=torch.long).to(points.device).fill_(-1)
                gt_bbox_targets = torch.zeros((len(points), 7), dtype=torch.float).to(points.device)
                centerness_targets = torch.zeros((len(points)), dtype=torch.float).to(points.device)
            else:
                n_boxes = len(select_inds)
                volumes = volume(gt_bboxes_ori).to(points.device)[select_inds]
                volumes = volumes.expand(n_points, n_boxes).contiguous()
                gt_bboxes = gt_bboxes_ori[select_inds].clone().to(points.device).expand(n_points, n_boxes, 7)
                gt_labels = gt_labels_ori[select_inds].clone()
                expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
                shift = torch.stack((
                    expanded_points[..., 0] - gt_bboxes[..., 0],
                    expanded_points[..., 1] - gt_bboxes[..., 1],
                    expanded_points[..., 2] - gt_bboxes[..., 2]
                ), dim=-1).permute(1, 0, 2)
                shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
                centers = gt_bboxes[..., :3] + shift
                dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
                dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
                dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
                dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
                dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
                dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
                bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

                # condition1: inside a gt bbox
                inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

                # condition3: limit topk locations per box by centerness
                centerness = compute_centerness(bbox_targets)
                centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
                top_centerness = torch.topk(centerness, min(self.topk + 1, len(centerness)), dim=0).values[-1]
                inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

                # if there are still more than one objects for a location,
                # we choose the one with minimal area
                volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
                volumes = torch.where(inside_top_centerness_mask, volumes, torch.ones_like(volumes) * float_max)
                min_area, min_area_inds = volumes.min(dim=1)

                labels = gt_labels[min_area_inds]
                # labels = torch.where(min_area == float_max, -1, labels)
                labels = torch.where(min_area == float_max, -labels.new_ones(labels.shape), labels)
                bbox_targets = bbox_targets[range(n_points), min_area_inds]
                centerness_targets = compute_centerness(bbox_targets)
                gt_bbox_targets = gt_bboxes[range(n_points), min_area_inds].clone()

            centerness_targets_all.append(centerness_targets)
            gt_bboxes_all.append(gt_bbox_targets)
            labels_all.append(labels)
        centerness_targets_all = torch.cat(centerness_targets_all)
        gt_bboxes_all = torch.cat(gt_bboxes_all)
        labels_all = torch.cat(labels_all)
        return centerness_targets_all, gt_bboxes_all, labels_all

    @classmethod
    def assign_semantic(self, points, gt_bboxes, gt_labels, n_classes):
        float_max = 1e8

        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = volume(gt_bboxes).to(points.device) # (n_box,)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        inside_gt_bbox_mask = find_points_in_boxes(points, gt_bboxes, expanded_volumes=volumes)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        bk_mask = inside_gt_bbox_mask.sum(dim=1) != 0
        min_area, min_area_inds = volumes.min(dim=1) # (n,)

        labels = gt_labels[min_area_inds]
        # labels = torch.where(min_area == float_max, -1, labels)
        labels = torch.where(min_area == float_max, -labels.new_ones(labels.shape), labels)
        ins_labels = (min_area_inds + 1) * bk_mask

        return labels, ins_labels