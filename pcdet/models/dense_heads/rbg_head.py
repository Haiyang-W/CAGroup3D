import numpy as np
import torch
import math

from functools import partial
from six.moves import map, zip
from torch.nn import functional as F
from torch import nn as nn
from pcdet.utils.loss_utils import AxisAlignedIoULoss, ChamferDistance, chamfer_distance
from pcdet.utils.box_coder_utils import RBGBBoxCoder
from pcdet.models.model_utils.vote_module import VoteModule
from pcdet.models.model_utils.rbgnet_utils import MLP, BasicBlock1D
from ...ops.pointnet2.pointnet2_batch.pointnet2_modules import PointnetSAModule
from ...ops.pointnet2.pointnet2_batch.pointnet2_utils import farthest_point_sample, grouping_operation, gather_operation, ball_query, three_nn, three_interpolate
from pcdet.models.backbones_3d.pointnet2_fbs_backbone import PointnetSAModuleSSGFBS
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.box_utils import boxes_to_corners_3d

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def generate_ray(ray_num):
    self_radius = 1.
    self_azimuthal = 0.  # [0, 2*pi)
    n = int(math.ceil(np.sqrt((ray_num - 2) / 4)))
    azimuthal = 0.5 * np.pi / n
    ray_vector = []
    for a in range(-n, n+1):
        self_polar = 0. # [0, pi)
        size = (n - abs(a)) * 4 or 1
        polar = 2 * math.pi / size
        for i in range(size):
            self_polar += polar
            # to Cartesian
            r = np.sin(self_azimuthal) * self_radius
            x = np.cos(self_polar) * r
            y = np.sin(self_polar) * r
            z = np.cos(self_azimuthal) * self_radius
            ray_vector.append([x, y, z])
        self_azimuthal += azimuthal
    return np.array(ray_vector)

def rotation_3d_in_axis(points, angles, axis=0):
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = torch.stack([
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos]),
            torch.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))

class RBGHead(nn.Module):
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__()
        self.num_classes = num_class
        self.train_cfg = model_cfg.TRAIN
        self.test_cfg = model_cfg.TEST
        self.loss_weight_cfg = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        self.gt_per_seed = model_cfg.VOTE_MODULE_CFG.GT_PER_SEED
        self.num_proposal = model_cfg.VOTE_AGGREGATION_CFG.NUM_POINTS
        self.ray_num = model_cfg.RAY_NUM

        self.objectness_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.8]), reduction='none')
        self.dir_res_loss = nn.SmoothL1Loss(reduction='none', beta=1./25.)
        self.dir_class_loss = nn.CrossEntropyLoss(reduction='none')
        self.size_res_loss = nn.SmoothL1Loss(reduction='none', beta=1./16.)
        self.semantic_loss = nn.CrossEntropyLoss(reduction='none')
        self.sample_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.8]), reduction='none')
        self.scale_res_loss = nn.SmoothL1Loss(reduction='none', beta=1./16.)
        self.intersection_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.5, 0.5]), reduction='none')
        self.iou_loss = AxisAlignedIoULoss(reduction='sum')
        self.center_loss = ChamferDistance(mode='l2', reduction='sum', loss_src_weight=10.0, loss_dst_weight=10.0)

        self.fps_num_sample = model_cfg.FPS_NUM_SAMPLE
        self.threshold = model_cfg.THRESHOLD
        self.sample_bin_num = model_cfg.SAMPLE_BIN_NUM
        self.fine_threshold = model_cfg.FINE_THRESHOLD
        self.fine_sample_bin_num = model_cfg.FINE_SAMPLE_BIN_NUM
        self.scale_ratio = model_cfg.SCALE_RATIO
        self.positive_weights = model_cfg.POSITIVE_WEIGHT

        self.scale_prediction = MLP(in_channel=model_cfg.PRED_LAYER_CFG.IN_CHANNELS,
                                    conv_channels=model_cfg.PRED_LAYER_CFG.SHARED_CONV_CHANNELS,
                                    bias=model_cfg.PRED_LAYER_CFG.BIAS)
        self.scale_prediction.mlp.add_module(f'conv_scale', nn.Conv1d(in_channels=model_cfg.PRED_LAYER_CFG.SHARED_CONV_CHANNELS[-1],
                                                                      out_channels=1, kernel_size=1))
        self.fuse_feat = MLP(in_channel=int(model_cfg.PRED_LAYER_CFG.IN_CHANNELS)*2, conv_channels=(int(model_cfg.PRED_LAYER_CFG.IN_CHANNELS),))
        self.num_sizes = model_cfg.BOX_CODER.NUM_SIZE
        self.num_dir_bins = model_cfg.BOX_CODER.NUM_DIR_BINS
        self.bbox_coder = RBGBBoxCoder(ray_num=self.ray_num, num_dir_bins=self.num_dir_bins, num_sizes=self.num_sizes, with_rot=model_cfg.BOX_CODER.WITH_ROT)
        self.vote_module = VoteModule(model_cfg.VOTE_MODULE_CFG)
        self.vote_aggregation_cfg = model_cfg.VOTE_AGGREGATION_CFG
        self.vote_aggregation = PointnetSAModule(mlp=self.vote_aggregation_cfg['MLP_CHANNELS'],
                                                 npoint=self.vote_aggregation_cfg['NUM_POINTS'],
                                                 radius=self.vote_aggregation_cfg['RADIUS'],
                                                 nsample=self.vote_aggregation_cfg['NUM_SAMPLE'],
                                                 use_xyz=self.vote_aggregation_cfg['USE_XYZ'])

        self.raybasedgrouping = RayBasedGrouping(model_cfg.RAY_BASED_GROUP)

        self.share_pred = MLP(in_channel=model_cfg.PRED_LAYER_CFG.IN_CHANNELS,
                              conv_channels=model_cfg.PRED_LAYER_CFG.SHARED_CONV_CHANNELS,
                              bias=model_cfg.PRED_LAYER_CFG.BIAS)
        self.conv_cls = nn.Conv1d(in_channels=model_cfg.PRED_LAYER_CFG.SHARED_CONV_CHANNELS[-1],
                                  out_channels=self._get_cls_out_channels(), kernel_size=1)
        self.conv_reg = nn.Conv1d(in_channels=model_cfg.PRED_LAYER_CFG.SHARED_CONV_CHANNELS[-1],
                                  out_channels=self._get_reg_out_channels(), kernel_size=1)

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs.
           Class numbers (k) + objectness (2)
        """
        return self.num_classes + 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs.
           center residual (3),
           heading class+residual (num_dir_bins*2),
           size (3)
        """
        return 3 + self.num_dir_bins * 2 + 3

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.
        Args:
            feat_dict (dict): Feature dict from backbone. 
        Returns:
            torch.Tensor: Coordinates of input points (seed points).
            torch.Tensor: Features of input points (seed features).
            torch.Tensor: Indices of input points (seed_indices).
            list[torch.Tensor]: Multi-stage foreground scores of backbone points (sa_masks_score).  
            list[torch.Tensor]: Multi-stage indices of backbone points (sa_masks_score).  
        """
        seed_points = feat_dict['fp_xyz'][-1]
        seed_features = feat_dict['fp_features'][-1]
        seed_indices = feat_dict['fp_indices'][-1]

        temp_sa_masks_score = feat_dict['sa_masks_score']
        sa_indices = feat_dict['sa_indices']
        sa_masks_score = []
        sa_masks_indices = []
        for i in range(len(temp_sa_masks_score)):
            if temp_sa_masks_score[i] is not None:
                sa_masks_score.append(temp_sa_masks_score[i])
                sa_masks_indices.append(sa_indices[i])
        return seed_points, seed_features, seed_indices, sa_masks_score, sa_masks_indices

    def forward(self, feat_dict):
        """Forward pass.
        Note:
            The forward of VoteHead is devided into 4 steps:
                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Ray-based grouping stage
                4. Predict bbox and score.
        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed", "random" and "spec".
        Returns:
            dict: Predictions of RBG head.
        """
        if self.training:
            sample_mod = self.train_cfg.SAMPLE_MODE
        else:
            sample_mod = self.test_cfg.SAMPLE_MODE
        # voting stage
        assert sample_mod in ['vote', 'seed', 'random', 'spec']
        seed_points, seed_features, seed_indices, sa_masks_score, sa_masks_indices = self._extract_input(
            feat_dict)

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, seed_features)
        batch_size = vote_features.shape[0]
        results = dict(
            seed_points=seed_points,
            seed_features=seed_features,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset,
            sa_masks_score=sa_masks_score,
            sa_masks_indices=sa_masks_indices)

        # 2. aggregate vote_points, follow VoteNet
        if sample_mod == 'vote':
            # Use fps in vote_aggregation, fps on vote space
            aggregation_inputs = dict(
                xyz=vote_points, features=vote_features)
        elif sample_mod == 'seed':
            # Use fps in vote_aggregation, fps on seed space and choose the votes corresponding to the seeds
            sample_indices = farthest_point_sample(seed_points, self.num_proposal)
            vote_points_flipped = vote_points.transpose(1, 2).contiguous()
            new_xyz = gather_operation(vote_points_flipped, sample_indices).transpose(1, 2).contiguous()
            aggregation_inputs = dict(
                xyz=vote_points,
                features=vote_features,
                new_xyz=new_xyz)
        elif sample_mod == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(
                torch.randint(0, num_seed, (batch_size, self.num_proposal)),
                dtype=torch.int32)
            vote_points_flipped = vote_points.transpose(1, 2).contiguous()
            new_xyz = gather_operation(vote_points_flipped, sample_indices).transpose(1, 2).contiguous()
            aggregation_inputs = dict(
                xyz=vote_points,
                features=vote_features,
                new_xyz=new_xyz)
        elif sample_mod == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(
                xyz=seed_points,
                features=seed_features,
                new_xyz=vote_points)
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mod} is not supported!')
        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, features = vote_aggregation_ret 
        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = features

        # 3. ray-based grouping stage
        scale_predictions = self.scale_prediction(features)
        decode_res = self.bbox_coder.scale_pred(scale_predictions, aggregated_points)
        results.update(decode_res)
        scale_pred = self.bbox_coder.decode_scale(results['scale_size_res'])
        results['ray_direction'] = scale_predictions.new_zeros(batch_size, self.num_proposal, 1)
        results['scale_pred'] = scale_pred
        pooled_feats, fine_intersec_scores, fine_query_indices, \
        coarse_intersec_scores, coarse_query_indices = self.raybasedgrouping(
            results['seed_points'],
            results['seed_features'],
            results['ray_direction'],
            results['scale_pred'],
            results['aggregated_points'],
            feat_dict['points_cat'],
            results['aggregated_features']
        )
        fused_feats = self.fuse_feat(torch.cat((results['aggregated_features'], pooled_feats), dim=1))
        results.update(fine_intersec_score=fine_intersec_scores)
        results.update(coarse_intersec_score=coarse_intersec_scores)
        results.update(fine_query_indices=fine_query_indices)
        results.update(coarse_query_indices=coarse_query_indices)
        results.update(fused_feats=fused_feats)

        # 4. predict bbox and score.
        pred_feats = self.share_pred(fused_feats)
        cls_predictions = self.conv_cls(pred_feats)
        reg_predictions = self.conv_reg(pred_feats)

        bbox_preds = self.bbox_coder.split_pred(
            cls_predictions, reg_predictions, results['aggregated_points'])
        results.update(bbox_preds)
        feat_dict.update(results)

        # post-precessing the bboxes (e.g., NMS)
        if not self.training:
            batch_size = feat_dict['batch_size']
            assert batch_size == 1, f'evalutation only supprots batch size = 1 but got {batch_size}'
            batch_box_preds, batch_score_preds, batch_cls_preds = self.generate_predicted_boxes(feat_dict['batch_size'], feat_dict)
            results['batch_cls_preds'] = batch_cls_preds
            results['batch_box_preds'] = batch_box_preds
            results['batch_score_preds'] = batch_score_preds
            results['cls_preds_normalized'] = False
        return results

    def loss(self,
             batch_dict,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None):
        with torch.no_grad():
            targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask, batch_dict, img_metas)

        (vote_targets, vote_target_masks, dir_class_targets, dir_res_targets, mask_targets, 
        objectness_targets, objectness_weights, box_loss_weights, center_targets, assigned_center_targets,
        valid_gt_weights, size_targets, scale_targets, fine_query_sample_targets, 
        coarse_query_sample_targets, fine_query_sample_weights, coarse_query_sample_weights) = targets

        batch_size, proposal_num = objectness_targets.shape
        # calculate vote loss
        vote_loss = self.vote_module.get_loss(batch_dict['seed_points'],
                                              batch_dict['vote_points'],
                                              batch_dict['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate scale residual loss
        scale_residual_norm = torch.exp(
            batch_dict['scale_res_norm']).squeeze(-1)
        box_loss_weights_scale_expand = box_loss_weights.unsqueeze(-1).repeat(
            1, 1, 1)

        scale_res_loss = self.scale_res_loss(
            scale_residual_norm,
            scale_targets)
        scale_res_loss = (scale_res_loss * box_loss_weights_scale_expand).sum()

        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            batch_dict['obj_scores'].transpose(2, 1),
            objectness_targets)
        objectness_loss = (objectness_loss * objectness_weights).sum()

        # calculate center loss
        source2target_loss, target2source_loss = self.center_loss(
            batch_dict['center'],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate direction class loss
        dir_class_targets = dir_class_targets.long()
        dir_class_loss = self.dir_class_loss(
            batch_dict['dir_class'].transpose(2, 1),
            dir_class_targets)
        dir_class_loss = (dir_class_loss * box_loss_weights).sum()

        # calculate direction residual loss
        heading_label_one_hot = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)

        dir_res_norm = torch.sum(
            batch_dict['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets)
        dir_res_loss = (dir_res_loss * box_loss_weights).sum()

        size_residual_norm = torch.exp(
            batch_dict['size_res_norm']).squeeze(-2)
        box_loss_weights_expand = box_loss_weights.unsqueeze(-1).repeat(
            1, 1, 3)

        size_res_loss = self.size_res_loss(
            size_residual_norm,
            size_targets)
        size_res_loss = (size_res_loss * box_loss_weights_expand).sum()

        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            batch_dict['sem_scores'].transpose(2, 1),
            mask_targets)
        semantic_loss = (semantic_loss * box_loss_weights).sum()

        # calculate intersection loss
        fine_query_sample_targets = fine_query_sample_targets.long()
        fine_intersec_score = batch_dict['fine_intersec_score']

        fine_intersec_loss = self.intersection_loss(
            fine_intersec_score.reshape(batch_size, 2, -1),
            fine_query_sample_targets.reshape(batch_size, -1))
        fine_intersec_loss = (fine_intersec_loss * fine_query_sample_weights.reshape(batch_size, -1)).sum()

        coarse_query_sample_targets = coarse_query_sample_targets.long()
        coarse_intersec_score = batch_dict['coarse_intersec_score']

        coarse_intersec_loss = self.intersection_loss(
            coarse_intersec_score.reshape(batch_size, 2, -1),
            coarse_query_sample_targets.reshape(batch_size, -1))
        coarse_intersec_loss = (coarse_intersec_loss * coarse_query_sample_weights.reshape(batch_size, -1)).sum()


        losses = dict(
            vote_loss=vote_loss,
            scale_res_loss=scale_res_loss * self.loss_weight_cfg['scale_loss_weight'],
            objectness_loss=objectness_loss * self.loss_weight_cfg['obj_loss_weight'],
            semantic_loss=semantic_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss * self.loss_weight_cfg['dir_class_loss_weight'],
            dir_res_loss=dir_res_loss * self.loss_weight_cfg['dir_res_loss_weight'],
            size_res_loss=size_res_loss * self.loss_weight_cfg['size_loss_weight'],
            fine_intersec_loss=fine_intersec_loss * self.loss_weight_cfg['intersection_loss_weight'],
            coarse_intersec_loss=coarse_intersec_loss * self.loss_weight_cfg['intersection_loss_weight'])

        # background
        if hasattr(self, 'sample_loss'):
            if self.bbox_coder.with_rot:
                num_points = points[0].shape[0]
                pts_semantic_mask = [points[b_].new_ones([num_points], dtype=torch.long) * self.num_classes
                                     for b_ in range(len(points))]
                for b_ in range(len(points)):
                    # TODO:rotationbug
                    pcdet_gt_bboxes_3d_batch = gt_bboxes_3d[b_].clone()
                    pcdet_gt_bboxes_3d_batch[:, -1] = -pcdet_gt_bboxes_3d_batch[:, -1]
                    box_indices_all = [roiaware_pool3d_utils.points_in_boxes_gpu(points=points[b_:b_+1],
                                                boxes=pcdet_gt_bboxes_3d_batch[t:t+1, :].unsqueeze(0)).squeeze(0) for t in range(pcdet_gt_bboxes_3d_batch.shape[0])]
                    box_indices_all = torch.stack(box_indices_all) > -1  
                    box_indices_all = box_indices_all.T
                    for i_ in range(gt_labels_3d[b_].shape[0]):
                        box_indices = box_indices_all[:, i_]
                        indices = torch.nonzero(
                            box_indices, as_tuple=False).squeeze(-1)

                        pts_semantic_mask[b_][indices] = gt_labels_3d[b_][i_].item()
            if not torch.is_tensor(pts_semantic_mask):
                pts_semantic_mask = torch.stack(pts_semantic_mask)
            foreground_mask = torch.where(pts_semantic_mask < self.num_classes,
                                      torch.ones_like(pts_semantic_mask),
                                      torch.zeros_like(pts_semantic_mask))
            foreground_mask = foreground_mask.unsqueeze(1)
            if len(batch_dict['sa_masks_score']) != 0:
                for i in range(len(batch_dict['sa_masks_score'])):
                    sa_masks_score = batch_dict['sa_masks_score'][i]
                    sa_masks_indices = batch_dict['sa_masks_indices'][i]

                    sa_masks_targets = gather_operation(foreground_mask.float(), sa_masks_indices.int())

                    sa_masks_targets = sa_masks_targets.squeeze(1).long()


                    sa_mask_weight = torch.ones_like(sa_masks_targets).float()
                    sa_mask_weight /= sa_mask_weight.sum()

                    loss_name = "sample_loss_"+str(i)
                    losses[loss_name] = (self.sample_loss(sa_masks_score,
                                sa_masks_targets) * sa_mask_weight).sum() * self.loss_weight_cfg['sample_loss_weight']

        if self.iou_loss:
            corners_pred = self.bbox_coder.decode_corners(
                batch_dict['center'], size_residual_norm)
            # one_hot_size_targets_expand)
            corners_target = self.bbox_coder.decode_corners(
                assigned_center_targets, size_targets)
            # one_hot_size_targets_expand)
            iou_loss = self.iou_loss(
                corners_pred, corners_target, weight=box_loss_weights)
            losses['iou_loss'] = iou_loss * self.loss_weight_cfg['iou_loss_weight']
        losses_sum = 0
        for k in losses.keys():
            losses_sum += losses[k]

        return losses_sum, losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    batch_dict=None,
                    img_metas=None):

        valid_gt_masks, gt_num = [], []
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].new_zeros(
                    1, gt_bboxes_3d[index].shape[-1])
                gt_bboxes_3d[index] = fake_box
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for _ in range(len(gt_labels_3d))]
        if pts_instance_mask is None:
            pts_instance_mask = [None for _ in range(len(gt_labels_3d))]

        aggregated_points = [batch_dict['aggregated_points'][i] for i in range(len(gt_labels_3d))]
        scale_size_res_pred = [batch_dict['scale_size_res'][i] for i in range(len(gt_labels_3d))]

        (vote_targets, vote_target_masks, 
             dir_class_targets, dir_res_targets, 
             mask_targets, objectness_targets, objectness_masks,
             dir_targets,
             center_targets, assigned_center_targets,
             size_targets, scale_targets,
             coarse_query_sample_targets, fine_query_sample_targets,
             coarse_valid_query_targets, fine_valid_query_targets) = multi_apply(
                self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
                pts_semantic_mask, pts_instance_mask, aggregated_points, scale_size_res_pred)

        for index in range(len(gt_labels_3d)):
            pad_num = max_gt_num - gt_labels_3d[index].shape[0]
            center_targets[index] = F.pad(center_targets[index],
                                          (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)
        valid_gt_masks = torch.stack(valid_gt_masks)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (torch.sum(objectness_targets).float() + 1e-6)

        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        dir_targets = torch.stack(dir_targets)
        mask_targets = torch.stack(mask_targets)
        valid_gt_weights = valid_gt_masks.float() / (torch.sum(valid_gt_masks.float()) + 1e-6)
        center_targets = torch.stack(center_targets)
        assigned_center_targets = torch.stack(assigned_center_targets)

        size_targets = torch.stack(size_targets)
        scale_targets = torch.stack(scale_targets)

        coarse_query_sample_targets = torch.stack(coarse_query_sample_targets)
        fine_query_sample_targets = torch.stack(fine_query_sample_targets)
        coarse_valid_query_targets = torch.stack(coarse_valid_query_targets)
        fine_valid_query_targets = torch.stack(fine_valid_query_targets)

        fine_query_sample_objectness_targets = objectness_targets.unsqueeze(-1).repeat(1, 1, self.ray_num * self.fine_sample_bin_num)
        fine_query_sample_objectness_targets *= fine_valid_query_targets
        fine_query_sample_weights = fine_query_sample_objectness_targets.float() / (
                torch.sum(fine_query_sample_objectness_targets).float() + 1e-6)

        coarse_query_sample_objectness_targets = objectness_targets.unsqueeze(-1).repeat(1, 1, self.ray_num * self.sample_bin_num)
        coarse_query_sample_objectness_targets *= coarse_valid_query_targets
        coarse_query_sample_weights = coarse_query_sample_objectness_targets.float() / (
                torch.sum(coarse_query_sample_objectness_targets).float() + 1e-6)

        return (vote_targets, vote_target_masks, dir_class_targets,
                dir_res_targets, mask_targets, objectness_targets, objectness_weights,
                box_loss_weights, center_targets, assigned_center_targets, valid_gt_weights,
                size_targets, scale_targets,
                fine_query_sample_targets, coarse_query_sample_targets,
                fine_query_sample_weights, coarse_query_sample_weights)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           scale_size_res_pred=None):
        """
        Generate targets for each batch.
        """

        (center_targets, size_half_targets, dir_class_targets,
         dir_res_targets, dir_targets, size_class_targets, size_targets,
         scale_class_targets, scale_targets) = self.bbox_coder.encode(
            gt_bboxes_3d, gt_labels_3d, ret_dir_target=True)

        # get predicted ray scales
        scale_size_res_pred = scale_size_res_pred.unsqueeze(0)
        scale_pred = self.bbox_coder.decode_scale(scale_size_res_pred)
        scale_pred = scale_pred.squeeze(0)

        # generate votes target and obj points
        selected_points_list = []
        max_points_num = 0
        num_points = points.shape[0]

        # sample points equal to latent point number
        points_xyz = points[:, :3].clone().contiguous().unsqueeze(0)
        latent_points_idx = farthest_point_sample(points_xyz, self.fps_num_sample).view(-1).long()
        sample_points_xyz = points_xyz[0][latent_points_idx]

        if self.bbox_coder.with_rot:
            # e.g., SUN RGB-D
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            # TODO:rotationbug
            pcdet_gt_bboxes_3d_batch = gt_bboxes_3d.clone()
            pcdet_gt_bboxes_3d_batch[:, -1] = -pcdet_gt_bboxes_3d_batch[:, -1]
            box_indices_all = [roiaware_pool3d_utils.points_in_boxes_gpu(points=points[None, ::],
                                                boxes=pcdet_gt_bboxes_3d_batch[t:t+1, :].unsqueeze(0)).squeeze(0) for t in range(pcdet_gt_bboxes_3d_batch.shape[0])]

            box_indices_all = torch.stack(box_indices_all) > -1  # gt_num, points_num
            box_indices_all = box_indices_all.T

            pts_instance_mask = points.new_zeros([num_points], dtype=torch.long)
            pts_semantic_mask = points.new_ones([num_points], dtype=torch.long) * self.num_classes

            # generate pts_instance_mask, pts_semantic_mask
            vaild_bbox = []
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                if len(indices) > 0:
                    vaild_bbox.append(True)
                else:
                    vaild_bbox.append(False)
                pts_instance_mask[indices] = i+1
                pts_semantic_mask[indices] = gt_labels_3d[i].item()
            sample_pts_instance_mask = pts_instance_mask[latent_points_idx]

            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d[i, :3].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                    int(j * 3):int(j * 3 +
                                   3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)

            for i in range(gt_labels_3d.shape[0]):
                indices = torch.nonzero(
                    pts_instance_mask == i+1, as_tuple=False).squeeze(-1)
                sample_indices = torch.nonzero(
                    sample_pts_instance_mask == i+1, as_tuple=False).squeeze(-1)

                # if pts_semantic_mask[indices[0]] < self.num_classes:
                if len(sample_indices) != 0 and vaild_bbox[i]:
                    sample_selected_points = sample_points_xyz[sample_indices].clone().detach()
                    sample_select_points_num = sample_selected_points.shape[0]
                    selected_points_list.append(sample_selected_points)
                    if sample_select_points_num > max_points_num:
                        max_points_num = sample_select_points_num
                else:
                    selected_points_list.append(sample_points_xyz.new_tensor([[100., 100., 100.]]))

        elif not self.bbox_coder.with_rot and pts_semantic_mask is not None:
            # e.g., ScanNet
            sample_pts_instance_mask = pts_instance_mask[latent_points_idx]
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points], dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                # TODO: random sampling in data augmentation will cause some objects no points. (bare case)
                indices = torch.nonzero(pts_instance_mask == i, as_tuple=False).squeeze(-1)
                sample_indices = torch.nonzero(sample_pts_instance_mask == i, as_tuple=False).squeeze(-1)
                # vote targets
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    object_points = points[indices, :3]
                    center = 0.5 * (object_points.min(0)[0] + object_points.max(0)[0])
                    vote_targets[indices, :] = center - object_points
                    vote_target_masks[indices] = 1
                    
                    if len(sample_indices) != 0:
                        sample_selected_points = sample_points_xyz[sample_indices].clone().detach()
                        sample_select_points_num = sample_selected_points.shape[0]
                        selected_points_list.append(sample_selected_points)
                        if sample_select_points_num > max_points_num:
                            max_points_num = sample_select_points_num
                    else:
                        # padding fake
                        selected_points_list.append(sample_points_xyz.new_tensor([[100., 100., 100.]]))
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[euclidean_distance1 < self.train_cfg['POS_DISTANCE_THR']] = 1.0
        objectness_masks[euclidean_distance1 > self.train_cfg['NEG_DISTANCE_THR']] = 1.0

        assigned_center_targets = center_targets[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        assigned_size_half_targets = size_half_targets[assignment]
        dir_targets = dir_targets[assignment]

        size_class_targets = size_class_targets[assignment]
        size_targets = size_targets[assignment]
        scale_class_targets = scale_class_targets[assignment]
        scale_targets = scale_targets[assignment]

        ray_vectors = points.new_tensor(generate_ray(self.ray_num)).unsqueeze(0).repeat(proposal_num, 1, 1)
        ray_vectors *= scale_pred[:, None, None]

        mask_targets = gt_labels_3d[assignment].long()

        # compute bbox targets
        canonical_xyz = aggregated_points - assigned_center_targets  # relative position
    
        if self.bbox_coder.with_rot:
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d[assignment, 6], 2).squeeze(1)

        # canonical_xyz to bound distance
        distance_front  = assigned_size_half_targets[:, 0] - canonical_xyz[:, 0]  # x+
        distance_left   = assigned_size_half_targets[:, 1] - canonical_xyz[:, 1]  # y+
        distance_top    = assigned_size_half_targets[:, 2] - canonical_xyz[:, 2]  # z+
        distance_back   = assigned_size_half_targets[:, 0] + canonical_xyz[:, 0]  # x-
        distance_right  = assigned_size_half_targets[:, 1] + canonical_xyz[:, 1]  # y-
        distance_bottom = assigned_size_half_targets[:, 2] + canonical_xyz[:, 2]  # z-

        distance_targets = torch.cat(
            (distance_front.unsqueeze(-1),
             distance_left.unsqueeze(-1),
             distance_top.unsqueeze(-1),
             distance_back.unsqueeze(-1),
             distance_right.unsqueeze(-1),
             distance_bottom.unsqueeze(-1)),
            dim=-1
        )

        inside_mask = (distance_targets >= 0.).all(dim=-1)
        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        pos_mask = (euclidean_distance1 < self.train_cfg['POS_DISTANCE_THR']) & inside_mask
        objectness_targets[pos_mask] = 1

        if max_points_num != 0 and len(selected_points_list) == len(gt_labels_3d):
            for i in range(len(selected_points_list)):
                selected_points_list[i] = F.pad(selected_points_list[i], 
                        (0, 0, 0, max_points_num - selected_points_list[i].shape[0]), mode='constant', value=100.)
            selected_points_list = torch.stack(selected_points_list)
            selected_points_list = selected_points_list[assignment]
        
            coarse_sample_rl_pos = torch.stack([ray_vectors * self.scale_ratio * bin_id / self.sample_bin_num
                                    for bin_id in range(self.sample_bin_num, 0, -1)], dim=1).view(proposal_num, -1, 3) 

            complete_sample_points_xyz = F.pad(sample_points_xyz, (0, 0, 1, 0), mode='constant', value=100.)[None, ::]

            selected_points_list = F.pad(selected_points_list, (0, 0, 1, 0, 0, 0), mode='constant', value=1000.)

            # fine query
            coarse_sample_abs_pos = aggregated_points.unsqueeze(1).repeat(1, self.sample_bin_num * self.ray_num, 1) \
                            + coarse_sample_rl_pos
            flaten_coarse_sample_abs_pos = coarse_sample_abs_pos.view(-1, 3)[None, ::]
            coarse_query_indices_complete = ball_query(self.threshold, 1, complete_sample_points_xyz, flaten_coarse_sample_abs_pos)
            coarse_query_indices_complete = coarse_query_indices_complete.view(proposal_num, self.sample_bin_num * self.ray_num)
            coarse_query_complete_targets = torch.where(coarse_query_indices_complete > 0,
                                            torch.ones_like(coarse_query_indices_complete), torch.zeros_like(coarse_query_indices_complete))
            coarse_query_complete_targets = coarse_query_complete_targets.view(proposal_num, self.sample_bin_num, self.ray_num)

            coarse_weights = (coarse_query_complete_targets.permute(0, 2, 1).clone() + 1e-5)  
            pdf = coarse_weights / torch.sum(coarse_weights, dim=-1, keepdim=True) 
            cdf = torch.cumsum(pdf, dim=-1) 
            cdf = torch.cat([torch.zeros_like(cdf[:, :, :1]), cdf], dim=-1)
            u = torch.linspace(0.+1e-4, 1.-1e-5, steps=self.fine_sample_bin_num)

            u = u.expand(list(cdf.shape[:-1]) + [self.fine_sample_bin_num]).to(cdf.device)  
            ## Invert CDF
            u = u.contiguous()
            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds-1), inds-1)
            above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  
            bins = torch.LongTensor([bin_id for bin_id in range(self.sample_bin_num, 0, -1)] + [0]).to(cdf.device)
            bins = bins[None, None, ...].expand(proposal_num, self.ray_num, self.sample_bin_num + 1)
            matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
            bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), 3, inds_g) 
            bin_center_positions = torch.FloatTensor([bin_id / self.sample_bin_num
                                            for bin_id in range(1, self.sample_bin_num+1, 1)]).to(cdf.device)
            bin_center_positions = bin_center_positions[None, None, ...].expand(proposal_num,
                                                                                self.ray_num, self.sample_bin_num)
            bins_g_above = torch.gather(bin_center_positions, 2, bins_g[..., 1]) + self.threshold
            bins_g_below = torch.gather(bin_center_positions, 2, bins_g[..., 1]) - self.threshold
            denom = (cdf_g[..., 1]-cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u-cdf_g[...,0])/denom

            fine_samples = bins_g_above.float() - t * (bins_g_above.float()-bins_g_below.float()) 
            fine_samples = fine_samples.permute(0, 2, 1).contiguous() 
            fine_sample_rl_pos = ray_vectors.unsqueeze(1).repeat(1, self.fine_sample_bin_num, 1, 1) * self.scale_ratio \
                                                                * fine_samples.unsqueeze(-1).repeat(1, 1, 1, 3)
            fine_sample_rl_pos = fine_sample_rl_pos.view(proposal_num, -1, 3)

            fine_sample_abs_pos = aggregated_points.unsqueeze(1).repeat(1, self.fine_sample_bin_num * self.ray_num, 1) \
                                                        + fine_sample_rl_pos
            fine_query_object_indices = ball_query(self.fine_threshold, 1, selected_points_list, fine_sample_abs_pos).squeeze(-1)
            fine_query_object_targets = torch.where(fine_query_object_indices > 0,
                                torch.ones_like(fine_query_object_indices), torch.zeros_like(fine_query_object_indices))

            # coarse query
            coarse_query_object_indices = ball_query(self.threshold, 1, selected_points_list, coarse_sample_abs_pos).squeeze(-1)
            coarse_query_object_targets = torch.where(coarse_query_object_indices > 0,
                                            torch.ones_like(coarse_query_object_indices), torch.zeros_like(coarse_query_object_indices))
            
            # generate vaild query masks
            flaten_fine_sample_abs_pos = fine_sample_abs_pos.view(1, -1, 3)  
            fine_query_indices_complete = ball_query(self.fine_threshold, 1, complete_sample_points_xyz, flaten_fine_sample_abs_pos)
            fine_vaild_query_targets = (fine_query_indices_complete != 0).view(proposal_num, self.fine_sample_bin_num * self.ray_num).long()
            coarse_vaild_query_targets = (coarse_query_indices_complete != 0).long()
        else:
            coarse_query_object_targets = torch.zeros((proposal_num, self.ray_num * self.sample_bin_num)).to(points.device)
            coarse_vaild_query_targets = torch.zeros((proposal_num, self.ray_num * self.sample_bin_num)).long().to(points.device)
            fine_query_object_targets = torch.zeros((proposal_num, self.ray_num * self.fine_sample_bin_num)).to(points.device)
            fine_vaild_query_targets = torch.zeros((proposal_num, self.ray_num * self.fine_sample_bin_num)).long().to(points.device)

        return (vote_targets, vote_target_masks, dir_class_targets, dir_res_targets, mask_targets, objectness_targets, 
                objectness_masks, dir_targets, center_targets, assigned_center_targets, size_targets, scale_targets, 
                coarse_query_object_targets, fine_query_object_targets, coarse_vaild_query_targets, fine_vaild_query_targets)

    def generate_predicted_boxes(self,
                                 batch_size,
                                 bbox_preds,
                                 use_nms=True):
        """Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        points_num = len(bbox_preds['points']) // batch_size
        points = bbox_preds['points'][:, 1:4].view(batch_size, points_num, 3)
        # decode boxes
        obj_scores = F.softmax(bbox_preds['obj_scores'], dim=-1)[..., -1]
        sem_scores = F.softmax(bbox_preds['sem_scores'], dim=-1)
        bbox3d = self.bbox_coder.decode_bbox(bbox_preds)

        if use_nms:
            batch_size = bbox3d.shape[0]
            batch_bbox_preds, batch_score_preds, batch_cls_preds = [], [], []
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3])
                batch_bbox_preds.append(bbox_selected)
                batch_score_preds.append(score_selected)
                batch_cls_preds.append(labels)
            batch_bbox_preds = torch.stack(batch_bbox_preds)
            batch_score_preds = torch.stack(batch_score_preds)
            batch_cls_preds = torch.stack(batch_cls_preds)

            return batch_bbox_preds, batch_score_preds, batch_cls_preds
        else:
            raise NotImplementedError

    def aligned_3d_nms(self, boxes, scores, classes, thresh):
        """3D NMS for aligned boxes.
        Args:
            boxes (torch.Tensor): Aligned box with shape [n, 6].
            scores (torch.Tensor): Scores of each box.
            classes (torch.Tensor): Class of each box.
            thresh (float): IoU threshold for nms.
        Returns:
            torch.Tensor: Indices of selected boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        z1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]
        z2 = boxes[:, 5]
        area = (x2 - x1) * (y2 - y1) * (z2 - z1)
        zero = boxes.new_zeros(1, )

        score_sorted = torch.argsort(scores)
        pick = []
        while (score_sorted.shape[0] != 0):
            last = score_sorted.shape[0]
            i = score_sorted[-1]
            pick.append(i)

            xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
            yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
            zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
            xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
            yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
            zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
            classes1 = classes[i]
            classes2 = classes[score_sorted[:last - 1]]
            inter_l = torch.max(zero, xx2 - xx1)
            inter_w = torch.max(zero, yy2 - yy1)
            inter_h = torch.max(zero, zz2 - zz1)

            inter = inter_l * inter_w * inter_h
            iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
            iou = iou * (classes1 == classes2).float()
            score_sorted = score_sorted[torch.nonzero(
                iou <= thresh, as_tuple=False).flatten()]

        indices = boxes.new_tensor(pick, dtype=torch.long)
        return indices
    
    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points):
        
        pcdet_bbox = bbox.clone()
        pcdet_bbox[:, -1] = -pcdet_bbox[:, -1]
        box_indices = [roiaware_pool3d_utils.points_in_boxes_gpu(points=points[None, ::],
                                                boxes=pcdet_bbox.unsqueeze(0)).squeeze(0) for t in range(bbox.shape[0])]

        box_indices = torch.stack(box_indices) > -1

        corner3d = boxes_to_corners_3d(bbox)
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = self.aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.NMS_THR)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.SCORE_THR)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.PER_CLASS_PROPOSAL:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected])
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected]
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

class RayBasedGrouping(nn.Module):
    def __init__(self,
                 model_cfg):
        super(RayBasedGrouping, self).__init__()
        self.ray_num = model_cfg['RAY_NUM']
        self.seed_feat_dim = model_cfg['SEED_FEAT_DIM']
        self.sample_bin_num = model_cfg['SAMPLE_BIN_NUM']
        self.sa_radius = model_cfg['SA_RADIUS']
        self.scale_ratio = model_cfg['SCALE_RATIO']
        self.fps_num_sample = model_cfg['FPS_NUM_SAMPLE']
        self.sa_num_sample = model_cfg['SA_NUM_SAMPLE']

        self.fine_sample_bin_num = model_cfg['FINE_SAMPLE_BIN_NUM']
        self.fine_sa_radius = model_cfg['FINE_SA_RADIUS']
        self.fine_sa_num_sample = model_cfg['FINE_SA_NUM_SAMPLE']
        self.reduce_seed_feat_dim = self.seed_feat_dim // 4
        self.num_seed_points = model_cfg['NUM_SEED_POINTS']
        self.fine_seed_aggregation = PointnetSAModuleSSGFBS(npoint=self.num_seed_points,
                                                            radii=self.fine_sa_radius,
                                                            nsamples=self.fine_sa_num_sample,
                                                            mlps=[self.reduce_seed_feat_dim, self.reduce_seed_feat_dim // 2],
                                                            zero_query=True)
        self.coarse_seed_aggregation = PointnetSAModuleSSGFBS(npoint=self.num_seed_points,
                                                            radii=self.sa_radius,
                                                            nsamples=self.sa_num_sample,
                                                            mlps=[self.reduce_seed_feat_dim, self.reduce_seed_feat_dim // 2],
                                                            zero_query=True)

        self.seed_feat_reduce = MLP(in_channel=self.seed_feat_dim, conv_channels=(self.seed_feat_dim // 2, self.reduce_seed_feat_dim))
        self.fine_intersection_module = MLP(in_channel=self.reduce_seed_feat_dim // 2 + self.seed_feat_dim // 2, conv_channels=(self.reduce_seed_feat_dim // 2, 2))
        self.coarse_intersection_module = MLP(in_channel=self.reduce_seed_feat_dim // 2 + self.seed_feat_dim // 2, conv_channels=(self.reduce_seed_feat_dim // 2, 2))
        self.fine_bin_reduce_dim = MLP(in_channel=self.fine_sample_bin_num * self.reduce_seed_feat_dim // 2, conv_channels=(self.reduce_seed_feat_dim // 2,))
        self.fine_ray_reduce_dim = MLP(in_channel=self.ray_num * self.reduce_seed_feat_dim // 2, conv_channels=(self.seed_feat_dim, self.seed_feat_dim // 2))
        self.coarse_bin_reduce_dim = MLP(in_channel=self.sample_bin_num * self.reduce_seed_feat_dim // 2, conv_channels=(self.reduce_seed_feat_dim // 2,))
        self.coarse_ray_reduce_dim = MLP(in_channel=self.ray_num * self.reduce_seed_feat_dim // 2, conv_channels=(self.seed_feat_dim, self.seed_feat_dim // 2))
        self.fuse_layer = MLP(in_channel=self.seed_feat_dim, conv_channels=(self.seed_feat_dim, self.seed_feat_dim // 2))
        self.ray_vector = torch.FloatTensor(generate_ray(self.ray_num))

    def forward(self,
                seed_xyz: torch.Tensor,
                seed_features: torch.Tensor,
                proposals: torch.Tensor,
                scale_pred: torch.Tensor,
                ref_points: torch.Tensor,
                points_cat: torch.Tensor,
                aggregated_features: torch.Tensor):
        batch_size, num_proposal, _ = proposals.shape
        # unsample
        target_sample_idx = farthest_point_sample(points_cat.contiguous(), self.fps_num_sample)

        points_cat = points_cat.transpose(1, 2).contiguous()
        target_sample_xyz = gather_operation(points_cat, target_sample_idx).transpose(1, 2).contiguous()

        dist, idx = three_nn(target_sample_xyz, seed_xyz)
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_features = three_interpolate(seed_features, idx, weight)
        
        # coarse
        coarse_ray_points, coarse_query_indices, coarse_query_sample_targets = \
            self._get_coarse_points(ref_points, scale_pred, target_sample_xyz)

        coarse_ray_points = coarse_ray_points.view(batch_size, -1, 3)
        interpolated_features = self.seed_feat_reduce(interpolated_features)
        
        _, coarse_ray_features, _ = self.coarse_seed_aggregation(target_sample_xyz, interpolated_features, new_xyz=coarse_ray_points)
        coarse_ray_features = coarse_ray_features.view(batch_size, -1, num_proposal, self.sample_bin_num * self.ray_num)
        coarse_repeat_aggregated_features = aggregated_features.unsqueeze(-1).repeat(1, 1, 1, self.ray_num*self.sample_bin_num)
        coarse_intersec_feats = torch.cat([coarse_repeat_aggregated_features, coarse_ray_features], dim=1)
        coarse_intersection_score = self.coarse_intersection_module(coarse_intersec_feats.view(batch_size, -1,
                                                                num_proposal*self.ray_num*self.sample_bin_num))
        coarse_intersection_mask = torch.argmax(coarse_intersection_score, dim=1).view(batch_size, num_proposal,
                                                                                       self.ray_num*self.sample_bin_num, 1)
        coarse_query_sample_targets *= coarse_intersection_mask
        coarse_query_indices *= coarse_intersection_mask.repeat(1, 1, 1, self.sa_num_sample)
        coarse_query_indices = coarse_query_indices.view(batch_size, num_proposal, -1)

        # concat method
        coarse_ray_features = F.pad(coarse_ray_features.unsqueeze(-1), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0), mode='constant', value=0.)
        coarse_intersection_mask = coarse_intersection_mask.unsqueeze(1).repeat(1, self.reduce_seed_feat_dim // 2, 1, 1, 1)
        coarse_ray_features = torch.gather(coarse_ray_features, -1, coarse_intersection_mask)
        coarse_ray_features = coarse_ray_features.squeeze(-1).transpose(2, 3).contiguous()
        coarse_ray_features = coarse_ray_features.reshape(batch_size, -1, self.ray_num * num_proposal)
        coarse_ray_features = self.coarse_bin_reduce_dim(coarse_ray_features)
        coarse_ray_features = coarse_ray_features.view(batch_size, -1, num_proposal)
        coarse_ray_features = self.coarse_ray_reduce_dim(coarse_ray_features)
        coarse_intersection_score = coarse_intersection_score.view(batch_size, 2, num_proposal, self.ray_num*self.sample_bin_num)

        # fine
        fine_ray_points, fine_query_indices = \
            self._get_fine_points(ref_points, scale_pred, target_sample_xyz, coarse_query_sample_targets)
        fine_ray_points = fine_ray_points.view(batch_size, -1, 3)
        _, fine_ray_features, _ = self.fine_seed_aggregation(target_sample_xyz, interpolated_features, new_xyz=fine_ray_points)

        fine_ray_features = fine_ray_features.view(batch_size, -1, num_proposal, self.fine_sample_bin_num * self.ray_num)
        fine_repeat_aggregated_features = aggregated_features.unsqueeze(-1).repeat(1, 1, 1, self.ray_num*self.fine_sample_bin_num)
        fine_intersection_features = torch.cat([fine_repeat_aggregated_features, fine_ray_features], dim=1)
        fine_intersection_score = self.fine_intersection_module(fine_intersection_features.view(batch_size, -1,
                                                                                                num_proposal*self.ray_num*self.fine_sample_bin_num))
        fine_intersection_mask = torch.argmax(fine_intersection_score, dim=1).view(batch_size, num_proposal,
                                                                                   self.ray_num*self.fine_sample_bin_num, 1)
        fine_query_indices *= fine_intersection_mask.repeat(1, 1, 1, self.fine_sa_num_sample)
        fine_query_indices = fine_query_indices.view(batch_size, num_proposal, -1)

        # concat method
        fine_ray_features = F.pad(fine_ray_features.unsqueeze(-1), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0), mode='constant', value=0.)
        fine_intersection_mask = fine_intersection_mask.unsqueeze(1).repeat(1, self.reduce_seed_feat_dim // 2, 1, 1, 1)
        fine_ray_features = torch.gather(fine_ray_features, -1, fine_intersection_mask)
        fine_ray_features = fine_ray_features.squeeze(-1).transpose(2, 3).contiguous()
        fine_ray_features = fine_ray_features.reshape(batch_size, -1, self.ray_num * num_proposal)
        fine_ray_features = self.fine_bin_reduce_dim(fine_ray_features)
        fine_ray_features = fine_ray_features.view(batch_size, -1, num_proposal)
        fine_ray_features = self.fine_ray_reduce_dim(fine_ray_features)
        fine_intersection_score = fine_intersection_score.view(batch_size, 2, num_proposal, self.fine_sample_bin_num*self.ray_num)

        fuse_ray_features = self.fuse_layer(torch.cat([fine_ray_features, coarse_ray_features], dim=1))

        return fuse_ray_features, fine_intersection_score, fine_query_indices, \
               coarse_intersection_score, coarse_query_indices

    def _get_coarse_points(self, aggregation_points, scale_pred, points_xyz):
        batch_size, num_proposal = scale_pred.shape[:2] 
        ray_vector = self.ray_vector.clone().to(scale_pred.device)
        ray_vector = ray_vector[None, None, :, :].repeat(batch_size, num_proposal, 1, 1)
        ray_vector *= scale_pred[:, :, None, None]

        sample_relative_positions = torch.stack([ray_vector * self.scale_ratio * bin_id / self.sample_bin_num
                                                 for bin_id in range(self.sample_bin_num, 0, -1)], dim=2)
        sample_relative_positions = sample_relative_positions.view(batch_size, num_proposal, self.sample_bin_num * self.ray_num, 3)
        center = aggregation_points.clone().unsqueeze(2).repeat(1, 1, self.sample_bin_num * self.ray_num, 1)

        coarse_sample_positions = sample_relative_positions + center 
        flatten_coarse_sample_positions = coarse_sample_positions.view(batch_size, -1, 3)
 
        points_xyz = F.pad(points_xyz, (0, 0, 1, 0, 0, 0), mode='constant', value=100.)

        coarse_ray_points = flatten_coarse_sample_positions.view(batch_size, num_proposal, self.sample_bin_num*self.ray_num, 3)
        coarse_query_indices = ball_query(self.sa_radius, self.sa_num_sample,
                                          points_xyz,
                                          flatten_coarse_sample_positions)
        coarse_query_indices = coarse_query_indices.view(batch_size, num_proposal, self.sample_bin_num * self.ray_num,
                                                         self.sa_num_sample)
        sum_coarse_query_indices = coarse_query_indices.sum(-1).unsqueeze(-1)
        coarse_query_sample_targets = torch.where(sum_coarse_query_indices > 0,
                                                    torch.ones_like(sum_coarse_query_indices), torch.zeros_like(sum_coarse_query_indices))

        return coarse_ray_points, coarse_query_indices, coarse_query_sample_targets

    def _get_fine_points(self, aggregation_points, scale_pred, points_xyz, coarse_query_sample_targets):
        batch_size, num_proposal = scale_pred.shape[:2]  # (B, N)
        ray_vector = self.ray_vector.clone().to(scale_pred.device)
        ray_vector = ray_vector[None, None, :, :].repeat(batch_size, num_proposal, 1, 1)
        ray_vector *= scale_pred[:, :, None, None]

        points_xyz = F.pad(points_xyz, (0, 0, 1, 0, 0, 0), mode='constant', value=100.)
        # fine sample positions
        # fine query
        coarse_query_sample_targets = coarse_query_sample_targets.view(batch_size, num_proposal, self.sample_bin_num, self.ray_num)
        coarse_weights = (coarse_query_sample_targets.clone().permute(0, 1, 3, 2) + 1e-5)  
        pdf = coarse_weights / torch.sum(coarse_weights, dim=-1, keepdim=True)  
        cdf = torch.cumsum(pdf, dim=-1) 
        cdf = torch.cat([torch.zeros_like(cdf[:, :, :, :1]), cdf], dim=-1)
        u = torch.linspace(0.+1e-4, 1.-1e-5, steps=self.fine_sample_bin_num)

        u = u.expand(list(cdf.shape[:-1]) + [self.fine_sample_bin_num]).to(cdf.device)  
        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1) 
        bins = torch.LongTensor([bin_id
                                 for bin_id in range(self.sample_bin_num, 0, -1)] + [0]).to(cdf.device)
        bins = bins[None, None, None, ...].expand(batch_size, num_proposal, self.ray_num, self.sample_bin_num + 1)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], inds_g.shape[3], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(3).expand(matched_shape), 4, inds_g)
        bins_g = torch.gather(bins.unsqueeze(3).expand(matched_shape), 4, inds_g) 
        bin_center_positions = torch.FloatTensor([bin_id / self.sample_bin_num for bin_id in range(1, self.sample_bin_num+1, 1)]).to(cdf.device)
        bin_center_positions = bin_center_positions[None, None, None, ...].expand(batch_size, num_proposal, self.ray_num, self.sample_bin_num)
        bins_g_above = torch.gather(bin_center_positions, 3, bins_g[..., 1]) + self.sa_radius
        bins_g_below = torch.gather(bin_center_positions, 3, bins_g[..., 1]) - self.sa_radius
        denom = (cdf_g[..., 1]-cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom

        fine_samples = bins_g_above.float() - t * (bins_g_above.float()-bins_g_below.float())  
        fine_samples = fine_samples.permute(0, 1, 3, 2).contiguous() 
        fine_relative_sample_positions = ray_vector.unsqueeze(2).repeat(1, 1, self.fine_sample_bin_num, 1, 1) * self.scale_ratio \
                                         * fine_samples.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        fine_relative_sample_positions = fine_relative_sample_positions.view(batch_size, num_proposal, -1, 3)
        fine_sample_positions = aggregation_points.clone().unsqueeze(2).repeat(1, 1, self.fine_sample_bin_num * self.ray_num, 1) \
                                + fine_relative_sample_positions

        fine_ray_points = fine_sample_positions
        fine_query_indices = ball_query(self.fine_sa_radius, self.fine_sa_num_sample, points_xyz,
                                        fine_sample_positions.view(batch_size, -1, 3))
        fine_query_indices = fine_query_indices.view(batch_size, num_proposal, self.fine_sample_bin_num * self.ray_num,
                                                     self.fine_sa_num_sample)

        return fine_ray_points, fine_query_indices
