from xml.sax.handler import all_properties
import torch
from torch import nn
import numpy as np
import MinkowskiEngine as ME
from pcdet.ops.knn import knn
from easydict import EasyDict as edict
from .target_assigner.cagroup3d_assigner import CAGroup3DAssigner, find_points_in_boxes
from pcdet.utils.loss_utils import CrossEntropy, SmoothL1Loss, FocalLoss 
from pcdet.utils.iou3d_loss import IoU3DLoss
from pcdet.models.model_utils.cagroup_utils import reduce_mean, parse_params, Scale, bias_init_with_prob
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu, nms_normal_gpu

class CAGroup3DHead(nn.Module):
    def __init__(self,
                 model_cfg,
                 yaw_parametrization='fcaf3d',
                 predict_boxes=True,
                 **kwargs,
                 ):
        super(CAGroup3DHead, self).__init__()
        n_classes = model_cfg.N_CLASSES
        in_channels = model_cfg.IN_CHANNELS
        out_channels = model_cfg.OUT_CHANNELS
        n_reg_outs = model_cfg.N_REG_OUTS
        voxel_size = model_cfg.VOXEL_SIZE
        semantic_threshold = model_cfg.SEMANTIC_THR
        expand_ratio = model_cfg.EXPAND_RATIO
        assigner = model_cfg.ASSIGNER
        with_yaw = model_cfg.WITH_YAW
        use_sem_score = model_cfg.USE_SEM_SCORE
        cls_kernel = model_cfg.CLS_KERNEL
        loss_centerness = model_cfg.get('LOSS_CENTERNESS', 
                                    edict(NAME='CrossEntropyLoss',
                                    USE_SIGMOID=True,
                                    LOSS_WEIGHT=1.0))
        loss_bbox = model_cfg.get('LOSS_BBOX', 
                                edict(NAME='IoU3DLoss', LOSS_WEIGHT=1.0))
        loss_cls = model_cfg.get('LOSS_CLS',
                                edict(
                                NAME='FocalLoss',
                                USE_SIGMOID=True,
                                GAMMA=2.0,
                                ALPHA=0.25,
                                LOSS_WEIGHT=1.0))
        loss_sem = model_cfg.get('LOSS_SEM',
                                edict(
                                NAME='FocalLoss',
                                USE_SIGMOID=True,
                                GAMMA=2.0,
                                ALPHA=0.25,
                                LOSS_WEIGHT=1.0))
        loss_offset = model_cfg.get('LOSS_OFFSET', 
                                    edict(NAME='SmoothL1Loss', BETA=0.04, 
                                    REDUCTION='sum', LOSS_WEIGHT=1.0))
        nms_config = model_cfg.get('NMS_CONFIG',
                                    edict(SCORE_THR=0.01,
                                        NMS_PRE=1000,
                                        IOU_THR=0.5,)) 
        self.voxel_size = voxel_size
        self.yaw_parametrization = yaw_parametrization
        self.cls_kernel = cls_kernel
        self.assigner = CAGroup3DAssigner(assigner)
        self.loss_centerness = CrossEntropy(**parse_params(loss_centerness))
        self.loss_bbox = IoU3DLoss(**parse_params(loss_bbox))
        self.loss_cls = FocalLoss(**parse_params(loss_cls))
        self.loss_sem = FocalLoss(**parse_params(loss_sem))
        self.loss_offset = SmoothL1Loss(**parse_params(loss_offset))

        self.nms_cfg = nms_config
        self.use_sem_score = use_sem_score
        self.semantic_threshold = semantic_threshold
        self.predict_boxes = predict_boxes
        self.n_classes = n_classes
        if self.n_classes == 18:
            self.voxel_size_list = [[0.2309, 0.2435, 0.2777],
                                    [0.5631, 0.5528, 0.3579],
                                    [0.1840, 0.1845, 0.2155],
                                    [0.4187, 0.4536, 0.2503],
                                    [0.2938, 0.3203, 0.1899],
                                    [0.1595, 0.1787, 0.5250],
                                    [0.2887, 0.2174, 0.3445],
                                    [0.2497, 0.3147, 0.5063],
                                    [0.0634, 0.1262, 0.1612],
                                    [0.4332, 0.5691, 0.0810],
                                    [0.3088, 0.4212, 0.2627],
                                    [0.4130, 0.1966, 0.5044],
                                    [0.1995, 0.2133, 0.3897],
                                    [0.1260, 0.1137, 0.5254],
                                    [0.1781, 0.1774, 0.2218],
                                    [0.1526, 0.1520, 0.0904],
                                    [0.3453, 0.3164, 0.1491],
                                    [0.1426, 0.1477, 0.1741]] # scannet
        else:
            self.voxel_size_list = [[0.6343, 0.4861, 0.2782],
                                    [0.2373, 0.3839, 0.2155],
                                    [0.2771, 0.5602, 0.2536],
                                    [0.1776, 0.1659, 0.2482],
                                    [0.2097, 0.1363, 0.2269],
                                    [0.2086, 0.4039, 0.2209],
                                    [0.1586, 0.3008, 0.3519],
                                    [0.1502, 0.1896, 0.2050],
                                    [0.1214, 0.3213, 0.5067],
                                    [0.2298, 0.4195, 0.1418]] # sunrgbd
        lower_size = 0.04
        self.voxel_size_list = np.clip(np.array(self.voxel_size_list) / 2., lower_size, 1.0).tolist()
        self.expand = expand_ratio
        self.gt_per_seed = 3 # only use for sunrgbd
        self.with_yaw = with_yaw
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)
        self.init_weights()

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_block_with_kernels(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.ModuleList([
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3),
            nn.Sequential(
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU(),
                ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU())])

    @staticmethod
    def _make_up_block_with_parameters(in_channels, out_channels, kernel_size, stride):
        return nn.ModuleList([
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3),
            nn.Sequential(
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU(),
                # ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                # ME.MinkowskiBatchNorm(out_channels),
                # ME.MinkowskiELU()
                )])

    # @staticmethod
    def _make_offset_block(self, in_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels, 9 if self.with_yaw else 3, kernel_size=1, dimension=3), # 3vote for sunrgbd
        )

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.__setattr__(f'offset_block', self._make_offset_block(out_channels))
        self.__setattr__(f'feature_offset', self._make_block(out_channels, 3*out_channels if self.with_yaw else out_channels)) # 3vote for sunrgbd

        # head layers
        self.semantic_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.centerness_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, dimension=3)
        self.reg_conv = ME.MinkowskiConvolution(out_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_classes)])
        self.cls_individual_out = nn.ModuleList([self._make_block_with_kernels(out_channels, out_channels, self.cls_kernel) for _ in range(n_classes)])
        self.cls_individual_up = nn.ModuleList([self._make_up_block_with_parameters(out_channels,
                                                        out_channels, self.expand, self.expand) for _ in range(n_classes)])
        self.cls_individual_fuse = nn.ModuleList([self._make_block_with_kernels(out_channels*2, out_channels, 1) for _ in range(n_classes)])
        self.cls_individual_expand_out = nn.ModuleList([self._make_block_with_kernels(out_channels, out_channels, 5) for _ in range(n_classes)])

    def init_weights(self):
        nn.init.normal_(self.centerness_conv.kernel, std=.01)
        nn.init.normal_(self.reg_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
        nn.init.normal_(self.semantic_conv.kernel, std=.01)
        nn.init.constant_(self.semantic_conv.bias, bias_init_with_prob(.01))
        for cls_id in range(self.n_classes):
            nn.init.normal_(self.cls_individual_out[cls_id][0].kernel, std=.01)

    def forward(self, input_dict, return_middle_feature=True):
        batch_size = input_dict['batch_size']
        outs = []
        out = input_dict['sp_tensor'] # semantic input from backbone3d
        decode_out = [None, None, None, out]
        semantic_scores = self.semantic_conv(out)

        pad_id = semantic_scores.C.new_tensor([permutation[0] for permutation in semantic_scores.decomposition_permutations]).long()
        # compute points range
        scene_coord = out.C[:, 1:].clone()
        max_bound = (scene_coord.max(0)[0] + out.coordinate_map_key.get_key()[0][0]) * self.voxel_size
        min_bound = (scene_coord.min(0)[0] - out.coordinate_map_key.get_key()[0][0]) * self.voxel_size

        voxel_offsets = self.__getattr__(f'offset_block')(out) # n,3x3
        offset_features = self.__getattr__(f'feature_offset')(out).F # n,3xc

        if not self.with_yaw: 
            voted_coordinates = out.C[:, 1:].clone() * self.voxel_size + voxel_offsets.F.clone().detach()
            voted_coordinates[:, 0] = torch.clamp(voted_coordinates[:, 0], max=max_bound[0], min=min_bound[0])
            voted_coordinates[:, 1] = torch.clamp(voted_coordinates[:, 1], max=max_bound[1], min=min_bound[1])
            voted_coordinates[:, 2] = torch.clamp(voted_coordinates[:, 2], max=max_bound[2], min=min_bound[2])
        else: # 3vote
            voted_coordinates = out.C[:, 1:].clone().view(-1, 1, 3).repeat(1,3,1) * self.voxel_size + voxel_offsets.F.clone().detach().view(-1,3,3)
            voted_coordinates[:, :, 0] = torch.clamp(voted_coordinates[:, :, 0], max=max_bound[0], min=min_bound[0])
            voted_coordinates[:, :, 1] = torch.clamp(voted_coordinates[:, :, 1], max=max_bound[1], min=min_bound[1])
            voted_coordinates[:, :, 2] = torch.clamp(voted_coordinates[:, :, 2], max=max_bound[2], min=min_bound[2])

        for cls_id in range(self.n_classes):
            with torch.no_grad():
                cls_semantic_scores = semantic_scores.F[:, cls_id].sigmoid()
                cls_selected_id = torch.nonzero(cls_semantic_scores > self.semantic_threshold).squeeze(1)
                cls_selected_id = torch.cat([cls_selected_id, pad_id])

            if not self.with_yaw:
                coordinates = out.C.float().clone()[cls_selected_id]
                coordinates[:, 1:4] = voted_coordinates[cls_selected_id]  # N,4 (b,x,y,z)
            else: # 3vote
                coordinates = out.C.float().clone()[cls_selected_id].view(-1, 1, 4).repeat(1, 3, 1) # n, 3, 4
                coordinates[:, :, 1:4] = voted_coordinates[cls_selected_id]  # N, 3, 4 (b,x,y,z)

            ori_coordinates = out.C.float().clone()[cls_selected_id]
            ori_coordinates[:, 1:4] *= self.voxel_size
            # 3 votes
            coordinates = coordinates.reshape([-1, 4])
            fuse_coordinates = torch.cat([coordinates, ori_coordinates], dim=0)

            if not self.with_yaw:
                fuse_features = torch.cat([offset_features[cls_selected_id], out.F[cls_selected_id]], dim=0)
            else: # 3vote
                offset_features = offset_features.view(offset_features.shape[0], 3, -1)
                select_offset_features = offset_features[cls_selected_id] # n, 3, c
                select_offset_features = select_offset_features.reshape([-1, select_offset_features.shape[-1]]) # 3n, c
                fuse_features = torch.cat([select_offset_features, out.F[cls_selected_id]], dim=0)
            
            voxel_size = torch.tensor(self.voxel_size_list[cls_id], device=fuse_features.device)
            voxel_coord = fuse_coordinates.clone().int()
            voxel_coord[:, 1:] = (fuse_coordinates[:, 1:] / voxel_size).floor()
            cls_individual_map = ME.SparseTensor(coordinates=voxel_coord, features=fuse_features,
                                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            cls_individual_map = self.cls_individual_out[cls_id](cls_individual_map)

            # expand feature map
            cls_voxel_coord = fuse_coordinates.clone().int()
            expand = self.expand
            cls_voxel_coord[:, 1:] = (fuse_coordinates[:, 1:] / (voxel_size * expand)).floor()
            cls_individual_map_expand = ME.SparseTensor(coordinates=cls_voxel_coord, features=fuse_features,
                                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            expand_coord = cls_individual_map_expand.C
            expand_coord[:, 1:] *= expand
            cls_individual_map_expand = ME.SparseTensor(coordinates=expand_coord, features=cls_individual_map_expand.F,
                                                        tensor_stride=expand,
                                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            cls_individual_map_expand = self.cls_individual_expand_out[cls_id](cls_individual_map_expand)
            cls_individual_map_up = self.cls_individual_up[cls_id][0](cls_individual_map_expand, cls_individual_map.C)
            cls_individual_map_up = self.cls_individual_up[cls_id][1](cls_individual_map_up)
            cls_individual_map_out = ME.SparseTensor(coordinates=cls_individual_map.C,
                                                    features=torch.cat([cls_individual_map_up.F, cls_individual_map.F], dim=-1))
            cls_individual_map_out = self.cls_individual_fuse[cls_id](cls_individual_map_out)

            prediction = self.forward_single(cls_individual_map_out, self.scales[cls_id], self.voxel_size_list[cls_id])
            scores = prediction[-1]
            outs.append(list(prediction[:-1]))

        all_prediction = zip(*outs)
        centernesses, bbox_preds, cls_scores, voxel_points = list(all_prediction)
        out_dict = dict()
        out_dict['one_stage_results'] = [centernesses, bbox_preds, cls_scores, voxel_points], semantic_scores, voxel_offsets
        if not return_middle_feature:
            out_dict['middle_feature_list'] = None 
        else:
            out_dict['middle_feature_list'] = decode_out
        
        # prepare for two-stage refinement
        if self.predict_boxes:
            img_metas = [None for _ in range(batch_size)]
            bbox_list = self.get_bboxes(centernesses, bbox_preds, cls_scores, voxel_points, img_metas, rescale=False)
            out_dict['pred_bbox_list'] = bbox_list
            if 'gt_boxes' in input_dict.keys() and 'gt_bboxes_3d' not in input_dict.keys():
                gt_bboxes_3d = []
                gt_labels_3d = []
                device = input_dict['points'].device
                for b in range(len(input_dict['gt_boxes'])):
                    gt_bboxes_b = []
                    gt_labels_b = []
                    for _item in input_dict['gt_boxes'][b]:
                        if not (_item == 0.).all(): 
                            gt_bboxes_b.append(_item[:7])  
                            gt_labels_b.append(_item[7:8]) 
                    if len(gt_bboxes_b) == 0:
                        gt_bboxes_b = torch.zeros((0, 7), dtype=torch.float32).to(device)
                        gt_labels_b = torch.zeros((0,), dtype=torch.int).to(device)
                    else:
                        gt_bboxes_b = torch.stack(gt_bboxes_b)
                        gt_labels_b = torch.cat(gt_labels_b).int()
                    gt_bboxes_3d.append(gt_bboxes_b)
                    gt_labels_3d.append(gt_labels_b)
                out_dict['gt_bboxes_3d'] = gt_bboxes_3d
                out_dict['gt_labels_3d'] = gt_labels_3d
        
        return out_dict

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             points, # fused points
             semantic_scores,
             voxel_offset,
             gt_bboxes,
             gt_labels,
             scene_points,
             img_metas,
             pts_semantic_mask,
             pts_instance_mask):

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for _ in range(len(centernesses[0]))]
            pts_instance_mask = pts_semantic_mask
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels) \
               == len(pts_instance_mask) == len(pts_semantic_mask) == len(scene_points)

        semantic_scores_list = []
        semantic_points_list = []
        for permutation in semantic_scores.decomposition_permutations:
            semantic_scores_list.append(semantic_scores.F[permutation])
            semantic_points_list.append(semantic_scores.C[permutation, 1:] * self.voxel_size)

        voxel_offset_list = []
        voxel_points_list = []
        for permutation in voxel_offset.decomposition_permutations:
            voxel_offset_list.append(voxel_offset.F[permutation])
            voxel_points_list.append(voxel_offset.C[permutation, 1:] * self.voxel_size)

        loss_centerness, loss_bbox, loss_cls, loss_sem, loss_vote = [], [], [], [], []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls, img_loss_sem, img_loss_vote = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                voxel_offset_preds=voxel_offset_list[i],
                original_points=voxel_points_list[i],
                semantic_scores=semantic_scores_list[i],
                semantic_points=semantic_points_list[i],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                scene_points=scene_points[i],
                pts_semantic_mask=pts_semantic_mask[i],
                pts_instance_mask=pts_instance_mask[i],
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
            loss_sem.append(img_loss_sem)
            loss_vote.append(img_loss_vote)
        
        
        loss_centerness=torch.mean(torch.stack(loss_centerness))
        loss_bbox=torch.mean(torch.stack(loss_bbox))
        loss_cls=torch.mean(torch.stack(loss_cls))
        loss_sem=torch.mean(torch.stack(loss_sem))
        loss_vote=torch.mean(torch.stack(loss_vote))

        loss = loss_centerness + loss_bbox + loss_cls + loss_sem + loss_vote
        tb_dict = dict(
            loss_centerness=loss_centerness.item(),
            loss_bbox=loss_bbox.item(),
            loss_cls=loss_cls.item(),
            loss_sem=loss_sem.item(),
            loss_vote=loss_vote.item()
        )
        tb_dict['one_stage_loss'] = loss.item()

        return loss, tb_dict

    # per image
    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     points,
                     voxel_offset_preds,
                     original_points,
                     semantic_scores,
                     semantic_points,
                     img_meta,
                     gt_bboxes,
                     gt_labels,
                     scene_points,
                     pts_semantic_mask,
                     pts_instance_mask):
        with torch.no_grad():
            semantic_labels, ins_labels = self.assigner.assign_semantic(semantic_points, gt_bboxes, gt_labels, self.n_classes)
            centerness_targets, bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)
            # compute offset targets
            if self.with_yaw:
                num_points = original_points.shape[0]
                vote_targets = original_points.new_zeros([num_points, 3 * self.gt_per_seed])
                vote_target_masks = original_points.new_zeros([num_points],
                                                    dtype=torch.long)
                vote_target_idx = original_points.new_zeros([num_points], dtype=torch.long)
                box_indices_all = find_points_in_boxes(points=original_points, gt_bboxes=gt_bboxes) # n_points. n_boxes
                for i in range(gt_labels.shape[0]):
                    box_indices = box_indices_all[:, i]
                    indices = torch.nonzero(
                        box_indices, as_tuple=False).squeeze(-1)
                    selected_points = original_points[indices]
                    vote_target_masks[indices] = 1
                    vote_targets_tmp = vote_targets[indices]
                    votes = gt_bboxes[i, :3].unsqueeze(
                        0).to(selected_points.device) - selected_points[:, :3]

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
                offset_targets = []
                offset_masks = []
                offset_targets.append(vote_targets)
                offset_masks.append(vote_target_masks)
            
            elif pts_semantic_mask is not None and pts_instance_mask is not None:
                allp_offset_targets = torch.zeros_like(scene_points[:, :3])
                allp_offset_masks = scene_points.new_zeros(len(scene_points))
                instance_center = scene_points.new_zeros((pts_instance_mask.max()+1, 3))
                instance_match_gt_id = -scene_points.new_ones((pts_instance_mask.max()+1)).long()
                for i in torch.unique(pts_instance_mask):
                    indices = torch.nonzero(
                        pts_instance_mask == i, as_tuple=False).squeeze(-1)
                    if pts_semantic_mask[indices[0]] < self.n_classes:
                        selected_points = scene_points[indices, :3]
                        center = 0.5 * (
                                selected_points.min(0)[0] + selected_points.max(0)[0])
                        allp_offset_targets[indices, :] = center - selected_points
                        allp_offset_masks[indices] = 1
                        match_gt_id = torch.argmin(torch.cdist(center.view(1, 1, 3),
                                                                gt_bboxes[:, :3].unsqueeze(0).to(center.device)).view(-1))
                        instance_match_gt_id[i] = match_gt_id
                        instance_center[i] = gt_bboxes[:, :3][match_gt_id].to(center.device)
                    else:
                        instance_center[i] = torch.ones_like(instance_center[i]) * (-10000.)
                        instance_match_gt_id[i] = -1

                # compute points offsets of each scale seed points
                offset_targets = []
                offset_masks = []
                knn_number = 1
                idx = knn(knn_number, scene_points[None, :, :3].contiguous(), original_points[None, ::])[0].long()
                instance_idx = pts_instance_mask[idx.view(-1)].view(idx.shape[0], idx.shape[1])

                # condition1: all the points must belong to one instance
                valid_mask = (instance_idx == instance_idx[0]).all(0)

                max_instance_num = pts_instance_mask.max()+1
                arange_tensor = torch.arange(max_instance_num).unsqueeze(1).unsqueeze(2).to(instance_idx.device)
                arange_tensor = arange_tensor.repeat(1, instance_idx.shape[0], instance_idx.shape[1]) # instance_num, k, points
                instance_idx = instance_idx[None, ::].repeat(max_instance_num, 1, 1)

                max_instance_idx = torch.argmax((instance_idx == arange_tensor).sum(1), dim=0)
                offset_t = instance_center[max_instance_idx] - original_points
                offset_m = torch.where(offset_t < -100., torch.zeros_like(offset_t), torch.ones_like(offset_t)).all(1)
                offset_t = torch.where(offset_t < -100., torch.zeros_like(offset_t), offset_t)
                offset_m *= valid_mask

                offset_targets.append(offset_t)
                offset_masks.append(offset_m)

            else:
                raise NotImplementedError

        centerness = torch.cat(centernesses)
        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        offset_targets = torch.cat(offset_targets) # num_points x 9
        offset_masks = torch.cat(offset_masks) # num_points

        # vote loss
        if self.with_yaw:
            offset_weights_expand = (offset_masks.float() / (offset_masks.float().sum() + 1e-6)).unsqueeze(1).repeat(1, 9)
            vote_points = original_points.repeat(1, self.gt_per_seed) + voxel_offset_preds # num_points, 9
            vote_gt = original_points.repeat(1, self.gt_per_seed) + offset_targets # num_points, 9
            loss_offset = self.loss_offset(vote_points, vote_gt, weight=offset_weights_expand)
        else:
            offset_weights_expand = (offset_masks.float() / torch.ones_like(offset_masks).float().sum() + 1e-6).unsqueeze(1).repeat(1, 3)
            loss_offset = self.loss_offset(voxel_offset_preds, offset_targets, weight=offset_weights_expand)
            
        # semantic loss
        sem_n_pos = torch.tensor(len(torch.nonzero(semantic_labels >= 0).squeeze(1)), dtype=torch.float, device=centerness.device)
        sem_n_pos = max(reduce_mean(sem_n_pos), 1.)
        loss_sem = self.loss_sem(semantic_scores, semantic_labels, avg_factor=sem_n_pos)

        # skip background
        # centerness loss and bbox loss
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        
        return loss_centerness, loss_bbox, loss_cls, loss_sem, loss_offset

    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i]
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta):
        mlvl_bboxes, mlvl_scores = [], []
        mlvl_sem_scores = []
        for centerness, bbox_pred, cls_score, point in zip(
            centernesses, bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid() * centerness.sigmoid()
            if self.use_sem_score:
                sem_scores = cls_score.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.nms_cfg.NMS_PRE > 0:
                _, ids = max_scores.topk(self.nms_cfg.NMS_PRE)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]
                if self.use_sem_score:
                    sem_scores = sem_scores[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if self.use_sem_score:
                mlvl_sem_scores.append(sem_scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        sem_scores = torch.cat(mlvl_sem_scores) if self.use_sem_score else None
        if self.use_sem_score:
            # if use class_agnostic nms when training
            if self.training and self.nms_cfg.get('SCORE_THR_AGNOSTIC', None) is not None:
                bboxes, scores, labels, sem_scores = self.class_agnostic_nms(bboxes, scores, img_meta, sem_scores=sem_scores)
            else:
                bboxes, scores, labels, sem_scores = self._nms(bboxes, scores, img_meta, sem_scores=sem_scores)
            return bboxes, scores, labels, sem_scores
        else:
            if self.training and self.nms_cfg.get('SCORE_THR_AGNOSTIC', None) is not None:
                bboxes, scores, labels = self.class_agnostic_nms(bboxes, scores, img_meta)
            else:
                bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
            return bboxes, scores, labels

    # per scale
    def forward_single(self, x, scale, voxel_size):
        centerness = self.centerness_conv(x).features
        scores = self.cls_conv(x)
        cls_score = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.reg_conv(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        voxel_size = torch.tensor(voxel_size, device=cls_score.device)
        for i in range(len(points)):
            points[i] = points[i] * voxel_size
            assert len(points[i]) > 0, "forward empty"

        return centernesses, bbox_preds, cls_scores, points, prune_scores

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        if self.yaw_parametrization == 'naive':
            # ..., alpha
            return torch.cat((
                base_bbox,
                bbox_pred[:, 6:7]
            ), -1)
        elif self.yaw_parametrization == 'sin-cos':
            # ..., sin(a), cos(a)
            norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
            sin = bbox_pred[:, 6:7] / norm
            cos = bbox_pred[:, 7:8] / norm
            return torch.cat((
                base_bbox,
                torch.atan2(sin, cos)
            ), -1)
        else:  # self.yaw_parametrization == 'fcaf3d'
            # ..., sin(2a)ln(q), cos(2a)ln(q)
            scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
            q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
            alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
            return torch.stack((
                x_center,
                y_center,
                z_center,
                scale / (1 + q),
                scale / (1 + q) * q,
                bbox_pred[:, 5] + bbox_pred[:, 4],
                alpha
            ), dim=-1)
    
    def class_agnostic_nms(self, bboxes, scores, img_meta, sem_scores=None):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        max_scores, labels = scores.max(dim=1)
        if yaw_flag:
            nms_function = nms_gpu
        else:
            bboxes = torch.cat((
                    bboxes, torch.zeros_like(bboxes[:, :1])), dim=1)
            nms_function = nms_normal_gpu
        ids = max_scores > self.nms_cfg.SCORE_THR_AGNOSTIC
        if not ids.any():
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))
            if sem_scores is not None:
                nms_sem_scores = bboxes.new_zeros((0, n_classes))
        else:
            class_bboxes = bboxes[ids]
            class_scores = max_scores[ids]
            class_labels = labels[ids]
            if sem_scores is not None:
                class_sem_scores = sem_scores[ids] # n, n_class
            # correct_heading
            correct_class_bboxes = class_bboxes.clone()
            if yaw_flag:
                correct_class_bboxes[..., 6] *= -1
            nms_ids, _ = nms_function(correct_class_bboxes, class_scores, self.nms_cfg.IOU_THR)
            nms_bboxes = class_bboxes[nms_ids]
            nms_scores = class_scores[nms_ids]
            nms_labels = class_labels[nms_ids]
            if sem_scores is not None:
                nms_sem_scores = class_sem_scores[nms_ids]

        if not yaw_flag:
            fake_heading = nms_bboxes.new_zeros(nms_bboxes.shape[0], 1)
            nms_bboxes = torch.cat([nms_bboxes[:, :6], fake_heading], dim=1)
        if sem_scores is not None:
            return nms_bboxes, nms_scores, nms_labels, nms_sem_scores
        else:
            return nms_bboxes, nms_scores, nms_labels

    def _nms(self, bboxes, scores, img_meta, sem_scores=None):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        nms_sem_scores = []
        for i in range(n_classes):
            ids = scores[:, i] > self.nms_cfg.SCORE_THR
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if sem_scores is not None:
                class_sem_scores = sem_scores[ids] # n,n_class
            if yaw_flag:
                nms_function = nms_gpu
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms_normal_gpu
            # check heading 
            correct_class_bboxes = class_bboxes.clone()
            if yaw_flag:
                correct_class_bboxes[..., 6] *= -1
            nms_ids, _ = nms_function(correct_class_bboxes, class_scores, self.nms_cfg.IOU_THR)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))
            if sem_scores is not None:
                nms_sem_scores.append(class_sem_scores[nms_ids])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
            if sem_scores is not None:
                nms_sem_scores = torch.cat(nms_sem_scores, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))
            if sem_scores is not None:
                nms_sem_scores = bboxes.new_zeros((0, n_classes))

        if not yaw_flag:
            fake_heading = nms_bboxes.new_zeros(nms_bboxes.shape[0], 1)
            nms_bboxes = torch.cat([nms_bboxes[:, :6], fake_heading], dim=1)
        if sem_scores is not None:
            return nms_bboxes, nms_scores, nms_labels, nms_sem_scores
        else:
            return nms_bboxes, nms_scores, nms_labels


