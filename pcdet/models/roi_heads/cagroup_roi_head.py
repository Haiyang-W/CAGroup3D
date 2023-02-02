from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
from pcdet.utils.iou3d_loss import IoU3DLoss
from pcdet.utils.loss_utils import WeightedSmoothL1Loss
from .target_assigner.cagroup_proposal_target_layer import ProposalTargetLayer
from pcdet.models.model_utils.cagroup_utils import CAGroupResidualCoder as ResidualCoder
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu, nms_normal_gpu
from pcdet.utils import common_utils

class SimplePoolingLayer(nn.Module):
    def __init__(self, channels=[128,128,128], grid_kernel_size = 5, grid_num = 7, voxel_size=0.04, coord_key=2,
                    point_cloud_range=[-5.12*3, -5.12*3, -5.12*3, 5.12*3, 5.12*3, 5.12*3], # simply use a large range
                    corner_offset_emb=False, pooling=False):
        super(SimplePoolingLayer, self).__init__()
        # build conv
        self.voxel_size = voxel_size
        self.coord_key = coord_key
        grid_size = [int((point_cloud_range[3] - point_cloud_range[0])/voxel_size), 
                     int((point_cloud_range[4] - point_cloud_range[1])/voxel_size), 
                     int((point_cloud_range[5] - point_cloud_range[2])/voxel_size)]
        self.grid_size = grid_size
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.grid_num = grid_num
        self.pooling = pooling
        self.count = 0
        self.grid_conv = ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=grid_kernel_size, dimension=3)
        self.grid_bn = ME.MinkowskiBatchNorm(channels[1])
        self.grid_relu = ME.MinkowskiELU()
        if self.pooling:
            self.pooling_conv = ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=grid_num, dimension=3)
            self.pooling_bn = ME.MinkowskiBatchNorm(channels[1])

        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.grid_conv.kernel, std=.01)
        if self.pooling:
            nn.init.normal_(self.pooling_conv.kernel, std=.01)

    def forward(self, sp_tensor, grid_points, grid_corners=None, box_centers=None, batch_size=None):
        """
        Args:
            sp_tensor: minkowski tensor
            grid_points: bxnum_roisx216, 4 (b,x,y,z)
            grid_corners (optional): bxnum_roisx216, 8, 3
            box_centers: bxnum_rois, 4 (b,x,y,z)
        """
        grid_coords = grid_points.long()
        grid_coords[:, 1:4] = torch.floor(grid_points[:, 1:4] / self.voxel_size) # get coords (grid conv center)
        grid_coords[:, 1:4] = torch.clamp(grid_coords[:, 1:4], min=-self.grid_size[0] / 2 + 1, max=self.grid_size[0] / 2 - 1) # -192 ~ 192
        grid_coords_positive = grid_coords[:, 1:4] + self.grid_size[0] // 2 
        merge_coords = grid_coords[:, 0] * self.scale_xyz + \
                        grid_coords_positive[:, 0] * self.scale_yz + \
                        grid_coords_positive[:, 1] * self.scale_z + \
                        grid_coords_positive[:, 2] 
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        unq_grid_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1) 
        unq_grid_coords[:, 1:4] -= self.grid_size[0] // 2
        unq_grid_coords[:, 1:4] *= self.coord_key
        unq_grid_sp_tensor = self.grid_relu(self.grid_bn(self.grid_conv(sp_tensor, unq_grid_coords.int()))) 
        unq_features = unq_grid_sp_tensor.F
        unq_coords = unq_grid_sp_tensor.C
        new_features = unq_features[unq_inv]

        if self.pooling:
            # fake grid
            fake_grid_coords = torch.ones(self.grid_num, self.grid_num, self.grid_num, device=unq_grid_coords.device)
            fake_grid_coords = torch.nonzero(fake_grid_coords) - self.grid_num // 2 
            fake_grid_coords = fake_grid_coords.unsqueeze(0).repeat(grid_coords.shape[0] // fake_grid_coords.shape[0], 1, 1) 
            # fake center
            fake_centers = fake_grid_coords.new_zeros(fake_grid_coords.shape[0], 3) 
            fake_batch_idx = torch.arange(fake_grid_coords.shape[0]).to(fake_grid_coords.device) 
            fake_center_idx = fake_batch_idx.reshape([-1, 1])
            fake_center_coords = torch.cat([fake_center_idx, fake_centers], dim=-1).int() 
            
            fake_grid_idx = fake_batch_idx.reshape([-1, 1, 1]).repeat(1, fake_grid_coords.shape[1], 1) 
            fake_grid_coords = torch.cat([fake_grid_idx, fake_grid_coords], dim=-1).reshape([-1, 4]).int()

            grid_sp_tensor = ME.SparseTensor(coordinates=fake_grid_coords, features=new_features)
            pooled_sp_tensor = self.pooling_conv(grid_sp_tensor, fake_center_coords) 
            pooled_sp_tensor = self.pooling_bn(pooled_sp_tensor) 
            return pooled_sp_tensor.F
        else:
            return new_features


class CAGroup3DRoIHead(nn.Module):
    def __init__(self, model_cfg, cls_loss_type='BinaryCrossEntropy', reg_loss_type='smooth-l1', **kwargs):
        super(CAGroup3DRoIHead, self).__init__()
        middle_feature_source = model_cfg.MIDDLE_FEATURE_SOURCE
        num_class = model_cfg.NUM_CLASSES
        code_size = model_cfg.CODE_SIZE
        grid_size = model_cfg.GRID_SIZE
        voxel_size = model_cfg.VOXEL_SIZE
        coord_key = model_cfg.COORD_KEY
        mlps = model_cfg.MLPS
        enlarge_ratio = model_cfg.ENLARGE_RATIO
        shared_fc = model_cfg.get('SHARED_FC', [256,256])
        cls_fc = model_cfg.get('CLS_FC', [256,256])
        reg_fc = model_cfg.get('REG_FC', [256,256])
        dp_ratio = model_cfg.get('DP_RATIO', 0.3)
        test_score_thr = model_cfg.get('TEST_SCORE_THR', 0.01)
        test_iou_thr = model_cfg.get('TEST_IOU_THR', 0.5)
        roi_per_image = model_cfg.get('ROI_PER_IMAGE', 128)
        roi_fg_ratio = model_cfg.get('ROI_FG_RATIO', 0.9)
        reg_fg_thresh = model_cfg.get('REG_FG_THRESH', 0.3)
        roi_conv_kernel = model_cfg.get('ROI_CONV_KERNEL', 5)
        encode_angle_by_sincos = model_cfg.get('ENCODE_SINCOS', False)
        use_iou_loss = model_cfg.get('USE_IOU_LOSS', False)
        use_grid_offset = model_cfg.get('USE_GRID_OFFSET', False)
        # pooling config
        use_simple_pooling = model_cfg.get('USE_SIMPLE_POOLING', True)
        use_center_pooling = model_cfg.get('USE_CENTER_POOLING', True)
        loss_weight = model_cfg.LOSS_WEIGHTS

        self.middle_feature_source = middle_feature_source # default [3] : only use semantic feature from backbone3d
        self.num_class = num_class
        self.code_size = code_size
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.enlarge_ratio = enlarge_ratio
        self.mlps = mlps
        self.shared_fc = shared_fc
        self.test_score_thr = test_score_thr
        self.test_iou_thr = test_iou_thr
        self.cls_fc = cls_fc
        self.reg_fc = reg_fc
        self.cls_loss_type = cls_loss_type
        self.reg_loss_type = reg_loss_type
        self.count = 0

        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.use_iou_loss = use_iou_loss
        if self.use_iou_loss:
            self.iou_loss_computer = IoU3DLoss(loss_weight=1.0, with_yaw=self.code_size > 6)
        self.use_grid_offset = use_grid_offset
        self.use_simple_pooling = use_simple_pooling
        self.use_center_pooling = use_center_pooling

        self.loss_weight = loss_weight
        self.proposal_target_layer = ProposalTargetLayer(roi_per_image=roi_per_image, 
                                                         fg_ratio=roi_fg_ratio, 
                                                         reg_fg_thresh=reg_fg_thresh,)
        self.box_coder = ResidualCoder(code_size=code_size, encode_angle_by_sincos=encode_angle_by_sincos)
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=loss_weight.CODE_WEIGHT)

        self.roi_grid_pool_layers = nn.ModuleList()
        for i in range(len(self.mlps)): # different feature source, default only use semantic feature
            mlp = self.mlps[i] 
            pool_layer = SimplePoolingLayer(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                            voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=self.use_center_pooling)
            self.roi_grid_pool_layers.append(pool_layer)
        
        if self.use_center_pooling:
            pre_channel = sum([x[-1] for x in self.mlps])
        else:
            raise NotImplementedError

        reg_fc_list = []
        for k in range(0, self.reg_fc.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.reg_fc[k], bias=False),
                nn.BatchNorm1d(self.reg_fc[k]),
                nn.ReLU()
            ])
            pre_channel = self.reg_fc[k]

            if k != self.reg_fc.__len__() - 1 and dp_ratio > 0:
                reg_fc_list.append(nn.Dropout(dp_ratio))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)

        if self.encode_angle_by_sincos:
            self.reg_pred_layer = nn.Linear(pre_channel, self.code_size+1, bias=True)
        else:
            self.reg_pred_layer = nn.Linear(pre_channel, self.code_size, bias=True)
        self.init_weights()
    
    def init_weights(self): 
        init_func = nn.init.xavier_normal_
        layers_list = [self.shared_fc_layer, self.reg_fc_layers] if not self.use_center_pooling else [self.reg_fc_layers]
        for module_list in layers_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
    
    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1]) 
        batch_size_rcnn = rois.shape[0]
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size) 
        if self.code_size > 6:
            global_roi_grid_points = common_utils.rotate_points_along_z(
                local_roi_grid_points.clone(), rois[:, 6]
            ).squeeze(dim=1)
        else:
            global_roi_grid_points = local_roi_grid_points

        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = global_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  
        
        return roi_grid_points

    def roi_grid_pool(self, input_dict):
        """
        Args:
            input_dict:
                rois: b, num_max_rois, 7
                batch_size: b
                middle_feature_list: List[mink_tensor]
        """
        rois = input_dict['rois']
        batch_size = input_dict['batch_size']
        middle_feature_list = [input_dict['middle_feature_list'][i] for i in self.middle_feature_source]
        if not isinstance(middle_feature_list, list):
            middle_feature_list = [middle_feature_list]
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size
        )  
        
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        batch_idx = rois.new_zeros(batch_size, roi_grid_xyz.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        
        pooled_features_list = []
        for k, cur_sp_tensors in enumerate(middle_feature_list):
            pool_layer = self.roi_grid_pool_layers[k]
            if self.use_simple_pooling:
                batch_grid_points = torch.cat([batch_idx, roi_grid_xyz], dim=-1) 
                batch_grid_points = batch_grid_points.reshape([-1, 4])
                new_features = pool_layer(cur_sp_tensors, grid_points=batch_grid_points)
            else:
                raise NotImplementedError
            pooled_features_list.append(new_features)
        ms_pooled_feature = torch.cat(pooled_features_list, dim=-1)
        return ms_pooled_feature

    def forward_train(self, input_dict):
        pred_boxes_3d = input_dict['pred_bbox_list']
        # preprocess rois, padding to same number
        rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d)
        if self.enlarge_ratio:
            rois[..., 3:6] *= self.enlarge_ratio
        input_dict['rois'] = rois
        input_dict['roi_scores'] = roi_scores
        input_dict['roi_labels'] = roi_labels
        input_dict['batch_size'] = batch_size

        # assign targets
        targets_dict = self.assign_targets(input_dict)
        input_dict.update(targets_dict)

        # roi pooling
        pooled_features = self.roi_grid_pool(input_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        if not self.use_center_pooling:
            shared_features = self.shared_fc_layer(pooled_features) 
        else:
            shared_features = pooled_features
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) # (BN, 6)

        input_dict['rcnn_reg'] = rcnn_reg

        return input_dict       

    def assign_targets(self, input_dict):
        with torch.no_grad():
            targets_dict = self.proposal_target_layer(input_dict)
        batch_size = input_dict['batch_size']
        rois = targets_dict['rois'] # b, num_max_rois, 7
        gt_of_rois = targets_dict['gt_of_rois'] # b, num_max_rois, 7
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        gt_label_of_rois = targets_dict['gt_label_of_rois'] # b, num_max_rois
        
        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        # also change gt angle to 0 ~ 2pi
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry # 0 - 0 = 0

        if self.code_size > 6:
            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
            ).view(batch_size, -1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            gt_of_rois[:, :, 6] = heading_label

        targets_dict['gt_of_rois'] = gt_of_rois

        return targets_dict
    
    def reoder_rois_for_refining(self, pred_boxes_3d):
        """
        Args:
            pred_boxes_3d: List[(box, score, label), (), ...]
        """
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds[0]) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0][0]

        if len(pred_boxes_3d[0]) == 4:
            use_sem_score = True
        else:
            use_sem_score = False

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        if use_sem_score:
            roi_sem_scores = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes_3d[0][3].shape[-1]))

        for bs_idx in range(batch_size):
            num_boxes = len(pred_boxes_3d[bs_idx][0])            
            rois[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx][0]
            roi_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][1]
            roi_labels[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][2]
            if use_sem_score:
                roi_sem_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][3]
        
        # converse heading to pcdet heading
        rois[..., 6] *= -1
        if use_sem_score:
            return rois, roi_scores, roi_labels, roi_sem_scores, batch_size
        else:
            return rois, roi_scores, roi_labels, batch_size

    def simple_test(self, input_dict):
        pred_boxes_3d = input_dict['pred_bbox_list']
        # preprocess rois, padding to same number
        if len(pred_boxes_3d[0]) == 4:
            use_sem_score = True
            rois, roi_scores, roi_labels, roi_sem_scores, batch_size = self.reoder_rois_for_refining(pred_boxes_3d) 
        else:
            rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d)
            use_sem_score = False
        input_dict['rois'] = rois
        input_dict['roi_scores'] = roi_scores
        input_dict['roi_labels'] = roi_labels
        if use_sem_score:
            input_dict['roi_sem_scores'] = roi_sem_scores
        input_dict['batch_size'] = batch_size        

        # roi pooling
        pooled_features = self.roi_grid_pool(input_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        if not self.use_center_pooling:
            shared_features = self.shared_fc_layer(pooled_features) 
        else:
            shared_features = pooled_features
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) 

        input_dict['rcnn_reg'] = rcnn_reg

        batch_size = input_dict['batch_size']
        img_meta = [None for _ in range(batch_size)]
        results = self.get_boxes(input_dict, img_meta) 
        pred_dict = dict(batch_box_preds=[], batch_score_preds=[], batch_cls_preds=[])
        for i in range(batch_size):
            pred_dict['batch_box_preds'].append(results[i][0])
            pred_dict['batch_score_preds'].append(results[i][1])
            pred_dict['batch_cls_preds'].append(results[i][2])
        
        input_dict.update(pred_dict)

        return input_dict
    
    def get_boxes(self, input_dict, img_meta):
        batch_size = input_dict['batch_size']
        rcnn_cls = None
        rcnn_reg = input_dict['rcnn_reg']
        roi_labels = input_dict['roi_labels']
        roi_scores = input_dict['roi_scores']
        roi_sem_scores = input_dict.get('roi_sem_scores', None)
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size, rois=input_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg, roi_labels=roi_labels, roi_sem_scores=roi_sem_scores
            )
        input_dict['cls_preds_normalized'] = False
        if not input_dict['cls_preds_normalized'] and batch_cls_preds is not None:
            batch_cls_preds = torch.sigmoid(batch_cls_preds)
        input_dict['batch_cls_preds'] = batch_cls_preds # B,N
        input_dict['batch_box_preds'] = batch_box_preds 

        results = []
        for bs_id in range(batch_size):
            # nms
            boxes = batch_box_preds[bs_id]
            # scores = batch_cls_preds[bs_id].squeeze(-1)
            scores = roi_scores[bs_id]
            labels = roi_labels[bs_id]

            result = self._nms(boxes, scores, labels, img_meta[bs_id])
            results.append(result)
        
        return results
    
    def _nms(self, bboxes, scores, labels, img_meta):
        n_classes = self.num_class
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            if scores.ndim == 2:
                # ids = (scores[:, i] > self.test_score_thr) & (bboxes.sum() != 0) # reclass
                ids = (labels == i) & (scores[:, i] > self.test_score_thr) & (bboxes.sum() != 0) # no reclass
            else:
                ids = (labels == i) & (scores > self.test_score_thr) & (bboxes.sum() != 0)
            if not ids.any():
                continue
            class_scores = scores[ids] if scores.ndim == 1 else scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms_gpu
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms_normal_gpu

            nms_ids, _ = nms_function(class_bboxes, class_scores, self.test_iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            # converse pcdet heading to original heading
            nms_bboxes[..., 6] *= -1
        else:
            fake_heading = nms_bboxes.new_zeros(nms_bboxes.shape[0], 1)
            nms_bboxes = torch.cat([nms_bboxes[:, :6], fake_heading], dim=1)

        return nms_bboxes, nms_scores, nms_labels
    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds, roi_labels=None, gt_bboxes_3d=None, gt_labels_3d=None, roi_sem_scores=None):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.code_size
        batch_cls_preds = None
        if self.encode_angle_by_sincos:
            batch_box_preds = box_preds.view(batch_size, -1, code_size+1) 
        else:
            batch_box_preds = box_preds.view(batch_size, -1, code_size) 

        # decode box
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()[..., :code_size]
        local_rois[:, :, 0:3] = 0
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)
        
        if self.code_size > 6:
            roi_ry = rois[:, :, 6].view(-1)
            batch_box_preds = common_utils.rotate_points_along_z(
                batch_box_preds.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz 
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)

        return batch_cls_preds, batch_box_preds
    
    def loss(self, input_dict):
        rcnn_loss_dict = {}
        # rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(input_dict)
        if not self.use_iou_loss:
            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(input_dict)
            rcnn_loss_dict['rcnn_loss_reg'] = rcnn_loss_reg
        else:
            rcnn_loss_reg, rcnn_loss_iou, reg_tb_dict = self.get_box_reg_layer_loss(input_dict)
            if self.loss_weight.RCNN_REG_WEIGHT > 0:
                rcnn_loss_dict['rcnn_loss_reg'] = rcnn_loss_reg
            rcnn_loss_dict['rcnn_loss_iou'] = rcnn_loss_iou
        # rcnn_loss_dict['rcnn_loss_cls'] = rcnn_loss_cls
        loss = 0.
        tb_dict = dict()
        for k in rcnn_loss_dict.keys():
            loss += rcnn_loss_dict[k]
            tb_dict[k] = rcnn_loss_dict[k].item()
        tb_dict['loss_two_stage'] = loss.item()
        return loss, tb_dict
    
    def get_box_cls_layer_loss(self, forward_ret_dict):
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if self.cls_loss_type == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif self.cls_loss_type == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * self.loss_weight.RCNN_CLS_WEIGHT
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict
    
    def get_box_reg_layer_loss(self, forward_ret_dict):
        code_size = self.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois'][..., 0:code_size]
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if self.reg_loss_type == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            if code_size > 6:
                rois_anchor[:, 6] = 0

            # encode box
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 6]

            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * self.loss_weight.RCNN_REG_WEIGHT
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
            loss_iou = torch.tensor(0., device=fg_mask.device)
            if self.use_iou_loss and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()

                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size+1 if self.encode_angle_by_sincos else code_size), batch_anchors
                ).view(-1, code_size)

                if self.code_size > 6:
                    roi_ry = batch_anchors[:, :, 6].view(-1)
                    rcnn_boxes3d = common_utils.rotate_points_along_z(
                        rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                    ).squeeze(dim=1)

                rcnn_boxes3d[:, 0:3] += roi_xyz
                loss_iou = self.iou_loss_computer(rcnn_boxes3d[:, 0:code_size],
                    gt_of_rois_src[fg_mask][:, 0:code_size])
                loss_iou = loss_iou * self.loss_weight.RCNN_IOU_WEIGHT
                tb_dict['rcnn_loss_iou'] = loss_iou.item()
        else:
            raise NotImplementedError
        
        if not self.use_iou_loss:
            return rcnn_loss_reg, tb_dict
        else:
            return rcnn_loss_reg, loss_iou, tb_dict

    def forward(self, input_dict):
        if self.training:
            return self.forward_train(input_dict)
        else:
            return self.simple_test(input_dict)