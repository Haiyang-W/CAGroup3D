from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ..model_utils.rbgnet_utils import BasicBlock2D, ZeroQueryAndGroup


class PointNet2_FBS_SSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        self.num_sa = len(self.model_cfg.SA_CONFIG.get('MLPS'))
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            mlps = [channel_in] + mlps
            channel_out = mlps[-1]
            if k != 0:
                fbs_use = True
            else:
                fbs_use = False
            self.SA_modules.append(
                PointnetSAModuleSSGFBS(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    fg_nsample=self.model_cfg.SA_CONFIG.FG_NSAMPLE[k],
                    topk=self.model_cfg.SA_CONFIG.TOPK[k],
                    fbs_mlps=self.model_cfg.SA_CONFIG.FBS_MLPS[k],
                    fbs_use=fbs_use,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()
        self.num_fp = len(self.model_cfg.FP_MLPS)

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[fp_source_channel + fp_target_channel] + self.model_cfg.FP_MLPS[k]
                )
            )
            if k != len(self.model_cfg.FP_MLPS) - 1:
                fp_source_channel = self.model_cfg.FP_MLPS[k][-1]
                fp_target_channel = skip_channel_list.pop()

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        batch_dict['points_cat'] = xyz.clone()
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        initial_indices = xyz.new_tensor(range(xyz.shape[1])).unsqueeze(0).repeat(batch_size, 1).long()
        l_xyz, l_features, l_indices, l_mask_scores = [xyz], [features], [initial_indices], [None]
        for i in range(len(self.SA_modules)):
            if i == 0:
                li_xyz, li_features, li_indices = self.SA_modules[i](l_xyz[i], l_features[i])
            else:
                li_xyz, li_features, li_indices, li_mask_scores = self.SA_modules[i](l_xyz[i], l_features[i])
                l_mask_scores.append(li_mask_scores)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_indices.append(torch.gather(l_indices[-1], 1, li_indices.long()))

        fp_xyz = [l_xyz[-1]]
        fp_features = [l_features[-1]]
        fp_indices = [l_indices[-1]]
    
        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                l_xyz[self.num_sa - i - 1], l_xyz[self.num_sa - i],
                l_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(l_xyz[self.num_sa - i - 1])
            fp_indices.append(l_indices[self.num_sa - i - 1])
        
        batch_dict['fp_xyz'] = fp_xyz
        batch_dict['fp_features'] = fp_features
        batch_dict['fp_indices'] = fp_indices
        batch_dict['sa_xyz'] = l_xyz
        batch_dict['sa_features'] = l_features
        batch_dict['sa_indices'] = l_indices
        batch_dict['sa_masks_score'] = l_mask_scores
        return batch_dict

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def _fbs_sample_points(self, samples_mask, points_xyz, foreground_sample_num):
        batch_size = samples_mask.shape[0]
        batch_indices = torch.arange(points_xyz.shape[-2]).to(points_xyz.device)
        new_xyz = []
        new_indices = []
        for b in range(batch_size):
            temp_indices = batch_indices.clone()
            positive_sample = torch.nonzero(samples_mask[b]).view(-1)  # n, 1
            positive_xyz = points_xyz[b][positive_sample]
            positive_indices = temp_indices[positive_sample]
            negative_sample = torch.nonzero(1 - samples_mask[b]).view(-1)  # n, 1
            negative_xyz = points_xyz[b][negative_sample]
            negative_indices = temp_indices[negative_sample]

            batch_xyz_flipped = points_xyz[b:b+1].clone().transpose(1, 2).contiguous()

            # positive
            if foreground_sample_num > 0:
                if positive_xyz.shape[0] < foreground_sample_num and positive_xyz.shape[0] > 0:
                    pad_num = foreground_sample_num - positive_xyz.shape[0]
                    random_sample = np.random.choice(positive_xyz.shape[0], pad_num, replace=True)
                    random_sample = positive_xyz.new_tensor(random_sample).long()
                    pad_positive_xyz = positive_xyz[random_sample]
                    pad_positive_indices = positive_indices[random_sample]
                    positive_xyz = torch.cat([positive_xyz, pad_positive_xyz], dim=0)
                    positive_indices = torch.cat([positive_indices, pad_positive_indices], dim=0)
                elif positive_xyz.shape[0] == 0:
                    positive_xyz = points_xyz[b]
                    positive_indices = temp_indices

                positive_xyz = positive_xyz.unsqueeze(0)
                select_positive_indices = pointnet2_utils.farthest_point_sample(positive_xyz, foreground_sample_num)
                batch_new_pos_indices = positive_indices[select_positive_indices.long().view(-1)].unsqueeze(0).int()
                batch_new_pos_xyz = pointnet2_utils.gather_operation(batch_xyz_flipped, batch_new_pos_indices).transpose(
                    1, 2).contiguous() if self.npoint is not None else None

            if self.npoint > foreground_sample_num:
                # negative
                if negative_xyz.shape[0] < (self.npoint - foreground_sample_num) and negative_xyz.shape[0] > 0:
                    pad_num = self.npoint - foreground_sample_num - negative_xyz.shape[0]
                    random_sample = np.random.choice(negative_xyz.shape[0], pad_num, replace=True)
                    random_sample = negative_xyz.new_tensor(random_sample).long()

                    pad_negative_xyz = negative_xyz[random_sample]
                    pad_negative_indices = negative_indices[random_sample]

                    negative_xyz = torch.cat([negative_xyz, pad_negative_xyz], dim=0)
                    negative_indices = torch.cat([negative_indices, pad_negative_indices], dim=0)
                elif negative_xyz.shape[0] == 0:
                    negative_xyz = points_xyz[b]
                    negative_indices = temp_indices

                negative_xyz = negative_xyz.unsqueeze(0)

                select_negative_indices = pointnet2_utils.farthest_point_sample(negative_xyz, self.npoint-foreground_sample_num)
                batch_new_neg_indices = negative_indices[select_negative_indices.long().view(-1)].unsqueeze(0).int()
                batch_new_neg_xyz = pointnet2_utils.gather_operation(batch_xyz_flipped, batch_new_neg_indices).transpose(
                    1, 2).contiguous() if self.npoint is not None else None

            if foreground_sample_num > 0 and self.npoint > foreground_sample_num:
                batch_new_xyz = torch.cat([batch_new_pos_xyz, batch_new_neg_xyz], dim=1)
                batch_new_indices = torch.cat([batch_new_pos_indices, batch_new_neg_indices], dim=1)
            elif foreground_sample_num == self.npoint:
                batch_new_xyz = batch_new_pos_xyz
                batch_new_indices = batch_new_pos_indices
            elif foreground_sample_num == 0:
                batch_new_xyz = batch_new_neg_xyz
                batch_new_indices = batch_new_neg_indices

            new_xyz.append(batch_new_xyz)
            new_indices.append(batch_new_indices)
        new_xyz = torch.cat(new_xyz, dim=0)
        new_indices = torch.cat(new_indices, dim=0)

        return new_xyz, new_indices

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        # :param features: (B, N, C) tensor of the descriptors of the the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            if not self.fbs_use:
                indices = pointnet2_utils.farthest_point_sample(xyz, self.npoint)
                new_xyz = pointnet2_utils.gather_operation(
                    xyz_flipped,
                    indices
                ).transpose(1, 2).contiguous() if self.npoint is not None else None
            else:
                mask_scores = self.fbs_mlps[0](features.unsqueeze(-1))  # B, 2, num_points, 1
                mask_scores = mask_scores.squeeze(-1)  # B,  2, num_points
                softmax_mask_score = F.softmax(mask_scores, dim=1)
                confidence_score_margin = softmax_mask_score[:, 1, :] - softmax_mask_score[:, 0, :]
                _, top_indices = torch.topk(confidence_score_margin, k=self.topk, dim=1)
                sample_masks = torch.zeros_like(confidence_score_margin).long()
                for b in range(sample_masks.shape[0]):
                    sample_masks[b][top_indices[b]] = 1
                new_xyz, indices = self._fbs_sample_points(sample_masks, xyz.clone(), self.fg_nsample)
                assert (new_xyz == pointnet2_utils.gather_operation(xyz_flipped, indices.int()).transpose(
                    1, 2).contiguous()).all()

        for i in range(len(self.groupers)):
            if self.zero_query:
                grouped_results, valid_idx = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                batch_size = grouped_results.shape[0]
                infeat_dim = self.mlp_spec[0]
                outfeat_dim = self.mlp_spec[-1]
                num_point = grouped_results.shape[2]
                nsample = grouped_results.shape[3]

                new_features = torch.zeros((batch_size, outfeat_dim, num_point, nsample)).to(features.device)
                new_features = new_features.permute(0, 2, 1, 3).reshape(batch_size*num_point, outfeat_dim, 1, nsample)
                grouped_results = grouped_results.permute(0, 2, 1, 3).reshape(batch_size*num_point, infeat_dim, 1, nsample)
                valid_idx = valid_idx.view(-1)
                valid_indices = torch.nonzero(valid_idx).view(-1)
                new_features[valid_indices] = self.mlps[i](grouped_results[valid_indices])
                new_features = new_features.view(batch_size, num_point, outfeat_dim, nsample)
                new_features = new_features.permute(0, 2, 1, 3).contiguous()
            else:
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
        if self.fbs_use:
            return new_xyz, torch.cat(new_features_list, dim=1), indices, mask_scores
        if self.zero_query:
            return new_xyz, torch.cat(new_features_list, dim=1), None
        return new_xyz, torch.cat(new_features_list, dim=1), indices


class PointnetSAModuleSSGFBS(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with singlescale grouping and foreground based sampling"""

    def __init__(self, *, npoint: int, radii: float, nsamples: int, mlps: List[int],
                 fg_nsample: int = None, topk: int = None, fbs_mlps: List[int] = None, zero_query: bool = False, fbs_use: bool = False,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: float, radii to group with
        :param nsamples: int, number of samples in each ball query
        :param mlps: list of int, spec of the pointnet before the global pooling for each scale
        :param topk: int
        :param fg_nsample: int
        :param fbs_mlps: list of int
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.npoint = npoint
        self.fbs_use = fbs_use
        self.zero_query = zero_query
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        radius = radii
        nsample = nsamples
        if zero_query:
            self.groupers.append(
                ZeroQueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            pass
        else:
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
        mlp_spec = mlps
        if use_xyz:
            mlp_spec[0] += 3
        self.mlp_spec = mlp_spec

        shared_mlps = []
        for k in range(len(mlp_spec) - 1):
            shared_mlps.extend([
                BasicBlock2D(
                    mlp_spec[k],
                    mlp_spec[k + 1],
                    kernel_size=1,
                    padding=0,
                    bias=False)
            ])
        self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

        if self.fbs_use:
            self.fbs_mlps = nn.ModuleList()
            self.fg_nsample = fg_nsample
            self.topk = topk
            fbs_shared_mlps = []
            if use_xyz:
                fbs_channels = [mlp_spec[0]-3] + fbs_mlps
            else:
                fbs_channels = [mlp_spec[0]] + fbs_mlps
            for k in range(len(fbs_channels) - 1):
                fbs_shared_mlps.extend([
                    BasicBlock2D(
                    fbs_channels[k],
                    fbs_channels[k + 1],
                    kernel_size=1,
                    padding=0,
                    bias=False)
                ])
            fbs_shared_mlps.extend([nn.Conv2d(fbs_channels[-1], 2, kernel_size=1, bias=True)])
            self.fbs_mlps.append(nn.Sequential(*fbs_shared_mlps))

