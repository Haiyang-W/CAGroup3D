import torch
import torch.nn as nn

from typing import Tuple
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils

class MLP(nn.Module):

    def __init__(self,
                 in_channel=18,
                 conv_channels=(256, 256),
                 bias=True):
        super().__init__()
        self.mlp = nn.Sequential()
        prev_channels = in_channel
        for i, conv_channel in enumerate(conv_channels):
            self.mlp.add_module(
                f'layer{i}',
                BasicBlock1D(
                    prev_channels,
                    conv_channels[i],
                    kernel_size=1,
                    padding=0,
                    bias=bias))
            prev_channels = conv_channels[i]

    def forward(self, img_features):
        return self.mlp(img_features)

class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
       
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    
    def init_weights(self):
        # only support relu
        nn.init.kaiming_normal_(
                self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
       
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    
    def init_weights(self):
        # only support relu
        nn.init.kaiming_normal_(
                self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ZeroQueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        batch_size = xyz.shape[0]
        feat_dim = features.shape[1]
        zero_xyz_padding = xyz.new_ones((batch_size, 1, 3)) * 1000.
        xyz = torch.cat([zero_xyz_padding, xyz], dim=1)
        zero_feature_padding = features.new_zeros((batch_size, feat_dim, 1))
        features = torch.cat([zero_feature_padding, features], dim=2)
        idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
        vaild_idx = (idx.sum(-1).float() != 0.)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        grouped_xyz = torch.where(idx.unsqueeze(1).repeat(1, 3, 1, 1) == 0, torch.zeros_like(grouped_xyz), grouped_xyz)

        if features is not None:
            grouped_features = pointnet2_utils.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, vaild_idx