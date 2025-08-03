"""
Module for point spatial convolution.
"""

from typing import List
import os
import sys
import torch
from torch import nn
from modules import pointnet2_utils

from local_aggregation import LocalAggregation
#, create_convblock2d

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

swith = False

class PointSpatialConv(nn.Module):
    def __init__(
        self,
        #in_channels: int,
        #mlp_channels: List[int],
        spatial_kernel_size: float,
        nsamples: int,
        #spatial_stride: int,

        backbone: str,
    ):
        super().__init__()

        #self.in_channels = in_channels
        #self.mlp_channels = mlp_channels
        self.spatial_kernel_size = spatial_kernel_size
        self.nsamples = nsamples
        self.spatial_stride = spatial_stride
        
        self.backbone = backbone
        # PointNet
        if backbone == "pointnet":
            self.local_pointnet = PointNetEncoder(in_channels=3, out_channels=1024)
            self.global_pointnet = PointNetEncoder(in_channels=3,out_channels=1024)

        # PointNet++
        elif backbone == "pointnet++":
            # Part Lv.
            channels_list = [[3, 64, 128], [128+3, 256, 256], [256+3, 512, 1024]]  # [[in_c, hidden, out_c], ...]
            
            #channels_list = [[3, 64, 128], [128+3, 512, 1024]]  # [[in_c, hidden, out_c], ...]
            conv_args = {'order': 'conv-norm-act'}
            norm_args = {'norm': 'bn'}
            act_args = {'act': 'relu'}
            aggr_args = {'NAME': 'convpool', 'reduction': 'max'}
            self.pointnet2 = HierarchicalAggregation(channels_list, conv_args, norm_args, act_args, aggr_args)

            self.conv_d = nn.Conv1d(
                in_channels=channels_list[1][0], out_channels=channels_list[2][2],
                #in_channels=channels_list[1][0], out_channels=channels_list[1][2],
                kernel_size=1, stride=1, padding=0, bias=False,
                )

    def forward(self, ptc: torch.Tensor, jnt: torch.Tensor=None): #-> (torch.Tensor, torch.Tensor):
        idx = pointnet2_utils.ball_query(self.spatial_kernel_size, self.nsamples*2, ptc, jnt)   # []
        local_ptc = pointnet2_utils.grouping_operation(ptc.transpose(1, 2).contiguous(), idx)   # [24, 3, 15, 64]

        local_ptc = local_ptc.permute(0, 2, 3, 1).contiguous()                                  # [24, 15, 64, 3]
        B, V ,N ,_ = local_ptc.shape

        # PointNet
        if self.backbone == "pointnet":
            global_ptc = local_ptc.view(B,-1,3)
            centroids = torch.mean(local_ptc, dim=-2)                                       # [24, 15, 3]
            local_ptc = local_ptc - centroids.unsqueeze(2)

            local_ptc = local_ptc.view(-1,N,3)                                              # [360, 64, 3]
            local_features = self.local_pointnet(local_ptc)                                 # [360, 1024]
            local_features = local_features.view(B,V,-1).transpose(1,2).contiguous()        # [24, 1024, 15]
            
            global_feature = self.global_pointnet(global_ptc)                                # [24, 1024]

            return global_feature, local_features                                           #[24, 1024], [24, 1024, 15]

        # PointNet++
        elif self.backbone == "pointnet++":
            global_feature, local_features = self.pointnet2(local_ptc)      # [24, 1024], [24, 128, 15]
            local_features = self.conv_d(local_features)                    # [24, 1024, 15]

            return global_feature, local_features
        

import matplotlib.pyplot as plt 
class HierarchicalAggregation(nn.Module):
    def __init__(self, channels_list, conv_args, norm_args, act_args, aggr_args):
        super().__init__()
        self.local_aggregations = nn.ModuleList()
        for channels in channels_list:
            self.local_aggregations.append(
                LocalAggregation(channels, aggr_args, conv_args, norm_args, act_args, use_res=False)
            )

    def forward(self, local_regions):
        # 1層目のPointNet                                                                    # [24, 15, 64, 3]
        local_regions = local_regions.permute(0, 3, 1, 2).contiguous()                      # [24, 3, 15, 64]
        centroids_1 = torch.mean(local_regions, dim=-1)                                     # [24, 3, 15]
        relative_1 = local_regions - centroids_1.unsqueeze(-1)                              # [24, 3, 15, 64]
        local_features_1 = self.local_aggregations[0](support_features=relative_1)          # [24, 128, 15]
        
        local_features = torch.cat((centroids_1, local_features_1), dim=1)                  # [24, 131, 15]

        #''' Part Lv.
        # 3. 5個の領域に分割 (B, 5, 3, C1)
        local_features_1 = self.split_regions(local_features_1.transpose(1,2)  )            # [24, 5, 3, 128]
        local_regions_2 = self.split_regions(centroids_1.transpose(1,2))                    # [24, 5, 3, 3]

        # 4. 2層目のPointNet
        centroids_2 = torch.mean(local_regions_2, dim=2)                                    # [24, 5, 3]
        relative_2 = local_regions_2 - centroids_2.unsqueeze(2)                             # [24, 5, 3, 3]

        local_features_1 = torch.cat((relative_2, local_features_1), dim=-1)                # [24, 5, 3, 131]
        local_features_1 = local_features_1.permute(0, 3, 1, 2).contiguous()                # [24, 131, 5, 3]
        local_features_2 = self.local_aggregations[1](support_features=local_features_1)    # [24, 256, 5]

        # 5. 3層目のPointNet
        centroids_3 = torch.mean(centroids_2, dim=1)                                        # [24, 3]
        relative_3 = centroids_2 - centroids_3.unsqueeze(1)                                 # [24, 5, 3]
        local_features_2 = torch.cat((relative_3.transpose(1,2), local_features_2), dim=1)  # [24, 259, 5]
        local_features_2 = local_features_2.unsqueeze(2)                                    # [24, 259, 1, 5]
        global_feature = self.local_aggregations[2](support_features=local_features_2)      # [24, 1024, 1]

        return global_feature.squeeze(2), local_features

    # 局所領域特徴を分割する関数 (仮実装 - 適切に実装してください)
    def split_regions(self, local_features):
        #B, V1, C1 = local_features.shape
        #num_combined = V1 // num_splits # 端数は切り捨て
        #local_features = local_features[:, :num_combined * num_splits, :].view(B, num_splits, num_combined, C1)

        indices = [[0, 1, 8], [2, 4, 6], [3, 5, 7], [9, 11, 13], [10, 12, 14]]
        local_features =  torch.stack([local_features[:, idx, :] for idx in indices], dim=1)            # [24, 5, 3, 128]

        return local_features

import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128,  out_channels, 1)
        self.fc1 = nn.Linear(out_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).reshape(1, 9)
        self.out_channels = out_channels

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_channels)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, in_channels=64, out_channels=1024):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_channels, 1)
        self.fc1 = nn.Linear(out_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, in_channels * in_channels)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.iden = torch.from_numpy(np.eye(self.in_channels).flatten().astype(np.float32)).reshape(1, self.in_channels * self.in_channels)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_channels)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)

        x = x + iden
        x = x.view(-1, self.in_channels, self.in_channels)
        return x

#@MODELS.register_module()
class PointNetEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_transform: bool=True,
                 feature_transform: bool=True,
                 #is_seg: bool=False,  
                 **kwargs
                 ):

        super().__init__()
        self.stn = STN3d(in_channels) if input_transform else None
        self.conv0_1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv0_2 = torch.nn.Conv1d(64, 64, 1)

        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_channels , 1)
        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.fstn = STNkd(in_channels=64, out_channels=out_channels) if feature_transform else None
        #self.out_channels = 1024 + 64 if is_seg else 1024 
        self.out_channels = out_channels 
         
    def forward_cls_feat(self, pos, x=None):
        if hasattr(pos, 'keys'):
            x = pos['x']
        if x is None:
            x = pos.transpose(1, 2).contiguous()
        
        B, D, N = x.size()
        #'''
        if self.stn is not None:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        #'''
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = F.relu(self.bn0_2(self.conv0_2(x)))

        #'''
        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        #'''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_channels)
        return x

    def forward_seg_feat(self, pos, x=None):
        if hasattr(pos, 'keys'):
            x = pos.get('x', None)
        if x is None:
            x = pos.transpose(1, 2).contiguous()

        B, D, N = x.size()
        if self.stn is not None:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = F.relu(self.bn0_2(self.conv0_2(x)))

        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024, 1).repeat(1, 1, N)
        return pos, torch.cat([pointfeat, x], 1)
    
    def forward(self, x, features=None):
        return self.forward_cls_feat(x)
