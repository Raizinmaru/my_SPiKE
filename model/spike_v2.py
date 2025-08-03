"""
Module for the SPiKE model.permute
"""

import sys
import os
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "modules"))
from modules.point_spat_convolution_v2 import PointSpatialConv
from modules.transformer import Transformer

class SPiKE(nn.Module):
    def __init__(
        self,
        radius,             # radius:0.2
        nsamples,           # nsamples:32
        dim,                # dim:1024
        depth,              # depth:5
        heads,              # heads:8
        dim_head,           # dim_head:256
        mlp_dim,            # mlp_dim:2048
        num_coord_joints,   # num_coord_joints:45
        dropout1=0.0,       # dropout1:0.0
        dropout2=0.0,       # dropout2:0.0

        backbone = "pointnet++"
    ):
        super().__init__()

        self.backbone = backbone
        print("backbone: ", self.backbone)

        self.stem = PointSpatialConv(
            spatial_kernel_size=radius,
            nsamples=nsamples,

            backbone=backbone
        )
        self.pos_embed = nn.Conv1d(
            in_channels=2,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)    
        self.mlp_head = self._build_mlp_head(dim, mlp_dim, 15*3, dropout2)
        self.jnt_embed = nn.Sequential(
            nn.Linear(in_features=3, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=dim)
        )

    def _build_mlp_head(self, dim, mlp_dim, num_coord_joints, dropout):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_coord_joints),
        )
        
    def forward(self, ptc, joints):   # [24, 1, 4096, 3], joints: [24, 2, 15, 3]or[24, 3, 15, 3]
        device, batch_size = ptc.device, ptc.shape[0]

        # Local Feature Extraction
        # pointnet
        if self.backbone == "pointnet":
            ptc_global_embd, ptc_local_embd, centroids = self.stem(ptc[:,-1].contiguous(), joints[:,-1].contiguous())   # [24, 1024], [24, 1024, 15], [24, 15, 3]
        # pointnet++
        elif self.backbone == "pointnet++":
            ptc_global_embd, ptc_local_embd = self.stem(ptc[:,-1].contiguous(), joints[:,-1].contiguous())              # [24, 1024], [24, 1024, 15]

        # Joints Embedding
        # pointnet
        if self.backbone == "pointnet":
            joints = torch.cat((joints[:,:-1], centroids.unsqueeze(1)), dim=1)                                          # [24, 3, 15, 3]
            jnt_embd = self.jnt_embed(joints.permute(0, 3, 1, 2).contiguous())                                          # [24, 1024, 3, 15]
            jnt_embd[:, :, -1, :] += ptc_local_embd                                                                     # [24, 1024, 3, 15]
            embd = jnt_embd.permute(0,2,3,1).contiguous()                                                               # [24, 3, 15, 1024]
            embd = embd.view(embd.shape[0], -1, embd.shape[-1]).contiguous()                                            # [24, 45, 1024]
        # pointnet++
        elif self.backbone == "pointnet++":
            joints = joints[:,:-1]                                                                                      # [24, 2, 15, 3]
            jnt_embd = self.jnt_embed(joints)                                                                           # [24, 2, 15, 1024]
            embd = torch.cat((jnt_embd, ptc_local_embd.transpose(1,2).contiguous().unsqueeze(1)), dim=1)                # [24, 3, 15, 1024]
            embd = embd.view(embd.shape[0], -1, embd.shape[-1]).contiguous()                                            # [24, 45, 1024]
        
        # Positional Embedding
        L, N = 3,15
        jnt_idx = torch.arange(N, device=device, dtype=torch.float).view(1,1,N,1).expand(batch_size,L,-1,-1) + 1    # [24, 3, 15, 1]
        t = torch.arange(L, device=device, dtype=torch.float).view(1,L,1,1).expand(batch_size,-1,N,-1) + 1          # [24, 3, 15, 1]

        pe = torch.cat((jnt_idx, t), dim=-1)                                                                        # [24, 3, 15, 2]
        pe = pe.view(batch_size, -1, pe.shape[-1]).contiguous()                                                     # [24, 45, 2]
        pe = self.pos_embed(pe.transpose(1,2)).transpose(1,2)                                                       # [24, 45, 1024]

        embd_pe = embd + pe                                                                                         # [24, 45, 1024]
        embd_pe_cls = torch.cat((ptc_global_embd.unsqueeze(1), embd_pe), dim=1)                                     # [24, 46, 1024]

        # Transformer
        output = self.transformer(embd_pe_cls)                                                                      # [24, 46, 1024]

        # cls tokenのみ使用（global feature）
        output = output[:,0]                                                                                        # [24, 1024]
        
        jnt_pred = self.mlp_head(output)                                                                            # [24, 45]

        return jnt_pred
