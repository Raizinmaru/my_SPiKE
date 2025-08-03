"""
Module for the SPiKE model.permute
"""

import sys
import os
import torch
from torch import nn

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "modules"))
from modules.point_spat_convolution_v2 import PointSpatialConv
from modules.transformer import Transformer

class SPiKE(nn.Module):
    """
    SPiKE model class.
    """

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
        #print(f"radius:{radius}, nsamples:{nsamples}, dim:{dim}, depth:{depth}, heads:{heads}, dim_head:{dim_head}, mlp_dim:{mlp_dim}, num_coord_joints:{num_coord_joints}, dropout1:{dropout1}, dropout2:{dropout2}")
        super().__init__()

        self.backbone = backbone
        print("backbone: ", self.backbone)

        self.stem = PointSpatialConv(
            #in_channels=0,
            #mlp_channels=[dim],
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

        '''
        self.jnt_embed = nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        '''
        self.jnt_embed = nn.Sequential(
            nn.Linear(in_features=3, out_features=128),
            nn.ReLU(),
            # learn
            #nn.GELU(),
            nn.Linear(in_features=128, out_features=dim)
        )
        #'''

    '''
    def _build_stem(self, radius, nsamples, dim):
        return PointSpatialConv(
            in_channels=0,
            mlp_channels=[dim],
            spatial_kernel_size=radius,
            nsamples=nsamples,
        )
    
    def _build_pos_embed(self, dim):
        return nn.Conv1d(
            #in_channels=4,
            in_channels=2,  #Add
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
    
    def _build_transformer(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        return Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
    '''

    def _build_mlp_head(self, dim, mlp_dim, num_coord_joints, dropout):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_coord_joints),
        )
    
    

    
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Forward pass of the SPiKE model.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    def forward(self, ptc, joints):   # [24, 1, 4096, 3], joints: [24, 2, 15, 3]or[24, 3, 15, 3]
        device, batch_size = ptc.device, ptc.shape[0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Local Feature Extraction
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #reference_pts, ptc_features = self.stem(ptc, joints[:,-1].unsqueeze(1))
        #ptc_centroids, ptc_embeddings, ref_pts = self.stem(ptc, joints)

        # pointnet
        if self.backbone == "pointnet":
            ptc_global_embd, ptc_local_embd, centroids = self.stem(ptc[:,-1].contiguous(), joints[:,-1].contiguous())
        # [24, 1024], [24, 1024, 15], [24, 15, 3]

        # pointnet++
        elif self.backbone == "pointnet++":
            ptc_global_embd, ptc_local_embd = self.stem(ptc[:,-1].contiguous(), joints[:,-1].contiguous())              # [24, 1024], [24, 1024, 15]


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Joints Embedding
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #''' pointnet
        if self.backbone == "pointnet":
            joints = torch.cat((joints[:,:-1], centroids.unsqueeze(1)), dim=1)                                          # [24, 3, 15, 3]
            jnt_embd = self.jnt_embed(joints.permute(0, 3, 1, 2).contiguous())                                          # [24, 1024, 3, 15]
            jnt_embd[:, :, -1, :] += ptc_local_embd                                                                     # [24, 1024, 3, 15]
            embd = jnt_embd.permute(0,2,3,1).contiguous()                                                               # [24, 3, 15, 1024]
            embd = embd.view(embd.shape[0], -1, embd.shape[-1]).contiguous()                                            # [24, 45, 1024]
        #'''

        #''' pointnet++
        elif self.backbone == "pointnet++":
            joints = joints[:,:-1]                                                                                      # [24, 2, 15, 3]
            '''
            jnt_embd = self.jnt_embed(joints.permute(0, 3, 1, 2).contiguous())                                          # [24, 1024, 2, 15]
            embd = torch.cat((jnt_embd, ptc_local_embd.unsqueeze(2)), dim=2)                                            # [24, 1024, 3, 15]
            embd = embd.permute(0,2,3,1).contiguous()                                                                   # [24, 3, 15, 1024]
            '''
            jnt_embd = self.jnt_embed(joints)                                                                           # [24, 2, 15, 1024]
            embd = torch.cat((jnt_embd, ptc_local_embd.transpose(1,2).contiguous().unsqueeze(1)), dim=1)                # [24, 3, 15, 1024]
            #'''
            embd = embd.view(embd.shape[0], -1, embd.shape[-1]).contiguous()                                            # [24, 45, 1024]
        #'''
        

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Positional Embedding
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #L, N = ptc.shape[1] + joints.shape[1], joints.shape[2]                                                      # 3, 15
        L, N = 3,15
        jnt_idx = torch.arange(N, device=device, dtype=torch.float).view(1,1,N,1).expand(batch_size,L,-1,-1) + 1    # [24, 3, 15, 1]
        t = torch.arange(L, device=device, dtype=torch.float).view(1,L,1,1).expand(batch_size,-1,N,-1) + 1          # [24, 3, 15, 1]

        pe = torch.cat((jnt_idx, t), dim=-1)                                                                        # [24, 3, 15, 2]
        pe = pe.view(batch_size, -1, pe.shape[-1]).contiguous()                                                     # [24, 45, 2]
        pe = self.pos_embed(pe.transpose(1,2)).transpose(1,2)                                                       # [24, 45, 1024]

        embd_pe = embd + pe                                                                                         # [24, 45, 1024]
        embd_pe_cls = torch.cat((ptc_global_embd.unsqueeze(1), embd_pe), dim=1)                                     # [24, 46, 1024]


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transformer
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = self.transformer(embd_pe_cls)                                                                      # [24, 46, 1024]

        # 全フレームから集約（SPiKE）
        #output = torch.max(output, dim=1, keepdim=False)[0]                                                        # [24, 1024]                                                                      # [24, 45]

        # 点群フレームのみ使用
        #output = output[:,1:16]                                                                                     # [24, 15, 1024]
        #output = torch.max(output, dim=1, keepdim=False)[0]                                                         # [24, 1024]

        # cls tokenのみ使用（global feature）
        output = output[:,0]                                                                                        # [24, 1024]
        
        jnt_pred = self.mlp_head(output)                                                                            # [24, 45]

        '''from utils import my_functions
        group_ptc = group_ptc.permute(0, 2, 3, 1)
        for n in range(batch_size):
            _joints_coord = joints_coord[n].view(joints[n, -1].shape)
            print(f"joints[n, -1].shape {joints[n, -1].shape} = _joints_coord: {_joints_coord}")
            fig = my_functions.plot_point_cloud_and_local_regions(point_cloud=ptc[n, -1], local_regions=group_ptc[n],
                                                                  pred_joints=_joints_coord,  pred_width=5,  pred_color=True, 
                                                                  label_joints=joints[n, -1], label_width=1, label_color=False)
            my_functions.draw_fig(fig=fig)'''
        
        return jnt_pred
        #return joints_coord, ref_pts #ptc_centroids
