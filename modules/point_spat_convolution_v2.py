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
        #self.spatial_stride = spatial_stride

        #self.conv_d = self._build_conv_d()
        #self.mlp = self._build_mlp()
        
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

    '''
    def _build_conv_d(self):
        conv_d = [
            nn.Conv2d(
                #in_channels=4,
                in_channels=3,  #Add
                out_channels=self.mlp_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            #, nn.BatchNorm2d(num_features=self.mlp_channels[0])
            #, nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*conv_d)

    def _build_mlp(self):
        mlp = []
        for i in range(1, len(self.mlp_channels)):
            print(f"self.mlp_channels[{i}]: {self.mlp_channels[i]}")
            if self.mlp_channels[i] != 0:
                mlp.append(
                    nn.Conv2d(
                        in_channels=self.mlp_channels[i - 1],
                        out_channels=self.mlp_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                )
            if self.mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=self.mlp_channels[i]))
            if self.mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        return nn.Sequential(*mlp)
    '''

    def forward(self, ptc: torch.Tensor, jnt: torch.Tensor=None): #-> (torch.Tensor, torch.Tensor):
        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Forward pass for the point spatial convolution layer.
        
        in: - ptcs (torch.Tensor)          [B, 1,  N, 3]: point cloud
            - reference_pts (torch.Tensor) [B, 2, 15, 3]: for sampling

        out:- reference_pts (torch.Tensor): for sampling
            - features (torch.Tensor)     : local featurea
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
        #print(f"ptc.shape: {ptc.shape}, jnt.shape{jnt.shape}")
        # ptc: [24, 4096, 3], joints: [24, 15, 3]

        ''' grouping empty test
        #ref = torch.full((24, 15, 3), 100.0, device=jnt.device)
        #idx = pointnet2_utils.ball_query(self.spatial_kernel_size, self.nsamples, ptc, ref)
        #'''

        idx = pointnet2_utils.ball_query(self.spatial_kernel_size, self.nsamples*2, ptc, jnt)   # []
        local_ptc = pointnet2_utils.grouping_operation(ptc.transpose(1, 2).contiguous(), idx)   # [24, 3, 15, 64]

        local_ptc = local_ptc.permute(0, 2, 3, 1).contiguous()                                  # [24, 15, 64, 3]
        B, V ,N ,_ = local_ptc.shape

        #''' rm duplication
        new_tensor = []
        flag = False

        for b in range(B):
            for v in range(V):
                slice = local_ptc[b, v]
                unique = torch.unique(slice, dim=0)
                if unique.shape[0] != N:
                    #print(f"{b}_{joint_indices[v]} rm duplication", unique.shape)
                    flag = True
                new_tensor.append(unique)
        if flag:
            padded_tensor = torch.zeros((B*V, N, 3), dtype=local_ptc.dtype, device=local_ptc.device)    #[360, 64, 3]
            for i, coords in enumerate(new_tensor):
                padded_tensor[i, :len(coords)] = coords
            local_ptc = padded_tensor.view(B,V,N,3).contiguous()
        #'''
        
        ''' viewer
        import matplotlib.pyplot as plt
        from const import skeleton_joints
        from utils import my_functions

        indices = [[0, 1, 8], [2, 4, 6], [3, 5, 7], [9, 11, 13], [10, 12, 14]]
        test_ptc =  torch.stack([local_ptc[:, idx, :, :] for idx in indices], dim=1)    # [24, 5, 3, 64, 3]
        for b in range(ptc.shape[0]):
            fig = plt.figure(f"batch: {b}")
            ax = fig.add_subplot(projection='3d')

            point_cloud  = ptc[b].to('cpu').detach().numpy().copy()
            ax.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray", alpha=0.2)
                
            #local_regions  = local_ptc[b].to('cpu').detach().numpy().copy()
            local_regions  = test_ptc[b].to('cpu').detach().numpy().copy()
            #_centroids = centroids[b].to('cpu').detach().numpy().copy()
            for i in range(local_regions.shape[0]):
                #ax.scatter(-local_regions[i,:,0], local_regions[i,:,1], local_regions[i,:,2], s = 14, c = skeleton_joints.joint_color[i])
                for j in range(local_regions[i].shape[0]):
                    ax.scatter(-local_regions[i,j,:,0], local_regions[i,j,:,1], local_regions[i,j,:,2], s = 14, c = skeleton_joints.joint_color[i])
                    
        
            jnt_label  = jnt[b].to('cpu').detach().numpy().copy()
            #my_functions.draw_sphere(ax=ax, r=self.spatial_kernel_size, centers=jnt_label)
            #my_functions.draw_circle(ax=ax, r=self.spatial_kernel_size, centers=jnt_label)           
            my_functions.draw_limbs(ax=ax, joints=jnt_label, linewidth=1)

            ax.set_aspect('equal')
            ax.view_init(elev=-90, azim=90)
            plt.grid(False)
            ax.axis("off")

            my_functions.draw_fig(fig)
        #'''

        # PointNet
        if self.backbone == "pointnet":
            global_ptc = local_ptc.view(B,-1,3)
            centroids = torch.mean(local_ptc, dim=-2)                                       # [24, 15, 3]
            local_ptc = local_ptc - centroids.unsqueeze(2)

            local_ptc = local_ptc.view(-1,N,3)                                              # [360, 64, 3]
            local_features = self.local_pointnet(local_ptc)                                 # [360, 1024]
            local_features = local_features.view(B,V,-1).transpose(1,2).contiguous()        # [24, 1024, 15]
            
            #global_feature = self.global_pointnet(ptc)                                      # [24, 1024]
            global_feature = self.global_pointnet(global_ptc)                                # [24, 1024]

            #return global_feature, local_features, centroids                               #[24, 1024], [24, 1024, 15], [24, 15, 3]
            return global_feature, local_features                                           #[24, 1024], [24, 1024, 15]
        #local_features = local_features.view(B,V,-1).contiguous()                       # [24, 15, 1021]
        #local_features = torch.cat((centroids, local_features), dim=-1)                 # [24, 15, 1024]
        #local_features = local_features.transpose(1,2).contiguous()                     # [24, 1024, 15]
    

        # PointNet++
        elif self.backbone == "pointnet++":
            global_feature, local_features = self.pointnet2(local_ptc)      # [24, 1024], [24, 128, 15]
            local_features = self.conv_d(local_features)                    # [24, 1024, 15]

            return global_feature, local_features
        

        """
        #  Transform Tensor2List
        ptcs = [ptc.squeeze(dim=1).contiguous() for ptc in torch.split(ptcs, 1, dim=1)]     # torch: [24, 3, 4096, 3] -> list: 3*[24, 4096, 3]
        reference_pts = [reference_pt.squeeze(dim=1).contiguous() for reference_pt in torch.split(reference_pts, 1, dim=1)]
        
        #new_reference_pts = []
        features = []
        for n in range(len(ptcs)):
            ptc = ptcs[n]

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  Sampling
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if reference_pts is None:
                # sampling by FPS
                #reference_idx = pointnet2_utils.furthest_point_sample(ptc, ptc.size(1) // self.spatial_stride)              # [24, 128]
                reference_idx = pointnet2_utils.furthest_point_sample(ptc, 15)                                              # [24, 15]
                reference_pt_flipped = pointnet2_utils.gather_operation(ptc.transpose(1, 2).contiguous(), reference_idx)    # [24, 3, 15]
                reference_pt = reference_pt_flipped.transpose(1, 2).contiguous()                                            # [24, 15, 3]
            else:
                reference_pt = reference_pts[-1]                                     # [24, 15, 3]
                reference_pt_flipped = reference_pt.transpose(1, 2).contiguous()    # [24, 3, 15]
            
            r = torch.norm(reference_pts[-2][:, :, :]-reference_pts[-1][:, :, :], p=2, dim=-1)      # [24, 15]
            r_maxs, _ = torch.max(r, dim=1)
            

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  Grouping
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #idx = pointnet2_utils.ball_query(self.spatial_kernel_size, self.nsamples, ptc, reference_pt)    # [24, 128 or 15, 32]
            idx_list = []
            for b in range(ptc.shape[0]):
                if r_maxs[b] > self.spatial_kernel_size:
                    idx = pointnet2_utils.ball_query(self.spatial_kernel_size+r_maxs[b], self.nsamples, ptc[b].unsqueeze(0), reference_pt[b].unsqueeze(0))
                else:
                    idx = pointnet2_utils.ball_query(self.spatial_kernel_size, self.nsamples, ptc[b].unsqueeze(0), reference_pt[b].unsqueeze(0))
                idx_list.append(idx)
            idx = torch.cat(idx_list, dim=0)
            
            '''
            idx_list = []

            #r[:, [0,1,2,3,8,9,10,11,12]] += 0.2
            #r[:, [4,5,13,14]] += 0.3
            #r[:, [6,7]] += 0.4
            r[:, [0,1,2,3,4,5,8,9,10,11,12]] += 0.2
            r[:, [6,7,13,14]] += 0.3
            for b in range(ptc.shape[0]):
                idx = torch.cat((pointnet2_utils.ball_query(r[b][0] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 0,:].reshape(1, 1, 3)), 
                                 pointnet2_utils.ball_query(r[b][1] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 1,:].reshape(1, 1, 3)), 
                                 pointnet2_utils.ball_query(r[b][2] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 2,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][3] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 3,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][4] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 4,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][5] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 5,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][6] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 6,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][7] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 7,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][8] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 8,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][9] , self.nsamples, ptc[b].unsqueeze(0), reference_pt[b, 9,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][10], self.nsamples, ptc[b].unsqueeze(0), reference_pt[b,10,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][11], self.nsamples, ptc[b].unsqueeze(0), reference_pt[b,11,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][12], self.nsamples, ptc[b].unsqueeze(0), reference_pt[b,12,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][13], self.nsamples, ptc[b].unsqueeze(0), reference_pt[b,13,:].reshape(1, 1, 3)),
                                 pointnet2_utils.ball_query(r[b][14], self.nsamples, ptc[b].unsqueeze(0), reference_pt[b,14,:].reshape(1, 1, 3)))
                                 ,dim=1,)
                idx_list.append(idx)
            idx = torch.cat(idx_list, dim=0)
            '''

            group_ptc = pointnet2_utils.grouping_operation(ptc.transpose(1, 2).contiguous(), idx)       # [24, 3, 15, 32]
            #global_ptc = group_ptc.reshape(group_ptc.shape[0], group_ptc.shape[1], -1)                  # [24, 3, 480]
            
            # STN
            #B, D, N = global_ptc.size()
            #trans = self.stn(global_ptc)
            #global_ptc = global_ptc.transpose(2, 1)
            #global_ptc = torch.bmm(global_ptc, trans)
            #global_ptc = global_ptc.transpose(2, 1).unsqueeze(2).contiguous()                           # [24, 3, 1, 480]

             
            #local_ptc = global_ptc.reshape(group_ptc.shape)                                             # [24, 3, 15, 32]
            local_ptc = group_ptc                                                                 # [24, 3, 15, 32]
            from utils import my_functions
            fig = my_functions.plot_point_cloud_and_local_regions(local_regions=local_ptc.permute(0, 2, 3, 1).contiguous()[0])
            my_functions.draw_fig(fig=fig)

            local_ptc_flip = local_ptc.permute(0, 2, 3, 1).contiguous()                                 # [24, 15, 32, 3]
            local_centroids = torch.mean(local_ptc_flip, dim=2)                                         # [24, 15, 3]
            global_centroid = torch.mean(local_ptc_flip.reshape(local_ptc_flip.shape[0], -1, 3), dim=1) # [24, 3]
            #print(global_centroid)

            '''
            import matplotlib.pyplot as plt
            from const import skeleton_joints
            from utils import my_functions

            for b in range(ptc.shape[0]):
                fig = plt.figure("test", figsize=(12,12))
                ax = fig.add_subplot(projection='3d')

                point_cloud  = ptc[b].to('cpu').detach().numpy().copy()
                ax.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray", alpha=0.4)
                
                local_regions  = group_ptc.permute(0, 2, 3, 1)[b].to('cpu').detach().numpy().copy()
                _centroids = centroids[b].to('cpu').detach().numpy().copy()
                for i in range(local_regions.shape[0]):
                    ax.scatter(-local_regions[i,:,0], local_regions[i,:,1], local_regions[i,:,2], s = 14, c = skeleton_joints.joint_color[i])
                    #ax.scatter(-_centroids[i,0], _centroids[i,1], _centroids[i,2], s = 50, c = skeleton_joints.joint_color[i])
        
                pred_joints  = reference_pt[b].to('cpu').detach().numpy().copy()
                _r = r[b].to('cpu').detach().numpy().copy()

                my_functions.draw_sphere(ax=ax, r=_r, centers=pred_joints)
                my_functions.draw_circle(ax=ax, r=_r, centers=pred_joints)           
                my_functions.draw_limbs(ax=ax, joints=pred_joints)

                #target1 = reference_pts[-1][b].to('cpu').detach().numpy().copy()
                #target2 = reference_pts[-2][b].to('cpu').detach().numpy().copy()
                #target3 = reference_pts[-3][b].to('cpu').detach().numpy().copy()
                #my_functions.draw_limbs(ax=ax, joints=target1, linewidth=5)
                #my_functions.draw_limbs(ax=ax, joints=target2, linewidth=1)
                #my_functions.draw_limbs(ax=ax, joints=target3, linewidth=1, color="black")
                #my_functions.draw_limbs(ax=ax, joints=_centroids, linewidth=1, color="black")

                # set gridsize
                ax.set_aspect('equal')
                ax.view_init(elev=-90, azim=90)
                plt.grid(False)
                ax.axis("off")

                my_functions.draw_fig(fig)
            #'''

            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  Local Feature Extraction
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ''' 4D: xyz + zeros
            displacement = torch.cat(
                (
                    #neighbor_xyz_grouped - reference_xyz_flipped.unsqueeze(3),
                    neighbor_xyz_grouped - local_regions_flipped.unsqueeze(3),
                    torch.zeros(
                        (
                            xyz.size(0),
                            1,
                            #xyz.size(1) // self.spatial_stride,
                            local_regions.size(1),
                            self.nsamples,
                        ),
                        device=device,
                    ),
                ),
                dim=1,
            )
            '''
            
            #local_ptc = group_ptc - local_centroids.transpose(1, 2).contiguous().unsqueeze(3)           # [24, 3, 15, 32]
            #local_ptc = self.conv_d(local_ptc)                                                          # [24, 1024, 15, 32]
            
            local_ptc = self.conv_d(local_ptc)                                                          # [24, 1024, 15, 32]
            #local_feature = torch.max(self.mlp(local_ptc), dim=-1, keepdim=False)[0]                    # [24, 1024, 15]
            local_feature = torch.max(local_ptc, dim=-1, keepdim=False)[0]                              # [24, 1024, 15]

            #global_ptc = self.conv_d(global_ptc)                                                        # [24, 1024, 1, 480]
            global_feature = torch.max(global_ptc, dim=-1, keepdim=False)[0]                            # [24, 1024, 1]

            feature = torch.cat((global_feature, local_feature), dim=-1)                                # [24, 1024, 16]

            #features.append(torch.max(torch.stack([feature], dim=1), dim=1, keepdim=False)[0])
            features.append(feature)

            #new_reference_pts.append(reference_pt)                                                      # [24, 15, 3]
            centroids = torch.cat((global_centroid.unsqueeze(1), local_centroids), dim=1)               #[24, 16, 3]

            new_reference_pts.append(centroids)                              # [24, 15, 3]
        """
            
        #return torch.stack(new_reference_pts, dim=1), torch.stack(features, dim=1), group_ptc.transpose(2, 1).contiguous()
        # List: 3*[24, 1024, 15] -> Tensor: [24, 3, 1024, 15]

import matplotlib.pyplot as plt 
class HierarchicalAggregation(nn.Module):
    def __init__(self, channels_list, conv_args, norm_args, act_args, aggr_args):
        super().__init__()
        self.local_aggregations = nn.ModuleList()
        for channels in channels_list:
            self.local_aggregations.append(
                #LocalAggregation(channels, aggr_args, conv_args, norm_args, act_args, group_args=None, use_res=False) # group_argsはNone
                LocalAggregation(channels, aggr_args, conv_args, norm_args, act_args, use_res=False)
            )

    def forward(self, local_regions):
        """
        Args:
            local_regions: (B,V,Nv,3)
        Returns:
            global_feature: (B, C3)
        """
        
        # 1st PointNet                                                                      # [24, 15, 64, 3]
        #ptcs = local_regions[0].to('cpu').detach().numpy().copy()

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
        
        '''

        # print(f'centroids_1.shape: {centroids_1.shape}')                                  # [24, 3, 15]
        centroids_2 = torch.mean(centroids_1, dim=2)                                        # [24, 3]
        relative_2 = centroids_1 - centroids_2.unsqueeze(2)                                 # [24, 3, 15]
        # print(f'local_features_1.shape: {local_features_1.shape}')                        # [24, 128, 15]
        local_features_2 = torch.cat((relative_2, local_features_1), dim=1)                 # [24, 131, 15]
        local_features_2 = local_features_2.unsqueeze(2)                                    # [24, 131, 1, 15]
        global_feature = self.local_aggregations[1](support_features=local_features_2)      # [24, 1024, 1]
        #'''

        ''' fig
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')

        from const import skeleton_joints
        for n in range(ptcs.shape[0]): ax.scatter(-ptcs[n,:,0], ptcs[n,:,1], ptcs[n,:,2], s = 10, c = skeleton_joints.joint_color[n], alpha=0.4)

        from utils import my_functions
        ctr1  = centroids_1.transpose(1,2).to('cpu').detach().numpy().copy()
        my_functions.draw_limbs(ax, ctr1[0], linewidth=5, color="red")

        ctr2  = centroids_2[0].to('cpu').detach().numpy().copy()
        ax.plot([-ctr2[0,0],-ctr2[1,0],-ctr2[2,0],-ctr2[3,0],-ctr2[4,0]], 
                [ ctr2[0,1], ctr2[1,1], ctr2[2,1], ctr2[3,1], ctr2[4,1]], 
                [ ctr2[0,2], ctr2[1,2], ctr2[2,2], ctr2[3,2], ctr2[4,2]],
                color="green", marker='o', markersize=10, linestyle='')

        ctr3  = centroids_3[0].to('cpu').detach().numpy().copy()
        ax.plot(-ctr3[0], ctr3[1], ctr3[2], color="blue", marker='o', markersize=20)

        ax.set_aspect('equal')
        ax.view_init(elev=-90, azim=90)
        plt.grid(False)
        ax.axis("off")
        my_functions.draw_fig(fig=fig)
        '''

        return global_feature.squeeze(2), local_features

    # 局所領域特徴を分割する関数 (仮実装 - 適切に実装してください)
    def split_regions(self, local_features):
        #B, V1, C1 = local_features.shape
        #num_combined = V1 // num_splits # 端数は切り捨て
        #local_features = local_features[:, :num_combined * num_splits, :].view(B, num_splits, num_combined, C1)

        indices = [[0, 1, 8], [2, 4, 6], [3, 5, 7], [9, 11, 13], [10, 12, 14]]
        local_features =  torch.stack([local_features[:, idx, :] for idx in indices], dim=1)            # [24, 5, 3, 128]

        return local_features


#"""PointNet
#Reference:
#https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py


#import torch
#import torch.nn as nn
import torch.nn.functional as F
#from ..build import MODELS
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
#"""


"""
class SharedMLP(nn.Module):
    def __init__(self, channels, conv_args, norm_args, act_args):
        super().__init__()
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_convblock1d(channels[i], channels[i+1],
                                            norm_args=norm_args, act_args=act_args, **conv_args))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, aggregation_type="max"):
        features = self.convs(x)
        if aggregation_type == "max":
            features = torch.max(features, dim=2)[0]
        elif aggregation_type == "avg":
            features = torch.mean(features, dim=2)
        else:
            raise ValueError("Invalid aggregation type")
        return features
"""