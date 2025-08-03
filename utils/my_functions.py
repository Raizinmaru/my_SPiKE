import math
import numpy as np
from const import skeleton_joints
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
#import os

"""
draw options
"""
def Rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return rot_pointcloud

def draw_limbs(ax, joints, linewidth=5, alpha=None, color=None, text_offset=0.0):
    for n, limb in enumerate(skeleton_joints.joint_connections):
        #limb: (idx1, idx2, color)
        p1, p2, _color = limb[0], limb[1], limb[2]
                        
        x1, y1, z1 = joints[p1] 
        x2, y2, z2 = joints[p2]
        
        if color is not None: _color = color
        ax.plot([-x1, -x2], [y1, y2], [z1, z2], color=_color, marker='o', linewidth=linewidth, alpha=alpha)

        #from math import floor
        #length = floor(np.linalg.norm(joints[p1] - joints[p2])*10**5) / 10**5
        #text limb[3].ljust(17, " ")
        #ax.text(2, 1.2+text_offset-n*0.2, 0, f"{text}:{length}", zdir='x', color=color)

def draw_circle(ax, r, centers, color=None):
    if color is None: color = skeleton_joints.joint_color
    else: color = [color]*centers.shape[0]
    for n in range(centers.shape[0]):
        # 円周上の点を計算
        theta = np.linspace(0, 2 * np.pi, 100)
        x = r * np.cos(theta) + centers[n, 0]
        y = r * np.sin(theta) + centers[n, 1]
        z = np.zeros_like(theta) + centers[n, 2]

        ax.plot(-x, y, z, color=color[n])

def draw_sphere(ax, r, centers, color=None, alpha=0.1):
    if color is None: color = skeleton_joints.joint_color
    else: color = [color]*centers.shape[0]
    for n in range(centers.shape[0]):
        theta_1_0 = np.linspace(0, np.pi, 100)                # θ_1は[0,π/2]の値をとる
        theta_2_0 = np.linspace(0, np.pi*2, 100)                # θ_2は[0,π/2]の値をとる
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)    # ２次元配列に変換
        x = centers[n, 0]+ np.cos(theta_2)*np.sin(theta_1) * r      # xの極座標表示
        y = centers[n, 1]+ np.sin(theta_2)*np.sin(theta_1) * r     # yの極座標表示
        z = centers[n, 2]+ np.cos(theta_1) * r                      # zの極座標表示
        
        ax.plot_surface(-x,y,z, alpha=alpha, color=color[n]) # 球を３次元空間に表示


"""
output options
"""
def save_fig(fig, file_name="./"):
    plt.savefig(file_name)
    fig.clear()
    plt.clf()
    plt.close()

class draw_fig:
    def __init__(self, fig):
        #self.fig = plot_point_cloud_and_joints_multiview(point_cloud=point_cloud)
        self.fig = fig
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def on_key_press(self, event):
        if event.key == ' ':
            self.fig.clear()
            plt.clf()
            plt.close(self.fig)
        elif event.key == 'escape':
            from sys import exit
            exit()


"""
draw define
"""
def plot_fig(point_cloud=None, label_joints=None):
    # create fig
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')

    theta = -17

    # draw point cloud
    if torch.is_tensor(point_cloud): point_cloud  = point_cloud.to('cpu').detach().numpy().copy()
    ax.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray")

    '''
    centroid = np.mean(point_cloud, axis=0)
    _point_cloud = point_cloud - centroid
    point_cloud_rot_y20 = Rotation_xyz(_point_cloud, 0, theta, 0)
    ax.scatter(-point_cloud_rot_y20[:,0], point_cloud_rot_y20[:,1], point_cloud_rot_y20[:,2], s = 1, c = "gray")
    #'''
    
    # draw joints
    if torch.is_tensor(label_joints): label_joints = label_joints.to('cpu').detach().numpy().copy()
    print(label_joints.shape)
    draw_limbs(ax, label_joints[0], linewidth=1, color='black')
    draw_limbs(ax, label_joints[1], linewidth=5, color='black')
    
    '''
    _label_joints = label_joints - centroid
    label_joints_rot20_y  = Rotation_xyz(_label_joints, 0, theta,  0)
    draw_limbs(ax, label_joints_rot20_y, linewidth=5)
    #draw_limbs(ax, label_joints, linewidth=1, color='black')
    #draw_sphere(ax=ax, r=0.1, centers=label_joints)
    #draw_circle(ax=ax, r=0.1, centers=label_joints)
    #'''

    # fig setting
    ax.set_aspect('equal')
    #ax.view_init(elev=-90, azim=90)
    ax.view_init(elev=-65, azim=93)
    plt.grid(False)
    ax.axis("off")

    return fig


def plot_point_cloud_and_joints(point_cloud=None, pred_joints=None, label_joints=None, view_angle='front'):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')

    if point_cloud is not None:
        if torch.is_tensor(point_cloud): point_cloud  = point_cloud.to('cpu').detach().numpy().copy()

        if view_angle == 'right':
            centroid = np.mean(point_cloud, axis=0)
            _point_cloud = point_cloud - centroid
            point_cloud = Rotation_xyz(_point_cloud, 0, 90, 0) + centroid
        elif view_angle == 'left':
            centroid = np.mean(point_cloud, axis=0)
            _point_cloud = point_cloud - centroid
            point_cloud = Rotation_xyz(_point_cloud, 0, -90, 0) + centroid

        ax.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray", alpha=0.6)

    if pred_joints is not None: 
        if torch.is_tensor(pred_joints): pred_joints  = pred_joints.to('cpu').detach().numpy().copy()

        if view_angle == 'right':
            _pred_joints = pred_joints - centroid
            pred_joints  = Rotation_xyz(_pred_joints, 0, 90,  0) + centroid
        elif view_angle == 'left':
            _pred_joints = pred_joints - centroid
            pred_joints  = Rotation_xyz(_pred_joints, 0, -90,  0) + centroid

        draw_limbs(ax, pred_joints, linewidth=5)

    if label_joints is not None: 
        if torch.is_tensor(label_joints): label_joints = label_joints.to('cpu').detach().numpy().copy()
        draw_limbs(ax, label_joints, linewidth=1, color='black')
        #draw_sphere(ax=ax, r=0.1, centers=label_joints)
        #draw_circle(ax=ax, r=0.1, centers=label_joints)

    # set viewpoint
    if view_angle == 'top': ax.view_init(elev=0, azim=90)
    else: ax.view_init(elev=-90, azim=90)
    
    ax.set_aspect('equal')
    ax.axis("off")
    plt.grid(False)
    
    return fig



def plot_local_regions_and_joints(point_cloud=None,
                                  local_pts=None,
                                  joint_pts=None,
                                  ref_pts  =None,
                                  targets  =None
                                  ):

    point_cloud  = point_cloud.to('cpu').detach().numpy().copy()    # [3, 4096, 3]
    local_pts    = local_pts.to('cpu').detach().numpy().copy()      # [15, 32, 3]
    joint_pts    = joint_pts.to('cpu').detach().numpy().copy()      # [3, 15, 32, 3]
    ref_pts      = ref_pts.to('cpu').detach().numpy().copy()        # [15, 3]
    targets      = targets.to('cpu').detach().numpy().copy()        # [15, 3]

    limbs = [
        (5,  3,  "#808080", "L Elbow-Shoulder"),    # L Elbow -- L Shoulder (gray)
        (7,  5,  "#FFA500", "L Hand-Elbow"),        # L Hand -- L Elbow (orange)
        (4,  2,  "#FFD700", "R Elbow-Shoulder"),    # R Elbow -- R Shoulder (gold again, for consistency with the hip)
        (6,  4,  "#808000", "R Hand-Elbow"),        # R Hand -- R Elbow (olive)
        (12, 10, "#FF6347", "L knee-Hip"),          # L knee -- L Hip (tomato red)
        (14, 12, "#FF69B4", "L foot-knee"),         # L foot -- L knee (bright pink)
        (11, 9,  "#FFD700", "R knee-Hip"),          # R knee -- R Hip (gold)
        (13, 11, "#32CD32", "R foot-knee"),         # R foot -- R knee (lime green)
    ]

    joint_connections = [
    (14, 12, "#FF69B4", "L foot-knee"),         # L foot -- L knee (bright pink)
    (12, 10, "#FF6347", "L knee-Hip"),          # L knee -- L Hip (tomato red)
    (13, 11, "#32CD32", "R foot-knee"),         # R foot -- R knee (lime green)
    (11, 9,  "#FFD700", "R knee-Hip"),          # R knee -- R Hip (gold)
    (10, 8,  "#6495ED", "L hip-Torso"),         # L hip -- Torso (cornflower blue)
    (9,  8,  "#FF00FF", "R hip-Torso"),         # R hip -- Torso (magenta)
    (8,  1,  "#00FFFF", "Torso-Neck"),          # Torso -- Neck (cyan)
    (1,  0,  "#ADFF2F", "Neck-Head"),           # Neck -- Head (green yellow)
    (7,  5,  "#FFA500", "L Hand-Elbow"),        # L Hand -- L Elbow (orange)
    (5,  3,  "#808080", "L Elbow-Shoulder"),    # L Elbow -- L Shoulder (gray)
    (3,  1,  "#708090", "L Shoulder-Neck"),     # L Shoulder -- Neck (slate gray)
    (6,  4,  "#808000", "R Hand-Elbow"),        # R Hand -- R Elbow (olive)
    (4,  2,  "#FFD700", "R Elbow-Shoulder"),    # R Elbow -- R Shoulder (gold again, for consistency with the hip)
    (2,  1,  "#4B0082", "R Shoulder-Neck"),     # R Shoulder -- Neck (indigo)
]

    fig = plt.figure("fig", figsize=(20,10))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 5, width_ratios=[4, 1, 1, 1, 1])#, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[:, 0],projection='3d')
    ax2 = fig.add_subplot(gs[0, 1],projection='3d')
    ax3 = fig.add_subplot(gs[0, 2],projection='3d')
    ax4 = fig.add_subplot(gs[0, 3],projection='3d')
    ax5 = fig.add_subplot(gs[0, 4],projection='3d')
    ax6 = fig.add_subplot(gs[1, 1],projection='3d')
    ax7 = fig.add_subplot(gs[1, 2],projection='3d')
    ax8 = fig.add_subplot(gs[1, 3],projection='3d')
    ax9 = fig.add_subplot(gs[1, 4],projection='3d')

    ax1.scatter(-point_cloud[0,:,0], point_cloud[0,:,1], point_cloud[0,:,2], s = 1, c = "gray", alpha=0.1)
    ax1.scatter(-point_cloud[1,:,0], point_cloud[1,:,1], point_cloud[1,:,2], s = 1, c = "gray", alpha=0.2)
    ax1.scatter(-point_cloud[2,:,0], point_cloud[2,:,1], point_cloud[2,:,2], s = 1, c = "red", alpha=0.2)
    draw_sphere(ax=ax1, r=0.2, centers=ref_pts)
    draw_limbs(ax=ax1, joints=joint_pts[0], color="black", linewidth=1, alpha=0.2)
    draw_limbs(ax=ax1, joints=joint_pts[1], color="black", linewidth=1, alpha=0.5)
    draw_limbs(ax=ax1, joints=joint_pts[2], color="red", linewidth=4, alpha=0.5)

    p1, p2, _color = limbs[0][0], limbs[0][1], limbs[0][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax2.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax2, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax2.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax2.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    
    p1, p2, _color = limbs[1][0], limbs[1][1], limbs[1][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax3.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax3, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax3.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax3.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    
    p1, p2, _color = limbs[2][0], limbs[2][1], limbs[2][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax4.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax4, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax4.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax4.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)



    p1, p2, _color = limbs[3][0], limbs[3][1], limbs[3][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax5.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax5, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax5.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax5.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[4][0], limbs[4][1], limbs[4][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax6.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax6, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax6.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax6.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[5][0], limbs[5][1], limbs[5][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax7.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax7, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax7.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax7.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[6][0], limbs[6][1], limbs[6][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax8.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax8, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax8.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax8.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[7][0], limbs[7][1], limbs[7][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax9.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax9, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax9.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax9.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')
    ax6.set_aspect('equal')
    ax7.set_aspect('equal')
    ax8.set_aspect('equal')
    ax9.set_aspect('equal')

    ax1.view_init(elev=-90, azim=90)
    ax2.view_init(elev=-90, azim=90)
    ax3.view_init(elev=-90, azim=90)
    ax4.view_init(elev=-90, azim=90)
    ax5.view_init(elev=-90, azim=90)
    ax6.view_init(elev=-90, azim=90)
    ax7.view_init(elev=-90, azim=90)
    ax8.view_init(elev=-90, azim=90)
    ax9.view_init(elev=-90, azim=90)

    draw_fig(fig=fig)

    return fig


def plot_local_regions(point_cloud=None,
                                  grp_pre=None,
                                  grp_tgt=None,
                                  jnt_pre  =None,
                                  jnt_tgt  =None
                                  ):

    point_cloud  = point_cloud.to('cpu').detach().numpy().copy()    # [3, 4096, 3]
    grp_pre = grp_pre.to('cpu').detach().numpy().copy()             # [15, 32, 3]
    grp_tgt = grp_tgt.to('cpu').detach().numpy().copy()             # [15, 32, 3]
    jnt_pre = jnt_pre.to('cpu').detach().numpy().copy()             # [15, 3]
    jnt_tgt = jnt_tgt.to('cpu').detach().numpy().copy()             # [15, 3]


    fig = plt.figure("fig", figsize=(20,10))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 6, width_ratios=[3,1,1,1,1,1])#, height_ratios=[2, 1])
    ax0  = fig.add_subplot(gs[:, 0],projection='3d')
    ax1  = fig.add_subplot(gs[0, 1],projection='3d')
    ax2  = fig.add_subplot(gs[0, 2],projection='3d')
    ax3  = fig.add_subplot(gs[0, 3],projection='3d')
    ax4  = fig.add_subplot(gs[0, 4],projection='3d')
    ax5  = fig.add_subplot(gs[0, 5],projection='3d')
    ax6  = fig.add_subplot(gs[1, 1],projection='3d')
    ax7  = fig.add_subplot(gs[1, 2],projection='3d')
    ax8  = fig.add_subplot(gs[1, 3],projection='3d')
    ax9  = fig.add_subplot(gs[1, 4],projection='3d')
    ax10 = fig.add_subplot(gs[1, 5],projection='3d')
    ax11 = fig.add_subplot(gs[2, 1],projection='3d')
    ax12 = fig.add_subplot(gs[2, 2],projection='3d')
    ax13 = fig.add_subplot(gs[2, 3],projection='3d')
    ax14 = fig.add_subplot(gs[2, 4],projection='3d')
    ax15 = fig.add_subplot(gs[2, 5],projection='3d')

    ax0.scatter(-point_cloud[-2,:,0], point_cloud[-2,:,1], point_cloud[-2,:,2], s = 1, c = "gray", alpha=0.2)
    ax0.scatter(-point_cloud[-1,:,0], point_cloud[-1,:,1], point_cloud[-1,:,2], s = 1, c = "red", alpha=0.2)
    draw_sphere(ax=ax0, r=0.2, centers=jnt_pre)
    draw_limbs(ax=ax0, joints=jnt_pre, color="black", linewidth=1, alpha=0.2)
    draw_limbs(ax=ax0, joints=jnt_tgt, color="red", linewidth=4, alpha=0.5)

    grp_pre = grp_pre - jnt_pre[:,np.newaxis,:]
    grp_tgt = grp_tgt - jnt_pre[:,np.newaxis,:]
    jnt_tgt = jnt_tgt -jnt_pre
    
    ax1.scatter(-grp_pre[0,:,0], grp_pre[0,:,1], grp_pre[0,:,2], s = 1, c = "gray", alpha=0.4)
    ax2.scatter(-grp_pre[1,:,0], grp_pre[1,:,1], grp_pre[1,:,2], s = 1, c = "gray", alpha=0.4)
    ax3.scatter(-grp_pre[2,:,0], grp_pre[2,:,1], grp_pre[2,:,2], s = 1, c = "gray", alpha=0.4)
    ax4.scatter(-grp_pre[3,:,0], grp_pre[3,:,1], grp_pre[3,:,2], s = 1, c = "gray", alpha=0.4)
    ax5.scatter(-grp_pre[4,:,0], grp_pre[4,:,1], grp_pre[4,:,2], s = 1, c = "gray", alpha=0.4)
    ax6.scatter(-grp_pre[5,:,0], grp_pre[5,:,1], grp_pre[5,:,2], s = 1, c = "gray", alpha=0.4)
    ax7.scatter(-grp_pre[6,:,0], grp_pre[6,:,1], grp_pre[6,:,2], s = 1, c = "gray", alpha=0.4)
    ax8.scatter(-grp_pre[7,:,0], grp_pre[7,:,1], grp_pre[7,:,2], s = 1, c = "gray", alpha=0.4)
    ax9.scatter(-grp_pre[8,:,0], grp_pre[8,:,1], grp_pre[8,:,2], s = 1, c = "gray", alpha=0.4)
    ax10.scatter(-grp_pre[9,:,0], grp_pre[9,:,1], grp_pre[9,:,2], s = 1, c = "gray", alpha=0.4)
    ax11.scatter(-grp_pre[10,:,0], grp_pre[10,:,1], grp_pre[10,:,2], s = 1, c = "gray", alpha=0.4)
    ax12.scatter(-grp_pre[11,:,0], grp_pre[11,:,1], grp_pre[11,:,2], s = 1, c = "gray", alpha=0.4)
    ax13.scatter(-grp_pre[12,:,0], grp_pre[12,:,1], grp_pre[12,:,2], s = 1, c = "gray", alpha=0.4)
    ax14.scatter(-grp_pre[13,:,0], grp_pre[13,:,1], grp_pre[13,:,2], s = 1, c = "gray", alpha=0.4)
    ax15.scatter(-grp_pre[14,:,0], grp_pre[14,:,1], grp_pre[14,:,2], s = 1, c = "gray", alpha=0.4)


    ax1.scatter(-grp_tgt[0,:,0], grp_tgt[0,:,1], grp_tgt[0,:,2], s = 2, c = "red", alpha=0.4)
    ax2.scatter(-grp_tgt[1,:,0], grp_tgt[1,:,1], grp_tgt[1,:,2], s = 2, c = "red", alpha=0.4)
    ax3.scatter(-grp_tgt[2,:,0], grp_tgt[2,:,1], grp_tgt[2,:,2], s = 2, c = "red", alpha=0.4)
    ax4.scatter(-grp_tgt[3,:,0], grp_tgt[3,:,1], grp_tgt[3,:,2], s = 2, c = "red", alpha=0.4)
    ax5.scatter(-grp_tgt[4,:,0], grp_tgt[4,:,1], grp_tgt[4,:,2], s = 2, c = "red", alpha=0.4)
    ax6.scatter(-grp_tgt[5,:,0], grp_tgt[5,:,1], grp_tgt[5,:,2], s = 2, c = "red", alpha=0.4)
    ax7.scatter(-grp_tgt[6,:,0], grp_tgt[6,:,1], grp_tgt[6,:,2], s = 2, c = "red", alpha=0.4)
    ax8.scatter(-grp_tgt[7,:,0], grp_tgt[7,:,1], grp_tgt[7,:,2], s = 2, c = "red", alpha=0.4)
    ax9.scatter(-grp_tgt[8,:,0], grp_tgt[8,:,1], grp_tgt[8,:,2], s = 2, c = "red", alpha=0.4)
    ax10.scatter(-grp_tgt[9,:,0], grp_tgt[9,:,1], grp_tgt[9,:,2], s = 2, c = "red", alpha=0.4)
    ax11.scatter(-grp_tgt[10,:,0], grp_tgt[10,:,1], grp_tgt[10,:,2], s = 2, c = "red", alpha=0.4)
    ax12.scatter(-grp_tgt[11,:,0], grp_tgt[11,:,1], grp_tgt[11,:,2], s = 2, c = "red", alpha=0.4)
    ax13.scatter(-grp_tgt[12,:,0], grp_tgt[12,:,1], grp_tgt[12,:,2], s = 2, c = "red", alpha=0.4)
    ax14.scatter(-grp_tgt[13,:,0], grp_tgt[13,:,1], grp_tgt[13,:,2], s = 2, c = "red", alpha=0.4)
    ax15.scatter(-grp_tgt[14,:,0], grp_tgt[14,:,1], grp_tgt[14,:,2], s = 2, c = "red", alpha=0.4)

    ax1.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax2.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax3.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax4.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax5.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax6.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax7.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax8.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax9.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax10.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax11.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax12.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax13.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax14.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)
    ax15.scatter(0, 0, 0, s=10, c="gray", alpha=0.6)

    ax1.scatter(jnt_tgt[0,0], jnt_tgt[0,1], jnt_tgt[0,2], s=30, c="red", alpha=0.6)
    ax2.scatter(jnt_tgt[1,0], jnt_tgt[1,1], jnt_tgt[1,2], s=30, c="red", alpha=0.6)
    ax3.scatter(jnt_tgt[2,0], jnt_tgt[2,1], jnt_tgt[2,2], s=30, c="red", alpha=0.6)
    ax4.scatter(jnt_tgt[3,0], jnt_tgt[3,1], jnt_tgt[3,2], s=30, c="red", alpha=0.6)
    ax5.scatter(jnt_tgt[4,0], jnt_tgt[4,1], jnt_tgt[4,2], s=30, c="red", alpha=0.6)
    ax6.scatter(jnt_tgt[5,0], jnt_tgt[5,1], jnt_tgt[5,2], s=30, c="red", alpha=0.6)
    ax7.scatter(jnt_tgt[6,0], jnt_tgt[6,1], jnt_tgt[6,2], s=30, c="red", alpha=0.6)
    ax8.scatter(jnt_tgt[7,0], jnt_tgt[7,1], jnt_tgt[7,2], s=30, c="red", alpha=0.6)
    ax9.scatter(jnt_tgt[8,0], jnt_tgt[8,1], jnt_tgt[8,2], s=30, c="red", alpha=0.6)
    ax10.scatter(jnt_tgt[9,0], jnt_tgt[9,1], jnt_tgt[9,2], s=30, c="red", alpha=0.6)
    ax11.scatter(jnt_tgt[10,0], jnt_tgt[10,1], jnt_tgt[10,2], s=30, c="red", alpha=0.6)
    ax12.scatter(jnt_tgt[11,0], jnt_tgt[11,1], jnt_tgt[11,2], s=30, c="red", alpha=0.6)
    ax13.scatter(jnt_tgt[12,0], jnt_tgt[12,1], jnt_tgt[12,2], s=30, c="red", alpha=0.6)
    ax14.scatter(jnt_tgt[13,0], jnt_tgt[13,1], jnt_tgt[13,2], s=30, c="red", alpha=0.6)
    ax15.scatter(jnt_tgt[14,0], jnt_tgt[14,1], jnt_tgt[14,2], s=30, c="red", alpha=0.6)

    O = np.zeros((1, 3))
    draw_sphere(ax=ax1, r=0.2, centers=O, color=skeleton_joints.joint_color[0])
    draw_sphere(ax=ax2, r=0.2, centers=O, color=skeleton_joints.joint_color[1])
    draw_sphere(ax=ax3, r=0.2, centers=O, color=skeleton_joints.joint_color[2])
    draw_sphere(ax=ax4, r=0.2, centers=O, color=skeleton_joints.joint_color[3])
    draw_sphere(ax=ax5, r=0.2, centers=O, color=skeleton_joints.joint_color[4])
    draw_sphere(ax=ax6, r=0.2, centers=O, color=skeleton_joints.joint_color[5])
    draw_sphere(ax=ax7, r=0.2, centers=O, color=skeleton_joints.joint_color[6])
    draw_sphere(ax=ax8, r=0.2, centers=O, color=skeleton_joints.joint_color[7])
    draw_sphere(ax=ax9, r=0.2, centers=O, color=skeleton_joints.joint_color[8])
    draw_sphere(ax=ax10, r=0.2, centers=O, color=skeleton_joints.joint_color[9])
    draw_sphere(ax=ax11, r=0.2, centers=O, color=skeleton_joints.joint_color[10])
    draw_sphere(ax=ax12, r=0.2, centers=O, color=skeleton_joints.joint_color[11])
    draw_sphere(ax=ax13, r=0.2, centers=O, color=skeleton_joints.joint_color[12])
    draw_sphere(ax=ax14, r=0.2, centers=O, color=skeleton_joints.joint_color[13])
    draw_sphere(ax=ax15, r=0.2, centers=O, color=skeleton_joints.joint_color[14])

    ax1.set_title("0: Head")
    ax2.set_title("1: Neck")
    ax3.set_title("2: R Shoulder")
    ax4.set_title("3: L Shoulder")
    ax5.set_title("4: R Elbow")
    ax6.set_title("5: L Elbow")
    ax7.set_title("6: R Hand")
    ax8.set_title("7: L Hand")
    ax9.set_title("8: Torso")
    ax10.set_title("9: R Hip")
    ax11.set_title("10: L Hip")
    ax12.set_title("11: R Knee")
    ax13.set_title("12: L Knee")
    ax14.set_title("13: R Foot")
    ax15.set_title("14: L Foot")

    ax0.set_aspect('equal')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')
    ax6.set_aspect('equal')
    ax7.set_aspect('equal')
    ax8.set_aspect('equal')
    ax9.set_aspect('equal')
    ax10.set_aspect('equal')
    ax11.set_aspect('equal')
    ax12.set_aspect('equal')
    ax13.set_aspect('equal')
    ax14.set_aspect('equal')
    ax15.set_aspect('equal')

    ax0.view_init(elev=-90, azim=90)
    ax1.view_init(elev=-90, azim=90)
    ax2.view_init(elev=-90, azim=90)
    ax3.view_init(elev=-90, azim=90)
    ax4.view_init(elev=-90, azim=90)
    ax5.view_init(elev=-90, azim=90)
    ax6.view_init(elev=-90, azim=90)
    ax7.view_init(elev=-90, azim=90)
    ax8.view_init(elev=-90, azim=90)
    ax9.view_init(elev=-90, azim=90)
    ax10.view_init(elev=-90, azim=90)
    ax11.view_init(elev=-90, azim=90)
    ax12.view_init(elev=-90, azim=90)
    ax13.view_init(elev=-90, azim=90)
    ax14.view_init(elev=-90, azim=90)
    ax15.view_init(elev=-90, azim=90)

    return fig
    

def plot_point_cloud_and_joints_multiview(point_cloud, pred_joints=None, label_joints=None):
    # cast torch -> numpy
    if torch.is_tensor(point_cloud):
        point_cloud  = point_cloud.to('cpu').detach().numpy().copy()

    centroid = np.mean(point_cloud, axis=0)
    _point_cloud = point_cloud - centroid

    # rotate pointcloud
    point_cloud_rot_y90 = Rotation_xyz(_point_cloud, 0, 90, 0)
    point_cloud_rot_y270 = Rotation_xyz(_point_cloud, 0, -90, 0)

    # return to original position
    point_cloud_rot_y90  = point_cloud_rot_y90 + centroid
    point_cloud_rot_y270 = point_cloud_rot_y270 + centroid

    # create fig
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax2 = fig.add_subplot(2,2,2,projection='3d')
    ax3 = fig.add_subplot(2,2,3,projection='3d')
    ax4 = fig.add_subplot(2,2,4,projection='3d')

    # plot pointcloud
    ax1.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray", alpha=0.2)
    ax2.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray", alpha=0.2)
    ax3.scatter(-point_cloud_rot_y90[:,0], point_cloud_rot_y90[:,1], point_cloud_rot_y90[:,2], s = 1, c = "gray", alpha=0.2)
    ax4.scatter(-point_cloud_rot_y270[:,0], point_cloud_rot_y270[:,1], point_cloud_rot_y270[:,2], s = 1, c = "gray", alpha=0.2)


    if pred_joints is not None: 
        if torch.is_tensor(pred_joints):
            pred_joints  = pred_joints.to('cpu').detach().numpy().copy()
        
        _pred_joints = pred_joints - centroid

        pred_rot90_y  = Rotation_xyz(_pred_joints, 0, 90,  0)
        pred_rot270_y = Rotation_xyz(_pred_joints, 0, -90, 0)

        pred_rot90_y  = pred_rot90_y  + centroid
        pred_rot270_y = pred_rot270_y + centroid

        draw_limbs(ax1, pred_joints,   linewidth=5)
        draw_limbs(ax2, pred_joints,   linewidth=5)
        draw_limbs(ax3, pred_rot90_y,  linewidth=5)
        draw_limbs(ax4, pred_rot270_y, linewidth=5)

    if label_joints is not None: 
        if torch.is_tensor(label_joints):
            label_joints = label_joints.to('cpu').detach().numpy().copy()
        
        _label_joints = label_joints - centroid

        label_rot90_y  = Rotation_xyz(_label_joints, 0, 90,  0)
        label_rot270_y = Rotation_xyz(_label_joints, 0, -90, 0)

        label_rot90_y  = label_rot90_y  + centroid
        label_rot270_y = label_rot270_y + centroid

        draw_limbs(ax1, label_joints,   linewidth=1, color="black")
        draw_limbs(ax2, label_joints,   linewidth=1, color="black")
        draw_limbs(ax3, label_rot90_y,  linewidth=1, color="black")
        draw_limbs(ax4, label_rot270_y, linewidth=1, color="black")

    # set gridsize
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')

    # set viewpoint
    ax1.view_init(elev=-90, azim=90)
    ax2.view_init(elev=0, azim=90)
    ax3.view_init(elev=-90, azim=90)
    ax4.view_init(elev=-90, azim=90)

    return fig




def plot_local_regions_and_joints(point_cloud=None,
                                  local_pts=None,
                                  joint_pts=None,
                                  ref_pts  =None,
                                  targets  =None
                                  ):

    point_cloud  = point_cloud.to('cpu').detach().numpy().copy()    # [3, 4096, 3]
    local_pts    = local_pts.to('cpu').detach().numpy().copy()      # [15, 32, 3]
    joint_pts    = joint_pts.to('cpu').detach().numpy().copy()      # [3, 15, 32, 3]
    ref_pts      = ref_pts.to('cpu').detach().numpy().copy()        # [15, 3]
    targets      = targets.to('cpu').detach().numpy().copy()        # [15, 3]

    limbs = [
        (5,  3,  "#808080", "L Elbow-Shoulder"),    # L Elbow -- L Shoulder (gray)
        (7,  5,  "#FFA500", "L Hand-Elbow"),        # L Hand -- L Elbow (orange)
        (4,  2,  "#FFD700", "R Elbow-Shoulder"),    # R Elbow -- R Shoulder (gold again, for consistency with the hip)
        (6,  4,  "#808000", "R Hand-Elbow"),        # R Hand -- R Elbow (olive)
        (12, 10, "#FF6347", "L knee-Hip"),          # L knee -- L Hip (tomato red)
        (14, 12, "#FF69B4", "L foot-knee"),         # L foot -- L knee (bright pink)
        (11, 9,  "#FFD700", "R knee-Hip"),          # R knee -- R Hip (gold)
        (13, 11, "#32CD32", "R foot-knee"),         # R foot -- R knee (lime green)
    ]

    joint_connections = [
    (14, 12, "#FF69B4", "L foot-knee"),         # L foot -- L knee (bright pink)
    (12, 10, "#FF6347", "L knee-Hip"),          # L knee -- L Hip (tomato red)
    (13, 11, "#32CD32", "R foot-knee"),         # R foot -- R knee (lime green)
    (11, 9,  "#FFD700", "R knee-Hip"),          # R knee -- R Hip (gold)
    (10, 8,  "#6495ED", "L hip-Torso"),         # L hip -- Torso (cornflower blue)
    (9,  8,  "#FF00FF", "R hip-Torso"),         # R hip -- Torso (magenta)
    (8,  1,  "#00FFFF", "Torso-Neck"),          # Torso -- Neck (cyan)
    (1,  0,  "#ADFF2F", "Neck-Head"),           # Neck -- Head (green yellow)
    (7,  5,  "#FFA500", "L Hand-Elbow"),        # L Hand -- L Elbow (orange)
    (5,  3,  "#808080", "L Elbow-Shoulder"),    # L Elbow -- L Shoulder (gray)
    (3,  1,  "#708090", "L Shoulder-Neck"),     # L Shoulder -- Neck (slate gray)
    (6,  4,  "#808000", "R Hand-Elbow"),        # R Hand -- R Elbow (olive)
    (4,  2,  "#FFD700", "R Elbow-Shoulder"),    # R Elbow -- R Shoulder (gold again, for consistency with the hip)
    (2,  1,  "#4B0082", "R Shoulder-Neck"),     # R Shoulder -- Neck (indigo)
]

    fig = plt.figure("fig", figsize=(20,10))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 5, width_ratios=[4, 1, 1, 1, 1])#, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[:, 0],projection='3d')
    ax2 = fig.add_subplot(gs[0, 1],projection='3d')
    ax3 = fig.add_subplot(gs[0, 2],projection='3d')
    ax4 = fig.add_subplot(gs[0, 3],projection='3d')
    ax5 = fig.add_subplot(gs[0, 4],projection='3d')
    ax6 = fig.add_subplot(gs[1, 1],projection='3d')
    ax7 = fig.add_subplot(gs[1, 2],projection='3d')
    ax8 = fig.add_subplot(gs[1, 3],projection='3d')
    ax9 = fig.add_subplot(gs[1, 4],projection='3d')

    ax1.scatter(-point_cloud[0,:,0], point_cloud[0,:,1], point_cloud[0,:,2], s = 1, c = "gray", alpha=0.1)
    ax1.scatter(-point_cloud[1,:,0], point_cloud[1,:,1], point_cloud[1,:,2], s = 1, c = "gray", alpha=0.2)
    ax1.scatter(-point_cloud[2,:,0], point_cloud[2,:,1], point_cloud[2,:,2], s = 1, c = "red", alpha=0.2)
    draw_sphere(ax=ax1, r=0.2, centers=ref_pts)
    draw_limbs(ax=ax1, joints=joint_pts[0], color="black", linewidth=1, alpha=0.2)
    draw_limbs(ax=ax1, joints=joint_pts[1], color="black", linewidth=1, alpha=0.5)
    draw_limbs(ax=ax1, joints=joint_pts[2], color="red", linewidth=4, alpha=0.5)

    p1, p2, _color = limbs[0][0], limbs[0][1], limbs[0][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax2.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax2, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax2.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax2.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    
    p1, p2, _color = limbs[1][0], limbs[1][1], limbs[1][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax3.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax3, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax3.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax3.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    
    p1, p2, _color = limbs[2][0], limbs[2][1], limbs[2][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax4.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax4, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax4.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax4.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)



    p1, p2, _color = limbs[3][0], limbs[3][1], limbs[3][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax5.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax5, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax5.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax5.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[4][0], limbs[4][1], limbs[4][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax6.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax6, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax6.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax6.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[5][0], limbs[5][1], limbs[5][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax7.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax7, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax7.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax7.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[6][0], limbs[6][1], limbs[6][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax8.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax8, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax8.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax8.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)


    p1, p2, _color = limbs[7][0], limbs[7][1], limbs[7][2]
    ref_start, ref_end, tgt_start, tgt_end = ref_pts[p1], ref_pts[p2], targets[p1], targets[p2]

    local_pts[p2] = local_pts[p2] - ref_end
    tgt_start = tgt_start - ref_end
    tgt_end   = tgt_end   - ref_end
    ref_start = ref_start - ref_end
    ref_end   = ref_end   - ref_end

    ax9.scatter(-local_pts[p2,:,0], local_pts[p2,:,1], local_pts[p2,:,2], s = 1, c = "gray", alpha=0.4)
    draw_sphere(ax=ax9, r=0.2, centers=ref_end[np.newaxis,:], color=_color)
    ax9.plot([-ref_start[0], -ref_end[0]], [ref_start[1], ref_end[1]], [ref_start[2], ref_end[2]], color="black", marker='o', linewidth=1, alpha=0.5)
    ax9.plot([-tgt_start[0], -tgt_end[0]], [tgt_start[1], tgt_end[1]], [tgt_start[2], tgt_end[2]], color=_color, marker='o', linewidth=4)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')
    ax6.set_aspect('equal')
    ax7.set_aspect('equal')
    ax8.set_aspect('equal')
    ax9.set_aspect('equal')

    ax1.view_init(elev=-90, azim=90)
    ax2.view_init(elev=-90, azim=90)
    ax3.view_init(elev=-90, azim=90)
    ax4.view_init(elev=-90, azim=90)
    ax5.view_init(elev=-90, azim=90)
    ax6.view_init(elev=-90, azim=90)
    ax7.view_init(elev=-90, azim=90)
    ax8.view_init(elev=-90, azim=90)
    ax9.view_init(elev=-90, azim=90)

    draw_fig(fig=fig)

    return fig


def plot_each_local_regions(point_cloud=None,
                                  grp_pre=None,
                                  grp_tgt=None,
                                  jnt_pre  =None,
                                  jnt_tgt  =None
                                  ):
    
    ctr_pre = torch.mean(grp_pre, dim=1)                            # [15, 3]
    ctr_tgt = torch.mean(grp_tgt, dim=1)                            # [15, 3]
    ctr_pre = ctr_pre.to('cpu').detach().numpy().copy()             # [15, 3]
    ctr_tgt = ctr_tgt.to('cpu').detach().numpy().copy()             # [15, 3]

    point_cloud  = point_cloud.to('cpu').detach().numpy().copy()    # [3, 4096, 3]
    grp_pre = grp_pre.to('cpu').detach().numpy().copy()             # [15, 32, 3]
    grp_tgt = grp_tgt.to('cpu').detach().numpy().copy()             # [15, 32, 3]
    jnt_pre = jnt_pre.to('cpu').detach().numpy().copy()             # [15, 3]
    jnt_tgt = jnt_tgt.to('cpu').detach().numpy().copy()             # [15, 3]


    fig = plt.figure("fig", figsize=(20,10))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, width_ratios=[2,1,1])#, height_ratios=[2, 1])
    ax0  = fig.add_subplot(gs[:, 0],projection='3d')
    ax1  = fig.add_subplot(gs[0, 1],projection='3d')
    ax2  = fig.add_subplot(gs[0, 2],projection='3d')
    ax3  = fig.add_subplot(gs[1, 1],projection='3d')
    ax4  = fig.add_subplot(gs[1, 2],projection='3d')

    ax0.scatter(-point_cloud[-2,:,0], point_cloud[-2,:,1], point_cloud[-2,:,2], s = 1, c = "gray", alpha=0.2)
    ax0.scatter(-point_cloud[-1,:,0], point_cloud[-1,:,1], point_cloud[-1,:,2], s = 1, c = "red", alpha=0.2)
    draw_sphere(ax=ax0, r=0.2, centers=jnt_pre)
    draw_limbs(ax=ax0, joints=jnt_pre, color="black", linewidth=1, alpha=0.2)
    draw_limbs(ax=ax0, joints=jnt_tgt, color="red", linewidth=4, alpha=0.5)


    limbs = [
        (4,  2,  "#FFD700", "R Elbow-Shoulder"),    # R Elbow -- R Shoulder (gold again, for consistency with the hip)
        (6,  4,  "#808000", "R Hand-Elbow"),        # R Hand -- R Elbow (olive)
        
        (5,  3,  "#808080", "L Elbow-Shoulder"),    # L Elbow -- L Shoulder (gray)
        (7,  5,  "#FFA500", "L Hand-Elbow"),        # L Hand -- L Elbow (orange)

        (11, 9,  "#FFD700", "R knee-Hip"),          # R knee -- R Hip (gold)
        (13, 11, "#32CD32", "R foot-knee"),         # R foot -- R knee (lime green)
    
        (12, 10, "#FF6347", "L knee-Hip"),          # L knee -- L Hip (tomato red)
        (14, 12, "#FF69B4", "L foot-knee"),         # L foot -- L knee (bright pink)
        
        #(10, 8,  "#6495ED", "L hip-Torso"),         # L hip -- Torso (cornflower blue)
        #(9,  8,  "#FF00FF", "R hip-Torso"),         # R hip -- Torso (magenta)
        #(8,  1,  "#00FFFF", "Torso-Neck"),          # Torso -- Neck (cyan)
        #(1,  0,  "#ADFF2F", "Neck-Head"),           # Neck -- Head (green yellow)
        #(3,  1,  "#708090", "L Shoulder-Neck"),     # L Shoulder -- Neck (slate gray)
        #(2,  1,  "#4B0082", "R Shoulder-Neck"),     # R Shoulder -- Neck (indigo)
    ]

    joint_indices = {
    0: "Head",
    1: "Neck",
    2: "R Shoulder",
    3: "L Shoulder",
    4: "R Elbow",
    5: "L Elbow",
    6: "R Hand",
    7: "L Hand",
    8: "Torso",
    9: "R Hip",
    10: "L Hip",
    11: "R Knee",
    12: "L Knee",
    13: "R Foot",
    14: "L Foot"
}
    
    draw_circle(ax=ax1, r=0.2, centers=(jnt_pre[limbs[0][0]]-jnt_pre[limbs[0][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax1, r=0.2, centers=(jnt_pre[limbs[1][0]]-jnt_pre[limbs[0][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax2, r=0.2, centers=(jnt_pre[limbs[2][0]]-jnt_pre[limbs[2][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax2, r=0.2, centers=(jnt_pre[limbs[3][0]]-jnt_pre[limbs[2][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax3, r=0.2, centers=(jnt_pre[limbs[4][0]]-jnt_pre[limbs[4][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax3, r=0.2, centers=(jnt_pre[limbs[5][0]]-jnt_pre[limbs[4][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax4, r=0.2, centers=(jnt_pre[limbs[6][0]]-jnt_pre[limbs[6][1]])[np.newaxis,:], color="gray")
    draw_circle(ax=ax4, r=0.2, centers=(jnt_pre[limbs[7][0]]-jnt_pre[limbs[6][1]])[np.newaxis,:], color="gray")

    # 隣接する関節から見た "1フレーム前" の関節領域点群
    lcl_grp_pre = []
    for n in range(len(limbs)): lcl_grp_pre.append(grp_pre[limbs[n][0],:]-jnt_pre[limbs[n//2*2][1]])
    #torch.stack(lcl_grp_pre, dim=0)
    lcl_grp_pre = np.array(lcl_grp_pre) # [8, 32, 3]
    ax1.scatter(-lcl_grp_pre[0,:,0], lcl_grp_pre[0,:,1], lcl_grp_pre[0,:,2], s = 1, c = "gray", alpha=0.4)
    ax1.scatter(-lcl_grp_pre[1,:,0], lcl_grp_pre[1,:,1], lcl_grp_pre[1,:,2], s = 1, c = "gray", alpha=0.4)
    ax2.scatter(-lcl_grp_pre[2,:,0], lcl_grp_pre[2,:,1], lcl_grp_pre[2,:,2], s = 1, c = "gray", alpha=0.4)
    ax2.scatter(-lcl_grp_pre[3,:,0], lcl_grp_pre[3,:,1], lcl_grp_pre[3,:,2], s = 1, c = "gray", alpha=0.4)
    ax3.scatter(-lcl_grp_pre[4,:,0], lcl_grp_pre[4,:,1], lcl_grp_pre[4,:,2], s = 1, c = "gray", alpha=0.4)
    ax3.scatter(-lcl_grp_pre[5,:,0], lcl_grp_pre[5,:,1], lcl_grp_pre[5,:,2], s = 1, c = "gray", alpha=0.4)
    ax4.scatter(-lcl_grp_pre[6,:,0], lcl_grp_pre[6,:,1], lcl_grp_pre[6,:,2], s = 1, c = "gray", alpha=0.4)
    ax4.scatter(-lcl_grp_pre[7,:,0], lcl_grp_pre[7,:,1], lcl_grp_pre[7,:,2], s = 1, c = "gray", alpha=0.4)

    # 隣接する関節から見た "ターゲットフレーム" の関節領域点群
    lcl_grp_tgt = []
    for n in range(len(limbs)): lcl_grp_tgt.append(grp_tgt[limbs[n][0],:]-jnt_pre[limbs[n//2*2][1]])
    #torch.stack(lcl_grp_pre, dim=0)
    lcl_grp_tgt = np.array(lcl_grp_tgt) # [8, 32, 3]
    ax1.scatter(-lcl_grp_tgt[0,:,0], lcl_grp_tgt[0,:,1], lcl_grp_tgt[0,:,2], s = 2, c = "blue", alpha=0.4)
    ax1.scatter(-lcl_grp_tgt[1,:,0], lcl_grp_tgt[1,:,1], lcl_grp_tgt[1,:,2], s = 2, c = "red", alpha=0.4)
    ax2.scatter(-lcl_grp_tgt[2,:,0], lcl_grp_tgt[2,:,1], lcl_grp_tgt[2,:,2], s = 2, c = "blue", alpha=0.4)
    ax2.scatter(-lcl_grp_tgt[3,:,0], lcl_grp_tgt[3,:,1], lcl_grp_tgt[3,:,2], s = 2, c = "red", alpha=0.4)
    ax3.scatter(-lcl_grp_tgt[4,:,0], lcl_grp_tgt[4,:,1], lcl_grp_tgt[4,:,2], s = 2, c = "blue", alpha=0.4)
    ax3.scatter(-lcl_grp_tgt[5,:,0], lcl_grp_tgt[5,:,1], lcl_grp_tgt[5,:,2], s = 2, c = "red", alpha=0.4)
    ax4.scatter(-lcl_grp_tgt[6,:,0], lcl_grp_tgt[6,:,1], lcl_grp_tgt[6,:,2], s = 2, c = "blue", alpha=0.4)
    ax4.scatter(-lcl_grp_tgt[7,:,0], lcl_grp_tgt[7,:,1], lcl_grp_tgt[7,:,2], s = 2, c = "red", alpha=0.4)

    #'''# 隣接する関節から見た "1フレーム前" の関節領域点群の "重心"
    lcl_ctr_pre = []
    for n in range(len(limbs)): lcl_ctr_pre.append(ctr_pre[limbs[n][0]]-jnt_pre[limbs[n//2*2][1]])
    #torch.stack(lcl_grp_pre, dim=0)
    lcl_ctr_pre = np.array(lcl_ctr_pre) # [8, 3]
    ax1.scatter(-lcl_ctr_pre[0,0], lcl_ctr_pre[0,1], lcl_ctr_pre[0,2], marker='x', s = 50, c = "gray")
    ax1.scatter(-lcl_ctr_pre[1,0], lcl_ctr_pre[1,1], lcl_ctr_pre[1,2], marker='x', s = 50, c = "gray")
    ax2.scatter(-lcl_ctr_pre[2,0], lcl_ctr_pre[2,1], lcl_ctr_pre[2,2], marker='x', s = 50, c = "gray")
    ax2.scatter(-lcl_ctr_pre[3,0], lcl_ctr_pre[3,1], lcl_ctr_pre[3,2], marker='x', s = 50, c = "gray")
    ax3.scatter(-lcl_ctr_pre[4,0], lcl_ctr_pre[4,1], lcl_ctr_pre[4,2], marker='x', s = 50, c = "gray")
    ax3.scatter(-lcl_ctr_pre[5,0], lcl_ctr_pre[5,1], lcl_ctr_pre[5,2], marker='x', s = 50, c = "gray")
    ax4.scatter(-lcl_ctr_pre[6,0], lcl_ctr_pre[6,1], lcl_ctr_pre[6,2], marker='x', s = 50, c = "gray")
    ax4.scatter(-lcl_ctr_pre[7,0], lcl_ctr_pre[7,1], lcl_ctr_pre[7,2], marker='x', s = 50, c = "gray")

    # 隣接する関節から見た "ターゲットフレーム" の関節領域点群の "重心"
    lcl_ctr_tgt = []
    for n in range(len(limbs)): lcl_ctr_tgt.append(ctr_tgt[limbs[n][0]]-jnt_pre[limbs[n//2*2][1]])
    #torch.stack(lcl_grp_pre, dim=0)
    lcl_ctr_tgt = np.array(lcl_ctr_tgt) # [8, 3]
    ax1.scatter(-lcl_ctr_tgt[0,0], lcl_ctr_tgt[0,1], lcl_ctr_tgt[0,2], marker='x', s = 50, c = "red")
    ax1.scatter(-lcl_ctr_tgt[1,0], lcl_ctr_tgt[1,1], lcl_ctr_tgt[1,2], marker='x', s = 50, c = "red")
    ax2.scatter(-lcl_ctr_tgt[2,0], lcl_ctr_tgt[2,1], lcl_ctr_tgt[2,2], marker='x', s = 50, c = "red")
    ax2.scatter(-lcl_ctr_tgt[3,0], lcl_ctr_tgt[3,1], lcl_ctr_tgt[3,2], marker='x', s = 50, c = "red")
    ax3.scatter(-lcl_ctr_tgt[4,0], lcl_ctr_tgt[4,1], lcl_ctr_tgt[4,2], marker='x', s = 50, c = "red")
    ax3.scatter(-lcl_ctr_tgt[5,0], lcl_ctr_tgt[5,1], lcl_ctr_tgt[5,2], marker='x', s = 50, c = "red")
    ax4.scatter(-lcl_ctr_tgt[6,0], lcl_ctr_tgt[6,1], lcl_ctr_tgt[6,2], marker='x', s = 50, c = "red")
    ax4.scatter(-lcl_ctr_tgt[7,0], lcl_ctr_tgt[7,1], lcl_ctr_tgt[7,2], marker='x', s = 50, c = "red")
    #'''

    # 隣接する関節から見た "1フレーム前" の骨
    lcl_jnt_pre = []
    for n in range(len(limbs)):
        tmp_lst = []
        for p in range(2): tmp_lst.append(jnt_pre[limbs[n][p]]-jnt_pre[limbs[n//2*2][1]])
        lcl_jnt_pre.append(np.array(tmp_lst))
    #torch.stack(lcl_grp_pre, dim=0)
    lcl_jnt_pre = np.array(lcl_jnt_pre) # [8, 2, 3]
    ax1.plot([-lcl_jnt_pre[0,0,0], -lcl_jnt_pre[0,1,0]], [lcl_jnt_pre[0,0,1], lcl_jnt_pre[0,1,1]], [lcl_jnt_pre[0,0,2], lcl_jnt_pre[0,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax1.plot([-lcl_jnt_pre[1,0,0], -lcl_jnt_pre[1,1,0]], [lcl_jnt_pre[1,0,1], lcl_jnt_pre[1,1,1]], [lcl_jnt_pre[1,0,2], lcl_jnt_pre[1,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax2.plot([-lcl_jnt_pre[2,0,0], -lcl_jnt_pre[2,1,0]], [lcl_jnt_pre[2,0,1], lcl_jnt_pre[2,1,1]], [lcl_jnt_pre[2,0,2], lcl_jnt_pre[2,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax2.plot([-lcl_jnt_pre[3,0,0], -lcl_jnt_pre[3,1,0]], [lcl_jnt_pre[3,0,1], lcl_jnt_pre[3,1,1]], [lcl_jnt_pre[3,0,2], lcl_jnt_pre[3,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax3.plot([-lcl_jnt_pre[4,0,0], -lcl_jnt_pre[4,1,0]], [lcl_jnt_pre[4,0,1], lcl_jnt_pre[4,1,1]], [lcl_jnt_pre[4,0,2], lcl_jnt_pre[4,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax3.plot([-lcl_jnt_pre[5,0,0], -lcl_jnt_pre[5,1,0]], [lcl_jnt_pre[5,0,1], lcl_jnt_pre[5,1,1]], [lcl_jnt_pre[5,0,2], lcl_jnt_pre[5,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax4.plot([-lcl_jnt_pre[6,0,0], -lcl_jnt_pre[6,1,0]], [lcl_jnt_pre[6,0,1], lcl_jnt_pre[6,1,1]], [lcl_jnt_pre[6,0,2], lcl_jnt_pre[6,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)
    ax4.plot([-lcl_jnt_pre[7,0,0], -lcl_jnt_pre[7,1,0]], [lcl_jnt_pre[7,0,1], lcl_jnt_pre[7,1,1]], [lcl_jnt_pre[7,0,2], lcl_jnt_pre[7,1,2]], color="gray", marker='o', linewidth=1, alpha=0.4)

    # 隣接する関節から見た "ターゲットフレーム" の骨
    lcl_jnt_tgt = []
    for n in range(len(limbs)):
        tmp_lst = []
        for p in range(2): tmp_lst.append(jnt_tgt[limbs[n][p]]-jnt_pre[limbs[n//2*2][1]])
        lcl_jnt_tgt.append(np.array(tmp_lst))
    #torch.stack(lcl_grp_pre, dim=0)
    lcl_jnt_tgt = np.array(lcl_jnt_tgt) # [8, 2, 3]
    ax1.plot([-lcl_jnt_tgt[0,0,0], -lcl_jnt_tgt[0,1,0]], [lcl_jnt_tgt[0,0,1], lcl_jnt_tgt[0,1,1]], [lcl_jnt_tgt[0,0,2], lcl_jnt_tgt[0,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax1.plot([-lcl_jnt_tgt[1,0,0], -lcl_jnt_tgt[1,1,0]], [lcl_jnt_tgt[1,0,1], lcl_jnt_tgt[1,1,1]], [lcl_jnt_tgt[1,0,2], lcl_jnt_tgt[1,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax2.plot([-lcl_jnt_tgt[2,0,0], -lcl_jnt_tgt[2,1,0]], [lcl_jnt_tgt[2,0,1], lcl_jnt_tgt[2,1,1]], [lcl_jnt_tgt[2,0,2], lcl_jnt_tgt[2,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax2.plot([-lcl_jnt_tgt[3,0,0], -lcl_jnt_tgt[3,1,0]], [lcl_jnt_tgt[3,0,1], lcl_jnt_tgt[3,1,1]], [lcl_jnt_tgt[3,0,2], lcl_jnt_tgt[3,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax3.plot([-lcl_jnt_tgt[4,0,0], -lcl_jnt_tgt[4,1,0]], [lcl_jnt_tgt[4,0,1], lcl_jnt_tgt[4,1,1]], [lcl_jnt_tgt[4,0,2], lcl_jnt_tgt[4,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax3.plot([-lcl_jnt_tgt[5,0,0], -lcl_jnt_tgt[5,1,0]], [lcl_jnt_tgt[5,0,1], lcl_jnt_tgt[5,1,1]], [lcl_jnt_tgt[5,0,2], lcl_jnt_tgt[5,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax4.plot([-lcl_jnt_tgt[6,0,0], -lcl_jnt_tgt[6,1,0]], [lcl_jnt_tgt[6,0,1], lcl_jnt_tgt[6,1,1]], [lcl_jnt_tgt[6,0,2], lcl_jnt_tgt[6,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    ax4.plot([-lcl_jnt_tgt[7,0,0], -lcl_jnt_tgt[7,1,0]], [lcl_jnt_tgt[7,0,1], lcl_jnt_tgt[7,1,1]], [lcl_jnt_tgt[7,0,2], lcl_jnt_tgt[7,1,2]], color="red", marker='o', linewidth=1, alpha=0.4)
    

    lcl_ctr_tgt = torch.from_numpy(lcl_ctr_tgt.astype(np.float32)).clone()
    lcl_jnt_pre = torch.from_numpy(lcl_jnt_pre.astype(np.float32)).clone()
    lcl_jnt_pre_rot = []
    for n in range(len(limbs)):
        #rotation_matrix = find_best_rotation_for_single_point(lcl_ctr_tgt[n], lcl_jnt_pre[n,0])
        rotation_matrix = find_best_rotation_for_single_point(lcl_jnt_pre[n,0], lcl_ctr_tgt[n])
        rot_pt = torch.matmul(lcl_jnt_pre[n,0], rotation_matrix)
        rot_pt = rot_pt.to('cpu').detach().numpy().copy()
        lcl_jnt_pre_rot.append(rot_pt)
    lcl_jnt_pre_rot = np.array(lcl_jnt_pre_rot)
    
    ax1.plot([-lcl_jnt_pre_rot[0,0], -lcl_jnt_pre[0,1,0]], [lcl_jnt_pre_rot[0,1], lcl_jnt_pre[0,1,1]], [lcl_jnt_pre_rot[0,2], lcl_jnt_pre[0,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax1.plot([-lcl_jnt_pre_rot[1,0], -lcl_jnt_pre[1,1,0]], [lcl_jnt_pre_rot[1,1], lcl_jnt_pre[1,1,1]], [lcl_jnt_pre_rot[1,2], lcl_jnt_pre[1,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax2.plot([-lcl_jnt_pre_rot[2,0], -lcl_jnt_pre[2,1,0]], [lcl_jnt_pre_rot[2,1], lcl_jnt_pre[2,1,1]], [lcl_jnt_pre_rot[2,2], lcl_jnt_pre[2,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax2.plot([-lcl_jnt_pre_rot[3,0], -lcl_jnt_pre[3,1,0]], [lcl_jnt_pre_rot[3,1], lcl_jnt_pre[3,1,1]], [lcl_jnt_pre_rot[3,2], lcl_jnt_pre[3,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax3.plot([-lcl_jnt_pre_rot[4,0], -lcl_jnt_pre[4,1,0]], [lcl_jnt_pre_rot[4,1], lcl_jnt_pre[4,1,1]], [lcl_jnt_pre_rot[4,2], lcl_jnt_pre[4,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax3.plot([-lcl_jnt_pre_rot[5,0], -lcl_jnt_pre[5,1,0]], [lcl_jnt_pre_rot[5,1], lcl_jnt_pre[5,1,1]], [lcl_jnt_pre_rot[5,2], lcl_jnt_pre[5,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax4.plot([-lcl_jnt_pre_rot[6,0], -lcl_jnt_pre[6,1,0]], [lcl_jnt_pre_rot[6,1], lcl_jnt_pre[6,1,1]], [lcl_jnt_pre_rot[6,2], lcl_jnt_pre[6,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)
    ax4.plot([-lcl_jnt_pre_rot[7,0], -lcl_jnt_pre[7,1,0]], [lcl_jnt_pre_rot[7,1], lcl_jnt_pre[7,1,1]], [lcl_jnt_pre_rot[7,2], lcl_jnt_pre[7,1,2]], color="green", marker='o', linewidth=2, alpha=0.6)

    ax1.set_title(limbs[0][3]+"-"+limbs[1][3])
    ax1.set_title(limbs[2][3]+"-"+limbs[3][3])
    ax1.set_title(limbs[4][3]+"-"+limbs[5][3])
    ax1.set_title(limbs[6][3]+"-"+limbs[7][3])

    ax0.set_aspect('equal')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')

    ax0.view_init(elev=-90, azim=90)
    ax1.view_init(elev=-90, azim=90)
    ax2.view_init(elev=-90, azim=90)
    ax3.view_init(elev=-90, azim=90)
    ax4.view_init(elev=-90, azim=90)

    return fig


def plot_point_cloud_and_shell(point_cloud=None, shell_points=None, radius=0.2,
                               jnt_pre=None, jnt_pre_width=1, jnt_pre_color="black", jnt_pre_alpha=0.4,
                               jnt_tgt=None, jnt_tgt_width=5, jnt_tgt_color=None, jnt_tgt_alpha=1.0, 
                               jnt_clb=None, jnt_clb_width=5, jnt_clb_color="black", jnt_clb_alpha=1.0):

    # create fig
    fig = plt.figure( figsize=(12,12))
    ax = fig.add_subplot(projection='3d')

    if point_cloud is not None:
        if torch.is_tensor(point_cloud): point_cloud  = point_cloud.to('cpu').detach().numpy().copy()
        ax.scatter(-point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s = 1, c = "gray", alpha=0.4)
    
    if shell_points is not None:
        #print("shell_points cluster: ", len(shell_points))
        #for i, shell_point in enumerate(shell_points):
        #    if torch.is_tensor(shell_point): shell_point = shell_point.to('cpu').detach().numpy().copy()
        #    ax.scatter(-shell_point[:,0], shell_point[:,1], shell_point[:,2], s = 14, c =cm.tab10(i))
        if torch.is_tensor(shell_points): shell_points = shell_points.to('cpu').detach().numpy().copy()
        ax.scatter(-shell_points[:,0], shell_points[:,1], shell_points[:,2], s = 30, c ="red")

    if jnt_pre is not None: 
        if torch.is_tensor(jnt_pre): jnt_pre = jnt_pre.to('cpu').detach().numpy().copy()
        #draw_sphere(ax=ax, r=radius, centers=jnt_pre)
        #draw_circle(ax=ax, r=radius, centers=jnt_pre)
        draw_limbs(ax=ax, joints=jnt_pre, linewidth=jnt_pre_width, color=jnt_pre_color, alpha=jnt_pre_alpha)

    if jnt_tgt is not None: 
        if torch.is_tensor(jnt_tgt): jnt_tgt = jnt_tgt.to('cpu').detach().numpy().copy()
        draw_limbs(ax=ax, joints=jnt_tgt, linewidth=jnt_tgt_width, color=jnt_tgt_color, alpha=jnt_tgt_alpha)

    if jnt_clb is not None: 
        if torch.is_tensor(jnt_clb): jnt_clb = jnt_clb.to('cpu').detach().numpy().copy()
        draw_limbs(ax=ax, joints=jnt_clb, linewidth=jnt_clb_width, color=jnt_clb_color, alpha=jnt_clb_alpha)

    
    # set gridsize
    ax.set_aspect('equal')

    # set viewpoint
    ax.view_init(elev=-90, azim=90)

    plt.grid(False)
    ax.axis("off")

    return fig


def find_best_rotation_batch(A_batch, B_batch):
    """
    A_batch: Bx1x3のテンソル（Bはバッチサイズ）
    B_batch: BxNx3のテンソル（Bはバッチサイズ、Nは点群の点数）
    """
    # 点群Bの重心を計算
    B_center = B_batch.mean(dim=1, keepdim=True)

    # AとBの重心を原点に揃える
    A_centered = A_batch - A_batch
    B_centered = B_center - A_batch

    # 共分散行列を計算
    H_batch = torch.matmul(B_centered.transpose(1, 2), A_centered)

    # 特異値分解を行う
    U, S, Vt = torch.linalg.svd(H_batch)

    # 回転行列Rを計算
    R_batch = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # 回転行列の行列式が負の場合、反射を防ぐために調整
    det_R = torch.linalg.det(R_batch)
    Vt[det_R < 0, :, -1] *= -1
    R_batch = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    return R_batch


def find_best_rotation_for_single_point(A, B):

    """
    Aは1x3のテンソルで、BはNx3のテンソルです。
    """
    # 点群Bの重心を計算
    #B_center = B.mean(dim=0)

    # AとBの重心を原点に揃える
    #A_centered = A - A
    #B_centered = B_center - A

    # 共分散行列を計算
    H = torch.outer(B, A)

    # 特異値分解を行う
    U, S, Vt = torch.linalg.svd(H)

    # 回転行列Rを計算
    R = torch.matmul(Vt.T, U.T)

    # 回転行列の行列式が負の場合、反射を防ぐために調整
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(Vt.T, U.T)

    return R


def plot_sequence(point_clouds=None, joints=None):
    """
    Plots a point cloud and joint coordinates on a 3D scatter plot.

    Args:
        point_cloud (torch): 

    Returns:
        figure (plt.figure)
    """
    if point_clouds is not None: num = len(point_clouds)
    elif joints is not None: num = len(joints)
    
    # 点群データの描画
    fig = plt.figure(figsize=(5 * num, 5))

    if num == 1: axes = [axes]  # 1つのプロットの場合はaxesが配列でなくAxesオブジェクトになるため

    if point_clouds is not None:
        if torch.is_tensor(point_clouds): point_clouds  = point_clouds.to('cpu').detach().numpy().copy()
        
    if joints is not None: 
        if torch.is_tensor(joints): joints = joints.to('cpu').detach().numpy().copy()
        

    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1, projection='3d')
        if point_clouds is not None: ax.scatter(-point_clouds[i,:,0], point_clouds[i,:,1], point_clouds[i,:,2], s = 1, c = "gray")
        if joints is not None: draw_limbs(ax, joints[i], linewidth=5)
        #draw_sphere(ax=ax, r=0.1, centers=label_joints)
        #draw_circle(ax=ax, r=0.1, centers=label_joints)

        # set gridsize
        ax.set_aspect('equal')

        # set viewpoint
        ax.view_init(elev=-90, azim=90)

        #plt.grid(False)
        #ax.axis("off")

    return fig