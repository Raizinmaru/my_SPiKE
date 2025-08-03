"""
Module for preprocessing the ITOP dataset to isolate the points belonging to the human body.
"""

import argparse
import os
import numpy as np
import h5py
import tqdm

try:
    import open3d as o3d
except ImportError as import_error:
    raise ImportError("Open3D is not installed. Please install it to use this script.") from import_error


#''' add
import math
import matplotlib.pyplot as plt

# difine joint <class 'dict'>: 0: 'Head' ~ 14: 'L_Foot'
joint_id_to_name = {
    0: 'Head',
    1: 'Neck',
    2: 'R_Shoulder',
    3: 'L_Shoulder',
    4: 'R_Elbow',
    5: 'L_Elbow',
    6: 'R_Hand',
    7: 'L_Hand',
    8: 'Torso',
    9: 'R_Hip',
    10: 'L_Hip',
    11: 'R_Knee',
    12: 'L_Knee',
    13: 'R_Foot',
    14: 'L_Foot',
}

# define joint name <class 'list'>:
partList = [joint_id_to_name[k] for k in joint_id_to_name.keys()]

# difine joint <class 'dict'>: 'Head': 0 ~ 'L_Foot': 14
jointMap = {}
i = 0
for joint in partList:
    jointMap[joint] = i
    i+=1

# define limbs by jointMap
limbList  = [[jointMap["Neck"],        jointMap["Head"]],
             [jointMap["Neck"],        jointMap["Torso"]],
             [jointMap["Torso"],       jointMap["L_Hip"]],
             [jointMap["L_Hip"],       jointMap["L_Knee"]],
             [jointMap["L_Knee"],      jointMap["L_Foot"]],
             [jointMap["Neck"],        jointMap["L_Shoulder"]],
             [jointMap["L_Shoulder"],  jointMap["L_Elbow"]],
             [jointMap["L_Elbow"],     jointMap["L_Hand"]],
             [jointMap["Torso"],       jointMap["R_Hip"]],
             [jointMap["R_Hip"],       jointMap["R_Knee"]],
             [jointMap["R_Knee"],      jointMap["R_Foot"]],
             [jointMap["Neck"],        jointMap["R_Shoulder"]],
             [jointMap["R_Shoulder"],  jointMap["R_Elbow"]],
             [jointMap["R_Elbow"],     jointMap["R_Hand"]]]

# define limbs colors
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170],
          [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255],
          [255, 0, 255], [255, 0, 170], [255, 0, 85]]
for n in range(len(colors)): colors[n] = [value/255.0 for value in colors[n]]

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

def draw_fig(fig, data, out_path):
    ax = fig.add_subplot(2,2,1,projection='3d')
    ax.scatter(-data[:,0], data[:,1], data[:,2], s = 1, c = "gray", alpha=0.2)
    ax.set_aspect('equal')
    ax.view_init(elev=-90, azim=90)

    plt.savefig(out_path)
    fig.clear()
    plt.clf()  
    plt.close()
#'''

# Constant defining the minimum number of points required for a cluster
CLUSTER_POINTS = 10

def remove_floor(point_cloud, num_bins=10, bins=100):
    """
    Remove the floor from the point cloud by analyzing point density along the vertical (Y) axis.

    Parameters:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        num_bins (int): Number of bins to analyze for potential floor points.
        bins (int): Total number of bins for the histogram.

    Returns:
        np.ndarray: Point cloud with the floor removed.
    """
    try:
        y_coords, y_bins = np.histogram(point_cloud[:, 1], bins=bins)
        first_n = y_coords[:num_bins]

        # Check if the first bins have significantly higher density
        if np.sum(first_n) > num_bins * np.mean(y_coords):
            max_idx = np.argmax(first_n)
            cut_idx = np.argmax(first_n[max_idx:] < np.mean(first_n[max_idx:]))
            y_limit = y_bins[max_idx + cut_idx]
            # Keep points above the identified floor threshold
            return point_cloud[point_cloud[:, 1] >= y_limit]
        return point_cloud
    except (ValueError, IndexError) as floor_error:
        print(f"Error in remove_floor: {floor_error}")
        return point_cloud

def get_bounding_box_3d(points, offset=0.3):
    """
    Calculate the bounding box of a 3D point cloud with an optional offset.

    Parameters:
        points (np.ndarray): The input point cloud with shape (N, 3).
        offset (float): Extra margin to add to the bounding box.

    Returns:
        tuple: Bounding box limits (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    try:
        x_min, x_max = np.min(points[:, 0]) - offset, np.max(points[:, 0]) + offset
        y_min, y_max = np.min(points[:, 1]) - offset, np.max(points[:, 1]) + 2 * offset
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2]) + offset
        return x_min, x_max, y_min, y_max, 1.7, z_max
    except (ValueError, IndexError) as bbox_error:
        print(f"Error in get_bounding_box_3d: {bbox_error}")
        return 0, 0, 0, 0, 0, 0

def preprocess_point_cloud(point_cloud, num_points=0):
    """
    Preprocess the point cloud by filtering, removing the floor, and clustering.

    Parameters:
        point_cloud (np.ndarray): The input point cloud with shape (N, 3).
        num_points (int): If greater than 0, randomly samples this number of points from the largest cluster.

    Returns:
        np.ndarray: Processed point cloud containing the largest cluster and nearby clusters.
    """
    try:
        x_min, x_max, y_min, y_max, z_min, z_max = -0.85, 0.95, -1.7, 1.0, 1.85, 3.7

        point_cloud = point_cloud[
            (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
            (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
            (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
        ]

        point_cloud = remove_floor(point_cloud)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Perform DBSCAN clustering
        dbscan_clusters = np.array(pcd.cluster_dbscan(eps=0.15, min_points=CLUSTER_POINTS, print_progress=False))

        num_clusters = dbscan_clusters.max() + 1

        if num_clusters == 0:
            print("No clusters found! Saving empty array!")
            return np.empty(0)

        clusters = [point_cloud[dbscan_clusters == cluster_id] for cluster_id in range(num_clusters)]
        largest_cluster = max(clusters, key=len)  
        clusters.remove(largest_cluster)

        x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box_3d(largest_cluster, offset=0.2)

        # Include additional clusters near the largest cluster
        largest_cluster_with_front = largest_cluster.copy()
        for cluster in clusters:
            if (
                (np.min(cluster[:, 0]) >= x_min and np.max(cluster[:, 0]) <= x_max and
                 np.min(cluster[:, 1]) >= y_min and np.max(cluster[:, 1]) <= y_max and
                 np.min(cluster[:, 2]) >= z_min and np.max(cluster[:, 2]) <= z_max) or
                (np.min(cluster[:, 0]) >= x_min + 0.2 and np.max(cluster[:, 0]) <= x_max - 0.2 and
                 np.max(cluster[:, 1]) <= y_max and np.min(cluster[:, 2]) >= z_min and
                 np.max(cluster[:, 2]) <= z_max)
            ):
                largest_cluster_with_front = np.concatenate((largest_cluster_with_front, cluster))

        # Downsample points if required
        if num_points > 0:
            random_indices = np.random.choice(largest_cluster.shape[0], size=num_points, replace=False)
            point_cloud = largest_cluster[random_indices, :]

        return largest_cluster_with_front
    except (ValueError, IndexError, RuntimeError) as preprocess_error:
        print(f"Error in preprocess_point_cloud: {preprocess_error}")
        return np.empty(0)

def process_as_npz(clouds_path, output_folder):
    """
    Process point clouds from an HDF5 file and save them as .npz files.

    Parameters:
        clouds_path (str): Path to the input HDF5 file containing point clouds.
        output_folder (str): Directory to save the processed .npz files.
    """
    try:
        with h5py.File(clouds_path, 'r') as clouds_file:
            point_clouds = clouds_file['data']
            ids = clouds_file['id']

            for idx, cloud in tqdm.tqdm(enumerate(point_clouds), desc="Preprocessing Point Clouds", total=len(point_clouds)):
                processed_cloud = preprocess_point_cloud(cloud)

                fig = plt.figure(ids[idx].decode('UTF-8'), figsize=(12,12))
                figsave_path = output_folder + "/fig/" + ids[idx].decode('UTF-8') + ".jpg"
                draw_fig(fig, processed_cloud, figsave_path)

                np.savez(os.path.join(output_folder, f"{idx}.npz"), processed_cloud)
    except (OSError, KeyError, ValueError) as process_error:
        print(f"Error in process_as_npz: {process_error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess point clouds from an HDF5 file and save them as .npz files.")
    parser.add_argument('input_path', type=str, help="Path to the input HDF5 file containing point clouds.")
    parser.add_argument('output_path', type=str, help="Directory to save the processed .npz files.")
    
    args = parser.parse_args()

    process_as_npz(args.input_path, args.output_path)
    