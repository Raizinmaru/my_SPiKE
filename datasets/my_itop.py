"""
Module for loading the ITOP dataset.
"""

import os
import numpy as np
import h5py
import torch
import tqdm
from torch.utils.data import Dataset
from augmentations.my_aug_pipeline import AugPipeline
from const import skeleton_joints

#Add
#import random
#random.seed(0)  # シード値を設定

class ITOP(Dataset):
    """ITOP dataset class."""

    def __init__(
        self,
        root,
        frames_per_clip=16,
        num_points=2048,
        train=True,
        use_valid_only=False,
        aug_list=None,
        target_frame="middle",
    ):
        super().__init__()

        self.target_frame = target_frame
        self.frames_per_clip = frames_per_clip
        self.num_points = num_points
        self.train = train
        self.root = root
        self.num_coord_joints = skeleton_joints.NUM_COORD_JOINTS

        self._load_data(use_valid_only)

        if aug_list is not None:
            self.aug_pipeline = AugPipeline()
            self.aug_pipeline.create_pipeline(aug_list)
        else:
            self.aug_pipeline = None

    def _get_valid_joints(self, use_valid_only, joints_dict, point_clouds_dict):
        """Cumbersome but necessary logic to create clips of only valid joints and their corresponding frames."""

        valid_joints_dict = {}
        list_joints_items = list(joints_dict.items())

        # Process joints based on if we are inputting only past timestamps (target_frame == 'last')
        # or past and future timestamps (target_frame == 'middle')
        if self.target_frame == "last":
            for identifier, (joints, is_valid) in joints_dict.items():
                # If we are only using valid joints, check if joints are valid and identifier is greater than or equal to frames_per_clip
                # (so we have enough frames to add)
                if (use_valid_only and is_valid and int(identifier[-5:]) >= self.frames_per_clip):
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3]
                            + str(
                                int(identifier[-5:]) - self.frames_per_clip + 1 + i
                            ).zfill(5),
                            None,
                        )
                        for i in range(self.frames_per_clip)
                    ]
                    #valid_joints_dict[identifier] = (joints, frames)
                    #'''~~~~~ Add ~~~~~
                    joints_list = [
                        joints_dict.get(
                            identifier[:3]
                            + str(
                                int(identifier[-5:]) - self.frames_per_clip + 1 + i
                            ).zfill(5),
                            None,
                        )[0]
                        for i in range(self.frames_per_clip)
                    ]
                    valid_joints_dict[identifier] = (joints_list, frames)
                    #~~~~~~~~~~~~~~~'''
                # If not using valid_only, consider all joints
                elif (not use_valid_only and int(identifier[-5:]) >= self.frames_per_clip):
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3]
                            + str(
                                int(identifier[-5:]) - self.frames_per_clip + 1 + i
                            ).zfill(5),
                            None,
                        )
                        for i in range(self.frames_per_clip)
                    ]
                    valid_joints_dict[identifier] = (joints, frames)
        # If we are considering past and future frames, we need to ensure that we have enough frames before and after,
        # and that they belong to the same person (see ITOP naming convention for more details)
        elif self.target_frame == "middle":
            for i, (identifier, (joints, is_valid)) in enumerate(list_joints_items):
                # Get the identifier of the next frames_per_clip // 2 (i + frames_per_clip // 2 ) to make sure they belong to the same person
                next_half_frames_per_clip_id_person, _ = (
                    list_joints_items[i + self.frames_per_clip // 2]
                    if i + self.frames_per_clip // 2 < len(list_joints_items)
                    else (None, None)
                )
                # If no next identifier available, we don't consider the current joints because we don't have enough frames
                if next_half_frames_per_clip_id_person is None:
                    continue
                # Check that the frames belong to the same person and that we have enough frames before and after
                if (use_valid_only and is_valid and int(identifier[-5:]) >= self.frames_per_clip // 2
                    and int(identifier[:2]) == int(next_half_frames_per_clip_id_person[:2])
                ):
                    middle_frame_starting_index = (int(identifier[-5:]) - self.frames_per_clip // 2)
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3]
                            + str(
                                middle_frame_starting_index + i
                            ).zfill(5),
                            None,
                        )
                        for i in range(self.frames_per_clip)
                    ]
                    #valid_joints_dict[identifier] = (joints, frames)
                    #'''~~~~~ Add ~~~~~
                    joints_list = [
                        joints_dict.get(
                            identifier[:3]
                            + str(
                                middle_frame_starting_index + i
                            ).zfill(5),
                        )[0]
                        for i in range(self.frames_per_clip)
                    ]
                    valid_joints_dict[identifier] = (joints_list, frames)
                    #~~~~~~~~~~~~~~~'''
                # If not using valid_only, consider all joints (except for validity, same conditions as above)
                elif (
                    not use_valid_only
                    and int(identifier[-5:]) >= self.frames_per_clip // 2
                    and int(identifier[:2])
                    == int(next_half_frames_per_clip_id_person[:2])
                ):
                    middle_frame_starting_index = (
                        int(identifier[-5:]) - self.frames_per_clip // 2
                    )
                    frames = [
                        point_clouds_dict.get(
                            identifier[:3]
                            + str(middle_frame_starting_index + i).zfill(5),
                            None,
                        )
                        for i in range(self.frames_per_clip)
                    ]
                    valid_joints_dict[identifier] = (joints, frames)
        return valid_joints_dict

    def _load_data(self, use_valid_only):
        """Load the data from the dataset."""

        point_clouds_folder = os.path.join(self.root, "train" if self.train else "test")
        labels_file = h5py.File(
            os.path.join(
                self.root, "train_labels.h5" if self.train else "test_labels.h5"
            ),
            "r",
        )
        identifiers = labels_file["id"][:]
        joints = labels_file["real_world_coordinates"][:]
        is_valid_flags = labels_file["is_valid"][:]
        labels_file.close()

        point_cloud_names = sorted(
            os.listdir(point_clouds_folder), key=lambda x: int(x.split(".")[0])
        )
        point_clouds = []

        for pc_name in tqdm.tqdm(
            point_cloud_names,
            f"Loading {'train' if self.train else 'test'} point clouds",
        ):
            point_clouds.append(
                np.load(os.path.join(point_clouds_folder, pc_name))["arr_0"]
            )

        point_clouds_dict = {
            identifier.decode("utf-8"): point_clouds[i]
            for i, identifier in enumerate(identifiers)
        }
        joints_dict = {
            identifier.decode("utf-8"): (joints[i], is_valid_flags[i])
            for i, identifier in enumerate(identifiers)
        }

        self.valid_joints_dict = self._get_valid_joints(use_valid_only, joints_dict, point_clouds_dict)

        self.valid_identifiers = list(self.valid_joints_dict.keys())

        if use_valid_only:
            print(
                f"Using only frames labeled as valid. From the total of {len(point_clouds)} "
                f"{'train' if self.train else 'test'} frames using {len(self.valid_identifiers)} valid joints"
            )

    #Add
    def remove_points_in_region(self, point_clouds, x_range=(-0.7,0.7), y_range=(-0.2,0.2), z_range=(-1,1), region_size=(0.5,0.5,10)):
        """
        指定した範囲内に指定した領域内の点群データを削除する関数。

        Args:
            point_clouds: 点群データのリスト。各要素はshapeが(N, 3)のNumPy配列。
            x_range: x座標の範囲 (min, max)。
            y_range: y座標の範囲 (min, max)。
            z_range: z座標の範囲 (min, max)。
            region_size: 削除する領域のサイズ (x, y, z)。

        Returns:
            点群データのリスト。領域内の点が削除されたデータ。
        """

        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        region_x, region_y, region_z = region_size

        # ランダムに領域の中心座標を生成
        region_x_center = random.uniform(x_min + region_x / 2, x_max - region_x / 2)
        region_y_center = random.uniform(y_min + region_y / 2, y_max - region_y / 2)
        region_z_center = random.uniform(z_min + region_z / 2, z_max - region_z / 2)


        filtered_point_clouds = []
        for point_cloud in point_clouds:
            mask = (
                (point_cloud[:, 0] < region_x_center - region_x / 2) |
                (point_cloud[:, 0] > region_x_center + region_x / 2) |
                (point_cloud[:, 1] < region_y_center - region_y / 2) |
                (point_cloud[:, 1] > region_y_center + region_y / 2) |
                (point_cloud[:, 2] < region_z_center - region_z / 2) |
                (point_cloud[:, 2] > region_z_center + region_z / 2)
            )
            filtered_point_clouds.append(point_cloud[mask])

        return filtered_point_clouds

    def _random_sample_pc(self, p):
        """Randomly sample points from the point cloud."""
        if p.shape[0] > self.num_points:
            r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
        elif p.shape[0] < self.num_points:
            repeat, residue = divmod(self.num_points, p.shape[0])
            r = np.concatenate(
                [np.arange(p.shape[0])] * repeat
                + [np.random.choice(p.shape[0], size=residue, replace=False)],
                axis=0,
            )
        else:
            return p
        return p[r, :]

    def __len__(self):
        return len(self.valid_identifiers)

    def __getitem__(self, idx):
        identifier = self.valid_identifiers[idx]
        # identifier ... str: id (e.g., 05_00463)

        joints, clip = self.valid_joints_dict.get(
            identifier, (None, [None] * self.frames_per_clip)
        )
        #joints ... ndarray: [15,3] -> list[ndarray]: 3*[15,3]
        # clip  ... list[ndarray]: 3*[n,3]

        if joints is None or any(frame is None for frame in clip):
            raise ValueError(f"Invalid joints or frames for identifier {identifier}")
        
        # Add
        #clip = self.remove_points_in_region(clip)
        
        clip = [self._random_sample_pc(p) for p in clip]
        clip = torch.FloatTensor(clip)
        #joints = torch.FloatTensor(joints).view(1, -1, 3)
        joints = torch.FloatTensor(joints)

        '''from utils import my_functions
        fig = my_functions.plot_point_cloud_and_joints_sequence(
            point_cloud=clip, 
            pred_joints=None,
            label_joints=joints
            )
        my_functions.show_fig(fig)
        #'''

        if self.aug_pipeline:
            clip, _, joints = self.aug_pipeline.augment(clip, joints)

        return clip, joints, np.array([tuple(map(int, identifier.split("_")))])