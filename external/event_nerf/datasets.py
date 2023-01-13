import os
import easydict
import numpy as np
import torch
from . import tensor_ops


class CameraPose(torch.utils.data.Dataset):
    CAMERA_POSES_FILENAME = "camera_poses.npz"
    CAMERA_POSES_KEYS = set(
        [ "T_wc_position", "T_wc_orientation", "T_wc_timestamp" ]
    )

    def __init__(
        self,
        root_directory,
        permutation_seed,
        downsampling_factor,
        upsampling_factor,
        upsampling_algo
    ):
        super().__init__()
        assert isinstance(downsampling_factor, int) and downsampling_factor > 0
        assert isinstance(upsampling_factor, int) and upsampling_factor > 0

        """
        NOTE:
            Camera poses are downsampled before upsampled.
        """
        self.camera_poses = self.load_camera_poses(root_directory)
        if downsampling_factor > 1:
            self.camera_poses = self.downsample_camera_poses(
                self.camera_poses, downsampling_factor
            )
        if upsampling_factor > 1:
            self.camera_poses = self.upsample_camera_poses(
                self.camera_poses, upsampling_factor, upsampling_algo
            )

        # randomly permute camera poses, if necessary
        if permutation_seed is None:
            return
        perm_indices = tensor_ops.randperm_manual_seed(
            len(self.camera_poses.T_wc_position), permutation_seed
        )
        for key, value in self.camera_poses.items():
            self.camera_poses[key] = value[perm_indices]

    @classmethod
    def load_camera_poses(cls, root_directory):
        camera_poses_path = os.path.join(
            root_directory, cls.CAMERA_POSES_FILENAME
        )
        camera_poses = easydict.EasyDict(np.load(camera_poses_path))
        assert set(camera_poses.keys()) == cls.CAMERA_POSES_KEYS
        
        # cast camera poses components to `torch.Tensor`
        for key, value in camera_poses.items():
            camera_poses[key] = torch.tensor(value)

        return camera_poses

    @staticmethod
    def downsample_camera_poses(camera_poses, downsampling_factor):
        downsampled_camera_poses = easydict.EasyDict(camera_poses)
        for key, value in downsampled_camera_poses.items():
            downsampled_camera_poses[key] = value[::downsampling_factor]
        return downsampled_camera_poses

    @staticmethod
    def upsample_camera_poses(
        camera_poses,
        upsampling_factor,
        upsampling_algo
    ):
        upsampled_camera_poses = easydict.EasyDict(camera_poses)
        if upsampling_algo == "linear":
            # linearly interpolate positions & timestamps
            upsampled_camera_poses.T_wc_position = tensor_ops.lerp_uniform(
                camera_poses.T_wc_position, upsampling_factor
            )
            upsampled_camera_poses.T_wc_timestamp = tensor_ops.lerp_uniform(
                camera_poses.T_wc_timestamp.to(torch.float64),
                upsampling_factor
            )
            upsampled_camera_poses.T_wc_timestamp = (
                upsampled_camera_poses.T_wc_timestamp.to(torch.int64)
            )

            # apply slerp to interpolate orientation
            upsampled_camera_poses.T_wc_orientation = tensor_ops.slerp_uniform(
                camera_poses.T_wc_orientation, upsampling_factor
            ) 
        else:
            raise NotImplementedError

        return upsampled_camera_poses

    def __getitem__(self, index):
        return {
            key: value[index] for key, value in self.camera_poses.items()
        }

    def __len__(self):
        return len(self.camera_poses.T_wc_position)
