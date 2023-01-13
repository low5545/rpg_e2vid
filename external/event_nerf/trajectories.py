import easydict
import torch
import roma
from . import datasets
from . import tensor_ops


class LinearTrajectory(torch.nn.Module):
    def __init__(self, camera_poses):
        super().__init__()
        assert isinstance(camera_poses, datasets.CameraPose)

        # define camera pose samples & sampling timestamps as a buffer
        self.register_buffer(
            "init_T_wc_position",                                               # (C, 3)
            camera_poses.camera_poses.T_wc_position, persistent=False
        )
        self.register_buffer(
            "init_T_wc_orientation_quat",                                       # (C, 4)
            camera_poses.camera_poses.T_wc_orientation, persistent=False
        )
        self.register_buffer(
            "T_wc_timestamp",                                                   # (C)
            camera_poses.camera_poses.T_wc_timestamp.contiguous(),              # contiguous required for efficient `torch.searchsorted()`
            persistent=False
        )
        self.register_buffer(                                                   # (C-1)
            "bin_width", self.T_wc_timestamp.diff(), persistent=False
        )

        # define camera pose refinement parameters
        self.delta_T_wc_position = torch.nn.parameter.Parameter(                # (C, 3)
            torch.zeros_like(self.init_T_wc_position)
        )
        # TODO: Is it better to parameterize orientation refinement with
        #       rotation vectors or unnormalized quaternions?
        self.delta_T_wc_orientation_rotvec = torch.nn.parameter.Parameter(      # (C, 3) rotation vectors
            torch.zeros(self.init_T_wc_orientation_quat.shape[0], 3)
        )
        
    def forward(self, input_timestamp):
        """
        Args:
            input_timestamp (torch.Tensor):
                Input timestamp(s) of shape (N) / () with dtype in the same
                units as `self.T_wc_timestamp` & contiguous
        Returns:
            input_T_wc_position (torch.Tensor):
                Linearly interpolated position of camera poses associated to
                the input timestamps
            input_T_wc_orientation (torch.Tensor):
                Linearly interpolated orientation of camera poses associated to
                the input timestamps
        """
        assert input_timestamp.dim() <= 1

        # deduce the camera pose bins that input timestamps should fall into
        bin_left = easydict.EasyDict()
        bin_right = easydict.EasyDict()

        bin_right.index = torch.searchsorted(                                   # ([N])
            sorted_sequence=self.T_wc_timestamp, input=input_timestamp
        )
        is_corner_case = (input_timestamp == self.T_wc_timestamp[0])
        bin_left.index = torch.where(
            is_corner_case, bin_right.index, bin_right.index - 1
        )
        assert (
            (bin_left.index >= 0)
            & (bin_right.index < len(self.T_wc_timestamp))
        ).all()

        # linearly interpolate camera positions
        weight = (input_timestamp - self.T_wc_timestamp[bin_left.index]) \
                 / self.bin_width[bin_left.index]                               # ([N])

        bin_left.T_wc_position = (                                              # ([N,] 3)
            self.init_T_wc_position[bin_left.index, :]
            + self.delta_T_wc_position[bin_left.index, :]
        )
        bin_right.T_wc_position = (                                             # ([N,] 3)
            self.init_T_wc_position[bin_right.index, :]
            + self.delta_T_wc_position[bin_right.index, :]
        )
        input_T_wc_position = torch.lerp(                                       # ([N,] 3) / (3)
            input=bin_left.T_wc_position,
            end=bin_right.T_wc_position,
            weight=weight.unsqueeze(dim=-1)                                     # ([N,] 1)
        )

        # apply slerp to interpolate orientation
        bin_left.T_wc_orientation_quat = roma.quat_product(                     # ([N,] 4)
            roma.rotvec_to_unitquat(
                self.delta_T_wc_orientation_rotvec[bin_left.index, :]
            ),
            self.init_T_wc_orientation_quat[bin_left.index, :]
        )
        bin_right.T_wc_orientation_quat = roma.quat_product(                    # ([N,] 4)
            roma.rotvec_to_unitquat(
                self.delta_T_wc_orientation_rotvec[bin_right.index, :]
            ),
            self.init_T_wc_orientation_quat[bin_right.index, :]
        )

        input_T_wc_orientation_quat = tensor_ops.unitquat_slerp(                # ([N,] 4)
            q0=bin_left.T_wc_orientation_quat,
            q1=bin_right.T_wc_orientation_quat,
            steps=weight, shortest_path=True
        )
        input_T_wc_orientation = roma.unitquat_to_rotmat(                       # ([N,] 3, 3)
            input_T_wc_orientation_quat
        )

        return input_T_wc_position, input_T_wc_orientation
