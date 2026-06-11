# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import apply_delta_pose, compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
import numpy as np

if TYPE_CHECKING:
    from .differential_ik_with_soft_rcm_cfg import DifferentialIKWithSoftRCMControllerCfg


@torch.jit.script
def apply_delta_yaw(
    source_pos: torch.Tensor, source_rot: torch.Tensor, des_rot: torch.Tensor, delta_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies delta yaw transformation on source pose.

    The first three elements of `delta_pose` are interpreted as cartesian position displacement.
    The remaining three elements of `delta_pose` are interpreted as orientation displacement
    in the angle-axis format.

    Args:
        source_pos: Position of source frame. Shape is (N, 3).
        source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4)..
        delta_pose: Position and orientation displacements. Shape is (N, 6).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # number of poses given
    num_poses = source_pos.shape[0]
    device = source_pos.device

    # interpret delta_pose[:, 0:3] as target position displacements
    target_pos = source_pos + delta_pose[:, 0:3]
    # interpret delta_pose[:, 3:6] as target rotation displacements
    source_roll, source_pitch, source_yaw = euler_xyz_from_quat(source_rot)
    target_roll = des_rot[:, 0:1]
    target_pitch = des_rot[:, 1:2]
    target_yaw = source_yaw.unsqueeze(dim=1) + delta_pose[:, 5:6]
    target_yaw = torch.fmod(target_yaw + np.pi, 2 * np.pi) - np.pi  # keep in [-pi, pi]
    # convert the euler angles back to quaternion
    target_rot = quat_from_euler_xyz(target_roll[:, 0], target_pitch[:, 0], target_yaw[:, 0])
    return target_pos, target_rot


@torch.jit.script
def apply_correct_ori(
    ee_quat: torch.Tensor, ee_pos: torch.Tensor, rcm_pos: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    """Applies delta pose transformation on source pose.

    The first three elements of `delta_pose` are interpreted as cartesian position displacement.
    The remaining three elements of `delta_pose` are interpreted as orientation displacement
    in the angle-axis format.

    Args:
        source_pos: Position of source frame. Shape is (N, 3).
        source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4)..
        delta_pose: Position and orientation displacements. Shape is (N, 6).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    
    # interpret delta_pose[:, 0:3] as target position displacements
    z_dir = ee_pos - rcm_pos
    z_axis = z_dir / torch.norm(z_dir, dim=1, keepdim=True)
    # convert the quat into rotation matrix
    R_marix = matrix_from_quat(ee_quat)
    y_temp = torch.cross(z_axis, R_marix[:, :, 0], dim=1)
    y_temp = y_temp / torch.norm(y_temp, dim=1, keepdim=True)
    x_axis = torch.cross(y_temp, z_axis, dim=1)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    R_correct = torch.stack((x_axis, y_axis, z_axis), dim=2)
    # convert the rotation matrix back to quat
    ee_quat_correct = quat_from_matrix(R_correct)
    return ee_quat_correct


class DifferentialIKWithSoftRCMController:
    r"""Differential inverse kinematics (IK) controller with soft RCM constraint.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    def __init__(self, cfg: DifferentialIKWithSoftRCMControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.rcm_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        self.ee_rot_des = torch.ones(self.num_envs, 3, device=self._device)
        self.ee_rot_des[:, 0] *= np.deg2rad(180)
        self.ee_rot_des[:, 1] *= 0.0
        self.ee_quat_ref = None  # ecm refer orientation
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        elif self.cfg.command_type == "yaw" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, 0, 0, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    # def set_command(
    #     self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    # ):
    #     """Set target end-effector pose command.

    #     Based on the configured command type and relative mode, the method computes the desired end-effector pose.
    #     It is up to the user to ensure that the command is given in the correct frame. The method only
    #     applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

    #     Args:
    #         command: The input command in shape (N, 3) or (N, 6) or (N, 7).
    #         ee_pos: The current end-effector position in shape (N, 3).
    #             This is only needed if the command type is ``position_rel`` or ``pose_rel``.
    #         ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
    #             This is only needed if the command type is ``position_*`` or ``pose_rel``.

    #     Raises:
    #         ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
    #         ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
    #         ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
    #     """
    #     # store command
    #     self._command[:] = command
    #     # compute the desired end-effector pose
    #     if self.cfg.command_type == "position":
    #         # we need end-effector orientation even though we are in position mode
    #         # this is only needed for display purposes
    #         if ee_quat is None:
    #             raise ValueError("End-effector orientation can not be None for `position_*` command type!")
    #         # compute targets
    #         if self.cfg.use_relative_mode:
    #             if ee_pos is None:
    #                 raise ValueError("End-effector position can not be None for `position_rel` command type!")
    #             self.ee_pos_des[:] = ee_pos + self._command
    #             self.ee_quat_des[:] = ee_quat
    #         else:
    #             self.ee_pos_des[:] = self._command
    #             self.ee_quat_des[:] = ee_quat
    #     else:
    #         # compute targets
    #         if self.cfg.use_relative_mode:
    #             if ee_pos is None or ee_quat is None:
    #                 raise ValueError(
    #                     "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
    #                 )
    #             if self.cfg.command_type == "yaw":
    #                 # only position and yaw are given
    #                 self._command = torch.cat([command[:, 0:3], command[:, 3:4]], dim=1)
    #                 self.ee_pos_des, self.ee_quat_des = apply_delta_yaw(ee_pos, ee_quat, self.ee_rot_des, self._command)
    #             else:
    #                 self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
    #         else:
    #             self.ee_pos_des = self._command[:, 0:3]
    #             self.ee_quat_des = self._command[:, 3:7]
    
    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        # print('command:', command[0, :])
        if self.ee_quat_ref is None:
            # compute the desired end-effector pose
            if self.cfg.command_type == "position":
                # we need end-effector orientation even though we are in position mode
                # this is only needed for display purposes
                if ee_quat is None:
                    raise ValueError("End-effector orientation can not be None for `position_*` command type!")
                # compute targets
                if self.cfg.use_relative_mode:
                    if ee_pos is None:
                        raise ValueError("End-effector position can not be None for `position_rel` command type!")
                    delta_pos = self._command
                    delta_pos_norm = torch.norm(delta_pos, dim=1, keepdim=True)
                    delta_pos_normalized = delta_pos / (delta_pos_norm + 1e-8)
                    clipped_delta_pos = torch.clamp(delta_pos_norm, max=1e-2) * delta_pos_normalized  # max step@1cm/s
                    self.ee_pos_des[:] = ee_pos + clipped_delta_pos
                    self.ee_quat_des[:] = ee_quat
                else:
                    self.ee_pos_des[:] = self._command
                    self.ee_quat_des[:] = ee_quat
            else:
                # compute targets
                if self.cfg.use_relative_mode:
                    if ee_pos is None or ee_quat is None:
                        raise ValueError(
                            "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                        )
                    delta_pos = command[:, 0:3]
                    delta_pos_norm = torch.norm(delta_pos, dim=1, keepdim=True)
                    delta_pos_normalized = delta_pos / (delta_pos_norm + 1e-8)
                    clipped_delta_pos = torch.clamp(delta_pos_norm, max=1e-2) * delta_pos_normalized  # max step@1cm/s
                    
                    if self.cfg.command_type == "yaw":
                        delta_yaw = command[:, 5:6]  # (N, 3) or (N, 1)
                        clipped_delta_yaw = torch.clamp(delta_yaw, max=0.05, min=-0.05)  # max step@0.5rad/s.
                        # only position and yaw are given
                        self._command = torch.cat([clipped_delta_pos, command[:, 3:5], clipped_delta_yaw], dim=1)
                        self.ee_pos_des, self.ee_quat_des = apply_delta_yaw(ee_pos, ee_quat, self.ee_rot_des, self._command)
                    else:
                        delta_ori = command[:, 3:]  # (N, 3) or (N, 1)
                        delta_ori_norm = torch.norm(delta_ori, dim=1, keepdim=True)
                        delta_ori_normalized = delta_ori / (delta_ori_norm + 1e-8)
                        clipped_delta_ori = torch.clamp(delta_ori_norm, max=5e-2) * delta_ori_normalized  # max step@0.05rad/s
                        self._command = torch.cat([clipped_delta_pos, clipped_delta_ori], dim=1)
                        self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
                else:
                    self.ee_pos_des = self._command[:, 0:3]
                    self.ee_quat_des = self._command[:, 3:7]
        else:
            # ecm control to keep the initial orientation
            # compute the desired end-effector pose
            if self.cfg.command_type == "position":
                # we need end-effector orientation even though we are in position mode
                # compute targets
                if self.cfg.use_relative_mode:
                    if ee_pos is None:
                        raise ValueError("End-effector position can not be None for `position_rel` command type!")
                    delta_pos = self._command
                    delta_pos_norm = torch.norm(delta_pos, dim=1, keepdim=True)
                    delta_pos_normalized = delta_pos / (delta_pos_norm + 1e-8)
                    clipped_delta_pos = torch.clamp(delta_pos_norm, max=1e-2) * delta_pos_normalized  # max step@1cm/s
                    self.ee_pos_des[:] = ee_pos + clipped_delta_pos
                    self.ee_quat_des = apply_correct_ori(self.ee_quat_ref, self.ee_pos_des, self.rcm_pos_des)
                else:
                    self.ee_pos_des = self._command
                    self.ee_quat_des = apply_correct_ori(self.ee_quat_ref, self.ee_pos_des, self.rcm_pos_des)
            else:
                # compute targets
                if self.cfg.use_relative_mode:
                    if ee_pos is None or ee_quat is None:
                        raise ValueError(
                            "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                        )
                    delta_pos = command[:, 0:3]
                    delta_pos_norm = torch.norm(delta_pos, dim=1, keepdim=True)
                    delta_pos_normalized = delta_pos / (delta_pos_norm + 1e-8)
                    clipped_delta_pos = torch.clamp(delta_pos_norm, max=5e-3) * delta_pos_normalized  # max step@1cm/s
                    delta_ori = command[:, 3:]  # (N, 3) or (N, 1)
                    delta_ori_norm = torch.norm(delta_ori, dim=1, keepdim=True)
                    delta_ori_normalized = delta_ori / (delta_ori_norm + 1e-8)
                    clipped_delta_ori = torch.clamp(delta_ori_norm, max=5e-2) * delta_ori_normalized  # max step@0.5rad/s
                    self._command = torch.cat([clipped_delta_pos, clipped_delta_ori], dim=1)
                    self.ee_pos_des, ee_quat_temp = apply_delta_pose(ee_pos, ee_quat, self._command)
                    self.ee_quat_des = apply_correct_ori(self.ee_quat_ref, self.ee_pos_des, self.rcm_pos_des)
                else:
                    self.ee_pos_des = self._command[:, 0:3]
                    self.ee_quat_des = apply_correct_ori(self.ee_quat_ref, self.ee_pos_des, self.rcm_pos_des)
        
    def set_rcm_target(
        self, rcm_pos: torch.Tensor | None = None, rcm_beta: torch.Tensor | None = None
    ):
        """Set the target RCM position by setting rcm position or its related coefficient.
        Args:
            rcm_pos: The target RCM position in shape (N, 3).
            rcm_beta: The beta of the RCM constraint (0: f1_pos, 1: f2_pos).
        """
        self.rcm_pos_des = rcm_pos
        self.rcm_beta = rcm_beta

    def set_ori_target(self, ee_quat: torch.Tensor):
        """Set target end-effector orientation command.

        Args:
            ee_quat: The desired end-effector orientation (w, x, y, z) in shape (N, 4).
        """
        self.ee_quat_ref = ee_quat

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, rcm_pos: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor, rcm_beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the target joint positions that will yield the desired end effector pose while maintaining RCM constraint.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        rcm_error = self.rcm_pos_des - rcm_pos
        # compute the delta in joint-space
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            merged_error = torch.cat((position_error, rcm_error), dim=1)
            jacobian_pos = jacobian[:, 0:3]
            jacobian_rcm = jacobian[:, 6:9]
            jacobian_tar = torch.concatenate((jacobian_pos, jacobian_rcm), dim=1)  # combination of jacobian_pos and jacobian_rcm
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=merged_error, jacobian=jacobian_tar)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error, rcm_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        # return the desired joint positions
        return joint_pos + delta_joint_pos[:, :-1], rcm_beta + delta_joint_pos[:, -1].unsqueeze(-1)

    """
    Helper functions.
    """

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            rank = S_inv.shape[-1]
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :rank] 
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_T @ torch.pinverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_joint_pos
