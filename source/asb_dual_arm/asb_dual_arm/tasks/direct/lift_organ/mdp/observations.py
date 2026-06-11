# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers import ObservationCfg
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, matrix_from_quat, quat_from_matrix, make_pose, unmake_pose
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def reset_object_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
):
    asset: RigidObject = env.scene[asset_cfg.name]
    
    default_root_state = asset.data.default_root_state[env_ids].clone()
    
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(torch.zeros_like(default_root_state[:, 7:]), env_ids)


def target_pose_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names='psm_tool_tip_Link'),
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[asset_cfg.name]
    # robot_sensor = env.scene["ee_1_frame"]

    # current_pos = robot_sensor.data.target_pos_w[..., 0, :] - env.scene.env_origins       # [B, 7]
    # current_ori = robot_sensor.data.target_quat_w[..., 0, :]
    current_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]] - robot.data.root_pos_w
    current_ori = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    # eef_roll, eef_pitch, eef_yaw = euler_xyz_from_quat(eef_ori)

    desired_pose = env.command_manager.get_command(command_name) # [B, 7]
    desired_pos = desired_pose[:, :3]
    desired_ori = desired_pose[:, 3:7]
    # target_roll, target_pitch, target_yaw = euler_xyz_from_quat(target_pose[:, 3:7])

    delta_pos, delta_ori = compute_pose_error(current_pos, current_ori, desired_pos, desired_ori)
    return torch.cat([delta_pos, delta_ori], dim=-1)


def ee_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names='psm_tool_tip_Link')
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[asset_cfg.name]
    # ee_frame_sensor = env.scene["ee_1_frame"] # .body_name = "psm_tool_tip_Link"
    # pos = ee_frame_sensor.data.target_pos_w[..., 0, :] - env.scene.env_origins
    # quat = ee_frame_sensor.data.target_quat_w[..., 0, :]
    current_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]] - robot.data.root_pos_w
    current_quat = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return torch.cat([current_pos, current_quat], dim=-1)


def jaw_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The gripper jaw status (open/close)."""
    asset: Articulation = env.scene[robot_cfg.name]
    joint_positions = asset.data.joint_pos[:, robot_cfg.joint_ids]  # Assuming last two joints are grippers
    # jaw_pos = (joint_positions[:, -1] - joint_positions[:, -2]) / 2
    jaw_pos = joint_positions[:, -2:]  # torch.cat((finger_joint_1, finger_joint_2), dim=1)
    return jaw_pos


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    grasp_offset: tuple,
    diff_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    object_quat = object.data.root_quat_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    end_effector_quat = ee_frame.data.target_quat_w[:, 0, :]
    
    # obtain the grasp pose
    offset_pos = torch.tensor(grasp_offset[:3], device=env.unwrapped.device)
    offset_quat = torch.tensor(grasp_offset[3:], device=env.unwrapped.device)
    wTo = make_pose(object_pos, matrix_from_quat(object_quat))
    oTg = make_pose(offset_pos, matrix_from_quat(offset_quat))
    wTg = wTo @ oTg
    grasp_pos, grasp_orientation = unmake_pose(wTg)
    
    pose_diff = torch.linalg.vector_norm(grasp_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    + torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )

    return grasped
