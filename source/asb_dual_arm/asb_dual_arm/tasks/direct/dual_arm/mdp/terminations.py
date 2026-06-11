# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dual_arm task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms, matrix_from_quat, quat_from_matrix, make_pose, unmake_pose

from .metrics import update_vessel_connectivity_diagnostics
from .rewards import _get_cd_only_connected_masks, _get_cd_only_connectivity_counter, _get_trisection_cached, _is_only_cd_connected


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "object_pose",
    grasp_offset: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    criterion: str = "pos",
    pos_threshold: float = 0.005,
    ori_threshold: float = 0.02,
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, des_quat_w = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    # distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w
    
    # compute target pos/ori
    # obtain the corresponding grasp pose if grasped
    offset_pos = torch.tensor(grasp_offset[:3], device=env.unwrapped.device)
    offset_quat = torch.tensor(grasp_offset[3:], device=env.unwrapped.device)
    wTo = make_pose(object_pos_w, matrix_from_quat(object_quat_w))
    oTg = make_pose(offset_pos, matrix_from_quat(offset_quat))
    wTg = wTo @ oTg
    grasp_pos, grasp_orientation = unmake_pose(wTg)
    grasp_quat = quat_from_matrix(grasp_orientation)
    
    delta_pos, delta_ori = compute_pose_error(grasp_pos, grasp_quat, des_pos_w, des_quat_w)
    distance = torch.norm(delta_pos, dim=1)
    oridiff = torch.norm(delta_ori, dim=1)
        
    oridiff = torch.norm(delta_ori, dim=1)
    if criterion == "pose":
        return (distance < pos_threshold) & (oridiff < ori_threshold)
    elif criterion == "pos":
        return distance < pos_threshold
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def vessel_gall_hard_connected(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
) -> torch.Tensor:
    """Terminate an episode when CD and gall become hard-connected.

    The connectivity test mirrors ``scripts/saved/scripts/reward3.py``:
    after trisection, the CD arm mask and gall mask are morphologically closed
    together and success is declared when the CD endpoint anchor and gall
    centroid lie in the same connected component.
    """
    results = _get_trisection_cached(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        prefer_semantic_info,
        vessel_color,
        gall_color,
        color_tolerance,
        gall_dilation_radius,
        compute_every,
    )
    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for i, result in enumerate(results):
        terminated[i] = _is_only_cd_connected(result)
    return terminated


def vessel_gall_hard_connected_stable(
    env: ManagerBasedRLEnv,
    camera_cfg_name: str = "camera",
    vessel_label: str = "vessel",
    gall_label: str = "gall",
    vessel_color: tuple = (25, 82, 255),
    gall_color: tuple = (255, 105, 180),
    color_tolerance: int = 10,
    prefer_semantic_info: bool = True,
    gall_dilation_radius: int = 8,
    compute_every: int = 1,
    stable_frames: int = 5,
    disconnect_decay: int = 1,
) -> torch.Tensor:
    """Terminate when hard connectivity is sustained for N consecutive frames.

    Connectivity is only counted on frames where trisection succeeds.
    If trisection fails or connectivity breaks, the streak resets to zero.
    The ``disconnect_decay`` argument is accepted for backward compatibility and
    has no effect in strict-consecutive mode.
    """
    counter = _get_cd_only_connectivity_counter(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
        disconnect_decay,
    )

    connected_mask, prev_connected_mask = _get_cd_only_connected_masks(
        env,
        camera_cfg_name,
        vessel_label,
        gall_label,
        vessel_color,
        gall_color,
        color_tolerance,
        prefer_semantic_info,
        gall_dilation_radius,
        compute_every,
    )
    update_vessel_connectivity_diagnostics(
        env,
        strict_counter=counter,
        connected_mask=connected_mask,
        prev_connected_mask=prev_connected_mask,
        success_frames=int(max(1, stable_frames)),
    )

    return counter >= int(max(1, stable_frames))
