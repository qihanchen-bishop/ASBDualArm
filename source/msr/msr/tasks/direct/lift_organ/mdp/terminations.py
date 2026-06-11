# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift_organ task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms, matrix_from_quat, quat_from_matrix, make_pose, unmake_pose


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
