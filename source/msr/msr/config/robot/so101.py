# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO101 / SO-ARM robot."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

SO101_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
"""SO101 arm joints used for end-effector pose IK."""

SO101_GRIPPER_JOINT_NAMES = ["gripper"]
"""SO101 gripper joint. Kept fixed by the first dual SO-ARM teleop task."""

SO101_ALL_JOINT_NAMES = SO101_ARM_JOINT_NAMES + SO101_GRIPPER_JOINT_NAMES
"""All SO101 joints present in the new-calibration USD/URDF."""

SO101_DEFAULT_JOINT_POS = {joint_name: 0.0 for joint_name in SO101_ALL_JOINT_NAMES}
"""Neutral SO101 joint positions in radians."""


SO101_CFG = ArticulationCfg(
    spawn=None,
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=SO101_DEFAULT_JOINT_POS,
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=SO101_ARM_JOINT_NAMES,
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=60.0,
            damping=8.0,
            friction=0.0,
            armature=0.005,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=SO101_GRIPPER_JOINT_NAMES,
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=80.0,
            damping=8.0,
            friction=0.0,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""SO101 articulation configuration for existing USD prims loaded with ``spawn=None``."""
