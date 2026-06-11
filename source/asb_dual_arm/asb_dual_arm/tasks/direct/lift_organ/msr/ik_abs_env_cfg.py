# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from asb_dual_arm.config.controllers import DifferentialIKWithSoftRCMControllerCfg
from asb_dual_arm.config.actions import DifferentialInverseKinematicsWithSoftRCMActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import asb_dual_arm.tasks.direct.lift_organ.mdp as mdp

from isaaclab.utils import configclass

from . import joint_pos_env_cfg
import numpy as np

##
# Pre-defined configs
##
from asb_dual_arm.config.robot import MSR_PSM_CFG, MSR_PSM_HIGH_PD_CFG   # isort: skip


@configclass
class MSRPSMLiftOrganNeedleEnvCfg(joint_pos_env_cfg.MSRPSMLiftOrganNeedleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1024

        # Set MSR-PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot_1 = MSR_PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_1.init_state.joint_pos = {
            "psm_shoulder_Joint": -np.pi / 2,
            "psm_upper_Joint": -3 * np.pi / 4,
            "psm_fore_Joint": 2./3. * np.pi,
            "psm_wrist1_Joint": 0,
            "psm_wrist2_Joint": np.pi / 2,
            "psm_wrist3_Joint": - 2* np.pi / 3,
            "psm_insertion_Joint": 0.12,
            "psm_roll_Joint": 0.0,
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.0,
            "psm_gripper2_Joint": 0.0,
        }
        self.scene.robot_2 = MSR_PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (0.5, 0.0, 0.0)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2.init_state.joint_pos = {
            "psm_shoulder_Joint": np.pi / 2,
            "psm_upper_Joint": -np.pi / 4,
            "psm_fore_Joint": -2./3. * np.pi,
            "psm_wrist1_Joint": np.pi,
            "psm_wrist2_Joint": -np.pi / 2,
            "psm_wrist3_Joint": - np.pi / 3,
            "psm_insertion_Joint": 0.12,
            "psm_roll_Joint": 0.0,
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.0,
            "psm_gripper2_Joint": 0.0,
        }

        # Set actions for the specific robot type (msr)
        self.actions.arm_1_action = DifferentialInverseKinematicsWithSoftRCMActionCfg(
            asset_name="robot_1",
            joint_names=["psm_shoulder_Joint", 
                         "psm_upper_Joint", 
                         "psm_fore_Joint", 
                         "psm_wrist1_Joint", 
                         "psm_wrist2_Joint", 
                         "psm_wrist3_Joint", 
                         "psm_insertion_Joint", 
                         "psm_roll_Joint", 
                         "psm_pitch_Joint", 
                         "psm_yaw_Joint"], 
            scale=0.5,
            body_name="psm_tool_tip_Link",
            f1_name="psm_insertion_Link",
            f2_name="psm_roll_Link",
            controller=DifferentialIKWithSoftRCMControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            rcm_beta=0.1,
        )
        self.actions.arm_2_action = DifferentialInverseKinematicsWithSoftRCMActionCfg(
            asset_name="robot_2",
            joint_names=["psm_shoulder_Joint", 
                         "psm_upper_Joint", 
                         "psm_fore_Joint", 
                         "psm_wrist1_Joint", 
                         "psm_wrist2_Joint", 
                         "psm_wrist3_Joint", 
                         "psm_insertion_Joint", 
                         "psm_roll_Joint", 
                         "psm_pitch_Joint", 
                         "psm_yaw_Joint"], 
            scale=0.5,
            body_name="psm_tool_tip_Link",
            f1_name="psm_insertion_Link",
            f2_name="psm_roll_Link",
            controller=DifferentialIKWithSoftRCMControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            rcm_beta=0.1,
        )

        # action penalty
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

@configclass
class MSRPSMLiftOrganNeedleEnvCfg_PLAY(MSRPSMLiftOrganNeedleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

@configclass
class MSRPSMLiftOrganBlockEnvCfg(joint_pos_env_cfg.MSRPSMLiftOrganBlockEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1024

        # Set MSR-PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot_1 = MSR_PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_1.init_state.joint_pos = {
            "psm_shoulder_Joint": -np.pi / 2,
            "psm_upper_Joint": -3 * np.pi / 4,
            "psm_fore_Joint": 2./3. * np.pi,
            "psm_wrist1_Joint": 0,
            "psm_wrist2_Joint": np.pi / 2,
            "psm_wrist3_Joint": - 2* np.pi / 3,
            "psm_insertion_Joint": 0.12,
            "psm_roll_Joint": 0.0,
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.0,
            "psm_gripper2_Joint": 0.0,
        }
        self.scene.robot_2 = MSR_PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (0.5, 0.0, 0.0)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2.init_state.joint_pos = {
            "psm_shoulder_Joint": np.pi / 2,
            "psm_upper_Joint": -np.pi / 4,
            "psm_fore_Joint": -2./3. * np.pi,
            "psm_wrist1_Joint": np.pi,
            "psm_wrist2_Joint": -np.pi / 2,
            "psm_wrist3_Joint": - np.pi / 3,
            "psm_insertion_Joint": 0.12,
            "psm_roll_Joint": 0.0,
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.0,
            "psm_gripper2_Joint": 0.0,
        }

        # Set actions for the specific robot type (msr)
        self.actions.arm_1_action = DifferentialInverseKinematicsWithSoftRCMActionCfg(
            asset_name="robot_1",
            joint_names=["psm_shoulder_Joint", 
                         "psm_upper_Joint", 
                         "psm_fore_Joint", 
                         "psm_wrist1_Joint", 
                         "psm_wrist2_Joint", 
                         "psm_wrist3_Joint", 
                         "psm_insertion_Joint", 
                         "psm_roll_Joint", 
                         "psm_pitch_Joint", 
                         "psm_yaw_Joint"], 
            scale=0.5,
            body_name="psm_tool_tip_Link",
            f1_name="psm_insertion_Link",
            f2_name="psm_roll_Link",
            controller=DifferentialIKWithSoftRCMControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            rcm_beta=0.1,
        )
        self.actions.arm_2_action = DifferentialInverseKinematicsWithSoftRCMActionCfg(
            asset_name="robot_2",
            joint_names=["psm_shoulder_Joint", 
                         "psm_upper_Joint", 
                         "psm_fore_Joint", 
                         "psm_wrist1_Joint", 
                         "psm_wrist2_Joint", 
                         "psm_wrist3_Joint", 
                         "psm_insertion_Joint", 
                         "psm_roll_Joint", 
                         "psm_pitch_Joint", 
                         "psm_yaw_Joint"], 
            scale=0.5,
            body_name="psm_tool_tip_Link",
            f1_name="psm_insertion_Link",
            f2_name="psm_roll_Link",
            controller=DifferentialIKWithSoftRCMControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            rcm_beta=0.1,
        )

        # action penalty
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-1e-3)
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class MSRPSMLiftOrganBlockEnvCfg_PLAY(MSRPSMLiftOrganBlockEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
