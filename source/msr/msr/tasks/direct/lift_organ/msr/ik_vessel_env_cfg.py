# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment configuration for vessel semantic segmentation based RL task.

This configuration uses:
- Single robot (PSM) with IK-based end-effector pose control
- Gripper kept closed (no gripper action)
- Reward based on visible Vessel area in semantic segmentation image
- Camera sensor for semantic segmentation observation
"""

from msr.config.controllers import DifferentialIKWithSoftRCMControllerCfg
from msr.config.actions import DifferentialInverseKinematicsWithSoftRCMActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg

import msr.tasks.direct.lift_organ.mdp as mdp
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
import numpy as np

##
# Pre-defined configs
##
from msr.config.robot import MSR_PSM_CFG, MSR_PSM_HIGH_PD_CFG   # isort: skip


# Adjustable episode reset horizon (seconds).
# Set this to any value in [5.0, 10.0] for your preferred reset frequency.
VESSEL_EPISODE_LENGTH_S = 5.0


@configclass
class MSRPSMVesselSemRewardEnvCfg(joint_pos_env_cfg.MSRPSMUpe6SingleRobotEnvCfg):
    """
    Single robot environment with IK-based action and vessel semantic segmentation reward.
    
    Key features:
    - Action: End-effector pose increment (6D: position + orientation delta)
    - Gripper: Kept closed (no gripper action in action space)
    - Reward: Vessel class area ratio in semantic segmentation image
    - Observation: End-effector pose + command pose + semantic image info
    """
    
    def __post_init__(self):
        # Post init of parent (MSRPSMUpe6SingleRobotEnvCfg)
        super().__post_init__()

        # Disable physics replication for this scene to avoid unsupported replication warnings
        # from complex deformable/attachment assets in the loaded USD.
        self.scene.replicate_physics = False
        
        # Configure number of environments for training
        # Note: semantic_segmentation TiledCamera creates a tiled buffer of size
        # ceil(sqrt(N))^2 * W * H * 4 bytes. With 640x480 and N=512 this overflows
        # signed int32. Keep resolution low (<=128x128) or reduce N accordingly.
        self.scene.num_envs = 256

        # PhysX GPU deformable/soft-body contact buffer safeguards.
        # This task uses deformable organ interactions and can overflow default buffers.
        self.sim.physx.gpu_max_soft_body_contacts = 8 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 8 * 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 2**29
        self.sim.physx.gpu_temp_buffer_capacity = 2**27
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.sim.physx.gpu_max_deformable_surface_contacts = 8 * 1024 * 1024
        
        # ========================================================================
        # ACTION CONFIGURATION: IK-based end-effector pose control
        # ========================================================================
        # Use DifferentialInverseKinematicsActionCfg for relative pose control
        # Action space: [dx, dy, dz, d_roll, d_pitch, d_yaw] (6D)
        self.actions.arm_1_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_shoulder_Joint", 
                "psm_upper_Joint", 
                "psm_fore_Joint", 
                "psm_wrist1_Joint", 
                "psm_wrist2_Joint", 
                "psm_wrist3_Joint", 
                "psm_insertion_Joint", 
                "psm_roll_Joint", 
                "psm_pitch_Joint", 
                "psm_yaw_Joint"
            ],
            body_name="psm_tool_tip_Link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",  # Control full 6D pose
                use_relative_mode=True,  # Use pose increment as action
                ik_method="dls",  # Damped Least Squares IK solver
                ik_params={"lambda_val": 0.05},
            ),
            scale=0.05,  # Scale factor for action (smaller for finer control)
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )
        
        # Gripper action: Set to None to keep gripper closed
        # The gripper will maintain its initial position (closed)
        self.actions.gripper_1_action = None
        
        # Disable robot_2 actions (single robot config)
        self.actions.arm_2_action = None
        self.actions.gripper_2_action = None
        
        # ========================================================================
        # REWARD CONFIGURATION: Vessel semantic segmentation coverage
        # ========================================================================
        # Disable existing tracking rewards (we don't need pose tracking)
        self.rewards.end_effector_1_position_tracking = None
        self.rewards.end_effector_1_orientation_tracking = None
        self.rewards.end_effector_1_position_tracking_fine_grained = None
        self.rewards.end_effector_1_orientation_tracking_fine_grained = None
        
        # Main reward: Vessel semantic segmentation coverage
        # Higher reward when more Vessel (blue) area is visible in camera
        self.rewards.vessel_coverage = RewTerm(
            func=mdp.vessel_semantic_coverage_reward,
            weight=10.0,  # High weight for main task objective
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),  # Blue-like color from saved semantic image
                "color_tolerance": 20,
                "debug": True,
                "debug_every": 20,
                "debug_to_file": True,
            },
        )
        
        # Smoother vessel coverage reward with tanh shaping
        self.rewards.vessel_coverage_shaped = RewTerm(
            func=mdp.vessel_semantic_coverage_reward_tanh,
            weight=5.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "color_tolerance": 20,
                "std": 0.1,
            },
        )

        # Vessel centerline trisection rewards
        self.rewards.vessel_trisection_success = RewTerm(
            func=mdp.vessel_trisection_reward,
            weight=3.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "gall_color": (255, 105, 180),
                "color_tolerance": 20,
                "compute_every": 2,
            },
        )

        self.rewards.vessel_segment_straightness = RewTerm(
            func=mdp.vessel_segment_straightness_reward,
            weight=2.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "gall_color": (255, 105, 180),
                "color_tolerance": 20,
                "compute_every": 2,
            },
        )

        self.rewards.vessel_gall_single_connection = RewTerm(
            func=mdp.vessel_gall_single_connection_reward,
            weight=20.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "gall_color": (255, 105, 180),
                "color_tolerance": 20,
                "gall_dilation_radius": 5,
                "compute_every": 2,
            },
        )
        
        # Action regularization (penalty for large actions)
        self.rewards.action_l2 = RewTerm(
            func=mdp.action_l2,
            weight=-0.01,  # Small penalty for large actions
        )
        
        # Action rate penalty (penalty for jerky movements)
        self.rewards.action_rate = RewTerm(
            func=mdp.action_rate_l2,
            weight=-0.005,
        )
        
        # Keep joint velocity penalty
        self.rewards.joint_1_vel.weight = -0.001
        
        # ========================================================================
        # OBSERVATION CONFIGURATION
        # ========================================================================
        # Observations include:
        # - End-effector pose (7D: position + quaternion)
        # - Previous action (for action smoothness)
        # Note: We don't include semantic image directly in observation
        # (the reward function accesses it directly from the camera)
        
        # Disable pose command observation (we're not tracking a target pose)
        self.observations.policy.pose_1_command = None
        self.observations.policy.pose_1_rel = None
        
        # ========================================================================
        # COMMAND CONFIGURATION
        # ========================================================================
        # Disable pose commands (we're not tracking target poses)
        self.commands.ee_1_pose = None
        
        # ========================================================================
        # CAMERA CONFIGURATION
        # ========================================================================
        # Ensure camera is configured for semantic segmentation
        # Camera should already be set in parent config, but verify settings
        if self.scene.camera is not None:
            # Update camera settings if needed
            self.scene.camera.data_types = ["rgb", "semantic_segmentation"]
            self.scene.camera.colorize_semantic_segmentation = True
            self.scene.camera.semantic_filter = "*:*"  # Capture all semantic labels
            # IMPORTANT: Reduce resolution to avoid int32 overflow in Warp kernel.
            # TiledCamera tiled buffer = ceil(sqrt(N))^2 * W * H * 16 bytes.
            # With 640x480@512 envs this exceeds INT32_MAX. Use 128x128 instead.
            self.scene.camera.width = 128
            self.scene.camera.height = 128
        
        # ========================================================================
        # CURRICULUM CONFIGURATION
        # ========================================================================
        # Disable unused curriculum terms
        self.curriculum.end_effector_1_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_1_position_tracking_fine_grained = None
        self.curriculum.action_rate = None
        self.curriculum.joint_1_vel = None
        
        # ========================================================================
        # RESET EVENTS: ensure joints return to init on episode timeout
        # ========================================================================
        # Parent (MSRPSMUpe6SingleRobotEnvCfg) disabled all reset events.
        # Re-enable scene-wide default reset + explicit joint reset so that
        # after episode_length_s the robot reliably goes back to init_state.
        self.events.reset_all = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
        )
        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),   # exact init positions, no randomization
                "velocity_range": (0.0, 0.0),   # zero velocity
            },
        )
        # robot_2 and object are unused in this env – keep them None
        
        # ========================================================================
        # TERMINATION CONFIGURATION
        # ========================================================================
        # time_out is inherited from LiftOrganEnvCfg and fires after
        # episode_length_s / (sim.dt * decimation) steps.
        
        # ========================================================================
        # ENVIRONMENT SETTINGS
        # ========================================================================
        self.episode_length_s = VESSEL_EPISODE_LENGTH_S
        self.decimation = 2  # Control frequency = sim_freq / decimation


@configclass
class MSRPSMVesselSemRewardEnvCfg_PLAY(MSRPSMVesselSemRewardEnvCfg):
    """Play configuration for vessel semantic reward environment."""
    
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Disable physics replication for this scene to avoid unsupported replication warnings
        # from complex deformable/attachment assets in the loaded USD.
        self.scene.replicate_physics = False
        
        # Smaller scene for play/visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        
        # Disable observation corruption for evaluation
        self.observations.policy.enable_corruption = False


@configclass
class MSRPSMVesselSemRewardWithRCMEnvCfg(joint_pos_env_cfg.MSRPSMUpe6SingleRobotEnvCfg):
    """
    Single robot environment with IK+RCM constraint and vessel semantic reward.
    
    Uses DifferentialInverseKinematicsWithSoftRCMActionCfg for surgical robot
    with Remote Center of Motion (RCM) constraint.
    """
    
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Disable physics replication for this scene to avoid unsupported replication
        # errors from deformable/attachment assets in the loaded USD.
        self.scene.replicate_physics = False
        
        self.scene.num_envs = 256

        # PhysX GPU deformable/soft-body contact buffer safeguards.
        self.sim.physx.gpu_max_soft_body_contacts = 8 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 8 * 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 2**29
        self.sim.physx.gpu_temp_buffer_capacity = 2**27
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.sim.physx.gpu_max_deformable_surface_contacts = 8 * 1024 * 1024
        
        # ========================================================================
        # ACTION CONFIGURATION: IK with soft RCM constraint
        # ========================================================================
        self.actions.arm_1_action = DifferentialInverseKinematicsWithSoftRCMActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_shoulder_Joint", 
                "psm_upper_Joint", 
                "psm_fore_Joint", 
                "psm_wrist1_Joint", 
                "psm_wrist2_Joint", 
                "psm_wrist3_Joint", 
                "psm_insertion_Joint", 
                "psm_roll_Joint", 
                "psm_pitch_Joint", 
                "psm_yaw_Joint"
            ],
            scale=0.05,
            body_name="psm_tool_tip_Link",
            f1_name="psm_insertion_Link",
            f2_name="psm_roll_Link",
            controller=DifferentialIKWithSoftRCMControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            rcm_beta=0.1,  # RCM constraint strength
        )
        
        # No gripper action (keep closed)
        self.actions.gripper_1_action = None
        self.actions.arm_2_action = None
        self.actions.gripper_2_action = None
        
        # ========================================================================
        # REWARD CONFIGURATION
        # ========================================================================
        self.rewards.end_effector_1_position_tracking = None
        self.rewards.end_effector_1_orientation_tracking = None
        self.rewards.end_effector_1_position_tracking_fine_grained = None
        self.rewards.end_effector_1_orientation_tracking_fine_grained = None
        
        self.rewards.vessel_coverage = RewTerm(
            func=mdp.vessel_semantic_coverage_reward,
            weight=10.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "color_tolerance": 20,
                "debug": True,
                "debug_every": 20,
                "debug_to_file": True,
            },
        )
        
        self.rewards.vessel_coverage_shaped = RewTerm(
            func=mdp.vessel_semantic_coverage_reward_tanh,
            weight=5.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "color_tolerance": 20,
                "std": 0.1,
            },
        )

        self.rewards.vessel_trisection_success = RewTerm(
            func=mdp.vessel_trisection_reward,
            weight=3.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "gall_color": (255, 105, 180),
                "color_tolerance": 20,
                "compute_every": 2,
            },
        )

        self.rewards.vessel_segment_straightness = RewTerm(
            func=mdp.vessel_segment_straightness_reward,
            weight=2.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "gall_color": (255, 105, 180),
                "color_tolerance": 20,
                "compute_every": 2,
            },
        )

        self.rewards.vessel_gall_single_connection = RewTerm(
            func=mdp.vessel_gall_single_connection_reward,
            weight=20.0,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": (25, 82, 255),
                "gall_color": (255, 105, 180),
                "color_tolerance": 20,
                "gall_dilation_radius": 5,
                "compute_every": 2,
            },
        )
        
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
        
        # ========================================================================
        # OBSERVATION CONFIGURATION
        # ========================================================================
        self.observations.policy.pose_1_command = None
        self.observations.policy.pose_1_rel = None
        
        # ========================================================================
        # COMMAND CONFIGURATION
        # ========================================================================
        self.commands.ee_1_pose = None

        # ========================================================================
        # CAMERA CONFIGURATION
        # ========================================================================
        if self.scene.camera is not None:
            self.scene.camera.data_types = ["rgb", "semantic_segmentation"]
            self.scene.camera.colorize_semantic_segmentation = True
            self.scene.camera.semantic_filter = "*:*"
            # Reduce resolution to prevent int32 overflow in Warp tiled buffer
            self.scene.camera.width = 128
            self.scene.camera.height = 128
        
        # ========================================================================
        # CURRICULUM CONFIGURATION
        # ========================================================================
        self.curriculum.end_effector_1_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_1_position_tracking_fine_grained = None
        self.curriculum.action_rate = None
        self.curriculum.joint_1_vel = None
        
        # ========================================================================
        # RESET EVENTS
        # ========================================================================
        self.events.reset_all = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
        )
        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        
        self.episode_length_s = VESSEL_EPISODE_LENGTH_S
        self.decimation = 2


@configclass
class MSRPSMVesselSemRewardWithRCMEnvCfg_PLAY(MSRPSMVesselSemRewardWithRCMEnvCfg):
    """Play configuration for vessel semantic reward with RCM environment."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
