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
from msr.config.actions import (
    DifferentialInverseKinematicsWorkspaceClampedActionCfg,
    DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedActionCfg,
)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg

import msr.tasks.direct.lift_organ_fixed.mdp as mdp
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
import numpy as np

##
# Pre-defined configs
##
from msr.config.robot import MSR_PSM_CFG, MSR_PSM_HIGH_PD_CFG   # isort: skip


# Adjustable episode reset horizon (seconds).
# Keep this long enough to include warmup + search + stable-connectivity hold.
VESSEL_EPISODE_LENGTH_S = 10.0
VESSEL_RGB = tuple(joint_pos_env_cfg.SEMANTIC_SEGMENTATION_MAPPING["class:vessel"][:3])
GALL_RGB = tuple(joint_pos_env_cfg.SEMANTIC_SEGMENTATION_MAPPING["class:gall"][:3])
GRIPPER_RGB = tuple(joint_pos_env_cfg.SEMANTIC_SEGMENTATION_MAPPING["class:robot,gripper"][:3])
VESSEL_CAMERA_RESOLUTION = 128
VESSEL_CONNECTIVITY_RADIUS = 8
VESSEL_CONNECTIVITY_COMPUTE_EVERY = 1
# Stable-connectivity target duration (seconds).
# RL step dt in this task is sim.dt * decimation = 0.005 * 2 = 0.01 s.
VESSEL_STABLE_CONNECTIVITY_SECONDS = 0.5
VESSEL_STABLE_CONNECTIVITY_FRAMES = max(
    1,
    int(round(VESSEL_STABLE_CONNECTIVITY_SECONDS / (0.005 * 2 * VESSEL_CONNECTIVITY_COMPUTE_EVERY))),
)
# Frame-by-frame curriculum targets for stable connectivity frames.
# Increase by exactly 1 frame each promotion, with an explicit cap at 50.
VESSEL_STABLE_CONNECTIVITY_CURRICULUM_MIN_FRAMES = 2
VESSEL_STABLE_CONNECTIVITY_CURRICULUM_MAX_FRAMES = min(50, VESSEL_STABLE_CONNECTIVITY_FRAMES)
VESSEL_STABLE_CONNECTIVITY_CURRICULUM_FRAMES = tuple(
    range(
        VESSEL_STABLE_CONNECTIVITY_CURRICULUM_MIN_FRAMES,
        VESSEL_STABLE_CONNECTIVITY_CURRICULUM_MAX_FRAMES + 1,
    )
)
# Keep the progress denominator fixed to the final curriculum target.
VESSEL_CONNECTIVITY_PROGRESS_TARGET_FRAMES = VESSEL_STABLE_CONNECTIVITY_CURRICULUM_MAX_FRAMES

# Optional global-step schedule kept aligned with frame stages.
# This is only used when promotion_mode="global_step".
VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEP_START = 0
VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEP_END = 320_000
_VESSEL_CONNECTIVITY_CURRICULUM_STAGE_COUNT = len(VESSEL_STABLE_CONNECTIVITY_CURRICULUM_FRAMES)
if _VESSEL_CONNECTIVITY_CURRICULUM_STAGE_COUNT <= 1:
    VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEPS = (VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEP_START,)
else:
    VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEPS = tuple(
        int(
            round(
                VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEP_START
                + i
                * (VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEP_END - VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEP_START)
                / (_VESSEL_CONNECTIVITY_CURRICULUM_STAGE_COUNT - 1)
            )
        )
        for i in range(_VESSEL_CONNECTIVITY_CURRICULUM_STAGE_COUNT)
    )
# Promotion policy for curriculum stage switching.
# Default is performance-driven: promote only when recent connectivity success
# rate is sufficiently high and enough successful episodes are observed.
VESSEL_STABLE_CONNECTIVITY_PROMOTION_MODE = "success_rate"
VESSEL_STABLE_CONNECTIVITY_PROMOTION_SUCCESS_RATE = 0.6
VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_SUCCESSES = 5
VESSEL_STABLE_CONNECTIVITY_PROMOTION_WINDOW_EPISODES = 20
VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_EPISODES = 5
VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_CONSECUTIVE = 0
# Count stable rounds from the success-rate window evaluation.
# Promote after sustained high-rate rounds and demote after sustained low-rate rounds.
VESSEL_STABLE_CONNECTIVITY_PROMOTION_REQUIRED_STABLE_ROUNDS = 20
VESSEL_STABLE_CONNECTIVITY_DEMOTION_ENABLED = True
VESSEL_STABLE_CONNECTIVITY_DEMOTION_SUCCESS_RATE = 0.4
VESSEL_STABLE_CONNECTIVITY_DEMOTION_MIN_EPISODES = 20
VESSEL_STABLE_CONNECTIVITY_DEMOTION_REQUIRED_STABLE_ROUNDS = 25
# Small cooldown prevents immediate oscillation right after an adjustment.
VESSEL_STABLE_CONNECTIVITY_ADJUSTMENT_COOLDOWN_ROUNDS = 2
VESSEL_CONNECTIVITY_DISCONNECT_DECAY = 0 
# Connectivity shaping (matches the dual-counter design in reward3).
VESSEL_CONNECTIVITY_HOLD_REWARD = 0.0
VESSEL_CONNECTIVITY_SHAPING_WEIGHT = 200.0
VESSEL_CONNECTIVITY_SHAPING_POWER = 2.0
VESSEL_CONNECTIVITY_BREAK_PENALTY = 0.0
VESSEL_CONNECTIVITY_DONE_BONUS = 1.0
VESSEL_CONNECTIVITY_REWARD_CLIP = 200.0
VESSEL_COVERAGE_WEIGHT = 40.0
GALL_COVERAGE_WEIGHT = 5.0
CONNECTIVITY_REWARD_WEIGHT = 1.0
STRAIGHTNESS_REWARD_WEIGHT = 1.0
BRANCH_ANGLE_PENALTY_WEIGHT = -0.5
GALL_COVERAGE_PENALIZE_DECREASE = True
ACTION_L2_WEIGHT = -0.003
END_EFFECTOR_WORKSPACE_WEIGHT = -0.005
END_EFFECTOR_WORKSPACE_WEIGHT_RCM = -0.1
GRIPPER_EDGE_SAFE_MARGIN_PX = 5.0
STEP_PENALTY_WEIGHT = -1.0
VESSEL_ARM_ACTION_SCALE = 0.03
VESSEL_EE_X_BOUNDS = (-0.15, 0.15)
VESSEL_EE_Y_BOUNDS = (0.2, 0.5)
VESSEL_EE_Z_BOUNDS = (0.01, 0.2)
VESSEL_STRAIGHTNESS_MODE = "length_ratio"  # options: "length_ratio", "pca"
VESSEL_MIN_BRANCH_ANGLE_DEG = 30.0
VESSEL_MAX_BRANCH_ANGLE_DEG = 120.0

# PhysX GPU memory/contact capacities for multi-deformable interaction scenes.
# Larger values reduce overflow risk at the cost of higher GPU memory usage.
PHYSX_DEFORMABLE_CONTACT_CAPACITY = 16 * 1024 * 1024
PHYSX_SOFT_BODY_CONTACT_CAPACITY = 16 * 1024 * 1024
PHYSX_PARTICLE_CONTACT_CAPACITY = 16 * 1024 * 1024
PHYSX_GPU_HEAP_CAPACITY = 2**30
PHYSX_GPU_TEMP_BUFFER_CAPACITY = 2**28
PHYSX_GPU_COLLISION_STACK_SIZE = 2**30


# Reward terms to disable when testing only hard-connectivity reward.
_NON_CONNECTIVITY_REWARD_NAMES = (
    "end_effector_1_position_tracking",
    "end_effector_1_orientation_tracking",
    "end_effector_1_position_tracking_fine_grained",
    "end_effector_1_orientation_tracking_fine_grained",
    "vessel_coverage",
    "gall_coverage",
    "vessel_coverage_shaped",
    "step_penalty",
    "vessel_trisection_success",
    "vessel_segment_straightness",
    "vessel_branch_angle_range_penalty",
    "vessel_cd_gall_boundary_distance",
    "vessel_gall_hard_connectivity",
    "vessel_gall_single_connection",
    "gripper_edge_penalty",
    "action_l2",
    "action_rate",
    "end_effector_workspace",
    "end_effector_workspace_boundary",
    "joint_1_vel",
)


def _keep_only_vessel_gall_hard_connectivity_reward(env_cfg) -> None:
    """Disable non-connectivity rewards for connectivity-focused isolation tests."""
    for reward_name in _NON_CONNECTIVITY_REWARD_NAMES:
        if hasattr(env_cfg.rewards, reward_name):
            setattr(env_cfg.rewards, reward_name, None)


@configclass
class MSRPSMVesselSemRewardEnvCfg(joint_pos_env_cfg.MSRPSMUpe6SingleRobotEnvCfg):
    """
    Single robot environment with IK-based action and vessel semantic segmentation reward.
    
    Key features:
    - Action: End-effector pose increment (6D: position + orientation delta)
    - Gripper: Kept closed (no gripper action in action space)
    - Reward: migrated from the offline depth/semantic analysis scripts
    - Observation: RGB image + end-effector pose
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
        self.sim.physx.gpu_max_soft_body_contacts = PHYSX_SOFT_BODY_CONTACT_CAPACITY
        self.sim.physx.gpu_max_particle_contacts = PHYSX_PARTICLE_CONTACT_CAPACITY
        self.sim.physx.gpu_heap_capacity = PHYSX_GPU_HEAP_CAPACITY
        self.sim.physx.gpu_temp_buffer_capacity = PHYSX_GPU_TEMP_BUFFER_CAPACITY
        self.sim.physx.gpu_collision_stack_size = PHYSX_GPU_COLLISION_STACK_SIZE
        self.sim.physx.gpu_max_deformable_surface_contacts = PHYSX_DEFORMABLE_CONTACT_CAPACITY
        
        # ========================================================================
        # ACTION CONFIGURATION: IK-based end-effector pose control
        # ========================================================================
        # Use DifferentialInverseKinematicsActionCfg for relative pose control
        # Action space: [dx, dy, dz, d_roll, d_pitch, d_yaw] (6D)
        self.actions.arm_1_action = DifferentialInverseKinematicsWorkspaceClampedActionCfg(
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
            scale=VESSEL_ARM_ACTION_SCALE,  # Smaller scale limits per-step motion magnitude
            body_offset=DifferentialInverseKinematicsWorkspaceClampedActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            enforce_workspace_bounds=True,
            x_bounds_world=VESSEL_EE_X_BOUNDS,
            y_bounds_world=VESSEL_EE_Y_BOUNDS,
            z_bounds_world=VESSEL_EE_Z_BOUNDS,
        )
        
        # Gripper action: Set to None to keep gripper closed
        # The gripper will maintain its initial position (closed)
        self.actions.gripper_1_action = None
        
        # Disable robot_2 actions (single robot config)
        self.actions.arm_2_action = None
        self.actions.gripper_2_action = None
        
        # ========================================================================
        # REWARD CONFIGURATION
        # ========================================================================
        # Disable existing tracking rewards (we don't need pose tracking)
        self.rewards.end_effector_1_position_tracking = None
        self.rewards.end_effector_1_orientation_tracking = None
        self.rewards.end_effector_1_position_tracking_fine_grained = None
        self.rewards.end_effector_1_orientation_tracking_fine_grained = None

        # Reward 1: EMA-normalized delta of depth-weighted vessel area.
        self.rewards.vessel_coverage = RewTerm(
            func=mdp.vessel_depth_weighted_area_delta_reward,
            weight=VESSEL_COVERAGE_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "depth_camera_cfg_name": "depth_camera",
                "vessel_color": VESSEL_RGB,
                "color_tolerance": 20,
                "normalizer_alpha": 0.99,
                "normalizer_clip": 5.0,
                "depth_exponent": 2.0,
            },
        )

        self.rewards.gall_coverage = RewTerm(
            func=mdp.gall_depth_weighted_area_delta_reward,
            weight=GALL_COVERAGE_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "depth_camera_cfg_name": "depth_camera",
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "normalizer_alpha": 0.99,
                "normalizer_clip": 5.0,
                "depth_exponent": 2.0,
                "penalize_decrease": GALL_COVERAGE_PENALIZE_DECREASE,
            },
        )

        self.rewards.vessel_coverage_shaped = None
        self.rewards.step_penalty = None
        # self.rewards.step_penalty = RewTerm(func=mdp.step_penalty, weight=STEP_PENALTY_WEIGHT)

        self.rewards.vessel_trisection_success = None

        # Reward 2: average straightness after successful trisection.
        self.rewards.vessel_segment_straightness = RewTerm(
            func=mdp.vessel_segment_straightness_reward,
            weight=STRAIGHTNESS_REWARD_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": VESSEL_RGB,
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
                "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
                "straightness_mode": VESSEL_STRAIGHTNESS_MODE,
            },
        )

        self.rewards.vessel_branch_angle_range_penalty = RewTerm(
            func=mdp.vessel_branch_angle_range_penalty,
            weight=BRANCH_ANGLE_PENALTY_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": VESSEL_RGB,
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
                "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
                "min_angle_deg": VESSEL_MIN_BRANCH_ANGLE_DEG,
                "max_angle_deg": VESSEL_MAX_BRANCH_ANGLE_DEG,
            },
        )

        # Reward 3: hard connectivity between CD and gall.
        # Split into 4 components for better TensorBoard diagnostics.
        self.rewards.vessel_cd_gall_boundary_distance = None
        connectivity_reward_params = {
            "camera_cfg_name": "camera",
            "vessel_color": VESSEL_RGB,
            "gall_color": GALL_RGB,
            "color_tolerance": 20,
            "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
            "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
            "stable_frames": VESSEL_STABLE_CONNECTIVITY_FRAMES,
            "progress_target_frames": VESSEL_CONNECTIVITY_PROGRESS_TARGET_FRAMES,
            "disconnect_decay": VESSEL_CONNECTIVITY_DISCONNECT_DECAY,
            "hold_reward": VESSEL_CONNECTIVITY_HOLD_REWARD,
            "shaping_weight": VESSEL_CONNECTIVITY_SHAPING_WEIGHT,
            "shaping_power": VESSEL_CONNECTIVITY_SHAPING_POWER,
            "break_penalty": VESSEL_CONNECTIVITY_BREAK_PENALTY,
            "done_bonus": VESSEL_CONNECTIVITY_DONE_BONUS,
            "reward_clip": VESSEL_CONNECTIVITY_REWARD_CLIP,
        }
        self.rewards.progress_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_progress_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.hold_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_hold_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.break_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_break_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.done_bonus_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_done_bonus_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        # Disable legacy aggregated term to avoid double-counting.
        self.rewards.vessel_gall_hard_connectivity = None
        self.rewards.vessel_gall_single_connection = None

        self.rewards.gripper_edge_penalty = RewTerm(
            func=mdp.gripper_edge_penalty,
            weight=1.0,
            params={
                "camera_cfg_name": "camera",
                "gripper_color": GRIPPER_RGB,
                "color_tolerance": 20,
                "edge_safe_margin_px": GRIPPER_EDGE_SAFE_MARGIN_PX,
            },
        )
        
        # Action regularization (penalty for large actions)
        self.rewards.action_l2 = RewTerm(
            func=mdp.action_l2,
            weight=ACTION_L2_WEIGHT,
        )
        
        # Action rate penalty (penalty for jerky movements)
        self.rewards.action_rate = RewTerm(
            func=mdp.action_rate_l2,
            weight=-0.005,
        )

        # Workspace box penalty for end-effector position in world frame.
        self.rewards.end_effector_workspace = RewTerm(
            func=mdp.end_effector_workspace_penalty,
            weight=END_EFFECTOR_WORKSPACE_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", body_names="psm_tool_tip_Link"),
                "x_bounds": VESSEL_EE_X_BOUNDS,
                "y_bounds": VESSEL_EE_Y_BOUNDS,
                "z_bounds": VESSEL_EE_Z_BOUNDS,
                "squared": True,
            },
        )
        self.rewards.end_effector_workspace_boundary = RewTerm(
            func=mdp.end_effector_workspace_boundary_penalty,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", body_names="psm_tool_tip_Link"),
                "x_bounds": VESSEL_EE_X_BOUNDS,
                "y_bounds": VESSEL_EE_Y_BOUNDS,
                "z_bounds": VESSEL_EE_Z_BOUNDS,
                "margin": 0.02,
                "squared": True,
            },
        )
        
        # Bounded and reset-safe joint velocity penalty.
        self.rewards.joint_1_vel = RewTerm(
            func=mdp.joint_vel_l2_limited,
            weight=-0.00001,
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "clip": 2000.0,
                "deadzone": 0.0,
                "reset_grace_steps": 8,
                "warmup_grace_steps": 100,
                "ema_alpha": 0.9,
            },
        )
        
        # ========================================================================
        # OBSERVATION CONFIGURATION
        # ========================================================================
        # Keep only the RGB image and the current end-effector pose in policy observations.
        self.observations.policy.pose_1_command = None
        self.observations.policy.pose_1_rel = None
        self.observations.policy.actions = None
        self.observations.policy.rgb_image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
        )
        self.observations.policy.depth_image = None
        self.observations.policy.semantic_image = None
        self.observations.policy.concatenate_terms = False
        self.observations.policy.enable_corruption = False
        
        # ========================================================================
        # COMMAND CONFIGURATION
        # ========================================================================
        # Disable pose commands (we're not tracking target poses)
        self.commands.ee_1_pose = None
        
        # ========================================================================
        # CAMERA CONFIGURATION
        # ========================================================================
        # Keep RGB as the policy input while preserving semantic/depth streams for rewards.
        if self.scene.camera is not None:
            self.scene.camera.data_types = ["rgb", "semantic_segmentation"]
            self.scene.camera.colorize_semantic_segmentation = True
            self.scene.camera.semantic_filter = "class:*"
            self.scene.camera.width = VESSEL_CAMERA_RESOLUTION
            self.scene.camera.height = VESSEL_CAMERA_RESOLUTION
        if self.scene.depth_camera is not None:
            self.scene.depth_camera.data_types = ["depth"]
            self.scene.depth_camera.width = VESSEL_CAMERA_RESOLUTION
            self.scene.depth_camera.height = VESSEL_CAMERA_RESOLUTION
        
        # ========================================================================
        # CURRICULUM CONFIGURATION
        # ========================================================================
        # Disable unused curriculum terms
        self.curriculum.end_effector_1_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_1_position_tracking_fine_grained = None
        self.curriculum.action_rate = None
        self.curriculum.joint_1_vel = None
        self.curriculum.connectivity_stable_frames = None
        # self.curriculum.connectivity_stable_frames = CurrTerm(
        #     func=mdp.schedule_connectivity_stable_frames,
        #     params={
        #         "phase_steps": VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEPS,
        #         "phase_frames": VESSEL_STABLE_CONNECTIVITY_CURRICULUM_FRAMES,
        #         "promotion_mode": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MODE,
        #         "promotion_success_rate_threshold": VESSEL_STABLE_CONNECTIVITY_PROMOTION_SUCCESS_RATE,
        #         "promotion_min_successes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_SUCCESSES,
        #         "promotion_window_episodes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_WINDOW_EPISODES,
        #         "promotion_min_episodes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_EPISODES,
        #         "promotion_min_consecutive_successes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_CONSECUTIVE,
        #         "promotion_required_stable_rounds": VESSEL_STABLE_CONNECTIVITY_PROMOTION_REQUIRED_STABLE_ROUNDS,
        #         "demotion_enabled": VESSEL_STABLE_CONNECTIVITY_DEMOTION_ENABLED,
        #         "demotion_success_rate_threshold": VESSEL_STABLE_CONNECTIVITY_DEMOTION_SUCCESS_RATE,
        #         "demotion_min_episodes": VESSEL_STABLE_CONNECTIVITY_DEMOTION_MIN_EPISODES,
        #         "demotion_required_stable_rounds": VESSEL_STABLE_CONNECTIVITY_DEMOTION_REQUIRED_STABLE_ROUNDS,
        #         "adjustment_cooldown_rounds": VESSEL_STABLE_CONNECTIVITY_ADJUSTMENT_COOLDOWN_ROUNDS,
        #         "reward_term_names": (
        #             "progress_connectivity",
        #             "hold_connectivity",
        #             "break_connectivity",
        #             "done_bonus_connectivity",
        #         ),
        #         "termination_term_name": "success",
        #     },
        # )
        
        # ========================================================================
        # RESET EVENTS: ensure joints return to init on episode timeout
        # ========================================================================
        # Parent (MSRPSMUpe6SingleRobotEnvCfg) disabled all reset events.
        # Re-enable scene-wide default reset + explicit joint reset so that
        # after episode_length_s the robot reliably goes back to init_state.
        # self.events.reset_all = EventTerm(
        #     func=mdp.reset_scene_to_default,
        #     mode="reset",
        # )
        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),   # exact init positions, no randomization
                "velocity_range": (0.0, 0.0),   # zero velocity
            },
        )
        self.events.reset_vessel_connectivity_state = EventTerm(
            func=mdp.reset_vessel_connectivity_state,
            mode="reset",
        )
        # robot_2 and object are unused in this env – keep them None
        
        # ========================================================================
        # TERMINATION CONFIGURATION
        # ========================================================================
        self.terminations.success = DoneTerm(
            func=mdp.vessel_gall_hard_connected_stable,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": VESSEL_RGB,
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
                "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
                "stable_frames": VESSEL_STABLE_CONNECTIVITY_FRAMES,
                "disconnect_decay": VESSEL_CONNECTIVITY_DISCONNECT_DECAY,
            },
        )
        
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
        self.sim.physx.gpu_max_soft_body_contacts = PHYSX_SOFT_BODY_CONTACT_CAPACITY
        self.sim.physx.gpu_max_particle_contacts = PHYSX_PARTICLE_CONTACT_CAPACITY
        self.sim.physx.gpu_heap_capacity = PHYSX_GPU_HEAP_CAPACITY
        self.sim.physx.gpu_temp_buffer_capacity = PHYSX_GPU_TEMP_BUFFER_CAPACITY
        self.sim.physx.gpu_collision_stack_size = PHYSX_GPU_COLLISION_STACK_SIZE
        self.sim.physx.gpu_max_deformable_surface_contacts = PHYSX_DEFORMABLE_CONTACT_CAPACITY
        
        # ========================================================================
        # ACTION CONFIGURATION: IK with soft RCM constraint
        # ========================================================================
        self.actions.arm_1_action = DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedActionCfg(
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
            f1_name="psm_insertion_Link",
            f2_name="psm_roll_Link",
            controller=DifferentialIKWithSoftRCMControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsWithSoftRCMWorkspaceClampedActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            rcm_beta=0.1,  # RCM constraint strength
            scale=VESSEL_ARM_ACTION_SCALE,
            enforce_workspace_bounds=True,
            x_bounds_world=VESSEL_EE_X_BOUNDS,
            y_bounds_world=VESSEL_EE_Y_BOUNDS,
            z_bounds_world=VESSEL_EE_Z_BOUNDS,
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
            func=mdp.vessel_depth_weighted_area_delta_reward,
            weight=VESSEL_COVERAGE_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "depth_camera_cfg_name": "depth_camera",
                "vessel_color": VESSEL_RGB,
                "color_tolerance": 20,
                "normalizer_alpha": 0.99,
                "normalizer_clip": 5.0,
                "depth_exponent": 2.0,
            },
        )
        self.rewards.gall_coverage = RewTerm(
            func=mdp.gall_depth_weighted_area_delta_reward,
            weight=GALL_COVERAGE_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "depth_camera_cfg_name": "depth_camera",
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "normalizer_alpha": 0.99,
                "normalizer_clip": 5.0,
                "depth_exponent": 2.0,
                "penalize_decrease": GALL_COVERAGE_PENALIZE_DECREASE,
            },
        )
        self.rewards.vessel_coverage_shaped = None
        self.rewards.step_penalty = None

        self.rewards.vessel_trisection_success = None

        self.rewards.vessel_segment_straightness = RewTerm(
            func=mdp.vessel_segment_straightness_reward,
            weight=STRAIGHTNESS_REWARD_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": VESSEL_RGB,
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
                "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
                "straightness_mode": VESSEL_STRAIGHTNESS_MODE,
            },
        )

        self.rewards.vessel_branch_angle_range_penalty = RewTerm(
            func=mdp.vessel_branch_angle_range_penalty,
            weight=BRANCH_ANGLE_PENALTY_WEIGHT,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": VESSEL_RGB,
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
                "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
                "min_angle_deg": VESSEL_MIN_BRANCH_ANGLE_DEG,
                "max_angle_deg": VESSEL_MAX_BRANCH_ANGLE_DEG,
            },
        )

        self.rewards.vessel_cd_gall_boundary_distance = None
        connectivity_reward_params = {
            "camera_cfg_name": "camera",
            "vessel_color": VESSEL_RGB,
            "gall_color": GALL_RGB,
            "color_tolerance": 20,
            "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
            "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
            "stable_frames": VESSEL_STABLE_CONNECTIVITY_FRAMES,
            "progress_target_frames": VESSEL_CONNECTIVITY_PROGRESS_TARGET_FRAMES,
            "disconnect_decay": VESSEL_CONNECTIVITY_DISCONNECT_DECAY,
            "hold_reward": VESSEL_CONNECTIVITY_HOLD_REWARD,
            "shaping_weight": VESSEL_CONNECTIVITY_SHAPING_WEIGHT,
            "shaping_power": VESSEL_CONNECTIVITY_SHAPING_POWER,
            "break_penalty": VESSEL_CONNECTIVITY_BREAK_PENALTY,
            "done_bonus": VESSEL_CONNECTIVITY_DONE_BONUS,
            "reward_clip": VESSEL_CONNECTIVITY_REWARD_CLIP,
        }
        self.rewards.progress_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_progress_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.hold_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_hold_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.break_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_break_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.done_bonus_connectivity = RewTerm(
            func=mdp.vessel_gall_hard_connectivity_done_bonus_reward,
            weight=CONNECTIVITY_REWARD_WEIGHT,
            params=connectivity_reward_params.copy(),
        )
        self.rewards.vessel_gall_hard_connectivity = None
        self.rewards.vessel_gall_single_connection = None

        self.rewards.gripper_edge_penalty = RewTerm(
            func=mdp.gripper_edge_penalty,
            weight=1.0,
            params={
                "camera_cfg_name": "camera",
                "gripper_color": GRIPPER_RGB,
                "color_tolerance": 20,
                "edge_safe_margin_px": GRIPPER_EDGE_SAFE_MARGIN_PX,
            },
        )
        
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=ACTION_L2_WEIGHT)
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
        self.rewards.end_effector_workspace = RewTerm(
            func=mdp.end_effector_workspace_penalty,
            weight=END_EFFECTOR_WORKSPACE_WEIGHT_RCM,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", body_names="psm_tool_tip_Link"),
                "x_bounds": VESSEL_EE_X_BOUNDS,
                "y_bounds": VESSEL_EE_Y_BOUNDS,
                "z_bounds": VESSEL_EE_Z_BOUNDS,
                "squared": True,
            },
        )
        self.rewards.end_effector_workspace_boundary = RewTerm(
            func=mdp.end_effector_workspace_boundary_penalty,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", body_names="psm_tool_tip_Link"),
                "x_bounds": VESSEL_EE_X_BOUNDS,
                "y_bounds": VESSEL_EE_Y_BOUNDS,
                "z_bounds": VESSEL_EE_Z_BOUNDS,
                "margin": 0.02,
                "squared": True,
            },
        )

        self.rewards.joint_1_vel = RewTerm(
            func=mdp.joint_vel_l2_limited,
            weight=-0.00001,
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "clip": 2000.0,
                "deadzone": 0.0,
                "reset_grace_steps": 8,
                "warmup_grace_steps": 100,
                "ema_alpha": 0.9,
            },
        )
        
        # ========================================================================
        # OBSERVATION CONFIGURATION
        # ========================================================================
        self.observations.policy.pose_1_command = None
        self.observations.policy.pose_1_rel = None
        self.observations.policy.actions = None
        self.observations.policy.rgb_image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
        )
        self.observations.policy.depth_image = None
        self.observations.policy.semantic_image = None
        self.observations.policy.concatenate_terms = False
        self.observations.policy.enable_corruption = False
        
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
            self.scene.camera.semantic_filter = "class:*"
            self.scene.camera.width = VESSEL_CAMERA_RESOLUTION
            self.scene.camera.height = VESSEL_CAMERA_RESOLUTION
        if self.scene.depth_camera is not None:
            self.scene.depth_camera.data_types = ["depth"]
            self.scene.depth_camera.width = VESSEL_CAMERA_RESOLUTION
            self.scene.depth_camera.height = VESSEL_CAMERA_RESOLUTION
        
        # ========================================================================
        # CURRICULUM CONFIGURATION
        # ========================================================================
        self.curriculum.end_effector_1_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_1_position_tracking_fine_grained = None
        self.curriculum.action_rate = None
        self.curriculum.joint_1_vel = None
        self.curriculum.connectivity_stable_frames = None
        # self.curriculum.connectivity_stable_frames = CurrTerm(
        #     func=mdp.schedule_connectivity_stable_frames,
        #     params={
        #         "phase_steps": VESSEL_STABLE_CONNECTIVITY_CURRICULUM_STEPS,
        #         "phase_frames": VESSEL_STABLE_CONNECTIVITY_CURRICULUM_FRAMES,
        #         "promotion_mode": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MODE,
        #         "promotion_success_rate_threshold": VESSEL_STABLE_CONNECTIVITY_PROMOTION_SUCCESS_RATE,
        #         "promotion_min_successes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_SUCCESSES,
        #         "promotion_window_episodes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_WINDOW_EPISODES,
        #         "promotion_min_episodes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_EPISODES,
        #         "promotion_min_consecutive_successes": VESSEL_STABLE_CONNECTIVITY_PROMOTION_MIN_CONSECUTIVE,
        #         "promotion_required_stable_rounds": VESSEL_STABLE_CONNECTIVITY_PROMOTION_REQUIRED_STABLE_ROUNDS,
        #         "demotion_enabled": VESSEL_STABLE_CONNECTIVITY_DEMOTION_ENABLED,
        #         "demotion_success_rate_threshold": VESSEL_STABLE_CONNECTIVITY_DEMOTION_SUCCESS_RATE,
        #         "demotion_min_episodes": VESSEL_STABLE_CONNECTIVITY_DEMOTION_MIN_EPISODES,
        #         "demotion_required_stable_rounds": VESSEL_STABLE_CONNECTIVITY_DEMOTION_REQUIRED_STABLE_ROUNDS,
        #         "adjustment_cooldown_rounds": VESSEL_STABLE_CONNECTIVITY_ADJUSTMENT_COOLDOWN_ROUNDS,
        #         "reward_term_names": (
        #             "progress_connectivity",
        #             "hold_connectivity",
        #             "break_connectivity",
        #             "done_bonus_connectivity",
        #         ),
        #         "termination_term_name": "success",
        #     },
        # )
        
        # ========================================================================
        # RESET EVENTS
        # ========================================================================
        # self.events.reset_all = EventTerm(
        #     func=mdp.reset_scene_to_default,
        #     mode="reset",
        # )
        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_vessel_connectivity_state = EventTerm(
            func=mdp.reset_vessel_connectivity_state,
            mode="reset",
        )

        self.terminations.success = DoneTerm(
            func=mdp.vessel_gall_hard_connected_stable,
            params={
                "camera_cfg_name": "camera",
                "vessel_color": VESSEL_RGB,
                "gall_color": GALL_RGB,
                "color_tolerance": 20,
                "gall_dilation_radius": VESSEL_CONNECTIVITY_RADIUS,
                "compute_every": VESSEL_CONNECTIVITY_COMPUTE_EVERY,
                "stable_frames": VESSEL_STABLE_CONNECTIVITY_FRAMES,
                "disconnect_decay": VESSEL_CONNECTIVITY_DISCONNECT_DECAY,
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


@configclass
class MSRPSMVesselSemRewardConnectivityOnlyEnvCfg(MSRPSMVesselSemRewardEnvCfg):
    """IK vessel environment with only hard-connectivity reward enabled."""

    def __post_init__(self):
        super().__post_init__()
        _keep_only_vessel_gall_hard_connectivity_reward(self)


@configclass
class MSRPSMVesselSemRewardConnectivityOnlyEnvCfg_PLAY(MSRPSMVesselSemRewardEnvCfg_PLAY):
    """Play config with only hard-connectivity reward enabled."""

    def __post_init__(self):
        super().__post_init__()
        _keep_only_vessel_gall_hard_connectivity_reward(self)


@configclass
class MSRPSMVesselSemRewardWithRCMConnectivityOnlyEnvCfg(MSRPSMVesselSemRewardWithRCMEnvCfg):
    """IK+RCM vessel environment with only hard-connectivity reward enabled."""

    def __post_init__(self):
        super().__post_init__()
        _keep_only_vessel_gall_hard_connectivity_reward(self)


@configclass
class MSRPSMVesselSemRewardWithRCMConnectivityOnlyEnvCfg_PLAY(MSRPSMVesselSemRewardWithRCMEnvCfg_PLAY):
    """Play IK+RCM config with only hard-connectivity reward enabled."""

    def __post_init__(self):
        super().__post_init__()
        _keep_only_vessel_gall_hard_connectivity_reward(self)
