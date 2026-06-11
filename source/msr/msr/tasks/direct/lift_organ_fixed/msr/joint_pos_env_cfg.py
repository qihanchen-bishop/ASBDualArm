# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg, FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from pxr import Usd

from msr.tasks.direct.lift_organ_fixed.lift_organ_fixed_env_cfg import LiftOrganEnvCfg
import msr.tasks.direct.lift_organ_fixed.mdp as mdp
##
# Pre-defined configs
## 
from msr.config.robot import MSR_PSM_HIGH_PD_CFG   # isort: skip
import numpy as np
import os

##
# Environment configuration
##
# organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/uperc_right.usd'
# organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/msr_organ.usd'
organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/dual_arm.usd'
ROBOT_1_PRIM_NAME_IN_USD = "msr_psm"
ROBOT_1_DEFAULT_INIT_POS = (0.35, 0.0, 0.0)
ROBOT_1_DEFAULT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOT_1_DEFAULT_INIT_JOINT_POS = {
    "psm_shoulder_Joint": -np.pi / 2,
    "psm_upper_Joint": -3 * np.pi / 4,
    "psm_fore_Joint": 2 * np.pi / 3,
    "psm_wrist1_Joint": 15.0 * np.pi / 180.0,
    "psm_wrist2_Joint": np.pi / 2,
    "psm_wrist3_Joint": -62.0 * np.pi / 180.0,
    "psm_insertion_Joint": 0.06,
    "psm_roll_Joint": 0.0,
    "psm_pitch_Joint": 0.0,
    "psm_yaw_Joint": 0.0,
    "psm_gripper1_Joint": 0.01,
    "psm_gripper2_Joint": -0.01,
}


def _to_vec3(value):
    try:
        if value is None:
            return None
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None


def _to_quat_wxyz(value):
    if value is None:
        return None
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imag = value.GetImaginary()
        return (float(value.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError, IndexError):
        return None


def _find_first_prim_by_name(stage: Usd.Stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim
    return None


def _find_joint_prim(stage: Usd.Stage, robot_prim, joint_name: str):
    direct = stage.GetPrimAtPath(f"{robot_prim.GetPath().pathString}/joints/{joint_name}")
    if direct.IsValid():
        return direct
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() == joint_name:
            return prim
    return None


def _read_joint_target_from_usd(joint_prim):
    # Priority: drive target first, then state fallback.
    attr_specs = (
        ("drive:angular:physics:targetPosition", True),
        ("drive:linear:physics:targetPosition", False),
        ("state:angular:physics:position", True),
        ("state:linear:physics:position", False),
    )
    for attr_name, is_angular in attr_specs:
        attr = joint_prim.GetAttribute(attr_name)
        if not attr.IsValid():
            continue
        value = attr.Get()
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if is_angular:
            scalar = float(np.deg2rad(scalar))
        return scalar, attr_name
    return None, None


def apply_robot_1_init_state_from_usd(env_cfg, usd_path: str, verbose: bool = True) -> bool:
    """Update robot_1 init state from the currently selected scene USD file."""
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot_1", None)
    if robot_cfg is None:
        if verbose:
            print("[Robot Init] scene.robot_1 is missing, skip USD init sync.")
        return False

    joint_fallback = getattr(robot_cfg.init_state, "joint_pos", None)
    if isinstance(joint_fallback, dict) and len(joint_fallback) > 0:
        fallback_joint_pos = dict(joint_fallback)
    else:
        fallback_joint_pos = dict(ROBOT_1_DEFAULT_INIT_JOINT_POS)

    fallback_pos = tuple(getattr(robot_cfg.init_state, "pos", ROBOT_1_DEFAULT_INIT_POS))
    fallback_rot = tuple(getattr(robot_cfg.init_state, "rot", ROBOT_1_DEFAULT_INIT_ROT))

    is_local = "://" not in usd_path
    if is_local and not os.path.isfile(usd_path):
        if verbose:
            print(f"[Robot Init] USD not found ({usd_path}), keep fallback init state.")
        robot_cfg.init_state.joint_pos = fallback_joint_pos
        robot_cfg.init_state.pos = fallback_pos
        robot_cfg.init_state.rot = fallback_rot
        return False

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        if verbose:
            print(f"[Robot Init] Failed to open USD ({usd_path}), keep fallback init state.")
        robot_cfg.init_state.joint_pos = fallback_joint_pos
        robot_cfg.init_state.pos = fallback_pos
        robot_cfg.init_state.rot = fallback_rot
        return False

    robot_prim = _find_first_prim_by_name(stage, ROBOT_1_PRIM_NAME_IN_USD)
    if robot_prim is None or not robot_prim.IsValid():
        if verbose:
            print(
                f"[Robot Init] Prim '{ROBOT_1_PRIM_NAME_IN_USD}' not found in USD ({usd_path}), "
                "keep fallback init state."
            )
        robot_cfg.init_state.joint_pos = fallback_joint_pos
        robot_cfg.init_state.pos = fallback_pos
        robot_cfg.init_state.rot = fallback_rot
        return False

    # Read root pose directly from authored xform ops on the robot root prim.
    pos = _to_vec3(robot_prim.GetAttribute("xformOp:translate").Get())
    quat = _to_quat_wxyz(robot_prim.GetAttribute("xformOp:orient").Get())
    if pos is None:
        pos = fallback_pos
    if quat is None:
        quat = fallback_rot

    joint_pos = dict(fallback_joint_pos)
    missing_joints = []
    loaded_joint_count = 0
    for joint_name in fallback_joint_pos.keys():
        joint_prim = _find_joint_prim(stage, robot_prim, joint_name)
        if joint_prim is None or not joint_prim.IsValid():
            missing_joints.append(joint_name)
            continue
        value, _ = _read_joint_target_from_usd(joint_prim)
        if value is None:
            missing_joints.append(joint_name)
            continue
        joint_pos[joint_name] = value
        loaded_joint_count += 1

    robot_cfg.init_state.joint_pos = joint_pos
    robot_cfg.init_state.pos = pos
    robot_cfg.init_state.rot = quat

    if verbose:
        print(f"[Robot Init] Synced from USD: {usd_path}")
        print(f"[Robot Init] Robot prim: {robot_prim.GetPath().pathString}")
        print(
            f"[Robot Init] Loaded joints from USD: {loaded_joint_count}/{len(fallback_joint_pos)}; "
            f"fallback joints: {len(missing_joints)}"
        )
        if missing_joints:
            print(f"[Robot Init] Missing joint authored values: {missing_joints}")

    return True

# Semantic class to RGBA mapping used directly by the camera annotator.
# SEMANTIC_SEGMENTATION_MAPPING = {
#     "class:vessel": (0, 90, 255, 255),
#     "class:gall": (255, 64, 180, 255),
#     "class:kidney": (60, 220, 120, 255),
#     "class:liver": (255, 130, 80, 255),
#     "class:table": (255, 210, 80, 255),
#     "class:ground": (255, 128, 32, 255),
#     "class:psm": (130, 255, 0, 255),
# }
SEMANTIC_SEGMENTATION_MAPPING = {
    "class:vessel": (255, 0, 0, 255),
    "class:gall": (0, 255, 0, 255),
    "class:liver": (139, 69, 19, 255),
    "class:table": (128, 128, 128, 255),
    "class:ground": (0, 0, 0, 255),
    "class:robot": (255, 255, 0, 255),
    "class:backplane": (255, 255, 255, 255), 
    "class:gripper": (0, 0, 255, 255),
    # Composite labels appear when parent/child carry multiple class semantics.
    "class:gripper,robot": (0, 0, 255, 255),
    "class:robot,gripper": (0, 0, 255, 255),
}
##
# New configurations for loading entire scene from USD file (uper.usd)
# All scene elements use spawn=None to reference existing prims in the USD
##


@configclass
class MSRPSMUpe6SingleRobotEnvCfg(LiftOrganEnvCfg):
    """Environment configuration that loads the entire scene from uper.usd.
    
    This configuration uses spawn=None for all scene elements, meaning all assets
    (ground, table, robot, cube, etc.) are pre-defined in the USD file at:
    /workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets/others/uper.usd
    
    Paths in the USD file:
    - Robot: {ENV_REGEX_NS}/Organ/msr_psm (directly under Organ)
    - Table: {ENV_REGEX_NS}/Organ/table (directly under Organ)
    - Ground: {ENV_REGEX_NS}/Organ/GroundPlane (directly under Organ)
    Only robot_1 is used for testing purposes.
    """
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1024
        
        # Load the entire scene from the USD file
        # This spawns the main USD containing all scene elements
        self.scene.organ = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Organ",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=organ_usd_path,
                scale=(1.0, 1.0, 1.0),
            ),
        )
        
        # Ground is now in the USD file - set to None to avoid spawning default ground
        self.scene.ground = None
        
        # Table is now in the USD file - set to None
        self.scene.table = None
        
        # Light is now in the USD file - set to None (or keep if you want additional light)
        # self.scene.light = None
        
        # Remove coordinate marker if not needed
        self.scene.coordinate_marker = None
        
        # Remove object since we're loading from USD
        self.scene.object = None
        
        # Robot 1 - use spawn=None to reference existing robot in USD
        # The robot is directly under Organ: {ENV_REGEX_NS}/Organ/msr_psm
        # Using MSR_PSM_HIGH_PD_CFG.replace() to reuse actuator config, then override spawn and joint_pos
        self.scene.robot_1 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Organ/msr_psm")
        self.scene.robot_1.spawn = None  # Robot already exists in the USD file
        # Fallback init state (legacy values) and then auto-sync from USD.
        self.scene.robot_1.init_state.joint_pos = dict(ROBOT_1_DEFAULT_INIT_JOINT_POS)
        self.scene.robot_1.init_state.pos = ROBOT_1_DEFAULT_INIT_POS
        self.scene.robot_1.init_state.rot = ROBOT_1_DEFAULT_INIT_ROT
        apply_robot_1_init_state_from_usd(self, self.scene.organ.spawn.usd_path, verbose=True)
        
        # Robot 2 - disabled for single robot testing
        # Set to a dummy configuration that won't be used
        self.scene.robot_2 = None

        # Override rewards for robot_1 only
        self.rewards.end_effector_1_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        
        # Disable robot_2 related rewards
        self.rewards.end_effector_2_position_tracking = None
        self.rewards.end_effector_2_orientation_tracking = None
        self.rewards.end_effector_2_position_tracking_fine_grained = None
        self.rewards.end_effector_2_orientation_tracking_fine_grained = None
        self.rewards.joint_2_vel = None
        
        # Override actions for robot_1 only
        self.actions.arm_1_action = mdp.JointPositionActionCfg(
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
                "psm_yaw_Joint"], 
            scale=0.5,
            use_default_offset=True
        )
        self.actions.gripper_1_action = None
        
        # Disable actions for robot_2
        self.actions.arm_2_action = None
        self.actions.gripper_2_action = None
        
        # Command generator for robot_1 only
        self.commands.ee_1_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_1",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.15, 0.15),
                pos_y=(0.40, 0.40),
                pos_z=(0.20, 0.20),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(1.57, 1.57),
            ),
        )
        self.commands.ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        
        # Disable commands for robot_2
        self.commands.ee_2_pose = None
        self.commands.lift_pose = None
        self.commands.object_pose = None
        
        # Disable observations for robot_2
        self.observations.policy.pose_2_command = None
        self.observations.policy.pose_2_rel = None
        self.observations.policy.ee_2_pose = None
        self.observations.policy.jaw_pos_1 = None
        self.observations.policy.jaw_pos_2 = None
        
        # Events - disable joint reset to preserve USD joint positions
        self.events.reset_robot_1_joints = None  # Preserve USD joint positions
        self.events.reset_robot_2_joints = None
        self.events.reset_object_position = None
        
        # Disable terminations that depend on removed components
        self.terminations.success = None
        
        # Disable curriculum terms for robot_2 and unused reward terms
        self.curriculum.end_effector_2_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_2_position_tracking_fine_grained = None
        self.curriculum.joint_2_vel = None
        self.curriculum.action_rate = None  # action_rate reward term is None in parent
        
        # Frame transformer for robot_1 only
        # Robot is at {ENV_REGEX_NS}/Organ/msr_psm
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Organ/msr_psm/psm_base_Link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Organ/msr_psm/psm_tool_tip_Link",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = None
        
        # Main camera sensor - use the aligned D455 color camera from the USD.
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Organ/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
            spawn=None,
            data_types=["rgb", "semantic_segmentation"],
            width=640,
            height=480,
            update_latest_camera_pose=True,
            colorize_semantic_segmentation=True,
            # Restrict to class semantics to avoid color conflicts from other semantic types.
            semantic_filter="class:*",
            semantic_segmentation_mapping=SEMANTIC_SEGMENTATION_MAPPING,
        )

        # D455 pseudo-depth camera loaded from USD with spawn=None.
        self.scene.depth_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Organ/rsd455/RSD455/Camera_Pseudo_Depth",
            spawn=None,
            data_types=["depth"],
            width=640,
            height=480,
            update_latest_camera_pose=True,
        )


@configclass
class MSRPSMUpe6SingleRobotEnvCfg_PLAY(MSRPSMUpe6SingleRobotEnvCfg):
    """Play configuration for single robot testing with upe6.usd scene."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
