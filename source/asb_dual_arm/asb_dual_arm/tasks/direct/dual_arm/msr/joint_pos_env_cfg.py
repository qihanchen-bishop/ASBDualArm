# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg, FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from pxr import Usd

from asb_dual_arm.tasks.direct.dual_arm.dual_arm_env_cfg import LiftOrganEnvCfg
import asb_dual_arm.tasks.direct.dual_arm.mdp as mdp
##
# Pre-defined configs
## 
from asb_dual_arm.config.robot import MSR_PSM_HIGH_PD_CFG   # isort: skip
import numpy as np
import os

##
# Environment configuration
##
# organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/asb_dual_arm/asb_dual_arm/assets/others/uperc_right.usd'
# organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/asb_dual_arm/asb_dual_arm/assets/others/msr_organ.usd'
organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/asb_dual_arm/asb_dual_arm/assets/others/dual_arm.usd'
ROBOT_1_PRIM_NAME_IN_USD = "msr_psm"
ROBOT_2_PRIM_NAME_IN_USD = "msr_psm_1"
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
ROBOT_2_DEFAULT_INIT_POS = (-0.35, 0.0, 0.0)
ROBOT_2_DEFAULT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOT_2_DEFAULT_INIT_JOINT_POS = {
    "psm_shoulder_Joint": np.pi / 2,
    "psm_upper_Joint": -np.pi / 4,
    "psm_fore_Joint": -2 * np.pi / 3,
    "psm_wrist1_Joint": np.pi,
    "psm_wrist2_Joint": -np.pi / 2,
    "psm_wrist3_Joint": -np.pi / 3,
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


def apply_robot_init_state_from_usd(
    env_cfg,
    robot_attr_name: str,
    robot_prim_name: str,
    default_joint_pos: dict,
    default_pos: tuple,
    default_rot: tuple,
    usd_path: str,
    verbose: bool = True,
) -> bool:
    """Update robot init state from the currently selected scene USD file."""
    robot_cfg = getattr(getattr(env_cfg, "scene", None), robot_attr_name, None)
    if robot_cfg is None:
        if verbose:
            print(f"[Robot Init] scene.{robot_attr_name} is missing, skip USD init sync.")
        return False

    joint_fallback = getattr(robot_cfg.init_state, "joint_pos", None)
    if isinstance(joint_fallback, dict) and len(joint_fallback) > 0:
        fallback_joint_pos = dict(joint_fallback)
    else:
        fallback_joint_pos = dict(default_joint_pos)

    fallback_pos = tuple(getattr(robot_cfg.init_state, "pos", default_pos))
    fallback_rot = tuple(getattr(robot_cfg.init_state, "rot", default_rot))

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

    robot_prim = _find_first_prim_by_name(stage, robot_prim_name)
    if robot_prim is None or not robot_prim.IsValid():
        if verbose:
            print(
                f"[Robot Init] Prim '{robot_prim_name}' not found in USD ({usd_path}), "
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
        print(f"[Robot Init] Robot prim ({robot_attr_name}): {robot_prim.GetPath().pathString}")
        print(
            f"[Robot Init] Loaded joints from USD: {loaded_joint_count}/{len(fallback_joint_pos)}; "
            f"fallback joints: {len(missing_joints)}"
        )
        if missing_joints:
            print(f"[Robot Init] Missing joint authored values: {missing_joints}")

    return True


def apply_robot_1_init_state_from_usd(env_cfg, usd_path: str, verbose: bool = True) -> bool:
    """Update robot_1 init state from the currently selected scene USD file."""
    return apply_robot_init_state_from_usd(
        env_cfg=env_cfg,
        robot_attr_name="robot_1",
        robot_prim_name=ROBOT_1_PRIM_NAME_IN_USD,
        default_joint_pos=ROBOT_1_DEFAULT_INIT_JOINT_POS,
        default_pos=ROBOT_1_DEFAULT_INIT_POS,
        default_rot=ROBOT_1_DEFAULT_INIT_ROT,
        usd_path=usd_path,
        verbose=verbose,
    )


def apply_robot_2_init_state_from_usd(env_cfg, usd_path: str, verbose: bool = True) -> bool:
    """Update robot_2 init state from the currently selected scene USD file."""
    return apply_robot_init_state_from_usd(
        env_cfg=env_cfg,
        robot_attr_name="robot_2",
        robot_prim_name=ROBOT_2_PRIM_NAME_IN_USD,
        default_joint_pos=ROBOT_2_DEFAULT_INIT_JOINT_POS,
        default_pos=ROBOT_2_DEFAULT_INIT_POS,
        default_rot=ROBOT_2_DEFAULT_INIT_ROT,
        usd_path=usd_path,
        verbose=verbose,
    )

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
class MSRPSMUpe6DualArmEnvCfg(LiftOrganEnvCfg):
    """Environment configuration that loads a dual-arm scene from dual_arm.usd.

    This configuration uses spawn=None for both robots, reading:
    - robot_1 from {ENV_REGEX_NS}/Organ/msr_psm
    - robot_2 from {ENV_REGEX_NS}/Organ/msr_psm_1
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1024

        # Load the entire scene from the USD file.
        self.scene.organ = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Organ",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=organ_usd_path,
                scale=(1.0, 1.0, 1.0),
                activate_contact_sensors=True,
            ),
        )

        # Ground/table are already provided by scene USD.
        self.scene.ground = None
        self.scene.table = None
        self.scene.coordinate_marker = None
        self.scene.object = None

        # Reference the target sphere already authored in the scene USD.
        self.scene.targetpoint = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Organ/Sphere_02",
            spawn=None,
        )

        # Contact sensors on robot_2 gripper links, filtered to the target sphere.
        # The psm_tool_tip_Link is an IK frame marker and does not carry collision contacts.
        self.scene.targetpoint_contact_gripper1 = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Organ/msr_psm_1/psm_gripper1_Link",
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Organ/Sphere_02"],
        )
        self.scene.targetpoint_contact_gripper2 = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Organ/msr_psm_1/psm_gripper2_Link",
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Organ/Sphere_02"],
        )

        # Robot 1 from USD (spawn=None).
        self.scene.robot_1 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Organ/msr_psm")
        self.scene.robot_1.spawn = None
        self.scene.robot_1.init_state.joint_pos = dict(ROBOT_1_DEFAULT_INIT_JOINT_POS)
        self.scene.robot_1.init_state.pos = ROBOT_1_DEFAULT_INIT_POS
        self.scene.robot_1.init_state.rot = ROBOT_1_DEFAULT_INIT_ROT
        apply_robot_1_init_state_from_usd(self, self.scene.organ.spawn.usd_path, verbose=True)

        # Robot 2 from USD (spawn=None) at msr_psm_1.
        self.scene.robot_2 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Organ/msr_psm_1")
        self.scene.robot_2.spawn = None
        self.scene.robot_2.init_state.joint_pos = dict(ROBOT_2_DEFAULT_INIT_JOINT_POS)
        self.scene.robot_2.init_state.pos = ROBOT_2_DEFAULT_INIT_POS
        self.scene.robot_2.init_state.rot = ROBOT_2_DEFAULT_INIT_ROT
        apply_robot_2_init_state_from_usd(self, self.scene.organ.spawn.usd_path, verbose=True)

        # Override rewards for both end-effectors.
        self.rewards.end_effector_1_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]

        # Joint-position actions for both robot arms.
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
                "psm_yaw_Joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.arm_2_action = mdp.JointPositionActionCfg(
            asset_name="robot_2",
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
                "psm_yaw_Joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_1_action = None
        self.actions.gripper_2_action = None

        # Command generators for both arms.
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
        self.commands.ee_2_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_2",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-0.15, -0.15),
                pos_y=(0.40, 0.40),
                pos_z=(0.20, 0.20),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(1.57, 1.57),
            ),
        )
        self.commands.ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_2_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_2_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.lift_pose = None
        self.commands.object_pose = None

        # Keep pose observations for both arms, but ignore jaw terms.
        self.observations.policy.jaw_pos_1 = None
        self.observations.policy.jaw_pos_2 = None

        # Preserve authored USD joint states on reset.
        self.events.reset_robot_1_joints = None
        self.events.reset_robot_2_joints = None
        self.events.reset_object_position = None

        # Object-related termination is unused in this scene variant.
        self.terminations.success = None
        self.curriculum.action_rate = None

        # Frame transformers for both robot end-effectors.
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
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Organ/msr_psm_1/psm_base_Link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Organ/msr_psm_1/psm_tool_tip_Link",
                    name="end_effector",
                ),
            ],
        )

        # Main camera sensor from USD.
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Organ/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
            spawn=None,
            data_types=["rgb", "semantic_segmentation"],
            width=640,
            height=480,
            update_latest_camera_pose=True,
            colorize_semantic_segmentation=True,
            semantic_filter="class:*",
            semantic_segmentation_mapping=SEMANTIC_SEGMENTATION_MAPPING,
        )

        # D455 pseudo-depth camera from USD.
        self.scene.depth_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Organ/rsd455/RSD455/Camera_Pseudo_Depth",
            spawn=None,
            data_types=["depth"],
            width=640,
            height=480,
            update_latest_camera_pose=True,
        )


@configclass
class MSRPSMUpe6DualArmEnvCfg_PLAY(MSRPSMUpe6DualArmEnvCfg):
    """Play configuration for dual-arm testing with dual_arm.usd scene."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# Backward compatibility aliases for existing imports in copied configs.
MSRPSMUpe6SingleRobotEnvCfg = MSRPSMUpe6DualArmEnvCfg
MSRPSMUpe6SingleRobotEnvCfg_PLAY = MSRPSMUpe6DualArmEnvCfg_PLAY
