# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual SO101 / SO-ARM task loaded from a full-scene USD."""

from __future__ import annotations

import os

from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from asb_dual_arm.config.actions import PinocchioInverseKinematicsActionCfg
from asb_dual_arm.config.actions.pinocchio_actions import PinocchioInverseKinematicsAction
from asb_dual_arm.config.robot import SO101_ALL_JOINT_NAMES, SO101_ARM_JOINT_NAMES, SO101_CFG, SO101_DEFAULT_JOINT_POS
from asb_dual_arm.tasks.direct.dual_arm.dual_arm_env_cfg import LiftOrganEnvCfg
import asb_dual_arm.tasks.direct.dual_arm.mdp as mdp


DUAL_SO_ARM_USD_PATH = "/workspace/isaaclab/source/ASBDualArm/source/asb_dual_arm/asb_dual_arm/assets/dual_arm/dual-so-arm.usd"
SO101_KINEMATICS_DIR = "/workspace/isaaclab/source/SO-ARM100/Simulation/SO101"
SO101_URDF_PATH = f"{SO101_KINEMATICS_DIR}/so101_new_calib.urdf"

ROBOT_1_PRIM_NAME_IN_USD = "so101_new_calib"
ROBOT_2_PRIM_NAME_IN_USD = "so101_new_calib_01"
TARGET_PRIM_NAME_IN_USD = "Target"
DEFORMABLE_OCCLUDER_PRIM_NAME_IN_USD = "DeformableOccluder"
ROBOT_1_DEFAULT_INIT_POS = (0.25, 0.5359415425704274, 0.9)
ROBOT_2_DEFAULT_INIT_POS = (-0.25, 0.5359415425704274, 0.9)
ROBOT_DEFAULT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

SO101_EE_LINK_NAME_URDF = "gripper_frame_link"
SO101_EE_FRAME_PRIM_NAME_USD = "gripperframe"
SO101_CAMERA_RESOLUTION = (640, 480)


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


def _find_first_prim_by_name(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim
    return None


def _find_joint_prim(stage, robot_prim, joint_name: str):
    direct = stage.GetPrimAtPath(f"{robot_prim.GetPath().pathString}/joints/{joint_name}")
    if direct.IsValid():
        return direct
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(robot_prim.GetPath().pathString) and prim.GetName() == joint_name:
            return prim
    return None


def _read_joint_target_from_usd(joint_prim):
    attr_specs = (
        ("drive:angular:physics:targetPosition", True),
        ("state:angular:physics:position", True),
    )
    import numpy as np

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
        return scalar
    return None


def _read_prim_pose_from_usd(
    usd_path: str,
    prim_name: str,
    fallback_pos: tuple[float, float, float],
    fallback_rot: tuple[float, float, float, float],
):
    """Read a prim pose from a USD file, including authored parent transforms."""

    if "://" not in usd_path and not os.path.isfile(usd_path):
        return fallback_pos, fallback_rot, None

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return fallback_pos, fallback_rot, None

    prim = _find_first_prim_by_name(stage, prim_name)
    if prim is None or not prim.IsValid():
        return fallback_pos, fallback_rot, None

    xformable = UsdGeom.Xformable(prim)
    transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = _to_vec3(transform.ExtractTranslation()) or _to_vec3(prim.GetAttribute("xformOp:translate").Get())
    rot = _to_quat_wxyz(transform.ExtractRotationQuat()) or _to_quat_wxyz(prim.GetAttribute("xformOp:orient").Get())
    return pos or fallback_pos, rot or fallback_rot, prim


def apply_rigid_object_init_state_from_usd(
    env_cfg,
    object_attr_name: str,
    object_prim_name: str,
    usd_path: str,
    verbose: bool = True,
) -> bool:
    """Update a rigid object's reset/default root pose from the selected scene USD."""

    object_cfg = getattr(getattr(env_cfg, "scene", None), object_attr_name, None)
    if object_cfg is None:
        return False

    fallback_pos = tuple(getattr(object_cfg.init_state, "pos", (0.0, 0.0, 0.0)))
    fallback_rot = tuple(getattr(object_cfg.init_state, "rot", ROBOT_DEFAULT_INIT_ROT))
    pos, rot, prim = _read_prim_pose_from_usd(usd_path, object_prim_name, fallback_pos, fallback_rot)
    if prim is None:
        object_cfg.init_state.pos = fallback_pos
        object_cfg.init_state.rot = fallback_rot
        return False

    object_cfg.init_state.pos = pos
    object_cfg.init_state.rot = rot

    if verbose:
        print(f"[RigidObject Init] {object_attr_name}: {prim.GetPath().pathString}")
        print(f"[RigidObject Init] pos={pos}, rot={rot}")

    return True


def apply_so101_init_state_from_usd(
    env_cfg,
    robot_attr_name: str,
    robot_prim_name: str,
    default_pos: tuple[float, float, float],
    usd_path: str,
    verbose: bool = True,
) -> bool:
    """Update a SO101 robot init state from the selected scene USD."""

    robot_cfg = getattr(getattr(env_cfg, "scene", None), robot_attr_name, None)
    if robot_cfg is None:
        return False

    fallback_joint_pos = dict(SO101_DEFAULT_JOINT_POS)
    fallback_pos = tuple(getattr(robot_cfg.init_state, "pos", default_pos))
    fallback_rot = tuple(getattr(robot_cfg.init_state, "rot", ROBOT_DEFAULT_INIT_ROT))

    if "://" not in usd_path and not os.path.isfile(usd_path):
        robot_cfg.init_state.joint_pos = fallback_joint_pos
        robot_cfg.init_state.pos = fallback_pos
        robot_cfg.init_state.rot = fallback_rot
        return False

    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        robot_cfg.init_state.joint_pos = fallback_joint_pos
        robot_cfg.init_state.pos = fallback_pos
        robot_cfg.init_state.rot = fallback_rot
        return False

    robot_prim = _find_first_prim_by_name(stage, robot_prim_name)
    if robot_prim is None or not robot_prim.IsValid():
        robot_cfg.init_state.joint_pos = fallback_joint_pos
        robot_cfg.init_state.pos = fallback_pos
        robot_cfg.init_state.rot = fallback_rot
        return False

    pos = _to_vec3(robot_prim.GetAttribute("xformOp:translate").Get()) or fallback_pos
    rot = _to_quat_wxyz(robot_prim.GetAttribute("xformOp:orient").Get()) or fallback_rot

    joint_pos = dict(fallback_joint_pos)
    loaded_joint_count = 0
    for joint_name in joint_pos.keys():
        joint_prim = _find_joint_prim(stage, robot_prim, joint_name)
        if joint_prim is None or not joint_prim.IsValid():
            continue
        value = _read_joint_target_from_usd(joint_prim)
        if value is None:
            continue
        joint_pos[joint_name] = value
        loaded_joint_count += 1

    robot_cfg.init_state.joint_pos = joint_pos
    robot_cfg.init_state.pos = pos
    robot_cfg.init_state.rot = rot

    if verbose:
        print(f"[SO101 Init] {robot_attr_name}: {robot_prim.GetPath().pathString}")
        print(f"[SO101 Init] Loaded joints from USD: {loaded_joint_count}/{len(joint_pos)}")

    return True


def _make_so101_pinocchio_action(asset_name: str) -> PinocchioInverseKinematicsActionCfg:
    return PinocchioInverseKinematicsActionCfg(
        class_type=PinocchioInverseKinematicsAction,
        asset_name=asset_name,
        joint_names=SO101_ARM_JOINT_NAMES,
        all_joint_names=SO101_ALL_JOINT_NAMES,
        urdf_path=SO101_URDF_PATH,
        mesh_path=SO101_KINEMATICS_DIR,
        base_body_name="base",
        ee_frame_name=SO101_EE_LINK_NAME_URDF,
        position_gain=0.5,
        orientation_gain=0.5,
        damping=0.05,
        max_delta=0.08,
    )


@configclass
class DualSOArmEnvCfg(LiftOrganEnvCfg):
    """Dual SO101 end-effector pose teleoperation environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 256
        self.scene.env_spacing = 2.5
        self.scene.replicate_physics = False
        self.episode_length_s = 60.0
        self.viewer.eye = (1.4, 1.6, 1.4)
        self.viewer.lookat = (0.0, 0.45, 0.9)

        self.scene.organ = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Scene",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=DUAL_SO_ARM_USD_PATH,
                scale=(1.0, 1.0, 1.0),
                activate_contact_sensors=False,
            ),
        )
        self.scene.ground = None
        self.scene.table = None
        self.scene.coordinate_marker = None
        self.scene.object = None
        self.scene.ring = None
        self.scene.plane = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Scene/Plane",
            spawn=None,
        )
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Scene/Target",
            spawn=None,
        )
        apply_rigid_object_init_state_from_usd(self, "target", TARGET_PRIM_NAME_IN_USD, DUAL_SO_ARM_USD_PATH)
        self.scene.deformable_occluder = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Scene/DeformableOccluder",
            spawn=None,
        )
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Scene/Camera",
            spawn=None,
            data_types=["rgb"],
            width=SO101_CAMERA_RESOLUTION[0],
            height=SO101_CAMERA_RESOLUTION[1],
            update_latest_camera_pose=True,
        )
        self.scene.depth_camera = None

        self.scene.robot_1 = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene/so101_new_calib")
        self.scene.robot_1.spawn = None
        self.scene.robot_1.init_state.joint_pos = dict(SO101_DEFAULT_JOINT_POS)
        self.scene.robot_1.init_state.pos = ROBOT_1_DEFAULT_INIT_POS
        self.scene.robot_1.init_state.rot = ROBOT_DEFAULT_INIT_ROT
        apply_so101_init_state_from_usd(
            self, "robot_1", ROBOT_1_PRIM_NAME_IN_USD, ROBOT_1_DEFAULT_INIT_POS, DUAL_SO_ARM_USD_PATH
        )

        self.scene.robot_2 = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene/so101_new_calib_01")
        self.scene.robot_2.spawn = None
        self.scene.robot_2.init_state.joint_pos = dict(SO101_DEFAULT_JOINT_POS)
        self.scene.robot_2.init_state.pos = ROBOT_2_DEFAULT_INIT_POS
        self.scene.robot_2.init_state.rot = ROBOT_DEFAULT_INIT_ROT
        apply_so101_init_state_from_usd(
            self, "robot_2", ROBOT_2_PRIM_NAME_IN_USD, ROBOT_2_DEFAULT_INIT_POS, DUAL_SO_ARM_USD_PATH
        )

        self.actions.arm_1_action = _make_so101_pinocchio_action("robot_1")
        self.actions.arm_2_action = _make_so101_pinocchio_action("robot_2")
        self.actions.gripper_1_action = None
        self.actions.gripper_2_action = None

        self.commands.ee_1_pose = None
        self.commands.ee_2_pose = None
        self.commands.lift_pose = None
        self.commands.object_pose = None

        self.observations.policy.pose_1_command = None
        self.observations.policy.ee_1_pose = None
        self.observations.policy.pose_1_rel = None
        self.observations.policy.jaw_pos_1 = None
        self.observations.policy.pose_2_command = None
        self.observations.policy.ee_2_pose = None
        self.observations.policy.pose_2_rel = None
        self.observations.policy.jaw_pos_2 = None
        self.observations.policy.rgb_image = None
        self.observations.policy.depth_image = None
        self.observations.policy.semantic_image = None
        self.observations.policy.enable_corruption = False

        for reward_name in tuple(self.rewards.__dict__.keys()):
            setattr(self.rewards, reward_name, None)
        self.terminations.success = None
        for curriculum_name in tuple(self.curriculum.__dict__.keys()):
            setattr(self.curriculum, curriculum_name, None)

        self.events.reset_object_position = None
        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_robot_2_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_2"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        # The authored gripperframe prim is an Xform, not a rigid body. Isaac Lab's FrameTransformer
        # only accepts rigid body targets, so the teleop script reconstructs the gripperframe pose from
        # the rigid gripper body plus the fixed URDF offset.
        self.scene.ee_1_frame = None
        self.scene.ee_2_frame = None


@configclass
class DualSOArmEnvCfg_PLAY(DualSOArmEnvCfg):
    """Play configuration for keyboard teleoperation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
