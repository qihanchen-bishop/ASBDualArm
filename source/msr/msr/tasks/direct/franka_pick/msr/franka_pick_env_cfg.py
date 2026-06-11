# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

import msr.tasks.direct.franka_pick.mdp as mdp


MSR_SURGICAL_ASSETS_DIR = "/workspace/isaaclab/source/ASBDualArm/source/msr/msr/assets"

TABLE_USD_POS = (0.0, 0.0, -0.457)
TABLE_USD_ROT = (0.7071068, 0.0, 0.0, 0.7071068)
TABLE_TOP_Z = 0.0
SHEET_USD_PATH = f"{MSR_SURGICAL_ASSETS_DIR}/others/plane.usd"
SHEET_USD_SCALE = (0.5, 0.5, 0.5)
SHEET_EFFECTIVE_SIZE = (0.30, 0.18, 0.003)
SHEET_INIT_POS = (0.50, 0.0, TABLE_TOP_Z + 0.5 * SHEET_EFFECTIVE_SIZE[2] + 0.003)
SHEET_INIT_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

ROBOT_BASE_X = 0.35
ROBOT_1_BASE_Y = -0.58
ROBOT_2_BASE_Y = 0.58

FRANKA_ARM_ACTION_SCALE = 0.2
FRANKA_EE_OFFSET = (0.0, 0.0, 0.107)
FRANKA_OPEN_POS = 0.04
FRANKA_CLOSED_POS = 0.0


@configclass
class FrankaPickSceneCfg(InteractiveSceneCfg):
    """Scene with two Franka Panda arms and a deformable rectangular sheet."""

    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_USD_POS, rot=TABLE_USD_ROT),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{MSR_SURGICAL_ASSETS_DIR}/Table/table.usd", scale=(1.0, 1.0, 1.0)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    deformable_sheet = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/DeformableSheet",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=SHEET_INIT_POS, rot=SHEET_INIT_ROT),
        spawn=sim_utils.UsdFileCfg(usd_path=SHEET_USD_PATH, scale=SHEET_USD_SCALE),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """No sampled commands are needed for teleoperation."""

    pass


@configclass
class ActionsCfg:
    """Dual-arm IK and binary gripper action specifications."""

    arm_1_action: ActionTerm = MISSING
    gripper_1_action: ActionTerm = MISSING
    arm_2_action: ActionTerm = MISSING
    gripper_2_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Minimal teleoperation observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_1_pose = ObsTerm(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("robot_1", body_names="panda_hand")})
        jaw_pos_1 = ObsTerm(
            func=mdp.jaw_pos,
            params={"robot_cfg": SceneEntityCfg("robot_1", joint_names=["panda_finger_joint.*"])},
        )
        ee_2_pose = ObsTerm(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("robot_2", body_names="panda_hand")})
        jaw_pos_2 = ObsTerm(
            func=mdp.jaw_pos,
            params={"robot_cfg": SceneEntityCfg("robot_2", joint_names=["panda_finger_joint.*"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset all articulations and deformables to their configured defaults."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class RewardsCfg:
    """No reward terms for the teleoperation-only first version."""

    pass


@configclass
class TerminationsCfg:
    """Terminate only on timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FrankaPickEnvCfg(ManagerBasedRLEnvCfg):
    """Dual-Franka deformable-sheet pick/exposure scene."""

    scene: FrankaPickSceneCfg = FrankaPickSceneCfg(num_envs=16, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0
        self.viewer.eye = (1.8, 1.4, 1.0)
        self.viewer.lookat = (0.5, 0.0, TABLE_TOP_Z + 0.08)

        self.sim.dt = 0.005
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_collision_stack_size = 2**26
        self.sim.physx.gpu_heap_capacity = 2**26
        self.sim.physx.gpu_temp_buffer_capacity = 2**24
        self.sim.physx.gpu_max_soft_body_contacts = 2**20
        self.sim.physx.gpu_max_particle_contacts = 2**10
        self.sim.physx.gpu_max_deformable_surface_contacts = 2**20

        self.scene.replicate_physics = False

        robot_1 = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_1.prim_path = "{ENV_REGEX_NS}/Robot_1"
        robot_1.init_state.pos = (ROBOT_BASE_X, ROBOT_1_BASE_Y, TABLE_TOP_Z)
        robot_1.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
        robot_1.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.45,
            "panda_joint3": 0.0,
            "panda_joint4": -2.35,
            "panda_joint5": 0.0,
            "panda_joint6": 2.25,
            "panda_joint7": 0.785,
            "panda_finger_joint.*": FRANKA_OPEN_POS,
        }
        robot_1.spawn.activate_contact_sensors = True

        robot_2 = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_2.prim_path = "{ENV_REGEX_NS}/Robot_2"
        robot_2.init_state.pos = (ROBOT_BASE_X, ROBOT_2_BASE_Y, TABLE_TOP_Z)
        robot_2.init_state.rot = (0.7071068, 0.0, 0.0, -0.7071068)
        robot_2.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.45,
            "panda_joint3": 0.0,
            "panda_joint4": -2.35,
            "panda_joint5": 0.0,
            "panda_joint6": 2.25,
            "panda_joint7": -0.785,
            "panda_finger_joint.*": FRANKA_OPEN_POS,
        }
        robot_2.spawn.activate_contact_sensors = True

        self.scene.robot_1 = robot_1
        self.scene.robot_2 = robot_2

        ik_controller = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self.actions.arm_1_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot_1",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=ik_controller,
            scale=FRANKA_ARM_ACTION_SCALE,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=FRANKA_EE_OFFSET),
        )
        self.actions.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": FRANKA_OPEN_POS},
            close_command_expr={"panda_finger_joint.*": FRANKA_CLOSED_POS},
        )
        self.actions.arm_2_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=ik_controller,
            scale=FRANKA_ARM_ACTION_SCALE,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=FRANKA_EE_OFFSET),
        )
        self.actions.gripper_2_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": FRANKA_OPEN_POS},
            close_command_expr={"panda_finger_joint.*": FRANKA_CLOSED_POS},
        )


@configclass
class FrankaPickEnvCfg_PLAY(FrankaPickEnvCfg):
    """Small play configuration for GUI teleoperation and smoke testing."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
