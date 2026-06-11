# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import TiledCameraCfg
# from isaaclab.markers import VisualizationMarkers
 
import asb_dual_arm.tasks.direct.lift_organ.mdp as mdp

##
# Scene definition
##


@configclass
class LiftOrganSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            # scale=(2, 2, 1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )
    
    ring: AssetBaseCfg | None = None
    coordinate_marker: AssetBaseCfg | None = None
    organ: AssetBaseCfg | RigidObjectCfg | None = None
    
    # Cube_02 inside organ USD for real-time pose tracking
    # This prim already exists in the organ USD and has RigidBody physics
    cube_02: RigidObjectCfg | None = None
    
    # robots
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING

    # end-effector sensor: will be populated by agent env cfg
    ee_1_frame: FrameTransformerCfg | None = None
    ee_2_frame: FrameTransformerCfg | None = None
    
    # camera sensor: will be populated by agent env cfg
    camera: TiledCameraCfg | None = None
    depth_camera: TiledCameraCfg | None = None

    object: RigidObjectCfg | DeformableObjectCfg | None = None

    # rcm marker visualization
    # rcm_marker: VisualizationMarkers | None = None
    # Raise RuntimeError: Pickling of "pxr.UsdGeom.PointInstancer" instances is not enabled 
    # (http://www.boost.org/libs/python/doc/v2/pickle.html)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_1_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_1",
        body_name=MISSING,
        # resampling_time_range=(2.0, 2.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )

    ee_2_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_2",
        body_name=MISSING,
        # resampling_time_range=(2.0, 2.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )

    lift_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), 
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_2",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

 
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_1_action: ActionTerm = MISSING
    gripper_1_action: ActionTerm | None = None

    arm_2_action: ActionTerm = MISSING
    gripper_2_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # joint_1_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot_1")})
        # joint_1_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot_1")})
        pose_1_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_1_pose"})
        ee_1_pose = ObsTerm(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("robot_1", body_names='psm_tool_tip_Link')})
        pose_1_rel = ObsTerm(func=mdp.target_pose_rel, params={"asset_cfg": SceneEntityCfg("robot_1", body_names='psm_tool_tip_Link'), "command_name": "ee_1_pose"})
        jaw_pos_1 = ObsTerm(func=mdp.jaw_pos, params={"robot_cfg": SceneEntityCfg("robot_1")})
        
        # joint_2_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot_2")})
        # joint_2_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot_2")})
        pose_2_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_2_pose"})
        ee_2_pose = ObsTerm(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("robot_2", body_names='psm_tool_tip_Link')})
        pose_2_rel = ObsTerm(func=mdp.target_pose_rel, params={"asset_cfg": SceneEntityCfg("robot_2", body_names='psm_tool_tip_Link'), "command_name": "ee_2_pose"})
        jaw_pos_2 = ObsTerm(func=mdp.jaw_pos, params={"robot_cfg": SceneEntityCfg("robot_2")})
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    reset_robot_1_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_2_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    end_effector_1_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING), "command_name": "ee_1_pose"},
    )
    end_effector_1_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING), "std": 0.2, "command_name": "ee_1_pose"},
    )

    end_effector_1_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING), "command_name": "ee_1_pose"},
    )
    end_effector_1_orientation_tracking_fine_grained = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING), "std": 0.3, "command_name": "ee_1_pose"},
    )

    end_effector_2_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING), "command_name": "ee_2_pose"},
    )
    end_effector_2_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING), "std": 0.2, "command_name": "ee_2_pose"},
    )

    end_effector_2_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING), "command_name": "ee_2_pose"},
    )
    end_effector_2_orientation_tracking_fine_grained = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING), "std": 0.3, "command_name": "ee_2_pose"},
    )
    # action penalty
    action_l2: RewTerm | None = None
    action_rate: RewTerm | None = None
    joint_1_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot_1")},
    )
    joint_2_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot_2")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    success = DoneTerm(
        func=mdp.object_reached_goal,
        params={
            "robot_cfg": SceneEntityCfg("robot_2"),
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "criterion": "pos",
            "pos_threshold": 0.005,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    end_effector_1_orientation_tracking_fine_grained = CurrTerm(
        func=mdp.anneal_reward_param,
        params={
            "term_name": "end_effector_1_orientation_tracking_fine_grained",
            "param_name": "std",
            "start": 0.3,
            "end": 0.15,
            "start_step": 10000,
            "end_step": 20000,
        },
    )
    end_effector_1_position_tracking_fine_grained = CurrTerm(
        func=mdp.anneal_reward_param,
        params={
            "term_name": "end_effector_1_position_tracking_fine_grained",
            "param_name": "std",
            "start": 0.2,
            "end": 0.1,
            "start_step": 10000,
            "end_step": 20000,
        },
    )

    end_effector_2_orientation_tracking_fine_grained = CurrTerm(
        func=mdp.anneal_reward_param,
        params={
            "term_name": "end_effector_2_orientation_tracking_fine_grained",
            "param_name": "std",
            "start": 0.3,
            "end": 0.15,
            "start_step": 10000,
            "end_step": 20000,
        },
    )
    end_effector_2_position_tracking_fine_grained = CurrTerm(
        func=mdp.anneal_reward_param,
        params={
            "term_name": "end_effector_2_position_tracking_fine_grained",
            "param_name": "std",
            "start": 0.2,
            "end": 0.1,
            "start_step": 10000,
            "end_step": 20000,
        },
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_1_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_1_vel", "weight": -0.001, "num_steps": 4500}
    )
    joint_2_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_2_vel", "weight": -0.001, "num_steps": 4500}
    )


# Environment configuration
##


@configclass
class LiftOrganEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: LiftOrganSceneCfg = LiftOrganSceneCfg(num_envs=4096, env_spacing=3)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 4024 * 4024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 4024 * 4024 * 4
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.sim.physx.gpu_max_soft_body_contacts = 4 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 4 * 1024 * 1024
