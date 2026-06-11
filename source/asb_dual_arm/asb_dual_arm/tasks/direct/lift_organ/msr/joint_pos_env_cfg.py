# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg, DeformableObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, DEFORMABLE_TARGET_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from asb_dual_arm.tasks.direct.lift_organ.lift_organ_env_cfg import LiftOrganEnvCfg
import asb_dual_arm.tasks.direct.lift_organ.mdp as mdp
##
# Pre-defined configs
## 
from asb_dual_arm.config.robot import MSR_PSM_CFG, MSR_PSM_HIGH_PD_CFG   # isort: skip
import numpy as np

##
# Environment configuration
##
from asb_dual_arm import PACKAGE_ROOT
needle_usd_path = str(PACKAGE_ROOT / 'assets' / 'Surgical_needle' / 'needlea.usd')
table_usd_path = str(PACKAGE_ROOT / 'assets' / 'Table' / 'table.usd')
block_usd_path = str(PACKAGE_ROOT / 'assets' /  'Surgical_block' / 'blockb.usd')
needle_w_rope_usd_path = str(PACKAGE_ROOT / 'assets' / 'Surgical_needle' / 'n_w_r_2.usd')
ring_usd_path = str(PACKAGE_ROOT / 'assets' /  'ring_cyl' / 'ring_cyl.usd')
coordinate_marker_usd_path = str(PACKAGE_ROOT / 'assets' /  'others' / 'Coordinate.usd')
organ_usd_path = '/workspace/isaaclab/source/ASBDualArm/source/asb_dual_arm/asb_dual_arm/assets/others/uperc.usd'


@configclass
class MSRPSMLiftOrganNeedleEnvCfg(LiftOrganEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0, -0.457)),
            spawn=UsdFileCfg(usd_path=table_usd_path,
            scale=(1, 1, 1)),
        )
        # switch robot to msr-psm
        self.scene.robot_1 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
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
        self.scene.robot_1.init_state.pos = (-0.2, 0.0, 0.0)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.scene.robot_2 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
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
        self.scene.robot_2.init_state.pos = (0.2, 0.0, 0.0)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # override rewards
        self.rewards.end_effector_1_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]

        # override actions/ JointPositionToLimitsActionCfg/JointPositionActionCfg
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
            clip={
                "psm_shoulder_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_upper_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_fore_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_wrist1_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_wrist2_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_wrist3_Joint": (-np.pi * 2, np.pi * 2), 
                'psm_insertion_Joint': (-0.045, 0.02),
                "psm_roll_Joint": (-3 * np.pi / 2, 3 * np.pi / 2), 
                "psm_pitch_Joint": (-np.pi / 2, np.pi / 2), 
                "psm_yaw_Joint": (-np.pi / 2, np.pi / 2)
                },
            use_default_offset=True
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
                "psm_yaw_Joint"], 
            scale=0.5,
            clip={
                "psm_shoulder_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_upper_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_fore_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_wrist1_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_wrist2_Joint": (-np.pi * 2, np.pi * 2), 
                "psm_wrist3_Joint": (-np.pi * 2, np.pi * 2), 
                'psm_insertion_Joint': (-0.045, 0.02),
                "psm_roll_Joint": (-3 * np.pi / 2, 3 * np.pi / 2), 
                "psm_pitch_Joint": (-np.pi / 2, np.pi / 2), 
                "psm_yaw_Joint": (-np.pi / 2, np.pi / 2)
                },
            use_default_offset=True
        )
        self.actions.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.01, "psm_gripper2_Joint": -0.01},
        )
        self.actions.gripper_2_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.01, "psm_gripper2_Joint": -0.01},
        )

        # Set Needle as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.1, 0.3, 0.025], rot=[0, 0.7071, 0.7071, 0]),
            spawn=UsdFileCfg(
                usd_path=needle_usd_path ,
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = "psm_yaw_Link"
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
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
        # set the scale of the visualization markers to (0.01, 0.01, 0.01)
        self.commands.ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

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
        # set the scale of the visualization markers to (0.01, 0.01, 0.01)
        self.commands.ee_2_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_2_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.commands.lift_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_1",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.25, 0.25),
                pos_y=(0.35, 0.35),
                pos_z=(0.10, 0.10),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(1.57, 1.57),
            ),
        )
        self.commands.lift_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.lift_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.commands.object_pose = mdp.UniformPoseCommandCfg(
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
        self.commands.object_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                'asset_cfg': SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_robot_2_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                'asset_cfg': SceneEntityCfg("robot_2"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.0), "z": (0.055, 0.055)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_base_Link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_Link",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/psm_base_Link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_Link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class MSRPSMLiftOrganNeedlewithRopeEnvCfg(LiftOrganEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0, -0.457)),
            spawn=UsdFileCfg(usd_path=table_usd_path,
            scale=(1, 1, 1)),
        )
        self.scene.coordinate_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CoordinateMarker",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(usd_path=coordinate_marker_usd_path, scale=(0.05, 0.05, 0.05)),
        )
        # Add organ to the scene (as static visual asset)
        self.scene.organ = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Organ",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.2, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071)),
            spawn=UsdFileCfg(
                usd_path=organ_usd_path,
                scale=(0.2, 0.2, 0.2),
            ),
        )
        
        # Track Cube_02 inside organ USD as a RigidObject for real-time pose access
        # spawn=None means the prim already exists in the scene (spawned by organ USD)
        # Cube_02 has RigidBodyAPI in the USD file
        self.scene.cube_02 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Organ/Cube_02",
            spawn=None,  # Don't spawn - already exists in organ USD
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),  # Will use current USD transform
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        
        # Remove object from the scene
        self.scene.object = None
        # switch robot to msr-psm
        self.scene.robot_1 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
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
        self.scene.robot_1.init_state.pos = (-0.35, 0.0, 0.0)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.scene.robot_2 = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
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
        self.scene.robot_2.init_state.pos = (0.35, 0.0, 0.0)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # override rewards
        self.rewards.end_effector_1_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_1_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_2_orientation_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]

        # override actions/ JointPositionToLimitsActionCfg/JointPositionActionCfg
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
                "psm_yaw_Joint"], 
            scale=0.5,
            use_default_offset=True
        )
        self.actions.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.01, "psm_gripper2_Joint": -0.01},
        )
        self.actions.gripper_2_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.01, "psm_gripper2_Joint": -0.01},
        )

        # Note: Command generators (ee_1_pose, ee_2_pose, lift_pose, object_pose) are removed.
        # Robot arm 1 now tracks Cube_02 pose directly via XFormPrim in the state machine.
        self.commands.ee_1_pose = None
        self.commands.ee_2_pose = None
        self.commands.lift_pose = None
        self.commands.object_pose = None

        # Disable observations that depend on removed command generators
        self.observations.policy.pose_1_command = None
        self.observations.policy.pose_1_rel = None
        self.observations.policy.pose_2_command = None
        self.observations.policy.pose_2_rel = None

        # Disable rewards that depend on removed command generators
        self.rewards.end_effector_1_position_tracking = None
        self.rewards.end_effector_1_position_tracking_fine_grained = None
        self.rewards.end_effector_1_orientation_tracking = None
        self.rewards.end_effector_1_orientation_tracking_fine_grained = None
        self.rewards.end_effector_2_position_tracking = None
        self.rewards.end_effector_2_position_tracking_fine_grained = None
        self.rewards.end_effector_2_orientation_tracking = None
        self.rewards.end_effector_2_orientation_tracking_fine_grained = None

        # Disable curriculum terms that depend on removed reward terms
        self.curriculum.end_effector_1_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_1_position_tracking_fine_grained = None
        self.curriculum.end_effector_2_orientation_tracking_fine_grained = None
        self.curriculum.end_effector_2_position_tracking_fine_grained = None

        self.events.reset_robot_1_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                'asset_cfg': SceneEntityCfg("robot_1"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_robot_2_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                'asset_cfg': SceneEntityCfg("robot_2"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_object_to_default,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        # Disable object-related event and termination since object is removed
        self.events.reset_object_position = None
        self.terminations.success = None

        # Note: ee_1_frame and ee_2_frame FrameTransformers are removed.
        # End-effector poses are now obtained directly from robot body data in the state machine.
        self.scene.ee_1_frame = None
        self.scene.ee_2_frame = None


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
class MSRPSMLiftOrganBlockEnvCfg(MSRPSMLiftOrganNeedleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set Block as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.05, 0.35, 0.055], rot=[0, 0.7071, -0.7071, 0]),
            spawn=UsdFileCfg(
                usd_path=block_usd_path,
                scale=(0.012, 0.012, 0.012),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=200,
                    max_linear_velocity=200,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )
        # Set the grasp action
        self.actions.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.08, "psm_gripper2_Joint": -0.08},
        )
        self.actions.gripper_2_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.3, "psm_gripper2_Joint": -0.3},
        )

        self.commands.ee_1_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_1",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.15, 0.15),
                pos_y=(0.40, 0.40),
                pos_z=(0.10, 0.10),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(-1.2, -1.2),
            ),
        )
        # set the scale of the visualization markers to (0.01, 0.01, 0.01)
        self.commands.ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
        self.commands.ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.commands.ee_2_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_2",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-0.15, -0.15),
                pos_y=(0.40, 0.40),
                pos_z=(0.10, 0.10),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(-1.2, -1.2),
            ),
        )
        # set the scale of the visualization markers to (0.01, 0.01, 0.01)
        self.commands.ee_2_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
        self.commands.ee_2_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.commands.lift_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_1",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.25, 0.25),
                pos_y=(0.35+0.05, 0.35+0.05),
                pos_z=(0.10-0.05, 0.10-0.05),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(-1.2, -1.2),
            ),
        )
        self.commands.lift_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
        self.commands.lift_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
        
        self.commands.object_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot_2",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-0.15, -0.15),
                pos_y=(0.40, 0.40),
                pos_z=(0.10, 0.10),
                roll=(3.14, 3.14),
                pitch=(0.0, 0.0),
                yaw=(-1.2, -1.2),
            ),
        )
        self.commands.object_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.terminations.success = DoneTerm(
            func=mdp.object_reached_goal,
            params={
                "robot_cfg": SceneEntityCfg("robot_2"),
                "object_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "grasp_offset": (0.0, -1.2 * 0.012, -0.06 * 0.012, 1.0, 0.0, 0.0, 0.0), 
                "criterion": "pos",
                "pos_threshold": 0.005,
            },
        )

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


##
# New configurations for loading entire scene from USD file (uper.usd)
# All scene elements use spawn=None to reference existing prims in the USD
##


@configclass
class MSRPSMUpe6SingleRobotEnvCfg(LiftOrganEnvCfg):
    """Environment configuration that loads the entire scene from uper.usd.
    
    This configuration uses spawn=None for all scene elements, meaning all assets
    (ground, table, robot, cube, etc.) are pre-defined in the USD file at:
    /workspace/isaaclab/source/ASBDualArm/source/asb_dual_arm/asb_dual_arm/assets/others/uper.usd
    
    Paths in the USD file:
    - Robot: {ENV_REGEX_NS}/Organ/msr_psm (directly under Organ)
    - Table: {ENV_REGEX_NS}/Organ/table (directly under Organ)
    - Ground: {ENV_REGEX_NS}/Organ/GroundPlane (directly under Organ)
    - Cube_02: {ENV_REGEX_NS}/Organ/upe6/Cube_02 (under Organ/upe6)
    
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
        # Joint positions read from USD file (converted from degrees to radians):
        self.scene.robot_1.init_state.joint_pos = {
            "psm_shoulder_Joint": -np.pi / 2,       # -90°
            "psm_upper_Joint": -3 * np.pi / 4,      # -135°
            "psm_fore_Joint": 2 * np.pi / 3,        # 120°
            "psm_wrist1_Joint": 15.0 * np.pi / 180.0,  # 13° from USD
            "psm_wrist2_Joint": np.pi / 2,          # 90°
            "psm_wrist3_Joint": -62.0 * np.pi / 180.0, # -62° from USD
            "psm_insertion_Joint": 0.06,
            "psm_roll_Joint": 0.0,
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.0,
            "psm_gripper2_Joint": 0.0,
        }
        self.scene.robot_1.init_state.pos = (0.35, 0.0, 0.0)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        
        # Robot 2 - disabled for single robot testing
        # Set to a dummy configuration that won't be used
        self.scene.robot_2 = None
        
        # Track Cube_02 inside organ USD as a RigidObject for real-time pose access
        # Cube_02 is under Organ/upe6: {ENV_REGEX_NS}/Organ/upe6/Cube_02
        self.scene.cube_02 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Organ/upe6/Cube_02",
            spawn=None,  # Don't spawn - already exists in organ USD
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),  # Will use current USD transform
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        
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
        self.actions.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            open_command_expr={"psm_gripper1_Joint": 0.5, "psm_gripper2_Joint": -0.5},
            close_command_expr={"psm_gripper1_Joint": 0.01, "psm_gripper2_Joint": -0.01},
        )
        
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
            semantic_filter="*:*",
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
