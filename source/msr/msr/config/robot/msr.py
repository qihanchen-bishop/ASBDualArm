# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the modular surgical research platform.

The following configurations are available:

* :obj:`MSR_PSM_CFG`: MSR_PSM robot with gripper
* :obj:`MSR_PSM_HIGH_PD_CFG`: MSR_PSM with gripper with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""
from msr import PACKAGE_ROOT
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import numpy as np

##
# Configuration
##

# usd_path = str(PACKAGE_ROOT / 'assets' / 'MSR_model' / 'MSR-URDF-Mortorpack-0522-3.usd')
usd_path = str(PACKAGE_ROOT / 'assets' / 'msr_psm' / 'msr_psm.usd')

MSR_PSM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=(1, 1, 1),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.0001,
        #     rest_offset=0.0001
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "psm_shoulder_Joint": np.pi / 2,
            "psm_upper_Joint": -np.pi / 8,
            "psm_fore_Joint": -3. / 4. * np.pi,
            "psm_wrist1_Joint": np.pi,
            "psm_wrist2_Joint": -np.pi / 2,
            "psm_wrist3_Joint": -2. / 3. * np.pi,
            "psm_insertion_Joint": 0.02,
            "psm_roll_Joint": 0.0,
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.5,
            "psm_gripper2_Joint": -0.5,
        },
    ),
    actuators={
        "psm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "psm_shoulder_Joint",
                "psm_upper_Joint",
                "psm_fore_Joint",
                "psm_wrist1_Joint",
                "psm_wrist2_Joint",
                "psm_wrist3_Joint",
            ],
            # effort_limit_sim=870.0,
            stiffness=500.0,
            damping=80.0,
            friction=0.0,
            armature=0.0,
        ),
        "psm_insertion": ImplicitActuatorCfg(
            joint_names_expr=["psm_insertion_Joint"],
            # effort_limit=2000,
            stiffness=800.0,
            damping=80.0,
            friction=0.0,
            armature=0.0,
        ),
        "psm_tool_rotation": ImplicitActuatorCfg(
            joint_names_expr=["psm_roll_Joint", "psm_pitch_Joint", "psm_yaw_Joint"],
            # effort_limit=800,
            # velocity_limit_sim=np.pi,
            stiffness=500.0,
            damping=80.0,
            friction=0.0,
            armature=0.0,
        ),
        "psm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["psm_gripper[1-2]_Joint"],
            # effort_limit=2000,
            # velocity_limit_sim=np.pi/2,
            stiffness=500.0, #100,
            damping=30, #0.1,
            friction=0.0,
            armature=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of MSR PSM robot."""


MSR_PSM_HIGH_PD_CFG = MSR_PSM_CFG.copy()
MSR_PSM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MSR_PSM_HIGH_PD_CFG.actuators["psm_arm"].stiffness = 4000.0 # 3000.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_arm"].damping = 200.0 # 20.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_insertion"].stiffness = 4000.0 # 4000.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_insertion"].damping = 200.0 # 50.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_tool_rotation"].stiffness = 4000.0 # 100.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_tool_rotation"].damping = 200.0 # 0.1
MSR_PSM_HIGH_PD_CFG.actuators["psm_gripper"].stiffness = 5000.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_gripper"].damping = 50.0
"""Configuration of MSR PSM robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""