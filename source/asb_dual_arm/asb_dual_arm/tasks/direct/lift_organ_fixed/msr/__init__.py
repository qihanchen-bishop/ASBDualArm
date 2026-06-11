# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# New environment configurations for loading entire scene from USD file (upe6.usd)
# Single robot configuration for testing
##

gym.register(
    id="Isaac-LiftOrganFixed-Upe6-SingleRobot-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMUpe6SingleRobotEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrganFixed-Upe6-SingleRobot-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMUpe6SingleRobotEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

##
# Vessel Semantic Segmentation Reward Environment
# Single robot with IK action and semantic segmentation based reward
##

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-RCM-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-RCM-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-ConnectivityOnly-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardConnectivityOnlyEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-ConnectivityOnly-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardConnectivityOnlyEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-RCM-ConnectivityOnly-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMConnectivityOnlyEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSemFixed-SingleRobot-IK-RCM-ConnectivityOnly-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMConnectivityOnlyEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

