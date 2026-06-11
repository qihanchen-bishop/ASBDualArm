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
# New environment configurations for loading entire scene from USD file (dual_arm.usd)
# Dual-arm configuration for testing
##

gym.register(
    id="Isaac-DualArm-Upe6-DualArm-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMUpe6DualArmEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-Upe6-DualArm-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMUpe6DualArmEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualSOArm-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_so_arm_env_cfg:DualSOArmEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualSOArm-IK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_so_arm_env_cfg:DualSOArmEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

##
# Vessel Semantic Segmentation Reward Environment
# Single robot with IK action and semantic segmentation based reward
##

gym.register(
    id="Isaac-DualArm-VesselSem-IK-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-RCM-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-RCM-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-ConnectivityOnly-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardConnectivityOnlyEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-ConnectivityOnly-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardConnectivityOnlyEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-RCM-ConnectivityOnly-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMConnectivityOnlyEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-IK-RCM-ConnectivityOnly-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMConnectivityOnlyEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


##
# Dual-arm vessel semantic IK variants (both arms active)
##

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-RCM-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardWithRCMEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-RCM-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardWithRCMEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-ConnectivityOnly-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardConnectivityOnlyEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-ConnectivityOnly-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardConnectivityOnlyEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-RCM-ConnectivityOnly-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardWithRCMConnectivityOnlyEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-DualArm-VesselSem-DualArm-IK-RCM-ConnectivityOnly-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMDualArmVesselSemRewardWithRCMConnectivityOnlyEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
