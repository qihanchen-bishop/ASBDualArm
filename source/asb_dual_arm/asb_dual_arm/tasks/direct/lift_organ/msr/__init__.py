# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-LiftOrgan-Needle-MSRPSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMLiftOrganNeedleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-LiftOrgan-Block-MSRPSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMLiftOrganBlockEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-LiftOrgan-Needle-MSRPSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMLiftOrganNeedleEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-LiftOrgan-Block-MSRPSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMLiftOrganBlockEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-LiftOrgan-Needle-MSRPSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:MSRPSMLiftOrganNeedleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Block-MSRPSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:MSRPSMLiftOrganBlockEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Needle-MSRPSM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:MSRPSMLiftOrganNeedleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Block-MSRPSM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:MSRPSMLiftOrganBlockEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-LiftOrgan-Needle-MSRPSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:MSRPSMLiftOrganNeedleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),

    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Block-MSRPSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:MSRPSMLiftOrganBlockEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganBlockPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),

    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Needle-MSRPSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:MSRPSMLiftOrganNeedleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Needle_with_Rope-MSRPSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:MSRPSMLiftOrganNeedlewithRopeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Block-MSRPSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:MSRPSMLiftOrganBlockEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MSRPSMLiftOrganNeedlePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),

    },
    disable_env_checker=True,
)

##
# New environment configurations for loading entire scene from USD file (upe6.usd)
# Single robot configuration for testing
##

gym.register(
    id="Isaac-LiftOrgan-Upe6-SingleRobot-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MSRPSMUpe6SingleRobotEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LiftOrgan-Upe6-SingleRobot-JointPos-Play-v0",
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
    id="Isaac-VesselSem-SingleRobot-IK-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSem-SingleRobot-IK-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSem-SingleRobot-IK-RCM-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VesselSem-SingleRobot-IK-RCM-Play-v0",
    entry_point=f"{__name__}.vessel_env:VesselSemEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_vessel_env_cfg:MSRPSMVesselSemRewardWithRCMEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_vessel_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

