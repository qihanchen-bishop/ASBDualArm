# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys
from types import MethodType

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--disable_kl_scheduler",
    action="store_true",
    default=False,
    help="Disable KLAdaptiveLR to run a no-scheduler control experiment.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import random
from datetime import datetime

import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import msr.tasks  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _patch_connectivity_tensorboard_prefix(runner: Runner) -> None:
    """Keep connectivity diagnostics under a dedicated top-level TensorBoard group."""
    agent = runner.agent
    if getattr(agent, "_msr_connectivity_tb_prefix_patched", False):
        return

    original_track_data = agent.track_data

    def _track_data_with_connectivity_prefix(tag: str, value):
        if isinstance(tag, str) and tag.startswith("Info / Episode_Connectivity/"):
            tag = tag[len("Info / ") :]
        return original_track_data(tag, value)

    agent.track_data = _track_data_with_connectivity_prefix
    setattr(agent, "_msr_connectivity_tb_prefix_patched", True)


def _configure_kl_scheduler(agent_cfg: dict) -> None:
    """Apply scheduler ablation/limits before runner instantiation."""
    cfg = agent_cfg.get("agent", {})
    scheduler = cfg.get("learning_rate_scheduler", None)

    if args_cli.disable_kl_scheduler:
        cfg["learning_rate_scheduler"] = None
        cfg["learning_rate_scheduler_kwargs"] = None
        print("[INFO] Disabled learning-rate scheduler (--disable_kl_scheduler).")
        return

    if str(scheduler) == "KLAdaptiveLR":
        kwargs = cfg.get("learning_rate_scheduler_kwargs") or {}
        kwargs["max_lr"] = min(float(kwargs.get("max_lr", 1.0e-3)), 1.0e-3)
        cfg["learning_rate_scheduler_kwargs"] = kwargs
        print(f"[INFO] KLAdaptiveLR max_lr is capped at {kwargs['max_lr']:.1e}.")


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _find_non_finite_tensor(obj, prefix: str = "") -> str | None:
    if torch.is_tensor(obj):
        if obj.is_floating_point() and not torch.isfinite(obj).all():
            return prefix or "tensor"
        return None
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = _find_non_finite_tensor(value, f"{prefix}.{key}" if prefix else str(key))
            if child is not None:
                return child
        return None
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            child = _find_non_finite_tensor(value, f"{prefix}[{index}]" if prefix else f"[{index}]")
            if child is not None:
                return child
        return None
    return None


def _find_non_finite_module_state(agent) -> str | None:
    for name, module in getattr(agent, "checkpoint_modules", {}).items():
        if module is None:
            continue
        if hasattr(module, "state_dict"):
            issue = _find_non_finite_tensor(module.state_dict(), str(name))
        else:
            issue = _find_non_finite_tensor(module, str(name))
        if issue is not None:
            return issue
    return None


def _repair_running_standard_scaler(module, module_name: str) -> list[str]:
    """Repair invalid RunningStandardScaler buffers in-place.

    Returns a list of repaired buffer names.
    """
    repaired: list[str] = []

    running_mean = getattr(module, "running_mean", None)
    if torch.is_tensor(running_mean) and running_mean.is_floating_point():
        if not torch.isfinite(running_mean).all():
            module.running_mean = torch.nan_to_num(running_mean, nan=0.0, posinf=0.0, neginf=0.0)
            repaired.append(f"{module_name}.running_mean")

    running_variance = getattr(module, "running_variance", None)
    if torch.is_tensor(running_variance) and running_variance.is_floating_point():
        invalid_variance = (~torch.isfinite(running_variance)) | (running_variance <= 0)
        if invalid_variance.any():
            cleaned = torch.nan_to_num(running_variance, nan=1.0, posinf=1.0, neginf=1.0)
            module.running_variance = torch.clamp(cleaned, min=1.0e-6)
            repaired.append(f"{module_name}.running_variance")

    current_count = getattr(module, "current_count", None)
    if torch.is_tensor(current_count) and current_count.is_floating_point():
        invalid_count = (~torch.isfinite(current_count)) | (current_count <= 0)
        if invalid_count.any():
            cleaned = torch.nan_to_num(current_count, nan=1.0, posinf=1.0, neginf=1.0)
            module.current_count = torch.clamp(cleaned, min=1.0)
            repaired.append(f"{module_name}.current_count")

    return repaired


def _repair_known_non_finite_modules(agent) -> list[str]:
    """Repair known non-finite states that can be safely recovered in-place."""
    repaired: list[str] = []
    for name, module in getattr(agent, "checkpoint_modules", {}).items():
        if module is None:
            continue
        # skrl RunningStandardScaler used by value/state preprocessors.
        if hasattr(module, "running_mean") and hasattr(module, "running_variance") and hasattr(module, "current_count"):
            repaired.extend(_repair_running_standard_scaler(module, str(name)))
    return repaired


def _find_non_finite_tracked_metric(agent) -> tuple[str, float] | None:
    tags = (
        "Loss / Policy loss",
        "Loss / Value loss",
        "Loss / Entropy loss",
        "Policy / Standard deviation",
    )
    for tag in tags:
        values = agent.tracking_data.get(tag, None)
        if not values:
            continue
        value = values[-1]
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = float(value.detach().item())
            else:
                value = float(value.detach().float().mean().item())
        if not _is_finite_number(value):
            try:
                return tag, float(value)
            except Exception:
                return tag, float("nan")
    return None


def _patch_finite_guard(runner: Runner, log_dir: str) -> None:
    """Abort training on non-finite losses/parameters and save last-good checkpoint."""
    if not args_cli.ml_framework.startswith("torch"):
        return

    agent = runner.agent
    if getattr(agent, "_msr_non_finite_guard_patched", False):
        return

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_good_path = os.path.join(checkpoint_dir, "last_good_agent.pt")

    original_post_interaction = agent.post_interaction
    rollouts = max(1, int(getattr(agent, "_rollouts", 1)))
    learning_starts = int(getattr(agent, "_learning_starts", 0))

    def _guarded_post_interaction(self, timestep: int, timesteps: int) -> None:
        will_update = ((int(getattr(self, "_rollout", 0)) + 1) % rollouts == 0) and timestep >= learning_starts

        if will_update:
            repaired_before = _repair_known_non_finite_modules(self)
            if repaired_before:
                print(
                    f"[WARN] Repaired non-finite preprocessor state before update at step {timestep + 1}: "
                    f"{', '.join(repaired_before)}"
                )

            pre_issue = _find_non_finite_module_state(self)
            if pre_issue is not None:
                raise RuntimeError(
                    f"[FiniteGuard] Non-finite state before update at step {timestep + 1}: {pre_issue}."
                )
            self.save(last_good_path)

        original_post_interaction(timestep=timestep, timesteps=timesteps)

        if not will_update:
            return

        repaired_after = _repair_known_non_finite_modules(self)
        if repaired_after:
            print(
                f"[WARN] Repaired non-finite preprocessor state after update at step {timestep + 1}: "
                f"{', '.join(repaired_after)}"
            )

        metric_issue = _find_non_finite_tracked_metric(self)
        module_issue = _find_non_finite_module_state(self)
        if metric_issue is None and module_issue is None:
            return

        bad_path = os.path.join(checkpoint_dir, f"non_finite_agent_t{timestep + 1}.pt")
        try:
            self.save(bad_path)
        except Exception as exc:
            print(f"[WARN] Failed to save non-finite snapshot: {exc}")
            bad_path = "<save_failed>"

        if metric_issue is not None:
            reason = f"non-finite tracked metric {metric_issue[0]}={metric_issue[1]}"
        else:
            reason = f"non-finite module state at {module_issue}"

        raise RuntimeError(
            f"[FiniteGuard] {reason}. Training stopped at step {timestep + 1}. "
            f"Last good checkpoint: {last_good_path}. Failing snapshot: {bad_path}."
        )

    agent.post_interaction = MethodType(_guarded_post_interaction, agent)
    setattr(agent, "_msr_non_finite_guard_patched", True)
    print(f"[INFO] Enabled finite guard. Last-good checkpoint: {last_good_path}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # configure LR scheduler safeguards / ablation flags
    _configure_kl_scheduler(agent_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    _patch_connectivity_tensorboard_prefix(runner)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    _patch_finite_guard(runner, log_dir)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
