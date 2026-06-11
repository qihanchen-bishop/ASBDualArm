# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate a trained skrl policy checkpoint on vessel connectivity task.

Metrics reported:
- success rate
- average steps to success
- safety metrics based on gripper_edge_penalty
- maximum consecutive connected frames
- vessel trisection success count

By default this script evaluates 100 episodes with 1 environment and enforces
an external max action-step limit per episode.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from statistics import mean
from contextlib import nullcontext

try:
    import yaml
except Exception:
    yaml = None

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained skrl checkpoint.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
    "--num_test_episodes",
    type=int,
    default=100,
    help="Number of test episodes.",
)
parser.add_argument(
    "--max_action_steps",
    type=int,
    default=None,
    help=(
        "Maximum action steps per episode. If omitted, the script uses the environment episode horizon. "
        "If provided, the environment timeout horizon is synchronized to this value."
    ),
)
parser.add_argument(
    "--stable_frames",
    type=int,
    required=True,
    help="Required. Stable connectivity K-frames used as success condition during evaluation.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate for evaluation. Recommended: 1.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/workspace/isaaclab/logs/skrl/testresult",
    help="Directory to store test results.",
)
parser.add_argument(
    "--save_success_video",
    action="store_true",
    help="Save RGB camera video for successful episodes.",
)
parser.add_argument(
    "--save_all_success_videos",
    action="store_true",
    help="If enabled, save videos for all successful episodes; otherwise only the first one.",
)
parser.add_argument(
    "--success_video_fps",
    type=int,
    default=30,
    help="FPS for saved success videos.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print per-frame trisection/connectivity debug status.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse args
args_cli, hydra_args = parser.parse_known_args()

# Force GUI by default unless `--headless` is explicitly requested.
if not getattr(args_cli, "headless", False):
    os.environ["HEADLESS"] = "0"

# This evaluation task relies on camera observations/rewards.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

# Keep GUI as the default mode. If no display server is available, warn and
# let the user explicitly choose headless mode via `--headless`.
if (not getattr(args_cli, "headless", False)) and os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
    print("[WARNING] No display server detected. GUI mode may fail. Use --headless if running remotely.")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import skrl
from packaging import version

try:
    import cv2 as _cv2

    _HAS_CV2 = True
except Exception:
    _cv2 = None
    _HAS_CV2 = False

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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import msr.tasks  # noqa: F401
import msr.tasks.direct.lift_organ_fixed.mdp as task_mdp
import msr.tasks.direct.lift_organ_fixed.mdp.rewards as task_mdp_rewards

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _sanitize_name(text: str) -> str:
    return text.replace(":", "_").replace("/", "_").replace(" ", "_")


def _align_model_cfg_with_checkpoint(experiment_cfg: dict, resume_path: str) -> None:
    """Align model architecture with checkpoint-time config to avoid state_dict mismatches."""
    if yaml is None:
        return

    run_dir = os.path.dirname(os.path.dirname(resume_path))
    params_agent_path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(params_agent_path):
        return

    try:
        with open(params_agent_path, "r", encoding="utf-8") as f:
            saved_agent_cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[WARNING] Failed to read checkpoint agent config: {params_agent_path} ({exc})")
        return

    saved_models = saved_agent_cfg.get("models", None)
    if not isinstance(saved_models, dict):
        return

    current_models = experiment_cfg.get("models", None)
    if current_models != saved_models:
        current_sep = current_models.get("separate", None) if isinstance(current_models, dict) else None
        saved_sep = saved_models.get("separate", None)
        print(
            "[INFO] Overriding test model architecture with checkpoint params/agent.yaml "
            f"(current separate={current_sep}, checkpoint separate={saved_sep})"
        )
        experiment_cfg["models"] = saved_models


def _to_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return bool(value[0])
    return bool(value)


def _override_stable_frames(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, stable_frames: int) -> None:
    stable = int(max(1, stable_frames))

    rewards_cfg = getattr(env_cfg, "rewards", None)
    reward_terms_with_stable_frames = (
        "vessel_gall_hard_connectivity",
        "progress_connectivity",
        "hold_connectivity",
        "break_connectivity",
        "done_bonus_connectivity",
    )
    for reward_name in reward_terms_with_stable_frames:
        rew_term = getattr(rewards_cfg, reward_name, None)
        if rew_term is not None and hasattr(rew_term, "params") and isinstance(rew_term.params, dict):
            if "stable_frames" in rew_term.params:
                rew_term.params["stable_frames"] = stable

    term_success = getattr(getattr(env_cfg, "terminations", None), "success", None)
    if term_success is not None and hasattr(term_success, "params") and isinstance(term_success.params, dict):
        if "stable_frames" in term_success.params:
            term_success.params["stable_frames"] = stable

    # Force deterministic evaluation criterion: do not let curriculum override
    # the CLI-specified stable frame threshold during env resets.
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None and hasattr(curriculum_cfg, "connectivity_stable_frames"):
        setattr(curriculum_cfg, "connectivity_stable_frames", None)


def _clear_vessel_connectivity_runtime_state(runtime_env) -> None:
    """Clear connectivity counters/caches that can leak across episode resets."""
    base_env = getattr(runtime_env, "unwrapped", runtime_env)

    counter = getattr(base_env, "_vessel_cd_gall_counter", None)
    if torch.is_tensor(counter):
        # Counter can be created during env.step under torch.inference_mode().
        # Replace it with a fresh regular tensor instead of in-place writes.
        setattr(
            base_env,
            "_vessel_cd_gall_counter",
            torch.zeros(counter.shape, device=counter.device, dtype=counter.dtype),
        )

    setattr(base_env, "_vessel_cd_gall_counter_fp", None)

    # Invalidate trisection cache to avoid stale frame reuse at episode start.
    if hasattr(base_env, "_trisect_cache"):
        setattr(base_env, "_trisect_cache", None)
    if hasattr(base_env, "_trisect_fp"):
        setattr(base_env, "_trisect_fp", None)
    if hasattr(base_env, "_trisect_step"):
        setattr(base_env, "_trisect_step", 0)


def _get_env_scalar_int(runtime_env, attr_name: str, default: int = 0) -> int:
    """Read a scalar integer attribute from the underlying environment."""
    base_env = getattr(runtime_env, "unwrapped", runtime_env)
    value = getattr(base_env, attr_name, None)

    if torch.is_tensor(value):
        if value.numel() == 0:
            return int(default)
        return int(value.detach().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)) and len(value) > 0:
        try:
            return int(value[0])
        except Exception:
            return int(default)
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _extract_rgb_frame_uint8(obs, runtime_env) -> np.ndarray | None:
    """Extract a single RGB frame as HWC uint8 numpy array."""
    candidate = None

    policy_obs = obs.get("policy", None) if isinstance(obs, dict) else obs
    if isinstance(policy_obs, dict) and "rgb_image" in policy_obs:
        candidate = policy_obs["rgb_image"]
    elif torch.is_tensor(policy_obs) and policy_obs.ndim == 4 and policy_obs.shape[-1] in (3, 4):
        candidate = policy_obs

    if candidate is None:
        base_env = getattr(runtime_env, "unwrapped", runtime_env)
        try:
            camera = base_env.scene["camera"]
            candidate = camera.data.output.get("rgb", None)
        except Exception:
            candidate = None

    if candidate is None:
        return None

    if torch.is_tensor(candidate):
        frame = candidate[0] if candidate.ndim == 4 else candidate
        frame = frame.detach().to(torch.float32).cpu()
        if frame.ndim != 3:
            return None
        if frame.shape[-1] not in (3, 4) and frame.shape[0] in (3, 4):
            frame = frame.permute(1, 2, 0)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            return None
        if float(frame.max().item()) <= 1.0:
            frame = frame * 255.0
        frame = torch.clamp(frame, 0.0, 255.0).to(torch.uint8).numpy()
    else:
        frame = np.asarray(candidate)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3:
            return None
        if frame.shape[-1] not in (3, 4) and frame.shape[0] in (3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            return None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

    return frame


def _save_rgb_video(frames_rgb: list[np.ndarray], video_path: str, fps: int) -> bool:
    """Save RGB frames to mp4 using OpenCV."""
    if len(frames_rgb) == 0 or not _HAS_CV2:
        return False

    first = frames_rgb[0]
    if first.ndim != 3 or first.shape[-1] != 3:
        return False

    height, width = int(first.shape[0]), int(first.shape[1])
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    writer = _cv2.VideoWriter(
        video_path,
        _cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (width, height),
    )
    if not writer.isOpened():
        return False

    try:
        for frame in frames_rgb:
            if frame.ndim != 3 or frame.shape[-1] != 3:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = _cv2.resize(frame, (width, height), interpolation=_cv2.INTER_AREA)
            writer.write(_cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return True


def _get_gripper_penalty_params(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> dict:
    default_params = {
        "camera_cfg_name": "camera",
        "gripper_color": (0, 0, 255),
        "color_tolerance": 20,
        "edge_safe_margin_px": 5.0,
    }

    term = getattr(getattr(env_cfg, "rewards", None), "gripper_edge_penalty", None)
    if term is None or not hasattr(term, "params") or not isinstance(term.params, dict):
        return default_params

    params = dict(term.params)
    for key, value in default_params.items():
        params.setdefault(key, value)
    return params


def _get_connectivity_eval_params(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> dict:
    default_params = {
        "camera_cfg_name": "camera",
        "vessel_label": "vessel",
        "gall_label": "gall",
        "vessel_color": (25, 82, 255),
        "gall_color": (255, 105, 180),
        "color_tolerance": 10,
        "prefer_semantic_info": True,
        "gall_dilation_radius": 8,
        "compute_every": 1,
    }

    term = getattr(getattr(env_cfg, "terminations", None), "success", None)
    params = dict(default_params)
    if term is not None and hasattr(term, "params") and isinstance(term.params, dict):
        params.update({k: v for k, v in term.params.items() if k in default_params})
    return params


def _get_frame_trisection_connectivity_status(runtime_env, connectivity_eval_params: dict) -> tuple[bool, bool]:
    base_env = getattr(runtime_env, "unwrapped", runtime_env)
    results = getattr(base_env, "_trisect_cache", None)

    if not isinstance(results, list) or len(results) == 0:
        try:
            results = task_mdp_rewards._get_trisection_cached(
                base_env,
                connectivity_eval_params["camera_cfg_name"],
                connectivity_eval_params["vessel_label"],
                connectivity_eval_params["gall_label"],
                bool(connectivity_eval_params["prefer_semantic_info"]),
                tuple(connectivity_eval_params["vessel_color"]),
                tuple(connectivity_eval_params["gall_color"]),
                int(connectivity_eval_params["color_tolerance"]),
                int(connectivity_eval_params["gall_dilation_radius"]),
                int(max(1, connectivity_eval_params["compute_every"])),
            )
        except Exception:
            return False, False

    if not isinstance(results, list) or len(results) == 0:
        return False, False

    result0 = results[0] if isinstance(results[0], dict) else {}
    tri_success = bool(result0.get("success", False))
    connect_success = bool(task_mdp_rewards._is_only_cd_connected(result0)) if isinstance(result0, dict) else False
    return tri_success, connect_success


def _deterministic_actions(outputs, env):
    # - multi-agent (deterministic) actions
    if hasattr(env, "possible_agents"):
        return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
    # - single-agent (deterministic) actions
    return outputs[-1].get("mean_actions", outputs[0])


def _get_inference_context(ml_framework: str):
    if ml_framework.startswith("torch"):
        return torch.inference_mode
    return nullcontext


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Evaluate a skrl policy checkpoint."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # Keep single-env evaluation for strict episode-level accounting.
    if args_cli.num_envs != 1:
        print(
            f"[WARNING] Overriding num_envs={args_cli.num_envs} to 1 for accurate per-episode testing."
        )
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Keep evaluation success criterion aligned with training termination:
    # hard connectivity stable for K frames.
    _override_stable_frames(env_cfg, args_cli.stable_frames)

    # set seeds
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    # Resolve step-time relationship and synchronize external/internal horizons.
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 0.0))
    decimation = int(max(1, getattr(env_cfg, "decimation", 1)))
    action_dt = sim_dt * float(decimation) if sim_dt > 0.0 else 0.0

    env_episode_action_steps = None
    episode_length_s = float(getattr(env_cfg, "episode_length_s", 0.0))
    if action_dt > 0.0 and episode_length_s > 0.0:
        env_episode_action_steps = int(round(episode_length_s / action_dt))

    max_action_steps_source = "env-default"
    if args_cli.max_action_steps is None:
        if env_episode_action_steps is None:
            max_action_steps = 600
            max_action_steps_source = "fallback-600"
            print(
                "[WARNING] Could not infer env episode horizon from episode_length_s/sim.dt/decimation. "
                "Falling back to max_action_steps=600."
            )
        else:
            max_action_steps = int(max(1, env_episode_action_steps))
    else:
        max_action_steps = int(max(1, args_cli.max_action_steps))
        max_action_steps_source = "cli-override"
        if action_dt > 0.0:
            env_cfg.episode_length_s = float(max_action_steps) * action_dt
            episode_length_s = float(env_cfg.episode_length_s)
            env_episode_action_steps = int(max_action_steps)
        else:
            print(
                "[WARNING] --max_action_steps was provided but sim.dt is unavailable; "
                "cannot synchronize env timeout horizon exactly."
            )

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # checkpoint resolve
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Pre-trained checkpoint unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

    # Keep test-time model architecture consistent with the checkpoint run.
    # This avoids state_dict key mismatches when local task config has changed
    # (e.g. shared vs separate policy/value model heads).
    _align_model_cfg_with_checkpoint(experiment_cfg, resume_path)

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    task_tag = _sanitize_name(args_cli.task.split(":")[-1])
    ckpt_tag = _sanitize_name(os.path.splitext(os.path.basename(resume_path))[0])

    # create env
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent if needed by algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # build runner and load checkpoint
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # safety term params
    gripper_penalty_params = _get_gripper_penalty_params(env_cfg)
    connectivity_eval_params = _get_connectivity_eval_params(env_cfg)

    # evaluate
    episode_records: list[dict] = []
    total_edge_sum = 0.0
    total_edge_steps = 0
    total_edge_violations = 0
    total_trisection_success_count = 0
    total_connect_success_count = 0
    inference_context = _get_inference_context(args_cli.ml_framework)

    target_episodes = int(max(1, args_cli.num_test_episodes))
    success_video_fps = int(max(1, args_cli.success_video_fps))
    save_success_video = bool(args_cli.save_success_video or args_cli.save_all_success_videos)
    save_all_success_videos = bool(args_cli.save_all_success_videos)
    success_video_dir = os.path.join(args_cli.output_dir, "success_videos")
    saved_success_video_paths: list[str] = []

    os.makedirs(args_cli.output_dir, exist_ok=True)
    if save_success_video and not _HAS_CV2:
        print("[WARNING] --save_success_video is enabled but OpenCV is unavailable. Video saving disabled.")
        save_success_video = False
    if save_success_video:
        os.makedirs(success_video_dir, exist_ok=True)

    print(
        "[INFO] Start testing: "
        f"episodes={target_episodes}, max_action_steps={max_action_steps}, "
        f"max_action_steps_source={max_action_steps_source}, stable_frames={int(max(1, args_cli.stable_frames))}"
    )
    if env_episode_action_steps is not None:
        print(
            "[INFO] Env episode horizon: "
            f"{env_episode_action_steps} action steps "
            f"(episode_length_s={episode_length_s}, sim.dt={sim_dt}, decimation={decimation})"
        )
        if max_action_steps != env_episode_action_steps:
            print(
                "[WARNING] External max_action_steps and env episode horizon differ. "
                f"external={max_action_steps}, env={env_episode_action_steps}."
            )

    for episode_idx in range(1, target_episodes + 1):
        if not simulation_app.is_running():
            print("[WARNING] Simulation closed before all episodes finished.")
            break

        obs, _ = env.reset()
        _clear_vessel_connectivity_runtime_state(env)

        action_steps = 0
        terminated = False
        truncated = False
        force_timeout = False
        edge_values: list[float] = []
        episode_trisection_success_count = 0
        episode_connect_success_count = 0
        # Local strict-consecutive counter is robust against env auto-reset on done.
        episode_strict_streak = 0
        episode_max_strict_streak = 0
        capture_this_episode = save_success_video and (save_all_success_videos or len(saved_success_video_paths) == 0)
        episode_rgb_frames: list[np.ndarray] = []

        if capture_this_episode:
            start_frame = _extract_rgb_frame_uint8(obs, env)
            if start_frame is not None:
                episode_rgb_frames.append(start_frame)

        while simulation_app.is_running() and action_steps < max_action_steps:
            with inference_context():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = _deterministic_actions(outputs, env)
                obs, _, terminated_t, truncated_t, _ = env.step(actions)

            action_steps += 1
            terminated = _to_bool(terminated_t)
            truncated = _to_bool(truncated_t)

            if capture_this_episode:
                frame = _extract_rgb_frame_uint8(obs, env)
                if frame is not None:
                    episode_rgb_frames.append(frame)

            # Evaluate safety indicator from reward term definition.
            edge_penalty = task_mdp.gripper_edge_penalty(env.unwrapped, **gripper_penalty_params)
            edge_value = float(edge_penalty.reshape(-1)[0].item())
            edge_values.append(edge_value)
            total_edge_sum += edge_value
            total_edge_steps += 1
            if edge_value < 0.0:
                total_edge_violations += 1

            tri_success_frame, connect_success_frame = _get_frame_trisection_connectivity_status(env, connectivity_eval_params)
            if tri_success_frame:
                episode_trisection_success_count += 1
                total_trisection_success_count += 1
            if connect_success_frame:
                episode_connect_success_count += 1
                total_connect_success_count += 1

            if connect_success_frame:
                episode_strict_streak += 1
            else:
                episode_strict_streak = 0
            if episode_strict_streak > episode_max_strict_streak:
                episode_max_strict_streak = episode_strict_streak

            strict_counter_now = _get_env_scalar_int(env, "_vessel_cd_gall_counter", default=0)
            episode_max_counter_now = _get_env_scalar_int(env, "_vcd_episode_max_streak", default=0)

            if args_cli.debug:
                print(
                    f"frame_id:{action_steps:04d} tri:{int(tri_success_frame)} connect:{int(connect_success_frame)} "
                    f"streak:{episode_strict_streak}/{int(max(1, args_cli.stable_frames))} "
                    f"max_streak:{episode_max_strict_streak} "
                    f"env_streak:{strict_counter_now} env_max:{episode_max_counter_now}"
                )

            if terminated or truncated:
                break

        if (not terminated) and (not truncated) and action_steps >= max_action_steps:
            force_timeout = True

        success = bool(terminated and (not truncated))
        episode_max_connected_frames_env = _get_env_scalar_int(env, "_vcd_episode_max_streak", default=0)
        episode_max_connected_frames = max(int(episode_max_connected_frames_env), int(episode_max_strict_streak))

        # Success implies stable-frames condition reached; env counters may already be reset.
        if success:
            episode_max_connected_frames = max(episode_max_connected_frames, int(max(1, args_cli.stable_frames)))

        if (not success) and episode_connect_success_count > 0 and episode_max_connected_frames == 0:
            print(
                "[WARNING] connect_success_count > 0 but max_consecutive_connected_frames == 0. "
                "This usually indicates connectivity debug status and strict-streak state are not aligned."
            )

        success_video_path = None

        if success and capture_this_episode and len(episode_rgb_frames) > 0:
            video_name = f"{task_tag}_{ckpt_tag}_ep{episode_idx:03d}_steps{action_steps}.mp4"
            success_video_path = os.path.join(success_video_dir, video_name)
            if _save_rgb_video(episode_rgb_frames, success_video_path, success_video_fps):
                saved_success_video_paths.append(success_video_path)
                file_size = os.path.getsize(success_video_path) if os.path.exists(success_video_path) else -1
                print(f"[INFO] Saved success video: {success_video_path} (bytes={file_size})")
            else:
                print(f"[WARNING] Failed to save success video for episode {episode_idx}.")
                success_video_path = None

        edge_mean = mean(edge_values) if len(edge_values) > 0 else 0.0
        edge_min = min(edge_values) if len(edge_values) > 0 else 0.0
        edge_violation_ratio = (
            float(sum(1 for v in edge_values if v < 0.0)) / float(len(edge_values)) if len(edge_values) > 0 else 0.0
        )

        record = {
            "episode": episode_idx,
            "success": success,
            "steps": action_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "force_timeout": bool(force_timeout),
            "gripper_edge_penalty_mean": edge_mean,
            "gripper_edge_penalty_min": edge_min,
            "gripper_edge_violation_ratio": edge_violation_ratio,
            "max_consecutive_connected_frames": episode_max_connected_frames,
            "vessel_trisection_success_count": int(episode_trisection_success_count),
            "connect_success_count": int(episode_connect_success_count),
            "success_video_path": success_video_path,
        }
        episode_records.append(record)

        print(
            f"[EP {episode_idx:03d}] success={success} steps={action_steps} "
            f"terminated={terminated} truncated={truncated} force_timeout={force_timeout} "
            f"edge_mean={edge_mean:.6f} edge_min={edge_min:.6f} "
            f"max_connected_frames={episode_max_connected_frames} "
            f"tri_success_count={episode_trisection_success_count} connect_success_count={episode_connect_success_count}"
        )

    env.close()

    # summarize
    num_episodes = len(episode_records)
    num_success = sum(1 for r in episode_records if r["success"])
    success_rate = (float(num_success) / float(num_episodes)) if num_episodes > 0 else 0.0

    successful_steps = [int(r["steps"]) for r in episode_records if r["success"]]
    avg_steps_success = float(mean(successful_steps)) if len(successful_steps) > 0 else None

    all_steps = [int(r["steps"]) for r in episode_records]
    avg_steps_all = float(mean(all_steps)) if len(all_steps) > 0 else None

    forced_timeouts = sum(1 for r in episode_records if r["force_timeout"])
    max_consecutive_connected_frames = (
        max(int(r["max_consecutive_connected_frames"]) for r in episode_records) if len(episode_records) > 0 else 0
    )

    edge_mean_all_steps = (total_edge_sum / float(total_edge_steps)) if total_edge_steps > 0 else 0.0
    edge_violation_rate = (
        float(total_edge_violations) / float(total_edge_steps) if total_edge_steps > 0 else 0.0
    )

    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "seed": args_cli.seed,
        "stable_frames": int(max(1, args_cli.stable_frames)),
        "max_action_steps": max_action_steps,
        "max_action_steps_source": max_action_steps_source,
        "num_test_episodes": num_episodes,
        "num_success": num_success,
        "vessel_trisection_success_count": int(total_trisection_success_count),
        "connect_success_count": int(total_connect_success_count),
        "success_rate": success_rate,
        "avg_steps_success": avg_steps_success,
        "avg_steps_all": avg_steps_all,
        "forced_timeouts": forced_timeouts,
        "max_consecutive_connected_frames": max_consecutive_connected_frames,
        "env_episode_action_steps": env_episode_action_steps,
        "gripper_edge_penalty_mean_all_steps": edge_mean_all_steps,
        "gripper_edge_violation_rate": edge_violation_rate,
        "num_success_videos": len(saved_success_video_paths),
        "first_success_video_path": saved_success_video_paths[0] if len(saved_success_video_paths) > 0 else None,
        "success_video_paths": saved_success_video_paths,
    }

    # output dir and files
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_prefix = f"{timestamp}_{task_tag}_{ckpt_tag}"

    json_path = os.path.join(args_cli.output_dir, f"{out_prefix}.json")
    txt_path = os.path.join(args_cli.output_dir, f"{out_prefix}.txt")

    payload = {
        "summary": summary,
        "episodes": episode_records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "========== Test Summary ==========",
        f"task: {summary['task']}",
        f"checkpoint: {summary['checkpoint']}",
        f"num_test_episodes: {summary['num_test_episodes']}",
        f"env_episode_action_steps: {summary['env_episode_action_steps']}",
        f"stable_frames: {summary['stable_frames']}",
        f"max_action_steps: {summary['max_action_steps']}",
        f"max_action_steps_source: {summary['max_action_steps_source']}",
        f"num_success: {summary['num_success']}",
        f"vessel_trisection_success_count: {summary['vessel_trisection_success_count']}",
        f"connect_success_count: {summary['connect_success_count']}",
        f"success_rate: {summary['success_rate']:.4f}",
        f"avg_steps_success: {summary['avg_steps_success']}",
        f"avg_steps_all: {summary['avg_steps_all']}",
        f"forced_timeouts: {summary['forced_timeouts']}",
        f"max_consecutive_connected_frames: {summary['max_consecutive_connected_frames']}",
        f"gripper_edge_penalty_mean_all_steps: {summary['gripper_edge_penalty_mean_all_steps']:.6f}",
        f"gripper_edge_violation_rate: {summary['gripper_edge_violation_rate']:.6f}",
        f"num_success_videos: {summary['num_success_videos']}",
        f"first_success_video_path: {summary['first_success_video_path']}",
        f"json_result: {json_path}",
    ]
    text_content = "\n".join(summary_lines) + "\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(text_content, end="")
    print(f"text_result: {txt_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
