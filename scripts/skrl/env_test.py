# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run environment tests with TensorBoard logging (no policy optimization)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate task constraints with random or zero actions and TensorBoard logging.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps", type=int, default=20000, help="Maximum environment steps for validation.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Scale random actions from [-1, 1].")
parser.add_argument("--log_interval", type=int, default=20, help="Log step metrics every N steps.")
parser.add_argument("--log_root", type=str, default="logs/env_test", help="Root directory for env-test logs.")
parser.add_argument(
    "--disable_ee_workspace_constraint",
    action="store_true",
    default=False,
    help="Disable IK end-effector workspace clamp in action config for validation.",
)
parser.add_argument(
    "--disable_physics_replication",
    action="store_true",
    default=False,
    help="Disable scene physics replication to avoid unsupported replication errors on deformable assets.",
)
parser.add_argument(
    "--joint_debug_interval",
    type=int,
    default=0,
    help="Optional interval (steps) for repeated joint diagnostics prints. 0 disables periodic prints.",
)
parser.add_argument(
    "--joint_debug_env_id",
    type=int,
    default=0,
    help="Environment index used for joint diagnostics output.",
)
action_mode_group = parser.add_mutually_exclusive_group(required=True)
action_mode_group.add_argument("--random", action="store_true", help="Use random actions in [-action_scale, action_scale].")
action_mode_group.add_argument("--zero", action="store_true", help="Use zero actions for static environment checks.")
parser.add_argument(
    "--x_bounds",
    nargs=2,
    type=float,
    default=None,
    metavar=("X_MIN", "X_MAX"),
    help="Optional EE world-space X bounds override.",
)
parser.add_argument(
    "--y_bounds",
    nargs=2,
    type=float,
    default=None,
    metavar=("Y_MIN", "Y_MAX"),
    help="Optional EE world-space Y bounds override.",
)
parser.add_argument(
    "--z_bounds",
    nargs=2,
    type=float,
    default=None,
    metavar=("Z_MIN", "Z_MAX"),
    help="Optional EE world-space Z bounds override.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from isaaclab.utils.math import subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import msr.tasks  # noqa: F401


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return None
    return None


def _log_numeric_dict(writer: SummaryWriter, prefix: str, data: dict, step: int):
    for key, value in data.items():
        tag = f"{prefix}/{key}"
        if isinstance(value, dict):
            _log_numeric_dict(writer, tag, value, step)
            continue

        scalar = _to_float(value)
        if scalar is not None:
            writer.add_scalar(tag, scalar, step)


def _find_workspace_bounds(env_cfg, device: torch.device):
    action_cfg = getattr(env_cfg.actions, "arm_1_action", None)
    x_bounds = getattr(action_cfg, "x_bounds_world", None) if action_cfg is not None else None
    y_bounds = getattr(action_cfg, "y_bounds_world", None) if action_cfg is not None else None
    z_bounds = getattr(action_cfg, "z_bounds_world", None) if action_cfg is not None else None

    if x_bounds is None or y_bounds is None or z_bounds is None:
        reward_cfg = getattr(env_cfg.rewards, "end_effector_workspace", None)
        if reward_cfg is not None and hasattr(reward_cfg, "params"):
            params = reward_cfg.params
            x_bounds = x_bounds if x_bounds is not None else params.get("x_bounds")
            y_bounds = y_bounds if y_bounds is not None else params.get("y_bounds")
            z_bounds = z_bounds if z_bounds is not None else params.get("z_bounds")

    if args_cli.x_bounds is not None:
        x_bounds = tuple(args_cli.x_bounds)
    if args_cli.y_bounds is not None:
        y_bounds = tuple(args_cli.y_bounds)
    if args_cli.z_bounds is not None:
        z_bounds = tuple(args_cli.z_bounds)

    if x_bounds is None or y_bounds is None or z_bounds is None:
        return None, None

    lower = torch.tensor([x_bounds[0], y_bounds[0], z_bounds[0]], device=device, dtype=torch.float32)
    upper = torch.tensor([x_bounds[1], y_bounds[1], z_bounds[1]], device=device, dtype=torch.float32)
    return lower, upper


def _get_workspace_constraint_states(env_cfg) -> dict[str, bool]:
    states = {}
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return states

    for action_name in ("arm_1_action", "arm_2_action"):
        action_cfg = getattr(actions_cfg, action_name, None)
        if action_cfg is None:
            continue
        if hasattr(action_cfg, "enforce_workspace_bounds"):
            states[action_name] = bool(getattr(action_cfg, "enforce_workspace_bounds"))

    return states


def _set_workspace_constraint_enabled(env_cfg, enabled: bool) -> dict[str, bool]:
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return {}

    for action_name in ("arm_1_action", "arm_2_action"):
        action_cfg = getattr(actions_cfg, action_name, None)
        if action_cfg is None:
            continue
        if hasattr(action_cfg, "enforce_workspace_bounds"):
            setattr(action_cfg, "enforce_workspace_bounds", bool(enabled))

    return _get_workspace_constraint_states(env_cfg)


def _build_joint_init_map(env_cfg) -> dict[str, float]:
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot_1", None)
    init_state = getattr(robot_cfg, "init_state", None)
    joint_pos = getattr(init_state, "joint_pos", None)
    if isinstance(joint_pos, dict):
        out = {}
        for joint_name, value in joint_pos.items():
            try:
                out[str(joint_name)] = float(value)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _print_joint_diagnostics(robot, env_cfg, step_tag: str, env_id: int = 0):
    cfg_joint_map = _build_joint_init_map(env_cfg)

    num_envs = int(robot.data.joint_pos.shape[0])
    safe_env_id = max(0, min(env_id, num_envs - 1))
    if safe_env_id != env_id:
        print(f"[WARNING] joint_debug_env_id={env_id} is out of range. Using {safe_env_id} instead.")

    print(f"[JOINT_DEBUG] {step_tag} | env_id={safe_env_id}")
    print("[JOINT_DEBUG] name | cfg_init | default_joint_pos | runtime_joint_pos | runtime-cfg | runtime-default")

    if hasattr(robot, "joint_names"):
        joint_names = list(robot.joint_names)
    else:
        joint_names = [f"joint_{i}" for i in range(robot.data.joint_pos.shape[1])]

    has_default_joint_pos = hasattr(robot.data, "default_joint_pos") and robot.data.default_joint_pos is not None

    for joint_idx, joint_name in enumerate(joint_names):
        runtime_val = float(robot.data.joint_pos[safe_env_id, joint_idx].item())
        default_val = float(robot.data.default_joint_pos[safe_env_id, joint_idx].item()) if has_default_joint_pos else None
        cfg_val = cfg_joint_map.get(joint_name)

        runtime_minus_cfg = runtime_val - cfg_val if cfg_val is not None else None
        runtime_minus_default = runtime_val - default_val if default_val is not None else None

        cfg_str = f"{cfg_val:+.6f}" if cfg_val is not None else "N/A"
        default_str = f"{default_val:+.6f}" if default_val is not None else "N/A"
        runtime_str = f"{runtime_val:+.6f}"
        d_cfg_str = f"{runtime_minus_cfg:+.6f}" if runtime_minus_cfg is not None else "N/A"
        d_default_str = f"{runtime_minus_default:+.6f}" if runtime_minus_default is not None else "N/A"

        print(
            f"[JOINT_DEBUG] {joint_name} | {cfg_str} | {default_str} | "
            f"{runtime_str} | {d_cfg_str} | {d_default_str}"
        )

    root_pos = robot.data.root_pos_w[safe_env_id]
    root_quat = robot.data.root_quat_w[safe_env_id]
    print(
        "[JOINT_DEBUG] root_pos_w="
        f"[{root_pos[0].item():+.6f}, {root_pos[1].item():+.6f}, {root_pos[2].item():+.6f}]"
    )
    print(
        "[JOINT_DEBUG] root_quat_w(wxyz)="
        f"[{root_quat[0].item():+.6f}, {root_quat[1].item():+.6f}, {root_quat[2].item():+.6f}, {root_quat[3].item():+.6f}]"
    )

    try:
        ee_body_ids, _ = robot.find_bodies("psm_tool_tip_Link")
        if len(ee_body_ids) > 0:
            ee_body_idx = ee_body_ids[0]
            ee_pos_w = robot.data.body_pos_w[safe_env_id, ee_body_idx]
            ee_quat_w = robot.data.body_quat_w[safe_env_id, ee_body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w[safe_env_id : safe_env_id + 1],
                robot.data.root_quat_w[safe_env_id : safe_env_id + 1],
                robot.data.body_pos_w[safe_env_id : safe_env_id + 1, ee_body_idx],
                robot.data.body_quat_w[safe_env_id : safe_env_id + 1, ee_body_idx],
            )
            ee_pos_b = ee_pos_b[0]
            ee_quat_b = ee_quat_b[0]
            print(
                "[JOINT_DEBUG] ee_pos_w="
                f"[{ee_pos_w[0].item():+.6f}, {ee_pos_w[1].item():+.6f}, {ee_pos_w[2].item():+.6f}]"
            )
            print(
                "[JOINT_DEBUG] ee_quat_w(wxyz)="
                f"[{ee_quat_w[0].item():+.6f}, {ee_quat_w[1].item():+.6f}, {ee_quat_w[2].item():+.6f}, {ee_quat_w[3].item():+.6f}]"
            )
            print(
                "[JOINT_DEBUG] ee_pos_base="
                f"[{ee_pos_b[0].item():+.6f}, {ee_pos_b[1].item():+.6f}, {ee_pos_b[2].item():+.6f}]"
            )
            print(
                "[JOINT_DEBUG] ee_quat_base(wxyz)="
                f"[{ee_quat_b[0].item():+.6f}, {ee_quat_b[1].item():+.6f}, {ee_quat_b[2].item():+.6f}, {ee_quat_b[3].item():+.6f}]"
            )
    except Exception as err:
        print(f"[WARNING] Failed to collect EE diagnostics: {err}")


def main():
    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.disable_physics_replication:
        scene_cfg = getattr(env_cfg, "scene", None)
        if scene_cfg is not None and hasattr(scene_cfg, "replicate_physics"):
            setattr(scene_cfg, "replicate_physics", False)
            print("[INFO] Disabled scene physics replication (scene.replicate_physics=False).")
        else:
            print("[WARNING] scene.replicate_physics is not available in env config. Nothing changed.")

    if args_cli.disable_ee_workspace_constraint:
        updated_states = _set_workspace_constraint_enabled(env_cfg, enabled=False)
        if updated_states:
            state_text = ", ".join(f"{name}={state}" for name, state in sorted(updated_states.items()))
            print(f"[INFO] Disabled EE workspace clamp. Action states: {state_text}")
        else:
            print("[WARNING] No action term with 'enforce_workspace_bounds' found. Nothing changed.")

    scene_cfg = getattr(env_cfg, "scene", None)
    scene_num_envs = getattr(scene_cfg, "num_envs", None)
    scene_replication = getattr(scene_cfg, "replicate_physics", None)
    if isinstance(scene_replication, bool) and scene_replication and isinstance(scene_num_envs, int) and scene_num_envs > 1:
        print(
            "[WARNING] scene.replicate_physics=True with num_envs>1 may trigger unsupported PhysX replication "
            "on deformable assets. Consider --disable_physics_replication or --num_envs 1."
        )

    workspace_constraint_states = _get_workspace_constraint_states(env_cfg)
    if workspace_constraint_states:
        state_text = ", ".join(f"{name}={state}" for name, state in sorted(workspace_constraint_states.items()))
        print(f"[INFO] EE workspace clamp states: {state_text}")

    action_mode = "random" if args_cli.random else "zero"
    task_tag = args_cli.task.replace(":", "_").replace("/", "_")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_env_test_{action_mode}_{task_tag}"
    log_dir = os.path.abspath(os.path.join(args_cli.log_root, run_name))
    tb_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg)
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"[INFO] Running env test for task: {args_cli.task}")
    print(f"[INFO] Action mode: {action_mode}")
    print(f"[INFO] Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO] Max steps: {args_cli.max_steps}")
    print(f"[INFO] TensorBoard log directory: {tb_dir}")

    try:
        robot = env.unwrapped.scene["robot_1"]
        ee_body_ids, _ = robot.find_bodies("psm_tool_tip_Link")
        ee_body_idx = ee_body_ids[0]
    except (KeyError, IndexError):
        robot = None
        ee_body_idx = None
        print("[WARNING] Failed to find robot_1/psm_tool_tip_Link. Workspace error metrics will be skipped.")

    lower_bounds, upper_bounds = _find_workspace_bounds(env_cfg, env.unwrapped.device)
    if lower_bounds is not None and upper_bounds is not None:
        print(
            "[INFO] Workspace bounds: "
            f"x=[{lower_bounds[0].item():.4f}, {upper_bounds[0].item():.4f}], "
            f"y=[{lower_bounds[1].item():.4f}, {upper_bounds[1].item():.4f}], "
            f"z=[{lower_bounds[2].item():.4f}, {upper_bounds[2].item():.4f}]"
        )
        if workspace_constraint_states and not any(workspace_constraint_states.values()):
            print("[INFO] Workspace bounds are in diagnostics-only mode (action clamp disabled).")
    else:
        print("[WARNING] No workspace bounds found. You can provide --x_bounds --y_bounds --z_bounds manually.")

    obs, info = env.reset()
    del obs, info

    if robot is not None:
        _print_joint_diagnostics(robot, env_cfg, step_tag="after_reset", env_id=args_cli.joint_debug_env_id)

    num_envs = env.unwrapped.num_envs
    episode_returns = torch.zeros(num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(num_envs, device=env.unwrapped.device)

    global_step = 0
    episode_count = 0

    while simulation_app.is_running() and global_step < args_cli.max_steps:
        with torch.inference_mode():
            if args_cli.random:
                actions = (2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0) * args_cli.action_scale
            else:
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            _, reward, terminated, truncated, info = env.step(actions)

        done = torch.logical_or(terminated, truncated)
        episode_returns += reward
        episode_lengths += 1

        if global_step % args_cli.log_interval == 0:
            reward_mean = reward.mean().item()
            reward_std = reward.std(unbiased=False).item()
            action_l2 = torch.linalg.norm(actions, dim=-1)
            writer.add_scalar("step/reward_mean", reward_mean, global_step)
            writer.add_scalar("step/reward_std", reward_std, global_step)
            writer.add_scalar("step/action_l2_mean", action_l2.mean().item(), global_step)
            writer.add_scalar("step/action_l2_max", action_l2.max().item(), global_step)

            if isinstance(info, dict):
                log_dict = info.get("log")
                if isinstance(log_dict, dict):
                    _log_numeric_dict(writer, "env_log", log_dict, global_step)

            if robot is not None and ee_body_idx is not None:
                ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
                writer.add_scalar("ee/pos_x_mean", ee_pos_w[:, 0].mean().item(), global_step)
                writer.add_scalar("ee/pos_y_mean", ee_pos_w[:, 1].mean().item(), global_step)
                writer.add_scalar("ee/pos_z_mean", ee_pos_w[:, 2].mean().item(), global_step)

                if lower_bounds is not None and upper_bounds is not None:
                    below = torch.relu(lower_bounds.unsqueeze(0) - ee_pos_w)
                    above = torch.relu(ee_pos_w - upper_bounds.unsqueeze(0))
                    violation_vec = below + above
                    violation_l2 = torch.linalg.norm(violation_vec, dim=-1)
                    in_bounds = violation_l2 <= 1e-8
                    box_center = 0.5 * (lower_bounds + upper_bounds)
                    center_error = torch.linalg.norm(ee_pos_w - box_center.unsqueeze(0), dim=-1)

                    writer.add_scalar("safety/workspace_violation_l2_mean", violation_l2.mean().item(), global_step)
                    writer.add_scalar("safety/workspace_violation_l2_max", violation_l2.max().item(), global_step)
                    writer.add_scalar("safety/workspace_violation_ratio", (~in_bounds).float().mean().item(), global_step)
                    writer.add_scalar("error/workspace_center_error_mean", center_error.mean().item(), global_step)
                    writer.add_scalar("error/workspace_center_error_max", center_error.max().item(), global_step)

        if args_cli.joint_debug_interval > 0 and global_step % args_cli.joint_debug_interval == 0 and robot is not None:
            _print_joint_diagnostics(
                robot,
                env_cfg,
                step_tag=f"step_{global_step}",
                env_id=args_cli.joint_debug_env_id,
            )

        if done.any():
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            for env_id in done_ids.tolist():
                writer.add_scalar("episode/return", episode_returns[env_id].item(), episode_count)
                writer.add_scalar("episode/length", episode_lengths[env_id].item(), episode_count)
                episode_count += 1

            episode_returns[done_ids] = 0.0
            episode_lengths[done_ids] = 0.0

        global_step += 1

    writer.flush()
    writer.close()
    env.close()

    print(f"[INFO] Env test finished at step {global_step}.")
    print("[INFO] Launch TensorBoard with:")
    print(f"       tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()