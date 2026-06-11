# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run a trained policy on the dual SO101 Isaac Lab scene.

Usage:
  ./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/so-arm-policy-test.py --num_envs 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


_ISAACLAB_SOURCE_DIR = Path(__file__).resolve().parents[4]
_LEROBOT_SRC = _ISAACLAB_SOURCE_DIR / "lerobot" / "src"
_LEROBOT_MYCODE_DIR = _ISAACLAB_SOURCE_DIR / "lerobot" / "mycode"
_DEFAULT_POLICY_PATH = (
    _ISAACLAB_SOURCE_DIR
    / "ASBDualArm"
    / "policy"
    / "act_sim_cube1"
    / "checkpoints"
    / "100000"
    / "pretrained_model"
)
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))
if _LEROBOT_MYCODE_DIR.is_dir() and str(_LEROBOT_MYCODE_DIR) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_MYCODE_DIR))


parser = argparse.ArgumentParser(description="Dual SO-ARM trained policy test.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DualSOArm-IK-Play-v0",
    help="Task ID used to create the environment. The script overrides its actions to joint-position control.",
)
parser.add_argument(
    "--policy-path",
    type=Path,
    default=_DEFAULT_POLICY_PATH,
    help=(
        "Path to a policy pretrained_model directory, a checkpoint directory containing pretrained_model, "
        "a training run directory containing checkpoints/, or a Mask-ACT run/checkpoint directory."
    ),
)
parser.add_argument(
    "--policy-type",
    type=str,
    default="auto",
    help="Policy type to load, such as auto, act, diffusion, mask_act, or a custom type registered in LeRobot.",
)
parser.add_argument(
    "--policy-class",
    type=str,
    default=None,
    help="Custom policy class import path, for example my_pkg.my_policy:MyPolicy. Overrides --policy-type.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint name/step. For Mask-ACT runs, examples: 100000 or checkpoint_step_100000.",
)
parser.add_argument(
    "--policy-image-key",
    type=str,
    default=None,
    help="Override the visual observation key sent to the policy. Defaults to the first VISUAL input in config.",
)
parser.add_argument(
    "--policy-state-key",
    type=str,
    default=None,
    help="Override the state observation key sent to the policy. Defaults to the first STATE input in config.",
)
parser.add_argument(
    "--mask-act-dataset-root",
    type=Path,
    default=None,
    help="Dataset root used to rebuild Mask-ACT feature metadata/stats. Overrides mask_act_run_config.json root.",
)
parser.add_argument(
    "--no-policy-processors",
    action="store_true",
    default=False,
    help="Skip LeRobot policy pre/post processors. Useful for custom policies that handle scaling internally.",
)
parser.add_argument("--test-times", type=int, default=100, help="Number of policy test episodes to run.")
parser.add_argument(
    "--reset-settle-seconds",
    type=float,
    default=0.75,
    help="Seconds to hold the scene still after each reset before policy execution. Not counted as test frames.",
)
parser.add_argument(
    "--max-frames-per-test",
    type=int,
    default=600,
    help="Maximum frames for each policy test episode.",
)
parser.add_argument(
    "--max-joint-delta",
    type=float,
    default=0.08,
    help="Maximum per-step joint target change in radians. Set <= 0 to disable.",
)
parser.add_argument(
    "--action-smoothing",
    type=float,
    default=1.0,
    help="Low-pass factor for policy joint targets in [0, 1].",
)
parser.add_argument(
    "--start-paused",
    action="store_true",
    default=False,
    help="Start with policy paused. Press T in GUI mode to run.",
)
parser.add_argument(
    "--interactive",
    action="store_true",
    default=False,
    help="Run GUI hotkey policy mode instead of the default automatic test loop.",
)
parser.add_argument(
    "--max-steps-headless",
    "--max_steps_headless",
    dest="max_steps_headless",
    type=int,
    default=None,
    help="Deprecated alias for --max-frames-per-test.",
)
parser.add_argument(
    "--target-random-x",
    type=float,
    default=0.015,
    help="Uniform target initial x randomization half-range in meters. Set 0 to disable.",
)
parser.add_argument(
    "--target-random-y",
    type=float,
    default=0.015,
    help="Uniform target initial y randomization half-range in meters. Set 0 to disable.",
)
parser.add_argument(
    "--none-random",
    "--none_random",
    "--none-randon",
    "--none_randon",
    dest="none_random",
    action="store_true",
    default=False,
    help=(
        "Disable random target placement during auto tests. Instead test a deterministic grid over "
        "--target-random-x/y."
    ),
)
parser.add_argument(
    "--grid-random",
    "--grid_random",
    dest="grid_random",
    action="store_true",
    default=False,
    help=(
        "Divide --target-random-x/y into a grid during auto tests, then randomly sample "
        "--grid-repeats-per-point target positions inside each grid cell."
    ),
)
parser.add_argument(
    "--grid-size",
    "--grid_size",
    type=int,
    default=5,
    help="Number of grid divisions per axis when --none-random or --grid-random is enabled.",
)
parser.add_argument(
    "--grid-repeats-per-point",
    "--grid_repeats_per_point",
    type=int,
    default=4,
    help="Number of tests to run at each grid point/cell when --none-random or --grid-random is enabled.",
)
parser.add_argument(
    "--task-status-print-interval",
    type=int,
    default=0,
    help="Print target/plane task status every N simulation steps. Set 0 to disable periodic status.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


class _TerminalLogSink:
    def __init__(self):
        self._buffer: list[str] = []
        self._file = None
        self.path: Path | None = None

    def write(self, text: str) -> None:
        if self._file is None:
            self._buffer.append(text)
            return
        self._file.write(text)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def attach(self, path: Path) -> None:
        if self._file is not None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self.path = path
        if self._buffer:
            self._file.write("".join(self._buffer))
            self._buffer.clear()
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class _TeeStream:
    def __init__(self, stream, sink: _TerminalLogSink):
        self._stream = stream
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.write(text)
        if not _TERMINAL_QUIET:
            return self._stream.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._sink.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


_TERMINAL_LOG_SINK = _TerminalLogSink()
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_TERMINAL_QUIET = False
sys.stdout = _TeeStream(sys.stdout, _TERMINAL_LOG_SINK)
sys.stderr = _TeeStream(sys.stderr, _TERMINAL_LOG_SINK)


def _set_terminal_quiet(enabled: bool) -> None:
    global _TERMINAL_QUIET
    _TERMINAL_QUIET = bool(enabled)


def _console_print(text: str = "", end: str = "\n") -> None:
    _ORIGINAL_STDOUT.write(text + end)
    _ORIGINAL_STDOUT.flush()


def _console_progress(current: int, total: int, success_count: int) -> None:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    width = 34
    filled = int(round(width * current / total))
    bar = "#" * filled + "-" * (width - filled)
    rate = success_count / current if current > 0 else 0.0
    _ORIGINAL_STDOUT.write(
        f"\r[grid_random] [{bar}] {current}/{total} "
        f"success={success_count} rate={rate:.2%}"
    )
    _ORIGINAL_STDOUT.flush()


def _attach_terminal_log(log_path: Path) -> None:
    try:
        _TERMINAL_LOG_SINK.attach(log_path)
        print(f"[Policy] Terminal output will be saved to: {log_path}")
    except OSError as exc:
        print(f"[Policy] WARNING: Could not create terminal log at {log_path}: {exc}", file=_ORIGINAL_STDERR)


def _close_terminal_log() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    _TERMINAL_LOG_SINK.close()
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR

if args_cli.grid_random:
    _set_terminal_quiet(True)

if not args_cli.enable_cameras:
    print("[Policy] Enabling cameras because the policy requires RGB input.")
    args_cli.enable_cameras = True

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401
from asb_dual_arm.config.robot import SO101_ALL_JOINT_NAMES
import asb_dual_arm.tasks  # noqa: F401
import asb_dual_arm.tasks.direct.dual_arm.mdp as dual_arm_mdp


SO101_JOINT_NAMES = list(SO101_ALL_JOINT_NAMES)
SO101_JOINT_LOWER = torch.tensor([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
SO101_JOINT_UPPER = torch.tensor([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

POLICY_STATE_KEY = "observation.state"
POLICY_IMAGE_KEY = "observation.images.camera"
POLICY_CAMERA_SHAPE = (3, 480, 640)
RESET_XFORM_PRIM_NAMES = ("DeformableOccluder",)
RESET_RIGID_OBJECT_NAMES = ("target",)
RESET_DEFORMABLE_OBJECT_NAMES = ("deformable_occluder",)


@dataclass
class PolicyRuntime:
    policy: object
    preprocessor: object
    postprocessor: object
    policy_path: Path
    policy_type: str
    state_key: str
    image_key: str


@dataclass(frozen=True)
class TargetGridCase:
    case_index: int
    grid_x_index: int
    grid_y_index: int
    repeat_index: int
    offset_x: float
    offset_y: float
    cell_x_min: float | None = None
    cell_x_max: float | None = None
    cell_y_min: float | None = None
    cell_y_max: float | None = None


class IdentityProcessor:
    def reset(self) -> None:
        pass

    def __call__(self, data):
        return data


class MaskACTInferencePolicy:
    """Inference adapter for checkpoints produced by train_mask_act_policy.py."""

    def __init__(
        self,
        model,
        preprocessor,
        postprocessor,
        state_key: str,
        rgb_key: str,
        device: torch.device | str,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state_key = state_key
        self.rgb_key = rgb_key
        self.device = torch.device(device)

    @property
    def config(self):
        return self.model.config

    def reset(self) -> None:
        _reset_policy(self.model.act_policy)
        self.preprocessor.reset()
        self.postprocessor.reset()

    @staticmethod
    def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim in {1, 3}:
            return tensor.unsqueeze(0)
        return tensor

    def select_action(self, observation: dict[str, torch.Tensor]):
        from lerobot.utils.constants import OBS_ENV_STATE
        from lerobot.policies.act.modeling_act import METRIC_SEED

        if self.rgb_key not in observation:
            raise KeyError(f"Mask-ACT policy requires raw RGB observation {self.rgb_key!r}.")
        if self.state_key not in observation:
            raise KeyError(f"Mask-ACT policy requires state observation {self.state_key!r}.")

        raw_state = observation[self.state_key].to(device=self.device, dtype=torch.float32)
        raw_rgb = observation[self.rgb_key].to(device=self.device, dtype=torch.float32)
        raw_batch = {
            self.state_key: self._ensure_batch(raw_state),
            self.rgb_key: self._ensure_batch(raw_rgb),
        }

        act_batch = self.preprocessor({self.state_key: raw_state})
        act_batch = {
            key: self._ensure_batch(value.to(device=self.device, dtype=torch.float32))
            for key, value in act_batch.items()
            if isinstance(value, torch.Tensor)
        }

        if self.model.uses_semantic_latents():
            rgb_main_latent, rgb_semantic_latents, _ = self.model.predict_semantic_latents(
                raw_batch,
                device=self.device,
                masks=None,
            )
            act_batch[OBS_ENV_STATE] = torch.cat(
                [rgb_main_latent, rgb_semantic_latents.reshape(rgb_semantic_latents.shape[0], -1)],
                dim=-1,
            )
            action = self.model.act_policy.select_action(act_batch)
            return self.postprocessor(action)

        mask_logits, rgb_latent = self.model.predict_masks_and_latent(raw_batch, device=self.device)
        if self.model.act_uses_latent():
            act_batch[OBS_ENV_STATE] = rgb_latent

        if self.model.act_uses_masks():
            mask_probs = torch.sigmoid(mask_logits)
            for idx, key in enumerate(self.model.mask_keys):
                mask_image = mask_probs[:, idx : idx + 1].repeat(1, 3, 1, 1)
                act_batch[key] = self.model.normalize_visual_like(mask_image, key)

            if self.model.uses_mask_metrics():
                metric_inputs = self.model.compute_mask_metrics(mask_probs)
                if self.model.experiment == "4A":
                    act_batch[OBS_ENV_STATE] = metric_inputs
                elif self.model.experiment == "4C":
                    act_batch[METRIC_SEED] = metric_inputs

        action = self.model.act_policy.select_action(act_batch)
        return self.postprocessor(action)


class TargetPlaneMonitor:
    """Reads static plane bounds and checks whether the target footprint is fully inside it."""

    def __init__(self, env):
        from pxr import Usd, UsdGeom
        import omni.usd

        self._env = env
        self._bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=False,
        )
        self._stage = omni.usd.get_context().get_stage()
        self._plane_prim = self._find_first_prim_by_name("Plane")
        self._target_prim = self._find_first_prim_by_name("Target")
        self._target_half_xy = self._read_target_half_xy()
        self.plane_min_xy, self.plane_max_xy = self._read_plane_xy_bounds()

        print(
            "[Task] Plane XY bounds: "
            f"min=({self.plane_min_xy[0]:.4f}, {self.plane_min_xy[1]:.4f}), "
            f"max=({self.plane_max_xy[0]:.4f}, {self.plane_max_xy[1]:.4f})"
        )
        print(
            "[Task] Target XY size: "
            f"({2.0 * self._target_half_xy[0]:.4f}, {2.0 * self._target_half_xy[1]:.4f})"
        )

    def _find_first_prim_by_name(self, prim_name: str):
        for prim in self._stage.Traverse():
            if prim.GetName() == prim_name:
                return prim
        raise RuntimeError(f"Could not find prim named {prim_name!r} in the stage.")

    def _read_aligned_bounds(self, prim):
        self._bbox_cache.Clear()
        box = self._bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        min_pt = box.GetMin()
        max_pt = box.GetMax()
        return np.array(min_pt, dtype=np.float64), np.array(max_pt, dtype=np.float64)

    def _read_plane_xy_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        min_pt, max_pt = self._read_aligned_bounds(self._plane_prim)
        return min_pt[:2], max_pt[:2]

    def _read_target_half_xy(self) -> np.ndarray:
        min_pt, max_pt = self._read_aligned_bounds(self._target_prim)
        half_xy = 0.5 * np.maximum(max_pt[:2] - min_pt[:2], 0.0)
        if np.any(half_xy <= 0.0):
            raise RuntimeError(f"Invalid target XY size read from USD bounds: half_xy={half_xy.tolist()}")
        return half_xy

    def read_status(self) -> dict[str, np.ndarray | bool]:
        target = self._env.unwrapped.scene["target"]
        pos = target.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        target_min_xy = pos[:2] - self._target_half_xy
        target_max_xy = pos[:2] + self._target_half_xy
        complete = bool(np.all(target_min_xy >= self.plane_min_xy) and np.all(target_max_xy <= self.plane_max_xy))
        return {
            "target_pos": pos,
            "target_size_xy": 2.0 * self._target_half_xy,
            "target_min_xy": target_min_xy,
            "target_max_xy": target_max_xy,
            "complete": complete,
        }

    def print_status(self, step_count: int, status: dict[str, np.ndarray | bool]) -> None:
        pos = status["target_pos"]
        size_xy = status["target_size_xy"]
        target_min_xy = status["target_min_xy"]
        target_max_xy = status["target_max_xy"]
        print(
            f"[Task] step={step_count} "
            f"target_pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
            f"target_size_xy=({size_xy[0]:.4f}, {size_xy[1]:.4f}) "
            f"target_xy_min=({target_min_xy[0]:.4f}, {target_min_xy[1]:.4f}) "
            f"target_xy_max=({target_max_xy[0]:.4f}, {target_max_xy[1]:.4f}) "
            f"plane_xy_min=({self.plane_min_xy[0]:.4f}, {self.plane_min_xy[1]:.4f}) "
            f"plane_xy_max=({self.plane_max_xy[0]:.4f}, {self.plane_max_xy[1]:.4f}) "
            f"complete={status['complete']} "
            f"{'TASK_COMPLETE' if status['complete'] else 'TASK_RUNNING'}"
        )


def _configure_joint_position_actions(env_cfg) -> None:
    """Replace the default SO101 task-space IK actions with direct joint-position actions."""

    env_cfg.actions.arm_1_action = dual_arm_mdp.JointPositionActionCfg(
        asset_name="robot_1",
        joint_names=SO101_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    env_cfg.actions.arm_2_action = dual_arm_mdp.JointPositionActionCfg(
        asset_name="robot_2",
        joint_names=SO101_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    env_cfg.actions.gripper_1_action = None
    env_cfg.actions.gripper_2_action = None


def _read_robot_joint_targets(env, robot_name: str) -> torch.Tensor:
    robot = env.unwrapped.scene[robot_name]
    joint_ids, joint_names = robot.find_joints(SO101_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(SO101_JOINT_NAMES):
        raise RuntimeError(f"Expected joints {SO101_JOINT_NAMES} on {robot_name}, found {joint_names}")
    return robot.data.joint_pos[:, joint_ids].clone()


def _write_joint_actions(actions: torch.Tensor, robot_1_targets: torch.Tensor, robot_2_targets: torch.Tensor) -> None:
    actions.zero_()
    if actions.ndim == 2:
        actions[:, 0:6] = robot_1_targets
        actions[:, 6:12] = robot_2_targets
    else:
        actions[0:6] = robot_1_targets[0]
        actions[6:12] = robot_2_targets[0]


def _read_policy_state(env) -> torch.Tensor:
    joints_1 = _read_robot_joint_targets(env, "robot_1")[0]
    joints_2 = _read_robot_joint_targets(env, "robot_2")[0]
    return torch.cat((joints_1, joints_2), dim=0).detach().cpu().to(dtype=torch.float32)


def _read_policy_image(env) -> torch.Tensor:
    try:
        camera = env.unwrapped.scene["camera"]
    except KeyError as err:
        raise RuntimeError("Policy camera is not available. Start with --enable_cameras.") from err
    if "rgb" not in camera.data.output:
        raise RuntimeError(f"Policy camera has no rgb output. Available: {list(camera.data.output.keys())}")

    image = camera.data.output["rgb"][0]
    if image.shape[-1] > 3:
        image = image[..., :3]
    image = image.detach().cpu().to(dtype=torch.float32) / 255.0
    return image.permute(2, 0, 1).contiguous()


def _make_policy_observation(env, runtime: PolicyRuntime) -> dict[str, torch.Tensor]:
    return {
        runtime.state_key: _read_policy_state(env),
        runtime.image_key: _read_policy_image(env),
    }


def _extract_action_tensor(action) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        return action
    if isinstance(action, dict):
        if "action" in action:
            return action["action"]
        if len(action) == 1:
            return next(iter(action.values()))
        raise RuntimeError(f"Policy action dict does not contain 'action'. Keys: {list(action.keys())}")
    raise RuntimeError(f"Unsupported policy action type: {type(action).__name__}")


def _split_policy_action(action, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    action = _extract_action_tensor(action)
    action = action.detach().to(device=device, dtype=torch.float32)
    if action.ndim == 2:
        action = action[0]
    if action.numel() != 12:
        raise RuntimeError(f"Expected 12D policy action, got shape {tuple(action.shape)}")
    lower = torch.cat((SO101_JOINT_LOWER, SO101_JOINT_LOWER), dim=0).to(device=device)
    upper = torch.cat((SO101_JOINT_UPPER, SO101_JOINT_UPPER), dim=0).to(device=device)
    action = torch.clamp(action, lower, upper)
    return action[:6].unsqueeze(0), action[6:12].unsqueeze(0)


def _limit_joint_delta(current: torch.Tensor, target: torch.Tensor, max_delta: float) -> torch.Tensor:
    if max_delta <= 0.0:
        return target
    return current + torch.clamp(target - current, min=-max_delta, max=max_delta)


def _blend_targets(current: torch.Tensor, new_target: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha >= 1.0:
        return new_target
    if alpha <= 0.0:
        return current
    return current + alpha * (new_target - current)


def _reset_robot_to_defaults(env, robot_name: str) -> None:
    robot = env.unwrapped.scene[robot_name]
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)


def _reset_rigid_object_to_defaults(env, object_name: str) -> None:
    try:
        rigid_object = env.unwrapped.scene[object_name]
    except KeyError:
        return
    root_state = rigid_object.data.default_root_state.clone()
    rigid_object.write_root_state_to_sim(root_state)


def _reset_scene_rigid_objects_to_defaults(env, object_names: tuple[str, ...]) -> None:
    for object_name in object_names:
        _reset_rigid_object_to_defaults(env, object_name)


def _reset_deformable_object_to_defaults(env, object_name: str) -> None:
    try:
        deformable_object = env.unwrapped.scene[object_name]
    except KeyError:
        return
    nodal_state = deformable_object.data.default_nodal_state_w.clone()
    deformable_object.write_nodal_state_to_sim(nodal_state)


def _reset_scene_deformable_objects_to_defaults(env, object_names: tuple[str, ...]) -> None:
    for object_name in object_names:
        _reset_deformable_object_to_defaults(env, object_name)


def _randomize_target_xy(env, x_half_range: float, y_half_range: float) -> None:
    x_half_range = max(0.0, float(x_half_range))
    y_half_range = max(0.0, float(y_half_range))
    if x_half_range == 0.0 and y_half_range == 0.0:
        return

    try:
        target = env.unwrapped.scene["target"]
    except KeyError:
        return

    root_state = target.data.root_state_w.clone()
    num_envs = root_state.shape[0]
    device = root_state.device
    if x_half_range > 0.0:
        root_state[:, 0] += torch.empty(num_envs, device=device).uniform_(-x_half_range, x_half_range)
    if y_half_range > 0.0:
        root_state[:, 1] += torch.empty(num_envs, device=device).uniform_(-y_half_range, y_half_range)
    target.write_root_state_to_sim(root_state)
    print(
        "[Reset] Target randomized in XY: "
        f"x_range=+/-{x_half_range:.4f} m, y_range=+/-{y_half_range:.4f} m, "
        f"env0_pos={root_state[0, 0:3].detach().cpu().tolist()}"
    )


def _set_target_xy_offset(env, offset_x: float, offset_y: float, label: str = "configured") -> None:
    try:
        target = env.unwrapped.scene["target"]
    except KeyError:
        return

    root_state = target.data.root_state_w.clone()
    root_state[:, 0] += float(offset_x)
    root_state[:, 1] += float(offset_y)
    target.write_root_state_to_sim(root_state)
    print(
        f"[Reset] Target {label} XY offset: "
        f"offset=({float(offset_x):+.4f}, {float(offset_y):+.4f}) m, "
        f"env0_pos={root_state[0, 0:3].detach().cpu().tolist()}"
    )


def _make_target_grid_cases(
    x_half_range: float,
    y_half_range: float,
    grid_size: int = 5,
    repeats_per_point: int = 4,
) -> list[TargetGridCase]:
    x_half_range = max(0.0, float(x_half_range))
    y_half_range = max(0.0, float(y_half_range))
    grid_size = max(1, int(grid_size))
    repeats_per_point = max(1, int(repeats_per_point))
    x_offsets = np.linspace(-x_half_range, x_half_range, grid_size, dtype=np.float64)
    y_offsets = np.linspace(-y_half_range, y_half_range, grid_size, dtype=np.float64)

    cases = []
    case_index = 0
    for grid_y_index, offset_y in enumerate(y_offsets):
        for grid_x_index, offset_x in enumerate(x_offsets):
            for repeat_index in range(repeats_per_point):
                cases.append(
                    TargetGridCase(
                        case_index=case_index,
                        grid_x_index=grid_x_index,
                        grid_y_index=grid_y_index,
                        repeat_index=repeat_index,
                        offset_x=float(offset_x),
                        offset_y=float(offset_y),
                    )
                )
                case_index += 1
    return cases


def _make_target_grid_random_cases(
    x_half_range: float,
    y_half_range: float,
    grid_size: int = 5,
    samples_per_cell: int = 4,
) -> list[TargetGridCase]:
    x_half_range = max(0.0, float(x_half_range))
    y_half_range = max(0.0, float(y_half_range))
    grid_size = max(1, int(grid_size))
    samples_per_cell = max(1, int(samples_per_cell))
    x_edges = np.linspace(-x_half_range, x_half_range, grid_size + 1, dtype=np.float64)
    y_edges = np.linspace(-y_half_range, y_half_range, grid_size + 1, dtype=np.float64)

    cases = []
    case_index = 0
    for grid_y_index in range(grid_size):
        cell_y_min = float(y_edges[grid_y_index])
        cell_y_max = float(y_edges[grid_y_index + 1])
        for grid_x_index in range(grid_size):
            cell_x_min = float(x_edges[grid_x_index])
            cell_x_max = float(x_edges[grid_x_index + 1])
            for repeat_index in range(samples_per_cell):
                offset_x = torch.empty((), dtype=torch.float64).uniform_(cell_x_min, cell_x_max).item()
                offset_y = torch.empty((), dtype=torch.float64).uniform_(cell_y_min, cell_y_max).item()
                cases.append(
                    TargetGridCase(
                        case_index=case_index,
                        grid_x_index=grid_x_index,
                        grid_y_index=grid_y_index,
                        repeat_index=repeat_index,
                        offset_x=float(offset_x),
                        offset_y=float(offset_y),
                        cell_x_min=cell_x_min,
                        cell_x_max=cell_x_max,
                        cell_y_min=cell_y_min,
                        cell_y_max=cell_y_max,
                    )
                )
                case_index += 1
    return cases


def _cache_initial_xforms(prim_names: tuple[str, ...]) -> dict[str, list[tuple[str, object]]]:
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cached = {}
    for prim in stage.Traverse():
        if prim.GetName() not in prim_names:
            continue
        xformable = UsdGeom.Xformable(prim)
        ops = []
        for op in xformable.GetOrderedXformOps():
            value = op.Get()
            if value is not None:
                ops.append((op.GetOpName(), value))
        if ops:
            cached[str(prim.GetPath())] = ops
    return cached


def _restore_cached_xforms(cached_xforms: dict[str, list[tuple[str, object]]]) -> None:
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for prim_path, ops in cached_xforms.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        existing_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        for op_name, value in ops:
            op = existing_ops.get(op_name)
            if op is not None:
                op.Set(value)


def _reset_environment(
    env,
    policy,
    preprocessor,
    postprocessor,
    cached_xforms: dict[str, list[tuple[str, object]]] | None = None,
    target_random_x: float = 0.0,
    target_random_y: float = 0.0,
    target_grid_case: TargetGridCase | None = None,
    target_grid_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()
    if cached_xforms is not None:
        _restore_cached_xforms(cached_xforms)
    _reset_scene_deformable_objects_to_defaults(env, RESET_DEFORMABLE_OBJECT_NAMES)
    _reset_scene_rigid_objects_to_defaults(env, RESET_RIGID_OBJECT_NAMES)
    if target_grid_case is not None:
        label = "grid-random" if target_grid_mode == "grid_random" else "deterministic-grid"
        _set_target_xy_offset(env, target_grid_case.offset_x, target_grid_case.offset_y, label=label)
    else:
        _randomize_target_xy(env, target_random_x, target_random_y)
    _reset_robot_to_defaults(env, "robot_1")
    _reset_robot_to_defaults(env, "robot_2")
    _reset_policy(policy)
    preprocessor.reset()
    postprocessor.reset()
    return _read_robot_joint_targets(env, "robot_1"), _read_robot_joint_targets(env, "robot_2")


def _get_env_step_dt(env) -> float:
    step_dt = getattr(env.unwrapped, "step_dt", None)
    if step_dt is None:
        cfg = getattr(env.unwrapped, "cfg", None)
        sim_cfg = getattr(cfg, "sim", None)
        step_dt = float(getattr(sim_cfg, "dt", 0.0)) * float(getattr(cfg, "decimation", 1))
    return float(step_dt)


def _settle_environment_after_reset(
    env,
    actions: torch.Tensor,
    robot_1_targets: torch.Tensor,
    robot_2_targets: torch.Tensor,
    settle_seconds: float,
) -> int:
    settle_seconds = max(0.0, float(settle_seconds))
    if settle_seconds <= 0.0:
        return 0

    step_dt = _get_env_step_dt(env)
    if step_dt <= 0.0:
        raise RuntimeError(f"Cannot compute reset settle frames because environment step_dt is invalid: {step_dt}")

    settle_frames = int(math.ceil(settle_seconds / step_dt))
    _write_joint_actions(actions, robot_1_targets, robot_2_targets)
    for _ in range(settle_frames):
        env.step(actions)
    return settle_frames


class PolicyHotkeys:
    """Keyboard hotkeys for policy testing."""

    def __init__(self):
        import carb
        import omni.appwindow

        self._carb = carb
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._last_down = {}

    def _is_down(self, key) -> bool:
        flags = self._input.get_keyboard_button_flags(self._keyboard, key)
        return bool(flags & self._carb.input.BUTTON_FLAG_DOWN)

    def _pressed(self, key) -> bool:
        is_down = self._is_down(key)
        was_down = self._last_down.get(key, False)
        self._last_down[key] = is_down
        return is_down and not was_down

    def poll(self) -> dict[str, bool]:
        key = self._carb.input.KeyboardInput
        return {
            "start": self._pressed(key.T),
            "pause": self._pressed(key.P),
            "reset": self._pressed(key.N),
        }


def _resolve_policy_path(policy_path: Path, checkpoint: str | None) -> Path:
    path = policy_path.expanduser()
    if checkpoint is not None:
        candidate = path / "checkpoints" / checkpoint / "pretrained_model"
        if candidate.is_dir():
            return candidate
        candidate = path / checkpoint / "pretrained_model"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Could not find checkpoint {checkpoint!r} under policy path: {path}")

    if (path / "config.json").is_file():
        return path
    if (path / "pretrained_model" / "config.json").is_file():
        return path / "pretrained_model"
    if (path / "checkpoints" / "last" / "pretrained_model" / "config.json").is_file():
        return path / "checkpoints" / "last" / "pretrained_model"
    raise FileNotFoundError(
        "Policy path must contain config.json, pretrained_model/config.json, "
        f"or checkpoints/last/pretrained_model/config.json: {path}"
    )


def _read_policy_config_json(policy_path: Path) -> dict:
    config_path = policy_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Policy config.json does not exist: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _infer_policy_type(policy_path: Path, policy_type: str) -> str:
    if policy_type != "auto":
        return policy_type
    config = _read_policy_config_json(policy_path)
    inferred = config.get("type")
    if not inferred:
        raise RuntimeError(f"Could not infer policy type from {policy_path / 'config.json'}; pass --policy-type.")
    return str(inferred)


def _load_policy_class(policy_type: str, policy_class: str | None):
    if policy_class is not None:
        module_name, sep, class_name = policy_class.partition(":")
        if not sep:
            module_name, sep, class_name = policy_class.rpartition(".")
        if not module_name or not class_name:
            raise ValueError("--policy-class must look like my_pkg.my_module:MyPolicy")
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    from lerobot.policies.factory import get_policy_class

    return get_policy_class(policy_type)


def _reset_policy(policy) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def _find_mask_act_run_dir(path: Path) -> Path | None:
    current = path.resolve() if path.exists() else path.expanduser()
    for candidate in (current, *current.parents):
        if (candidate / "mask_act_run_config.json").is_file():
            return candidate
    return None


def _mask_act_checkpoint_step(checkpoint_dir: Path) -> int:
    name = checkpoint_dir.name
    if name.startswith("checkpoint_step_"):
        try:
            return int(name.removeprefix("checkpoint_step_"))
        except ValueError:
            pass
    return -1


def _resolve_mask_act_paths(policy_path: Path, checkpoint: str | None) -> tuple[Path, Path] | None:
    path = policy_path.expanduser()

    if (path / "training_state.pt").is_file():
        run_dir = _find_mask_act_run_dir(path)
        if run_dir is None:
            raise FileNotFoundError(f"Found Mask-ACT checkpoint but no mask_act_run_config.json above: {path}")
        return run_dir, path

    run_dir = _find_mask_act_run_dir(path)
    if run_dir is None:
        return None

    if checkpoint is not None:
        checkpoint_candidates = [
            run_dir / checkpoint,
            run_dir / f"checkpoint_step_{checkpoint}",
            run_dir / f"checkpoint_step_{int(checkpoint):06d}" if checkpoint.isdigit() else run_dir / checkpoint,
        ]
        for candidate in checkpoint_candidates:
            if (candidate / "training_state.pt").is_file():
                return run_dir, candidate
        raise FileNotFoundError(f"Could not find Mask-ACT checkpoint {checkpoint!r} under: {run_dir}")

    checkpoint_dirs = [
        candidate
        for candidate in run_dir.glob("checkpoint_step_*")
        if candidate.is_dir() and (candidate / "training_state.pt").is_file()
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint_step_*/training_state.pt found under Mask-ACT run: {run_dir}")
    checkpoint_dir = max(checkpoint_dirs, key=_mask_act_checkpoint_step)
    return run_dir, checkpoint_dir


def _path_candidates(value: str | None, base_dirs: list[Path]) -> list[Path]:
    if not value:
        return []
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return [raw]
    return [base / raw for base in base_dirs]


def _looks_like_lerobot_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() or (path / "data").is_dir()


def _resolve_mask_act_dataset_root(run_dir: Path, run_config: dict, dataset_root_override: Path | None) -> Path:
    if dataset_root_override is not None:
        dataset_root = dataset_root_override.expanduser()
        if not _looks_like_lerobot_root(dataset_root):
            raise FileNotFoundError(f"--mask-act-dataset-root is not a LeRobot dataset root: {dataset_root}")
        return dataset_root

    output_dir_value = run_config.get("output_dir")
    base_dirs = [Path.cwd(), _ISAACLAB_SOURCE_DIR, run_dir]
    repo_id = str(run_config.get("repo_id") or "").strip()
    output_dir_candidates = _path_candidates(output_dir_value, base_dirs)
    view_candidates = []
    for output_dir in output_dir_candidates:
        view_candidates.append(output_dir.parent / "dataset_views" / output_dir.name)
    view_candidates.append(run_dir.parent / "dataset_views" / run_dir.name)

    source_candidates = _path_candidates(run_config.get("root"), base_dirs)
    if repo_id:
        source_candidates.extend(
            [
                _ISAACLAB_SOURCE_DIR / "ASBDualArm" / "saved_data" / repo_id,
                _ISAACLAB_SOURCE_DIR / "ASBDualArm" / "source" / "ASBDualArm" / "saved_data" / repo_id,
                run_dir.parent / "saved_data" / repo_id,
            ]
        )
    for candidate in [*view_candidates, *source_candidates]:
        if _looks_like_lerobot_root(candidate):
            return candidate
    searched = [str(candidate) for candidate in [*view_candidates, *source_candidates]]
    raise FileNotFoundError(
        "Could not find the LeRobot dataset root needed to rebuild Mask-ACT stats. "
        f"Searched: {searched}"
    )


def _default_visual_stats(count: int = 1) -> dict[str, list[float]]:
    zeros = [[[0.0]], [[0.0]], [[0.0]]]
    ones = [[[1.0]], [[1.0]], [[1.0]]]
    halves = [[[0.5]], [[0.5]], [[0.5]]]
    return {
        "min": zeros,
        "max": ones,
        "mean": halves,
        "std": halves,
        "count": [count],
        "q01": zeros,
        "q10": zeros,
        "q50": halves,
        "q90": ones,
        "q99": ones,
    }


def _with_mask_act_feature_fallbacks(meta, run_config: dict):
    from types import SimpleNamespace

    rgb_key = run_config.get("rgb_key", POLICY_IMAGE_KEY)
    mask_keys = list(run_config.get("mask_target_keys") or [])
    features = dict(meta.features)
    stats = dict(meta.stats)

    rgb_feature = features.get(rgb_key)
    if rgb_feature is None:
        raise KeyError(f"Mask-ACT RGB feature {rgb_key!r} is missing from dataset metadata.")

    count = 1
    for value in stats.values():
        if isinstance(value, dict) and value.get("count"):
            count = int(value["count"][0])
            break

    for mask_key in mask_keys:
        features[mask_key] = dict(rgb_feature) if isinstance(rgb_feature, dict) else rgb_feature
        stats[mask_key] = _default_visual_stats(count=count)

    if mask_keys:
        print(
            "[Policy] Mask-ACT uses internally predicted masks only; external mask observations are not read. "
            f"Synthesized binary mask feature/stats for: {mask_keys}"
        )

    return SimpleNamespace(features=features, stats=stats, fps=meta.fps)


def _load_mask_act_policy(
    run_dir: Path,
    checkpoint_dir: Path,
    device: str,
    dataset_root_override: Path | None,
) -> PolicyRuntime:
    from argparse import Namespace

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from train_mask_act_policy import make_policy

    run_config_path = run_dir / "mask_act_run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    dataset_root = _resolve_mask_act_dataset_root(run_dir, run_config, dataset_root_override)
    repo_id = str(run_config.get("repo_id") or "cube1")
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    meta_for_policy = _with_mask_act_feature_fallbacks(meta, run_config)

    policy_args = Namespace(
        experiment=run_config["experiment"],
        rgb_key=run_config.get("rgb_key", POLICY_IMAGE_KEY),
        state_keys=list(run_config.get("state_keys") or [POLICY_STATE_KEY]),
        mask_target_keys=list(run_config["mask_target_keys"]),
        latent_dim=int(run_config["latent_dim"]),
        unet_base_channels=int(run_config["unet_base_channels"]),
        seg_loss_weight=float(run_config.get("seg_loss_weight", 1.0)),
        action_loss_weight=float(run_config.get("action_loss_weight", 1.0)),
        semantic_loss_weight=float(run_config.get("semantic_loss_weight", 1.0)),
        metric_loss_weight=float(run_config.get("metric_loss_weight", 1.0)),
        metric_eps=float(run_config.get("metric_eps", 1e-6)),
        chunk_size=int(run_config.get("chunk_size", 100)),
        n_action_steps=int(run_config.get("n_action_steps", 100)),
        pretrained_backbone_weights=None,
        device=device,
    )

    model = make_policy(policy_args, meta_for_policy, stats=meta_for_policy.stats).to(device)
    training_state = torch.load(checkpoint_dir / "training_state.pt", map_location=device)
    model.load_state_dict(training_state["model"])
    model.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        dataset_stats=meta.stats,
    )
    adapter = MaskACTInferencePolicy(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        state_key=policy_args.state_keys[0],
        rgb_key=policy_args.rgb_key,
        device=device,
    )
    adapter.reset()
    return PolicyRuntime(
        policy=adapter,
        preprocessor=IdentityProcessor(),
        postprocessor=IdentityProcessor(),
        policy_path=checkpoint_dir,
        policy_type=f"mask_act_{policy_args.experiment}",
        state_key=policy_args.state_keys[0],
        image_key=policy_args.rgb_key,
    )


def _first_feature_key(config: dict, feature_type: str, fallback: str) -> str:
    for key, feature in config.get("input_features", {}).items():
        if str(feature.get("type", "")).upper() == feature_type.upper():
            return key
    return fallback


def _load_policy(
    policy_path: Path,
    policy_type: str,
    policy_class: str | None,
    checkpoint: str | None,
    use_processors: bool,
) -> PolicyRuntime:
    mask_act_paths = _resolve_mask_act_paths(policy_path, checkpoint)
    if mask_act_paths is not None:
        if policy_class is not None:
            raise ValueError("--policy-class is not used for Mask-ACT training checkpoints.")
        run_dir, checkpoint_dir = mask_act_paths
        if policy_type not in {"auto", "mask_act"}:
            raise ValueError(
                f"Mask-ACT checkpoint detected at {checkpoint_dir}, but --policy-type={policy_type!r}. "
                "Use --policy-type auto or --policy-type mask_act."
            )
        return _load_mask_act_policy(run_dir, checkpoint_dir, args_cli.device, args_cli.mask_act_dataset_root)

    resolved_path = _resolve_policy_path(policy_path, checkpoint)
    config_json = _read_policy_config_json(resolved_path)
    if policy_class is not None and policy_type == "auto":
        resolved_type = str(config_json.get("type") or "custom")
    else:
        resolved_type = _infer_policy_type(resolved_path, policy_type)
    policy_cls = _load_policy_class(resolved_type, policy_class)

    policy = policy_cls.from_pretrained(resolved_path)
    if use_processors:
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(resolved_path))
    else:
        preprocessor = IdentityProcessor()
        postprocessor = IdentityProcessor()
    _reset_policy(policy)
    preprocessor.reset()
    postprocessor.reset()

    state_key = args_cli.policy_state_key or _first_feature_key(config_json, "STATE", POLICY_STATE_KEY)
    image_key = args_cli.policy_image_key or _first_feature_key(config_json, "VISUAL", POLICY_IMAGE_KEY)
    return PolicyRuntime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_path=resolved_path,
        policy_type=resolved_type,
        state_key=state_key,
        image_key=image_key,
    )


def _summarize_test_results(
    total_tests: int,
    detailed_results: dict[str, dict],
) -> dict[str, float | int | None]:
    success_frames = [
        int(result["completion_frames"])
        for result in detailed_results.values()
        if result.get("success") and result.get("completion_frames") is not None
    ]
    success_count = len(success_frames)
    success_rate = success_count / total_tests if total_tests > 0 else 0.0
    return {
        "success_count": success_count,
        "success_rate": success_rate,
        "average_completion_frames": sum(success_frames) / success_count if success_frames else None,
        "max_completion_frames": max(success_frames) if success_frames else None,
        "min_completion_frames": min(success_frames) if success_frames else None,
    }


def _format_test_results_for_terminal(entry: dict) -> str:
    lines = [
        "Policy Test Results",
        f"Test time: {entry['test_time']}",
        f"Policy path: {entry['policy_path']}",
        f"Requested test count: {entry['requested_test_count']}",
        f"Completed test count: {entry['completed_test_count']}",
        f"Max frames per test: {entry['max_frames_per_test']}",
        f"Reset settle: {entry['reset_settle_seconds']:.3f}s ({entry['reset_settle_frames']} frames, not counted)",
        f"Success count: {entry['success_count']}",
        f"Success rate: {entry['success_rate']:.2%}",
    ]
    if entry["success_count"] > 0:
        lines.extend(
            [
                f"Average completion frames among successes: {entry['average_completion_frames']:.2f}",
                f"Min completion frames among successes: {entry['min_completion_frames']}",
                f"Max completion frames among successes: {entry['max_completion_frames']}",
            ]
        )
    else:
        lines.extend(
            [
                "Average completion frames among successes: N/A",
                "Min completion frames among successes: N/A",
                "Max completion frames among successes: N/A",
            ]
        )
    return "\n".join(lines) + "\n"


def _next_result_key(results: dict) -> str:
    max_index = -1
    for key in results.keys():
        if not key.startswith("test_"):
            continue
        try:
            max_index = max(max_index, int(key.removeprefix("test_")))
        except ValueError:
            continue
    return f"test_{max_index + 1}"


def _save_test_results_json_at(result_path: Path, entry: dict) -> tuple[Path, str]:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.is_file():
        try:
            results = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = {}
        if not isinstance(results, dict):
            results = {}
    else:
        results = {}

    result_key = _next_result_key(results)
    results[result_key] = entry
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path, result_key


def _save_test_results_json(policy_path: Path, entry: dict) -> tuple[Path, str]:
    return _save_test_results_json_at(policy_path / "test_result.json", entry)


def _make_test_result_entry(
    runtime: PolicyRuntime,
    requested_test_count: int,
    completed_test_count: int,
    max_frames_per_test: int,
    reset_settle_seconds: float,
    reset_settle_frames: int,
    detailed_results: dict[str, dict],
    target_sampling: dict | None = None,
) -> dict:
    summary = _summarize_test_results(completed_test_count, detailed_results)
    entry = {
        "test_time": datetime.now(timezone.utc).isoformat(),
        "policy_path": str(runtime.policy_path),
        "policy_type": runtime.policy_type,
        "requested_test_count": requested_test_count,
        "completed_test_count": completed_test_count,
        "max_frames_per_test": max_frames_per_test,
        "reset_settle_seconds": reset_settle_seconds,
        "reset_settle_frames": reset_settle_frames,
        "success_count": summary["success_count"],
        "success_rate": summary["success_rate"],
        "average_completion_frames": summary["average_completion_frames"],
        "max_completion_frames": summary["max_completion_frames"],
        "min_completion_frames": summary["min_completion_frames"],
        "detailed_results": detailed_results,
    }
    if target_sampling is not None:
        entry["target_sampling"] = target_sampling
    return entry


def _legacy_test_result_txt_path(policy_path: Path) -> Path:
    return policy_path / "test_result.txt"


def _warn_if_legacy_txt_exists(policy_path: Path) -> None:
    legacy_path = _legacy_test_result_txt_path(policy_path)
    if legacy_path.is_file():
        print(f"[Policy] Existing legacy text result kept unchanged: {legacy_path}")


def _generate_grid_result_artifacts(result_path: Path, result_key: str, output_dir: Path) -> None:
    analyzer_path = _ISAACLAB_SOURCE_DIR / "ASBDualArm" / "policy" / "analyze_grid_test_results.py"
    if not analyzer_path.is_file():
        raise FileNotFoundError(f"Grid result analyzer not found: {analyzer_path}")

    spec = importlib.util.spec_from_file_location("analyze_grid_test_results", analyzer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import grid result analyzer: {analyzer_path}")
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)

    entry = analyzer.load_test_entry(result_path, result_key)
    grid_size, repeats = analyzer.require_grid_entry(entry, result_key)
    cells = analyzer.collect_grid_stats(entry, grid_size)
    summary = analyzer.make_summary(entry, result_key, grid_size, repeats, cells)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grid_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    analyzer.write_csv(output_dir / "grid_summary.csv", summary["cells"])
    analyzer.write_markdown(output_dir / "grid_summary.md", summary, cells, ndigits=4)
    analyzer.write_heatmap_svg(
        output_dir / "success_rate_heatmap.svg",
        f"{result_key} Success Rate",
        cells,
        grid_size,
        "success_rate",
        ndigits=4,
    )
    analyzer.write_heatmap_svg(
        output_dir / "average_success_frames_heatmap.svg",
        f"{result_key} Average Success Frames",
        cells,
        grid_size,
        "average_success_frames",
        ndigits=4,
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_joint_position_actions(env_cfg)

    print("\n" + "=" * 80)
    print(f"Creating Environment: {args_cli.task}")
    print("=" * 80)
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {args_cli.device}")
    print(f"  Policy path: {args_cli.policy_path}")
    print(f"  Policy type: {args_cli.policy_type}")
    if args_cli.policy_class is not None:
        print(f"  Policy class: {args_cli.policy_class}")
    if args_cli.checkpoint is not None:
        print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Policy processors: {'disabled' if args_cli.no_policy_processors else 'enabled'}")
    print("  Control: trained policy -> dual SO101 joint-position targets")
    print("=" * 80 + "\n")

    runtime = _load_policy(
        args_cli.policy_path,
        args_cli.policy_type,
        args_cli.policy_class,
        args_cli.checkpoint,
        use_processors=not args_cli.no_policy_processors,
    )
    policy = runtime.policy
    preprocessor = runtime.preprocessor
    postprocessor = runtime.postprocessor
    grid_random_output_dir = runtime.policy_path / "grid_random" if args_cli.grid_random else None
    terminal_log_path = (grid_random_output_dir / "log.txt") if grid_random_output_dir is not None else runtime.policy_path / "log.txt"
    _attach_terminal_log(terminal_log_path)
    if args_cli.grid_random:
        _set_terminal_quiet(True)
        _console_print(f"[grid_random] detailed log: {terminal_log_path}")
    print(f"[Policy] Loaded {runtime.policy_type!r} from: {runtime.policy_path}")
    print(f"[Policy] Observation keys: state={runtime.state_key!r}, image={runtime.image_key!r}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    task_monitor = TargetPlaneMonitor(env)
    cached_xforms = _cache_initial_xforms(RESET_XFORM_PRIM_NAMES)
    if cached_xforms:
        print(f"[Reset] Cached initial xforms for: {', '.join(sorted(cached_xforms))}")
    else:
        print("[Reset] No DeformableOccluder xforms found to cache.")

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    action_dim = actions.shape[-1] if actions.ndim > 1 else actions.shape[0]
    if action_dim != 12:
        raise RuntimeError(f"Expected 12D action space for dual SO101 joint-position control, got {action_dim}.")

    smoothing = max(0.0, min(1.0, float(args_cli.action_smoothing)))
    max_joint_delta = max(0.0, float(args_cli.max_joint_delta))
    max_frames_per_test = int(args_cli.max_frames_per_test)
    if args_cli.max_steps_headless is not None:
        max_frames_per_test = int(args_cli.max_steps_headless)
    max_frames_per_test = max(1, max_frames_per_test)
    test_times = max(1, int(args_cli.test_times))
    auto_test = not (args_cli.interactive or args_cli.start_paused)
    grid_size = max(1, int(args_cli.grid_size))
    grid_repeats_per_point = max(1, int(args_cli.grid_repeats_per_point))
    if args_cli.none_random and args_cli.grid_random:
        raise ValueError("--none-random and --grid-random are mutually exclusive. Choose one target sampling mode.")
    target_grid_mode = None
    target_grid_cases = None
    if auto_test and args_cli.none_random:
        target_grid_mode = "deterministic_grid"
        target_grid_cases = _make_target_grid_cases(
            args_cli.target_random_x,
            args_cli.target_random_y,
            grid_size=grid_size,
            repeats_per_point=grid_repeats_per_point,
        )
    elif auto_test and args_cli.grid_random:
        target_grid_mode = "grid_random"
        target_grid_cases = _make_target_grid_random_cases(
            args_cli.target_random_x,
            args_cli.target_random_y,
            grid_size=grid_size,
            samples_per_cell=grid_repeats_per_point,
        )
    if target_grid_cases is not None:
        if int(args_cli.test_times) != len(target_grid_cases):
            print(
                f"[Policy] --{target_grid_mode.replace('_', '-')} enabled: overriding --test-times "
                f"from {int(args_cli.test_times)} to {len(target_grid_cases)} "
                f"({grid_size}x{grid_size} grid, {grid_repeats_per_point} repeats per point)."
            )
        test_times = len(target_grid_cases)
    task_status_print_interval = max(0, int(args_cli.task_status_print_interval))
    reset_settle_seconds = max(0.0, float(args_cli.reset_settle_seconds))
    reset_settle_frames = int(math.ceil(reset_settle_seconds / _get_env_step_dt(env))) if reset_settle_seconds > 0 else 0

    if auto_test:
        mode = "HEADLESS" if args_cli.headless else "GUI"
        print(
            f"[{mode} AUTO TEST] Policy rollout active. "
            f"tests={test_times}, max_frames_per_test={max_frames_per_test}, "
            f"reset_settle={reset_settle_seconds:.3f}s/{reset_settle_frames} frames"
        )
        if target_grid_cases is not None:
            placement = (
                f"deterministic {grid_size}x{grid_size} grid points"
                if target_grid_mode == "deterministic_grid"
                else f"random samples inside {grid_size}x{grid_size} grid cells"
            )
            print(
                f"[Policy] Target placement: {placement}, "
                f"{grid_repeats_per_point} repeats per grid point, "
                f"x_range=+/-{max(0.0, float(args_cli.target_random_x)):.4f} m, "
                f"y_range=+/-{max(0.0, float(args_cli.target_random_y)):.4f} m"
            )
    else:
        if args_cli.headless:
            raise RuntimeError("--interactive/--start-paused require GUI mode. Remove --headless to use hotkeys.")
        policy_active = not args_cli.start_paused
        hotkeys = PolicyHotkeys()
        print(f"[Policy] {'Active' if policy_active else 'Paused'}. Hotkeys: T=start, P=pause, N=reset")

    try:
        if auto_test:
            detailed_results = {}
            tests_run = 0
            progress_success_count = 0
            if args_cli.grid_random:
                _console_progress(0, test_times, 0)
            for test_index in range(1, test_times + 1):
                if not simulation_app.is_running():
                    break

                tests_run += 1
                test_key = f"test_{test_index - 1}"
                target_grid_case = target_grid_cases[test_index - 1] if target_grid_cases is not None else None
                target_joints_1, target_joints_2 = _reset_environment(
                    env,
                    policy,
                    preprocessor,
                    postprocessor,
                    cached_xforms,
                    args_cli.target_random_x,
                    args_cli.target_random_y,
                    target_grid_case,
                    target_grid_mode,
                )
                actual_settle_frames = _settle_environment_after_reset(
                    env,
                    actions,
                    target_joints_1,
                    target_joints_2,
                    reset_settle_seconds,
                )
                _reset_policy(policy)
                preprocessor.reset()
                postprocessor.reset()
                if target_grid_case is None:
                    print(f"\n[Test] {test_index}/{test_times} started.")
                else:
                    cell_text = ""
                    if target_grid_case.cell_x_min is not None:
                        cell_text = (
                            " "
                            f"cell_x=[{target_grid_case.cell_x_min:+.4f}, {target_grid_case.cell_x_max:+.4f}], "
                            f"cell_y=[{target_grid_case.cell_y_min:+.4f}, {target_grid_case.cell_y_max:+.4f}],"
                        )
                    print(
                        f"\n[Test] {test_index}/{test_times} started. "
                        f"grid=({target_grid_case.grid_x_index + 1}/{grid_size}, "
                        f"{target_grid_case.grid_y_index + 1}/{grid_size}), "
                        f"repeat={target_grid_case.repeat_index + 1}/{grid_repeats_per_point}, "
                        f"{cell_text} "
                        f"offset=({target_grid_case.offset_x:+.4f}, {target_grid_case.offset_y:+.4f}) m"
                    )

                success = False
                completion_frames = None
                frames_run = 0
                termination_reason = "max_frames_reached"
                for frame_count in range(1, max_frames_per_test + 1):
                    frames_run = frame_count
                    if not simulation_app.is_running():
                        termination_reason = "simulation_stopped"
                        break

                    observation = _make_policy_observation(env, runtime)
                    observation = preprocessor(observation)
                    with torch.inference_mode():
                        policy_action = policy.select_action(observation)
                    policy_action = postprocessor(policy_action)
                    policy_target_1, policy_target_2 = _split_policy_action(policy_action, env.unwrapped.device)
                    policy_target_1 = _limit_joint_delta(target_joints_1, policy_target_1, max_joint_delta)
                    policy_target_2 = _limit_joint_delta(target_joints_2, policy_target_2, max_joint_delta)
                    target_joints_1 = _blend_targets(
                        target_joints_1,
                        policy_target_1.expand_as(target_joints_1),
                        smoothing,
                    )
                    target_joints_2 = _blend_targets(
                        target_joints_2,
                        policy_target_2.expand_as(target_joints_2),
                        smoothing,
                    )

                    _write_joint_actions(actions, target_joints_1, target_joints_2)
                    _, _, terminated, truncated, _ = env.step(actions)

                    task_status = task_monitor.read_status()
                    task_complete = bool(task_status["complete"])
                    if task_status_print_interval > 0 and frame_count % task_status_print_interval == 0:
                        task_monitor.print_status(frame_count, task_status)

                    if task_complete:
                        success = True
                        completion_frames = frame_count
                        termination_reason = "task_complete"
                        print(f"[Test] {test_index}/{test_times} SUCCESS at frame {frame_count}.")
                        break

                    done = bool(torch.any(terminated) or torch.any(truncated))
                    if done:
                        termination_reason = "environment_terminated"
                        print(f"[Test] {test_index}/{test_times} ended by env termination at frame {frame_count}.")
                        break

                detailed_results[test_key] = {
                    "success": success,
                    "completion_frames": completion_frames,
                    "frames_run": frames_run,
                    "reset_settle_frames": actual_settle_frames,
                    "termination_reason": termination_reason,
                }
                if target_grid_case is not None:
                    detailed_results[test_key]["target_grid"] = {
                        "case_index": target_grid_case.case_index,
                        "grid_x_index": target_grid_case.grid_x_index,
                        "grid_y_index": target_grid_case.grid_y_index,
                        "repeat_index": target_grid_case.repeat_index,
                        "offset_x": target_grid_case.offset_x,
                        "offset_y": target_grid_case.offset_y,
                    }
                    if target_grid_case.cell_x_min is not None:
                        detailed_results[test_key]["target_grid"].update(
                            {
                                "cell_x_min": target_grid_case.cell_x_min,
                                "cell_x_max": target_grid_case.cell_x_max,
                                "cell_y_min": target_grid_case.cell_y_min,
                                "cell_y_max": target_grid_case.cell_y_max,
                            }
                        )
                if not success:
                    print(f"[Test] {test_index}/{test_times} FAILED within {max_frames_per_test} frames.")
                if args_cli.grid_random:
                    if success:
                        progress_success_count += 1
                    _console_progress(tests_run, test_times, progress_success_count)

            if args_cli.grid_random:
                _console_print()

            target_sampling = None
            if target_grid_cases is not None:
                target_sampling = {
                    "mode": target_grid_mode,
                    "grid_size": grid_size,
                    "repeats_per_point": grid_repeats_per_point,
                    "target_random_x": max(0.0, float(args_cli.target_random_x)),
                    "target_random_y": max(0.0, float(args_cli.target_random_y)),
                    "total_cases": len(target_grid_cases),
                }
            result_entry = _make_test_result_entry(
                runtime=runtime,
                requested_test_count=test_times,
                completed_test_count=tests_run,
                max_frames_per_test=max_frames_per_test,
                reset_settle_seconds=reset_settle_seconds,
                reset_settle_frames=reset_settle_frames,
                detailed_results=detailed_results,
                target_sampling=target_sampling,
            )
            if args_cli.grid_random:
                if grid_random_output_dir is None:
                    raise RuntimeError("grid_random output directory was not initialized.")
                result_path, result_key = _save_test_results_json_at(grid_random_output_dir / "result.json", result_entry)
                artifact_dir = grid_random_output_dir / result_key
                _generate_grid_result_artifacts(result_path, result_key, artifact_dir)
            else:
                result_path, result_key = _save_test_results_json(runtime.policy_path, result_entry)
                artifact_dir = None
                _warn_if_legacy_txt_exists(runtime.policy_path)
            result_text = _format_test_results_for_terminal(result_entry)
            print("\n" + result_text, end="")
            print(f"[Policy] Test results saved to: {result_path} ({result_key})")
            if artifact_dir is not None:
                print(f"[Policy] Grid summary artifacts saved to: {artifact_dir}")
                _console_print(f"[grid_random] result: {result_path} ({result_key})")
                _console_print(f"[grid_random] artifacts: {artifact_dir}")
                _console_print(f"[grid_random] log: {terminal_log_path}")
            return

        target_joints_1, target_joints_2 = _reset_environment(
            env,
            policy,
            preprocessor,
            postprocessor,
            cached_xforms,
            args_cli.target_random_x,
            args_cli.target_random_y,
        )
        _settle_environment_after_reset(env, actions, target_joints_1, target_joints_2, reset_settle_seconds)
        _reset_policy(policy)
        preprocessor.reset()
        postprocessor.reset()
        step_count = 0
        task_complete_notified = False
        while simulation_app.is_running():
            if hotkeys is not None:
                events = hotkeys.poll()
                if events["start"]:
                    policy_active = True
                    _reset_policy(policy)
                    preprocessor.reset()
                    postprocessor.reset()
                    print("[Policy] Started.")
                if events["pause"]:
                    policy_active = False
                    print("[Policy] Paused.")
                if events["reset"]:
                    target_joints_1, target_joints_2 = _reset_environment(
                        env,
                        policy,
                        preprocessor,
                        postprocessor,
                        cached_xforms,
                        args_cli.target_random_x,
                        args_cli.target_random_y,
                    )
                    _settle_environment_after_reset(env, actions, target_joints_1, target_joints_2, reset_settle_seconds)
                    _reset_policy(policy)
                    preprocessor.reset()
                    postprocessor.reset()
                    task_complete_notified = False
                    print("[Reset] Environment reset to initial robot defaults.")

            if policy_active:
                observation = _make_policy_observation(env, runtime)
                observation = preprocessor(observation)
                with torch.inference_mode():
                    policy_action = policy.select_action(observation)
                policy_action = postprocessor(policy_action)
                policy_target_1, policy_target_2 = _split_policy_action(policy_action, env.unwrapped.device)
                policy_target_1 = _limit_joint_delta(target_joints_1, policy_target_1, max_joint_delta)
                policy_target_2 = _limit_joint_delta(target_joints_2, policy_target_2, max_joint_delta)
                target_joints_1 = _blend_targets(target_joints_1, policy_target_1.expand_as(target_joints_1), smoothing)
                target_joints_2 = _blend_targets(target_joints_2, policy_target_2.expand_as(target_joints_2), smoothing)

            _write_joint_actions(actions, target_joints_1, target_joints_2)
            _, _, terminated, truncated, _ = env.step(actions)

            task_status = task_monitor.read_status()
            task_complete = bool(task_status["complete"])
            if task_status_print_interval > 0 and step_count % task_status_print_interval == 0:
                task_monitor.print_status(step_count, task_status)
            if task_complete and not task_complete_notified:
                task_complete_notified = True
                print("[Task Complete] Target is fully inside the plane area.", flush=True)
            elif not task_complete:
                task_complete_notified = False

            done = bool(torch.any(terminated) or torch.any(truncated))
            if done:
                target_joints_1, target_joints_2 = _reset_environment(
                    env,
                    policy,
                    preprocessor,
                    postprocessor,
                    cached_xforms,
                    args_cli.target_random_x,
                    args_cli.target_random_y,
                )
                task_complete_notified = False

            if step_count % 50 == 0:
                print(
                    f"step={step_count} active={policy_active} "
                    f"robot_1_target={target_joints_1[0].tolist()} "
                    f"robot_2_target={target_joints_2[0].tolist()}"
                )

            step_count += 1
    finally:
        env.close()
        print("\n[Environment Closed]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
        _close_terminal_log()
