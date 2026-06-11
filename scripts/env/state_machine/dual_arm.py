# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dual-arm keyboard teleoperation script.

Usage (GUI mode):
  ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/dual_arm.py --num_envs 1

Usage (headless smoke test):
  ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/dual_arm.py --num_envs 1 --headless

Keyboard mapping:
  Arm 1 position:
    W/S -> +/- Y
    A/D -> -/+ X
    Q/E -> +/- Z
  Arm 2 position:
    I/K -> +/- Y
    J/L -> -/+ X
    U/O -> +/- Z
  Arm 1 orientation (axis-angle increments):
    Z -> +Rx,  Shift+Z -> -Rx
    X -> +Ry,  Shift+X -> -Ry
    C -> +Rz,  Shift+C -> -Rz
  Arm 2 orientation (axis-angle increments):
    B -> +Rx,  Shift+B -> -Rx
    N -> +Ry,  Shift+N -> -Ry
    M -> +Rz,  Shift+M -> -Rz
  Common:
        R -> reset environment
"""

import argparse
import os
import re

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Dual-arm environment keyboard teleoperation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor for keyboard control.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-DualArm-VesselSem-DualArm-IK-Play-v0",
    help="Task ID used to create the environment.",
)
parser.add_argument(
    "--usd-path",
    "--usd_path",
    dest="usd_path",
    type=str,
    default=None,
    help="Optional scene USD override.",
)
parser.add_argument(
    "--max-steps-headless",
    "--max_steps_headless",
    dest="max_steps_headless",
    type=int,
    default=500,
    help="Maximum simulation steps in headless mode.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import msr.tasks  # noqa: F401


def _resolve_usd_path(raw_path: str) -> str:
    expanded = os.path.expanduser(raw_path)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", expanded):
        return expanded
    return os.path.abspath(expanded)


def apply_organ_usd_override(env_cfg, usd_path_override: str | None):
    if not usd_path_override:
        return

    resolved_path = _resolve_usd_path(usd_path_override)
    if "://" not in resolved_path and not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"USD file does not exist: {resolved_path}")

    organ_cfg = getattr(getattr(env_cfg, "scene", None), "organ", None)
    if organ_cfg is None:
        raise RuntimeError("Environment scene has no 'organ' asset to override.")

    spawn_cfg = getattr(organ_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "usd_path"):
        raise RuntimeError("Environment scene.organ.spawn has no configurable usd_path.")

    default_path = spawn_cfg.usd_path
    spawn_cfg.usd_path = resolved_path
    print("  USD override applied:")
    print(f"    default: {default_path}")
    print(f"    active : {spawn_cfg.usd_path}")

    # Keep both robots' init_state aligned with the overridden USD.
    try:
        from msr.tasks.direct.dual_arm.msr import joint_pos_env_cfg as dual_arm_joint_cfg

        dual_arm_joint_cfg.apply_robot_1_init_state_from_usd(env_cfg, spawn_cfg.usd_path, verbose=True)
        dual_arm_joint_cfg.apply_robot_2_init_state_from_usd(env_cfg, spawn_cfg.usd_path, verbose=True)
    except Exception as err:
        print(f"  [WARNING] Failed to refresh robot init states from overridden USD: {err}")


class DualArmKeyboardController:
    """Poll-based keyboard controller with shift-modified orientation keys."""

    def __init__(self, device: torch.device | str, sensitivity: float):
        import carb
        import omni.appwindow

        self._carb = carb
        self._device = device
        self._pos_step = 0.005 * sensitivity
        self._rot_step = 0.05 * sensitivity
        self._last_reset_down = False

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def _is_down(self, key) -> bool:
        flags = self._input.get_keyboard_button_flags(self._keyboard, key)
        return bool(flags & self._carb.input.BUTTON_FLAG_DOWN)

    def _shift_down(self) -> bool:
        key = self._carb.input.KeyboardInput
        return self._is_down(key.LEFT_SHIFT) or self._is_down(key.RIGHT_SHIFT)

    def _rotation_sign(self, key) -> float:
        if not self._is_down(key):
            return 0.0
        return -1.0 if self._shift_down() else 1.0

    def poll(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        key = self._carb.input.KeyboardInput

        delta_1 = torch.zeros(6, device=self._device)
        delta_2 = torch.zeros(6, device=self._device)

        # Arm 1 translation (x, y, z)
        if self._is_down(key.A):
            delta_1[0] -= self._pos_step
        if self._is_down(key.D):
            delta_1[0] += self._pos_step
        if self._is_down(key.W):
            delta_1[1] += self._pos_step
        if self._is_down(key.S):
            delta_1[1] -= self._pos_step
        if self._is_down(key.Q):
            delta_1[2] += self._pos_step
        if self._is_down(key.E):
            delta_1[2] -= self._pos_step

        # Arm 2 translation (x, y, z)
        if self._is_down(key.J):
            delta_2[0] -= self._pos_step
        if self._is_down(key.L):
            delta_2[0] += self._pos_step
        if self._is_down(key.I):
            delta_2[1] += self._pos_step
        if self._is_down(key.K):
            delta_2[1] -= self._pos_step
        if self._is_down(key.U):
            delta_2[2] += self._pos_step
        if self._is_down(key.O):
            delta_2[2] -= self._pos_step

        # Arm 1 orientation: Z/X/C -> Rx/Ry/Rz, shift for reverse.
        delta_1[3] += self._rotation_sign(key.Z) * self._rot_step
        delta_1[4] += self._rotation_sign(key.X) * self._rot_step
        delta_1[5] += self._rotation_sign(key.C) * self._rot_step

        # Arm 2 orientation: B/N/M -> Rx/Ry/Rz, shift for reverse.
        delta_2[3] += self._rotation_sign(key.B) * self._rot_step
        delta_2[4] += self._rotation_sign(key.N) * self._rot_step
        delta_2[5] += self._rotation_sign(key.M) * self._rot_step

        # Edge-triggered reset on key R.
        reset_down = self._is_down(key.R)
        reset_requested = reset_down and not self._last_reset_down
        self._last_reset_down = reset_down

        return delta_1, delta_2, reset_requested


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_organ_usd_override(env_cfg, args_cli.usd_path)

    print("\n" + "=" * 80)
    print(f"Creating Environment: {args_cli.task}")
    print("=" * 80)
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {args_cli.device}")
    print("=" * 80 + "\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    action_dim = actions.shape[-1] if actions.ndim > 1 else actions.shape[0]
    if action_dim < 12:
        raise RuntimeError(f"Expected action dimension >= 12 for dual-arm IK, got {action_dim}.")

    arm_1_scale = float(getattr(getattr(env_cfg.actions, "arm_1_action", None), "scale", 1.0))
    arm_2_scale = float(getattr(getattr(env_cfg.actions, "arm_2_action", None), "scale", 1.0))
    if arm_1_scale == 0.0 or arm_2_scale == 0.0:
        raise RuntimeError(f"Action scale cannot be zero (arm_1={arm_1_scale}, arm_2={arm_2_scale}).")

    print("Action configuration:")
    print(f"  action_dim: {action_dim}")
    print(f"  arm_1_scale: {arm_1_scale}")
    print(f"  arm_2_scale: {arm_2_scale}")

    teleop = None
    if args_cli.headless:
        print("\n[HEADLESS MODE] Keyboard control disabled. Running smoke test loop.")
    else:
        try:
            teleop = DualArmKeyboardController(env.unwrapped.device, args_cli.sensitivity)
            print("\nKeyboard controls active.")
            print("  Arm 1 translation: W/S (Y), A/D (X), Q/E (Z)")
            print("  Arm 2 translation: I/K (Y), J/L (X), U/O (Z)")
            print("  Arm 1 orientation: Z/X/C (Rx/Ry/Rz), Shift+key reverse")
            print("  Arm 2 orientation: B/N/M (Rx/Ry/Rz), Shift+key reverse")
            print("  R: reset")
        except Exception as err:
            print(f"[WARNING] Keyboard setup failed ({err}), fallback to headless behavior.")

    step_count = 0
    while simulation_app.is_running():
        if teleop is not None:
            delta_1, delta_2, reset_requested = teleop.poll()
        else:
            delta_1 = torch.zeros(6, device=env.unwrapped.device)
            delta_2 = torch.zeros(6, device=env.unwrapped.device)
            reset_requested = False
            if args_cli.headless and step_count >= args_cli.max_steps_headless:
                print(f"\n[INFO] Headless test completed after {step_count} steps.")
                break

        if reset_requested:
            env.reset()

        actions.zero_()
        raw_1 = delta_1 / arm_1_scale
        raw_2 = delta_2 / arm_2_scale

        if actions.ndim == 2:
            actions[:, 0:6] = raw_1.unsqueeze(0).expand(actions.shape[0], -1)
            actions[:, 6:12] = raw_2.unsqueeze(0).expand(actions.shape[0], -1)
        else:
            actions[0:6] = raw_1
            actions[6:12] = raw_2

        _, _, terminated, truncated, _ = env.step(actions)

        if torch.is_tensor(terminated):
            done = bool(torch.any(terminated) or torch.any(truncated))
        else:
            done = bool(terminated or truncated)
        if done:
            env.reset()

        if step_count % 200 == 0 and step_count > 0:
            print(f"step={step_count} arm1_delta={delta_1.tolist()} arm2_delta={delta_2.tolist()}")

        step_count += 1

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
