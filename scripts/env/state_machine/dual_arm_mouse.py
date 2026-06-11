"""Dual-arm teleoperation with keyboard for arm 1 and SpaceMouse for arm 2.

Usage (GUI mode):
  ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/dual_arm_mouse.py --num_envs 1

Usage (headless smoke test):
  ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/dual_arm_mouse.py --num_envs 1 --headless

Controls:
  Arm 1 position:
    W/S -> +/- Y
    A/D -> -/+ X
    Q/E -> +/- Z
  Arm 1 orientation (axis-angle increments):
    Z -> +Rx,  Shift+Z -> -Rx
    X -> +Ry,  Shift+X -> -Ry
    C -> +Rz,  Shift+C -> -Rz
  Arm 2:
    3Dconnexion SpaceMouse controls the end-effector pose incrementally.
  Common:
    R -> reset environment
    I -> arm 2 gripper open (symmetric +15deg / -15deg jaw target)
    O -> arm 2 gripper close
    Target finish:
        Arm 2 gripper contact with Sphere_02 prints "task finish" and freezes motion until reset or timeout.
"""

import argparse
import math
import os
import re

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Dual-arm environment with keyboard + SpaceMouse teleoperation.")
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
parser.add_argument(
    "--spacemouse-device",
    type=str,
    default=None,
    help="Optional SpaceMouse device name to open. If omitted, the first supported device is used.",
)
parser.add_argument(
    "--spacemouse-device-index",
    type=int,
    default=0,
    help="Which matching SpaceMouse device to open if multiple are connected.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import asb_dual_arm.tasks  # noqa: F401


GRIPPER_OPEN_ANGLE_DEG = 15.0
GRIPPER_OPEN_ANGLE_RAD = math.radians(GRIPPER_OPEN_ANGLE_DEG)
GRIPPER_CLOSE_ANGLE_RAD = 0.0
ROBOT_1_GRIPPER_CLOSED_POS = (0.01, -0.01)
TARGETPOINT_CONTACT_SENSOR_NAMES = ("targetpoint_contact_gripper1", "targetpoint_contact_gripper2")
TARGETPOINT_CONTACT_FORCE_THRESHOLD = 0.0


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

    try:
        from asb_dual_arm.tasks.direct.dual_arm.msr import joint_pos_env_cfg as dual_arm_joint_cfg

        dual_arm_joint_cfg.apply_robot_1_init_state_from_usd(env_cfg, spawn_cfg.usd_path, verbose=True)
        dual_arm_joint_cfg.apply_robot_2_init_state_from_usd(env_cfg, spawn_cfg.usd_path, verbose=True)
    except Exception as err:
        print(f"  [WARNING] Failed to refresh robot init states from overridden USD: {err}")


def enable_robot_2_gripper_action(env_cfg) -> None:
    """Enable a binary gripper action term for robot 2."""
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        raise RuntimeError("Environment configuration has no 'actions' section.")

    from asb_dual_arm.tasks.direct.dual_arm import mdp as dual_arm_mdp

    actions_cfg.gripper_2_action = dual_arm_mdp.BinaryJointPositionActionCfg(
        asset_name="robot_2",
        joint_names=["psm_gripper1_Joint", "psm_gripper2_Joint"],
        open_command_expr={
            "psm_gripper1_Joint": GRIPPER_OPEN_ANGLE_RAD,
            "psm_gripper2_Joint": -GRIPPER_OPEN_ANGLE_RAD,
        },
        close_command_expr={
            "psm_gripper1_Joint": GRIPPER_CLOSE_ANGLE_RAD,
            "psm_gripper2_Joint": -GRIPPER_CLOSE_ANGLE_RAD,
        },
    )


def find_action_slice(env, term_name: str) -> slice | None:
    term_names = list(env.unwrapped.action_manager.active_terms)
    term_dims = list(env.unwrapped.action_manager.action_term_dim)

    offset = 0
    for name, dim in zip(term_names, term_dims):
        next_offset = offset + dim
        if name == term_name:
            return slice(offset, next_offset)
        offset = next_offset
    return None


def setup_robot_1_gripper_lock(env) -> tuple[object, list[int], torch.Tensor]:
    """Prepare robot_1 gripper indices and a batched closed target tensor."""
    robot_1 = env.unwrapped.scene["robot_1"]
    gripper_joint_ids, _ = robot_1.find_joints(["psm_gripper1_Joint", "psm_gripper2_Joint"], preserve_order=True)
    num_envs = robot_1.data.joint_pos.shape[0]
    closed_joint_pos = torch.tensor([ROBOT_1_GRIPPER_CLOSED_POS], device=env.unwrapped.device).repeat(num_envs, 1)
    return robot_1, gripper_joint_ids, closed_joint_pos


def enforce_robot_1_gripper_closed(robot_1, gripper_joint_ids, closed_joint_pos: torch.Tensor, write_state: bool = False) -> None:
    """Force robot_1 gripper closed by setting targets every step; optionally hard-reset state."""
    if write_state:
        closed_joint_vel = torch.zeros_like(closed_joint_pos)
        robot_1.write_joint_state_to_sim(closed_joint_pos, closed_joint_vel, joint_ids=gripper_joint_ids)
    robot_1.set_joint_position_target(closed_joint_pos, joint_ids=gripper_joint_ids)


def _contact_sensor_has_force(contact_sensor) -> bool:
    force_matrix_w_history = getattr(contact_sensor.data, "force_matrix_w_history", None)
    if force_matrix_w_history is not None:
        contact_force = torch.linalg.vector_norm(force_matrix_w_history, dim=-1)
    else:
        net_forces_w_history = getattr(contact_sensor.data, "net_forces_w_history", None)
        if net_forces_w_history is None:
            return False
        contact_force = torch.linalg.vector_norm(net_forces_w_history, dim=-1)

    contact_force = torch.nan_to_num(contact_force, nan=0.0)
    return bool(torch.any(contact_force > TARGETPOINT_CONTACT_FORCE_THRESHOLD))


def _targetpoint_contact_detected(env) -> bool:
    """Return True once robot_2 gripper contact with Sphere_02 is observed."""
    scene_sensors = getattr(env.unwrapped.scene, "sensors", {})
    for sensor_name in TARGETPOINT_CONTACT_SENSOR_NAMES:
        contact_sensor = scene_sensors.get(sensor_name)
        if contact_sensor is not None and _contact_sensor_has_force(contact_sensor):
            return True
    return False


def _to_int16(low_byte: int, high_byte: int) -> int:
    value = low_byte | (high_byte << 8)
    if value >= 32768:
        value = -(65536 - value)
    return value


def _scale_axis(raw_value: int, scale: float = 350.0) -> float:
    return max(min(raw_value / scale, 1.0), -1.0)


class DualArmKeyboardController:
    """Poll-based keyboard controller for arm 1."""

    def __init__(self, device: torch.device | str, sensitivity: float):
        import carb
        import omni.appwindow

        self._carb = carb
        self._device = device
        self._pos_step = 0.005 * sensitivity
        self._rot_step = 0.05 * sensitivity
        self._last_reset_down = False
        self._last_gripper_open_down = False
        self._last_gripper_close_down = False

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

    def poll(self) -> tuple[torch.Tensor, bool, bool, bool]:
        key = self._carb.input.KeyboardInput

        delta_1 = torch.zeros(6, device=self._device)

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

        delta_1[3] += self._rotation_sign(key.Z) * self._rot_step
        delta_1[4] += self._rotation_sign(key.X) * self._rot_step
        delta_1[5] += self._rotation_sign(key.C) * self._rot_step

        reset_down = self._is_down(key.R)
        reset_requested = reset_down and not self._last_reset_down
        self._last_reset_down = reset_down

        gripper_open_down = self._is_down(key.I)
        gripper_close_down = self._is_down(key.O)
        gripper_open_requested = gripper_open_down and not self._last_gripper_open_down
        gripper_close_requested = gripper_close_down and not self._last_gripper_close_down
        self._last_gripper_open_down = gripper_open_down
        self._last_gripper_close_down = gripper_close_down

        return delta_1, reset_requested, gripper_open_requested, gripper_close_requested


class SpaceMouseController:
    """Direct hidapi reader for arm 2."""

    def __init__(
        self,
        sim_device: torch.device | str,
        sensitivity: float,
        device_name: str | None = None,
        device_index: int = 0,
    ):
        try:
            import hid
        except ImportError as exc:
            raise SystemExit("hidapi is not available in this Python environment.") from exc

        self._hid = hid
        self._sim_device = sim_device
        self._pos_step = 0.005 * sensitivity
        self._rot_step = 0.05 * sensitivity
        self._device = hid.device()
        self._device_name, self._device_path = self._find_device(device_name, device_index)

        try:
            self._device.open_path(self._device_path)
            self._device.set_nonblocking(True)
        except Exception as exc:
            raise SystemExit(self._device_open_help()) from exc

    def _device_open_help(self) -> str:
        return (
            "Failed to open the SpaceMouse device. "
            "If you are running inside Docker, pass the HID device through with a compose device mount, "
            'for example: devices: ["/dev/hidraw<#>:/dev/hidraw<#>"] . '
            "On a local Ubuntu host, you may also need to grant access with: sudo chmod 666 /dev/hidraw<#>. "
            "To identify the right device, run: ls -l /dev/hidraw* and check /sys/class/hidraw/hidraw<#>/device/uevent."
        )

    def _find_device(self, device_name: str | None, device_index: int) -> tuple[str, bytes]:
        device_names = {
            "SpaceMouse Compact",
            "SpaceMouse Wireless",
            "SpaceMouse Pro",
            "SpaceNavigator",
            "SpaceExplorer",
            "SpaceMouse Pro Wireless",
        }
        device_ids = {
            (0x046D, 0xC626),
            (0x046D, 0xC627),
            (0x046D, 0xC62B),
            (0x256F, 0xC635),
            (0x256F, 0xC62E),
            (0x256F, 0xC632),
            (0x256F, 0xC638),
            (0x256F, 0xC641),
            (0x256F, 0xC652),
            (0x256F, 0xC633),
        }

        matches: list[dict[str, object]] = []
        for device_info in self._hid.enumerate():
            product_name = device_info.get("product_string", "")
            vendor_id = device_info.get("vendor_id")
            product_id = device_info.get("product_id")
            if device_name is not None:
                if product_name == device_name:
                    matches.append(device_info)
            elif (vendor_id, product_id) in device_ids or product_name in device_names:
                matches.append(device_info)

        if not matches:
            raise SystemExit("No supported 3Dconnexion device was found by hidapi.")

        selected_index = max(0, min(device_index, len(matches) - 1))
        selected = matches[selected_index]
        path = selected.get("path")
        if isinstance(path, str):
            path_bytes = path.encode("utf-8")
        elif isinstance(path, (bytes, bytearray)):
            path_bytes = bytes(path)
        else:
            raise SystemExit(self._device_open_help())

        return str(selected.get("product_string", "unknown device")), path_bytes

    def poll(self) -> torch.Tensor:
        delta_2 = torch.zeros(6, device=self._sim_device)

        while True:
            data = self._device.read(64)
            if not data:
                break

            if data[0] == 1:
                delta_2[0] = self._pos_step * _scale_axis(_to_int16(data[3], data[4]))
                delta_2[1] = self._pos_step * _scale_axis(_to_int16(data[1], data[2]))
                delta_2[2] = -self._pos_step * _scale_axis(_to_int16(data[5], data[6]))
            elif data[0] == 2:
                delta_2[3] = self._rot_step * _scale_axis(_to_int16(data[3], data[4]))
                delta_2[4] = self._rot_step * _scale_axis(_to_int16(data[1], data[2]))
                delta_2[5] = -self._rot_step * _scale_axis(_to_int16(data[5], data[6]))

        return delta_2

    def close(self) -> None:
        self._device.close()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_organ_usd_override(env_cfg, args_cli.usd_path)
    enable_robot_2_gripper_action(env_cfg)

    print("\n" + "=" * 80)
    print(f"Creating Environment: {args_cli.task}")
    print("=" * 80)
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {args_cli.device}")
    print("=" * 80 + "\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot_1, robot_1_gripper_joint_ids, robot_1_closed_joint_pos = setup_robot_1_gripper_lock(env)
    enforce_robot_1_gripper_closed(
        robot_1,
        robot_1_gripper_joint_ids,
        robot_1_closed_joint_pos,
        write_state=True,
    )
    print(
        "[Gripper 1] Forced closed lock enabled: "
        f"psm_gripper1={ROBOT_1_GRIPPER_CLOSED_POS[0]:.4f} rad, "
        f"psm_gripper2={ROBOT_1_GRIPPER_CLOSED_POS[1]:.4f} rad"
    )

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

    gripper_2_action_slice = find_action_slice(env, "gripper_2_action")
    if gripper_2_action_slice is not None:
        print(f"  gripper_2_action slice: [{gripper_2_action_slice.start}:{gripper_2_action_slice.stop})")
    else:
        print("  [WARNING] gripper_2_action is not active; I/O gripper control is unavailable.")

    scene_sensors = getattr(env.unwrapped.scene, "sensors", {})
    missing_contact_sensors = [name for name in TARGETPOINT_CONTACT_SENSOR_NAMES if name not in scene_sensors]
    targetpoint_contact_enabled = len(missing_contact_sensors) < len(TARGETPOINT_CONTACT_SENSOR_NAMES)

    keyboard_teleop = None
    spacemouse_teleop = None
    gripper_2_action_value = -1.0
    task_finished = False
    if args_cli.headless:
        print("\n[HEADLESS MODE] Teleoperation disabled. Running smoke test loop.")
    else:
        try:
            keyboard_teleop = DualArmKeyboardController(env.unwrapped.device, args_cli.sensitivity)
            spacemouse_teleop = SpaceMouseController(
                env.unwrapped.device,
                args_cli.sensitivity,
                device_name=args_cli.spacemouse_device,
                device_index=args_cli.spacemouse_device_index,
            )
            print("\nTeleoperation active.")
            print("  Arm 1 keyboard: W/S, A/D, Q/E for translation; Z/X/C with Shift for rotation")
            print("  Arm 2 SpaceMouse: push/pull/tilt/twist the 3D mouse")
            print("  Arm 2 gripper keyboard: I=open (15deg each jaw), O=close (one-shot toggle)")
            print("  R: reset")
            print("  [Gripper 2] Initial state: CLOSED")
            if targetpoint_contact_enabled:
                print("  Task finish: robot_2 gripper contact with Sphere_02 freezes motion until reset or timeout.")
            else:
                print(
                    "  [WARNING] targetpoint contact sensors are missing; "
                    "finish detection is disabled."
                )
        except Exception as err:
            print(f"[WARNING] Teleoperation setup failed ({err}), fallback to headless behavior.")

    step_count = 0
    while simulation_app.is_running():
        if keyboard_teleop is not None and spacemouse_teleop is not None:
            delta_1, reset_requested, gripper_open_requested, gripper_close_requested = keyboard_teleop.poll()
            delta_2 = spacemouse_teleop.poll()

            if task_finished:
                delta_1.zero_()
                delta_2.zero_()
                gripper_open_requested = False
                gripper_close_requested = False
            else:
                if gripper_open_requested:
                    gripper_2_action_value = 1.0
                    print(
                        f"[Gripper 2] OPEN latched from key I: "
                        f"psm_gripper1={GRIPPER_OPEN_ANGLE_RAD:.4f} rad, psm_gripper2={-GRIPPER_OPEN_ANGLE_RAD:.4f} rad"
                    )
                if gripper_close_requested:
                    gripper_2_action_value = -1.0
                    print(
                        f"[Gripper 2] CLOSED latched from key O: "
                        f"psm_gripper1={GRIPPER_CLOSE_ANGLE_RAD:.4f} rad, psm_gripper2={-GRIPPER_CLOSE_ANGLE_RAD:.4f} rad"
                    )
        else:
            delta_1 = torch.zeros(6, device=env.unwrapped.device)
            delta_2 = torch.zeros(6, device=env.unwrapped.device)
            reset_requested = False
            if args_cli.headless and step_count >= args_cli.max_steps_headless:
                print(f"\n[INFO] Headless test completed after {step_count} steps.")
                break

        if reset_requested:
            env.reset()
            enforce_robot_1_gripper_closed(
                robot_1,
                robot_1_gripper_joint_ids,
                robot_1_closed_joint_pos,
                write_state=True,
            )
            gripper_2_action_value = -1.0
            task_finished = False
            if keyboard_teleop is not None and spacemouse_teleop is not None:
                delta_1.zero_()
                delta_2.zero_()

        actions.zero_()
        raw_1 = delta_1 / arm_1_scale
        raw_2 = delta_2 / arm_2_scale

        if actions.ndim == 2:
            actions[:, 0:6] = raw_1.unsqueeze(0).expand(actions.shape[0], -1)
            actions[:, 6:12] = raw_2.unsqueeze(0).expand(actions.shape[0], -1)
            if gripper_2_action_slice is not None:
                actions[:, gripper_2_action_slice] = gripper_2_action_value
        else:
            actions[0:6] = raw_1
            actions[6:12] = raw_2
            if gripper_2_action_slice is not None:
                actions[gripper_2_action_slice] = gripper_2_action_value

        enforce_robot_1_gripper_closed(
            robot_1,
            robot_1_gripper_joint_ids,
            robot_1_closed_joint_pos,
            write_state=False,
        )

        _, _, terminated, truncated, _ = env.step(actions)

        if not task_finished and targetpoint_contact_enabled and _targetpoint_contact_detected(env):
            task_finished = True
            print("task finish")

        if torch.is_tensor(terminated):
            done = bool(torch.any(terminated) or torch.any(truncated))
        else:
            done = bool(terminated or truncated)
        if done:
            env.reset()
            enforce_robot_1_gripper_closed(
                robot_1,
                robot_1_gripper_joint_ids,
                robot_1_closed_joint_pos,
                write_state=True,
            )
            gripper_2_action_value = -1.0
            task_finished = False

        if step_count % 200 == 0 and step_count > 0:
            print(f"step={step_count} arm1_delta={delta_1.tolist()} arm2_delta={delta_2.tolist()}")

        step_count += 1

    env.close()
    if spacemouse_teleop is not None:
        spacemouse_teleop.close()
    print("\n[Environment Closed]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
