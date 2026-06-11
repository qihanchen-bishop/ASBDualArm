"""Dual-Franka deformable-sheet teleoperation with keyboard and SpaceMouse.

Usage (GUI mode):
  ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/franka_pick_mouse.py --num_envs 1

Usage (headless smoke test):
  ./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/franka_pick_mouse.py --num_envs 1 --headless

Controls:
  Arm 1 position:
    W/S -> +/- Y
    A/D -> -/+ X
    Q/E -> +/- Z
  Arm 1 orientation:
    Z -> +Rx, Shift+Z -> -Rx
    X -> +Ry, Shift+X -> -Ry
    C -> +Rz, Shift+C -> -Rz
  Arm 1 gripper:
    J -> open, K -> close
  Arm 2:
    3Dconnexion SpaceMouse controls the end-effector pose incrementally.
  Arm 2 gripper:
    I -> open, O -> close
  Common:
    G -> scripted physical corner-grasp attempt with arm 1
    R -> reset environment
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Dual-Franka deformable-sheet keyboard + SpaceMouse teleoperation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Legacy sensitivity multiplier for teleoperation.")
parser.add_argument(
    "--motion-speed",
    "--motion_speed",
    dest="motion_speed",
    type=float,
    default=2.5,
    help="Motion speed multiplier for keyboard, SpaceMouse, and scripted grasp motion.",
)
parser.add_argument(
    "--attach",
    action="store_true",
    default=False,
    help="Use kinematic node attachment after G closes the gripper. By default, G uses physical grasping only.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-FrankaPick-DualArm-IK-Play-v0",
    help="Task ID used to create the environment.",
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

from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import asb_dual_arm.tasks  # noqa: F401
from asb_dual_arm.tasks.direct.franka_pick.msr.franka_pick_env_cfg import (
    FRANKA_EE_OFFSET,
    SHEET_EFFECTIVE_SIZE,
    SHEET_INIT_POS,
)


FRANKA_OPEN_POS = 0.04
FRANKA_CLOSED_POS = 0.0
SHEET_GRASP_LOCAL_FRACTION = (-0.52, -0.35)
SHEET_PIN_LOCAL_FRACTIONS = ((0.45, -0.45), (0.45, 0.45))
SHEET_GRASP_Z_OFFSET = 0.014
ATTACH_RADIUS = 0.065
ATTACH_MAX_NODES = 24
PIN_RADIUS = 0.06
PIN_MAX_NODES_PER_CORNER = 18


def _to_int16(low_byte: int, high_byte: int) -> int:
    value = low_byte | (high_byte << 8)
    if value >= 32768:
        value = -(65536 - value)
    return value


def _scale_axis(raw_value: int, scale: float = 350.0) -> float:
    return max(min(raw_value / scale, 1.0), -1.0)


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


class FrankaKeyboardController:
    """Poll-based keyboard controller for arm 1."""

    def __init__(self, device: torch.device | str, sensitivity: float):
        import carb
        import omni.appwindow

        self._carb = carb
        self._device = device
        self._pos_step = 0.005 * sensitivity
        self._rot_step = 0.05 * sensitivity
        self._last_reset_down = False
        self._last_gripper_1_open_down = False
        self._last_gripper_1_close_down = False
        self._last_gripper_2_open_down = False
        self._last_gripper_2_close_down = False
        self._last_scripted_grasp_down = False

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

    def _edge_requested(self, key, attr_name: str) -> bool:
        is_down = self._is_down(key)
        was_down = getattr(self, attr_name)
        setattr(self, attr_name, is_down)
        return is_down and not was_down

    def poll(self) -> tuple[torch.Tensor, bool, bool, bool, bool, bool, bool]:
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

        reset_requested = self._edge_requested(key.R, "_last_reset_down")
        gripper_1_open_requested = self._edge_requested(key.J, "_last_gripper_1_open_down")
        gripper_1_close_requested = self._edge_requested(key.K, "_last_gripper_1_close_down")
        gripper_2_open_requested = self._edge_requested(key.I, "_last_gripper_2_open_down")
        gripper_2_close_requested = self._edge_requested(key.O, "_last_gripper_2_close_down")
        scripted_grasp_requested = self._edge_requested(key.G, "_last_scripted_grasp_down")

        return (
            delta_1,
            reset_requested,
            gripper_1_open_requested,
            gripper_1_close_requested,
            gripper_2_open_requested,
            gripper_2_close_requested,
            scripted_grasp_requested,
        )


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
            "On a local Ubuntu host, you may also need to grant access with: sudo chmod 666 /dev/hidraw<#>."
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


class Arm1CornerGraspSequence:
    """Scripted physical grasp attempt; it only commands arm 1 and its gripper."""

    def __init__(self, env, motion_speed: float, pos_step: float = 0.006):
        self._env = env
        self._device = env.unwrapped.device
        self._robot = env.unwrapped.scene["robot_1"]
        body_ids, _ = self._robot.find_bodies(["panda_hand"], preserve_order=True)
        self._hand_body_id = body_ids[0]
        self._pos_step = pos_step * motion_speed
        self._ee_offset = torch.tensor(FRANKA_EE_OFFSET, device=self._device)
        self._stage = "idle"
        self._hold_steps = 0
        self._attach_requested = False
        sheet_x, sheet_y, sheet_z = SHEET_INIT_POS
        size_x, size_y, size_z = SHEET_EFFECTIVE_SIZE
        local_x = SHEET_GRASP_LOCAL_FRACTION[0] * size_x
        local_y = SHEET_GRASP_LOCAL_FRACTION[1] * size_y
        corner_x = sheet_x - local_y
        corner_y = sheet_y + local_x
        grasp_z = sheet_z + 0.5 * size_z + SHEET_GRASP_Z_OFFSET
        self._targets = {
            "pregrasp": torch.tensor([corner_x, corner_y - 0.04, grasp_z + 0.12], device=self._device),
            "descend": torch.tensor([corner_x, corner_y - 0.015, grasp_z], device=self._device),
            "lift": torch.tensor([corner_x, corner_y - 0.015, grasp_z + 0.04], device=self._device),
        }

    @property
    def active(self) -> bool:
        return self._stage != "idle"

    def start(self) -> None:
        self._stage = "pregrasp"
        self._hold_steps = 0
        print("[Arm 1] Scripted physical corner-grasp sequence started.")

    def reset(self) -> None:
        self._stage = "idle"
        self._hold_steps = 0
        self._attach_requested = False

    def consume_attach_request(self) -> bool:
        attach_requested = self._attach_requested
        self._attach_requested = False
        return attach_requested

    def _current_ee_pos(self) -> torch.Tensor:
        hand_pos = self._robot.data.body_pos_w[0, self._hand_body_id]
        hand_quat = self._robot.data.body_quat_w[0, self._hand_body_id]
        offset_w = math_utils.quat_apply(hand_quat.unsqueeze(0), self._ee_offset.unsqueeze(0))[0]
        return hand_pos + offset_w

    def _move_toward(self, target: torch.Tensor) -> tuple[torch.Tensor, bool]:
        delta_w = target - self._current_ee_pos()
        delta_b = math_utils.quat_apply_inverse(self._robot.data.root_quat_w[0].unsqueeze(0), delta_w.unsqueeze(0))[0]
        dist = torch.linalg.vector_norm(delta_b)
        command = torch.zeros(6, device=self._device)
        if dist < 0.01:
            return command, True
        step = torch.clamp(dist, max=self._pos_step)
        command[:3] = delta_b / torch.clamp(dist, min=1e-6) * step
        return command, False

    def poll(self) -> tuple[torch.Tensor, float | None]:
        if self._stage == "idle":
            return torch.zeros(6, device=self._device), None

        if self._stage == "pregrasp":
            command, reached = self._move_toward(self._targets["pregrasp"])
            if reached:
                self._stage = "descend"
                print("[Arm 1] Pregrasp reached; descending to sheet corner.")
            return command, 1.0

        if self._stage == "descend":
            command, reached = self._move_toward(self._targets["descend"])
            if reached:
                self._stage = "close"
                self._hold_steps = 80
                print("[Arm 1] Closing gripper on the sheet corner.")
            return command, 1.0

        if self._stage == "close":
            self._hold_steps -= 1
            if self._hold_steps <= 0:
                self._stage = "lift"
                self._attach_requested = True
                print("[Arm 1] Lifting after physical grasp attempt.")
            return torch.zeros(6, device=self._device), -1.0

        if self._stage == "lift":
            command, reached = self._move_toward(self._targets["lift"])
            if reached:
                self._stage = "idle"
                print("[Arm 1] Scripted sequence complete; keyboard control restored.")
            return command, -1.0

        self.reset()
        return torch.zeros(6, device=self._device), None


class SheetKinematicController:
    """Pin inner sheet corners and optionally attach a grasp corner to arm 1."""

    def __init__(self, env, robot, hand_body_id: int, ee_offset: torch.Tensor):
        self._env = env
        self._device = env.unwrapped.device
        self._sheet = env.unwrapped.scene["deformable_sheet"]
        self._robot = robot
        self._hand_body_id = hand_body_id
        self._ee_offset = ee_offset
        self._pin_node_ids: torch.Tensor | None = None
        self._pin_positions: torch.Tensor | None = None
        self._attach_node_ids: torch.Tensor | None = None
        self._attach_node_offsets: torch.Tensor | None = None
        self._attach_active = False

    @property
    def active(self) -> bool:
        return self._attach_active

    def _current_ee_pos(self) -> torch.Tensor:
        hand_pos = self._robot.data.body_pos_w[0, self._hand_body_id]
        hand_quat = self._robot.data.body_quat_w[0, self._hand_body_id]
        offset_w = math_utils.quat_apply(hand_quat.unsqueeze(0), self._ee_offset.unsqueeze(0))[0]
        return hand_pos + offset_w

    def _local_sheet_point_to_world(self, local_fraction: tuple[float, float], z_offset: float = 0.0) -> torch.Tensor:
        sheet_x, sheet_y, sheet_z = SHEET_INIT_POS
        size_x, size_y, size_z = SHEET_EFFECTIVE_SIZE
        local_x = local_fraction[0] * size_x
        local_y = local_fraction[1] * size_y
        # Sheet is rotated +90 deg about world Z.
        return torch.tensor(
            [sheet_x - local_y, sheet_y + local_x, sheet_z + 0.5 * size_z + z_offset],
            device=self._device,
        )

    def _nearest_nodes(
        self,
        target_pos: torch.Tensor,
        radius: float,
        max_nodes: int,
        exclude_node_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nodal_pos = self._sheet.data.nodal_pos_w[0]
        dist = torch.linalg.vector_norm(nodal_pos - target_pos.unsqueeze(0), dim=-1)
        if exclude_node_ids is not None and exclude_node_ids.numel() > 0:
            dist[exclude_node_ids] = torch.inf
        node_ids = torch.nonzero(dist < radius, as_tuple=False).squeeze(-1)
        if node_ids.numel() == 0:
            node_ids = torch.argsort(dist)[:max_nodes]
        elif node_ids.numel() > max_nodes:
            node_ids = node_ids[torch.argsort(dist[node_ids])[:max_nodes]]
        return node_ids

    def reset_pins(self) -> None:
        nodal_pos = self._sheet.data.nodal_pos_w[0]
        pin_ids = []
        for local_fraction in SHEET_PIN_LOCAL_FRACTIONS:
            target = self._local_sheet_point_to_world(local_fraction)
            exclude = torch.cat(pin_ids) if pin_ids else None
            pin_ids.append(self._nearest_nodes(target, PIN_RADIUS, PIN_MAX_NODES_PER_CORNER, exclude))
        self._pin_node_ids = torch.unique(torch.cat(pin_ids))
        self._pin_positions = nodal_pos[self._pin_node_ids].clone()
        self._attach_node_ids = None
        self._attach_node_offsets = None
        self._attach_active = False
        self.update()
        print(f"[Sheet] Pinned {self._pin_node_ids.numel()} inner-corner nodes to the table.")

    def attach_grasp_corner(self) -> None:
        nodal_pos = self._sheet.data.nodal_pos_w[0]
        corner_pos = self._local_sheet_point_to_world(SHEET_GRASP_LOCAL_FRACTION, SHEET_GRASP_Z_OFFSET)
        node_ids = self._nearest_nodes(corner_pos, ATTACH_RADIUS, ATTACH_MAX_NODES, self._pin_node_ids)
        ee_pos = self._current_ee_pos()
        self._attach_node_ids = node_ids
        self._attach_node_offsets = nodal_pos[node_ids] - ee_pos.unsqueeze(0)
        self._attach_active = True
        self.update()
        print(f"[Sheet] Attached {node_ids.numel()} corner nodes to arm 1.")

    def release_grasp_corner(self) -> None:
        self._attach_node_ids = None
        self._attach_node_offsets = None
        self._attach_active = False
        self.update()

    def update(self) -> None:
        targets = self._sheet.data.nodal_kinematic_target.clone()
        targets[..., :3] = self._sheet.data.nodal_pos_w
        targets[..., 3] = 1.0
        if self._pin_node_ids is not None and self._pin_positions is not None:
            targets[0, self._pin_node_ids, :3] = self._pin_positions
            targets[0, self._pin_node_ids, 3] = 0.0
        if self._attach_active and self._attach_node_ids is not None and self._attach_node_offsets is not None:
            ee_pos = self._current_ee_pos()
            targets[0, self._attach_node_ids, :3] = ee_pos.unsqueeze(0) + self._attach_node_offsets
            targets[0, self._attach_node_ids, 3] = 0.0
        self._sheet.write_nodal_kinematic_target_to_sim(targets)


def _assign_slice(actions: torch.Tensor, action_slice: slice | None, value: torch.Tensor | float) -> None:
    if action_slice is None:
        return
    if actions.ndim == 2:
        if torch.is_tensor(value):
            actions[:, action_slice] = value.unsqueeze(0).expand(actions.shape[0], -1)
        else:
            actions[:, action_slice] = value
    else:
        actions[action_slice] = value


def main():
    teleop_speed = args_cli.sensitivity * args_cli.motion_speed
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    print("\n" + "=" * 80)
    print(f"Creating Environment: {args_cli.task}")
    print("=" * 80)
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {args_cli.device}")
    print("=" * 80 + "\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    arm_1_slice = find_action_slice(env, "arm_1_action")
    gripper_1_slice = find_action_slice(env, "gripper_1_action")
    arm_2_slice = find_action_slice(env, "arm_2_action")
    gripper_2_slice = find_action_slice(env, "gripper_2_action")

    if arm_1_slice is None or arm_2_slice is None:
        raise RuntimeError("Expected active arm_1_action and arm_2_action terms for dual-arm IK teleoperation.")

    arm_1_scale = float(getattr(getattr(env_cfg.actions, "arm_1_action", None), "scale", 1.0))
    arm_2_scale = float(getattr(getattr(env_cfg.actions, "arm_2_action", None), "scale", 1.0))
    if arm_1_scale == 0.0 or arm_2_scale == 0.0:
        raise RuntimeError(f"Action scale cannot be zero (arm_1={arm_1_scale}, arm_2={arm_2_scale}).")

    print("Action configuration:")
    print(f"  active_terms: {list(env.unwrapped.action_manager.active_terms)}")
    print(f"  arm_1_action slice: [{arm_1_slice.start}:{arm_1_slice.stop}) scale={arm_1_scale}")
    print(f"  arm_2_action slice: [{arm_2_slice.start}:{arm_2_slice.stop}) scale={arm_2_scale}")
    if gripper_1_slice is not None:
        print(f"  gripper_1_action slice: [{gripper_1_slice.start}:{gripper_1_slice.stop})")
    if gripper_2_slice is not None:
        print(f"  gripper_2_action slice: [{gripper_2_slice.start}:{gripper_2_slice.stop})")

    keyboard_teleop = None
    spacemouse_teleop = None
    scripted_grasp = Arm1CornerGraspSequence(env, motion_speed=args_cli.motion_speed)
    sheet_kinematics = SheetKinematicController(
        env,
        scripted_grasp._robot,
        scripted_grasp._hand_body_id,
        scripted_grasp._ee_offset,
    )
    sheet_kinematics.reset_pins()
    gripper_1_action_value = 1.0
    gripper_2_action_value = 1.0

    if args_cli.headless:
        print("\n[HEADLESS MODE] Teleoperation disabled. Running smoke test loop.")
    else:
        try:
            keyboard_teleop = FrankaKeyboardController(env.unwrapped.device, teleop_speed)
            spacemouse_teleop = SpaceMouseController(
                env.unwrapped.device,
                teleop_speed,
                device_name=args_cli.spacemouse_device,
                device_index=args_cli.spacemouse_device_index,
            )
            print("\nTeleoperation active.")
            print(f"  Motion speed multiplier: {args_cli.motion_speed:g} (combined={teleop_speed:g})")
            print("  Arm 1 keyboard: W/S, A/D, Q/E for translation; Z/X/C with Shift for rotation")
            print("  Arm 1 gripper: J=open, K=close")
            print("  Arm 2 SpaceMouse: push/pull/tilt/twist the 3D mouse")
            print("  Arm 2 gripper: I=open, O=close")
            print("  G: scripted physical corner-grasp attempt with arm 1")
            if args_cli.attach:
                print("  --attach enabled: G will kinematically attach the grasped sheet corner after closing")
            else:
                print("  --attach disabled: G uses physical gripper contact only")
            print("  R: reset")
        except Exception as err:
            print(f"[WARNING] Teleoperation setup failed ({err}), fallback to headless behavior.")

    step_count = 0
    while simulation_app.is_running():
        if keyboard_teleop is not None and spacemouse_teleop is not None:
            (
                delta_1,
                reset_requested,
                gripper_1_open_requested,
                gripper_1_close_requested,
                gripper_2_open_requested,
                gripper_2_close_requested,
                scripted_grasp_requested,
            ) = keyboard_teleop.poll()
            delta_2 = spacemouse_teleop.poll()

            if scripted_grasp_requested and not scripted_grasp.active:
                scripted_grasp.start()

            if scripted_grasp.active:
                delta_1, scripted_gripper_value = scripted_grasp.poll()
                if scripted_gripper_value is not None:
                    gripper_1_action_value = scripted_gripper_value
                if scripted_grasp.consume_attach_request() and args_cli.attach and not sheet_kinematics.active:
                    sheet_kinematics.attach_grasp_corner()
            else:
                if gripper_1_open_requested:
                    gripper_1_action_value = 1.0
                    print(f"[Gripper 1] OPEN: panda_finger_joint.*={FRANKA_OPEN_POS:.4f} m")
                if gripper_1_close_requested:
                    gripper_1_action_value = -1.0
                    print(f"[Gripper 1] CLOSED: panda_finger_joint.*={FRANKA_CLOSED_POS:.4f} m")

            if gripper_2_open_requested:
                gripper_2_action_value = 1.0
                print(f"[Gripper 2] OPEN: panda_finger_joint.*={FRANKA_OPEN_POS:.4f} m")
            if gripper_2_close_requested:
                gripper_2_action_value = -1.0
                print(f"[Gripper 2] CLOSED: panda_finger_joint.*={FRANKA_CLOSED_POS:.4f} m")
        else:
            delta_1 = torch.zeros(6, device=env.unwrapped.device)
            delta_2 = torch.zeros(6, device=env.unwrapped.device)
            reset_requested = False
            if args_cli.headless and step_count >= args_cli.max_steps_headless:
                print(f"\n[INFO] Headless test completed after {step_count} steps.")
                break

        if reset_requested:
            env.reset()
            scripted_grasp.reset()
            sheet_kinematics.reset_pins()
            gripper_1_action_value = 1.0
            gripper_2_action_value = 1.0
            if keyboard_teleop is not None and spacemouse_teleop is not None:
                delta_1.zero_()
                delta_2.zero_()

        actions.zero_()
        _assign_slice(actions, arm_1_slice, delta_1 / arm_1_scale)
        _assign_slice(actions, arm_2_slice, delta_2 / arm_2_scale)
        _assign_slice(actions, gripper_1_slice, gripper_1_action_value)
        _assign_slice(actions, gripper_2_slice, gripper_2_action_value)
        sheet_kinematics.update()

        _, _, terminated, truncated, _ = env.step(actions)

        if torch.is_tensor(terminated):
            done = bool(torch.any(terminated) or torch.any(truncated))
        else:
            done = bool(terminated or truncated)
        if done:
            env.reset()
            scripted_grasp.reset()
            sheet_kinematics.reset_pins()
            gripper_1_action_value = 1.0
            gripper_2_action_value = 1.0

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
