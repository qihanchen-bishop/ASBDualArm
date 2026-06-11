# Copyright (c) 2026, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dual SO-ARM leader-arm joint teleoperation script.

Usage:
  ./isaaclab.sh -p source/ASBDualArm/scripts/env/state_machine/dual-so-arm-tel.py --num_envs 1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import types

from isaaclab.app import AppLauncher


_ISAACLAB_SOURCE_DIR = Path(__file__).resolve().parents[4]
_LEROBOT_SRC = _ISAACLAB_SOURCE_DIR / "lerobot" / "src"
_DEFAULT_LEADER_CALIBRATION_DIR = (
    _ISAACLAB_SOURCE_DIR
    / "ASBDualArm"
    / "source"
    / "msr"
    / "msr"
    / "tasks"
    / "direct"
    / "dual_arm"
    / "so-arm"
)
_DEFAULT_RECORD_DIR = _ISAACLAB_SOURCE_DIR / "ASBDualArm" / "saved_data"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))


parser = argparse.ArgumentParser(description="Dual SO-ARM leader-arm joint teleoperation.")
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
    "--leader-ports",
    nargs="+",
    default=["/dev/ttyACM0", "/dev/ttyACM1"],
    help="Serial ports for SO101 leader arms. One port controls robot_1; two ports control robot_1 and robot_2.",
)
parser.add_argument(
    "--leader-ids",
    nargs="+",
    default=["leader_1", "leader_2"],
    help="LeRobot calibration IDs for the leader arms. Keep these stable to reuse calibration files.",
)
parser.add_argument(
    "--leader-calibration-dir",
    type=Path,
    default=_DEFAULT_LEADER_CALIBRATION_DIR,
    help="Directory used to load/save leader-arm calibration JSON files.",
)
parser.add_argument(
    "--no-leader-calibration",
    action="store_false",
    dest="leader_calibration",
    default=True,
    help="Connect without running LeRobot calibration. Use only after calibration files/motor offsets are valid.",
)
parser.add_argument(
    "--joint-smoothing",
    type=float,
    default=1.0,
    help="Low-pass factor for leader joint targets in [0, 1]. 1.0 follows the leader directly.",
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
    "--record-dir",
    type=Path,
    default=_DEFAULT_RECORD_DIR,
    help="Directory where timestamped LeRobot datasets are saved.",
)
parser.add_argument(
    "--record-dataset-path",
    type=Path,
    default=None,
    help=(
        "Specific LeRobot dataset directory to create or append to. "
        "If omitted, a timestamped dataset is created under --record-dir."
    ),
)
parser.add_argument("--record-fps", type=int, default=30, help="FPS written into the LeRobot dataset metadata.")
parser.add_argument(
    "--record-task",
    type=str,
    default="dual so-arm teleoperation",
    help="Task string saved with each LeRobot frame.",
)
parser.add_argument(
    "--record-image-writer-threads",
    type=int,
    default=4,
    help="Number of LeRobot async image-writer threads.",
)
parser.add_argument(
    "--record-video-codec",
    type=str,
    choices=("h264", "hevc", "libsvtav1"),
    default="h264",
    help="Video codec used by LeRobot when encoding recorded episodes. h264 is best for browser preview.",
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
    "--task-status-print-interval",
    type=int,
    default=30,
    help="Print target/plane task status every N simulation steps. Set 1 for every step.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.enable_cameras:
    print("[Record] Enabling cameras because DualSOArmEnvCfg registers the USD camera.")
    args_cli.enable_cameras = True

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401
from msr.config.robot import SO101_ALL_JOINT_NAMES
import msr.tasks  # noqa: F401
import msr.tasks.direct.dual_arm.mdp as dual_arm_mdp


SO101_JOINT_NAMES = list(SO101_ALL_JOINT_NAMES)
SO101_ARM_JOINT_NAMES = SO101_JOINT_NAMES[:5]
SO101_GRIPPER_JOINT = SO101_JOINT_NAMES[5]

SO101_JOINT_LOWER = torch.tensor([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
SO101_JOINT_UPPER = torch.tensor([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

RECORD_CAMERA_KEY = "observation.images.camera"
RECORD_STATE_KEY = "observation.state"
RECORD_ACTION_KEY = "action"
RECORD_CAMERA_SHAPE = (480, 640, 3)
RESET_XFORM_PRIM_NAMES = ("DeformableOccluder",)
RESET_RIGID_OBJECT_NAMES = ("target",)
RESET_DEFORMABLE_OBJECT_NAMES = ("deformable_occluder",)


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


def _install_lerobot_motor_utils_stub() -> None:
    """Provide the two LeRobot terminal helpers needed by motors_bus without importing training deps."""

    if "lerobot.utils.utils" in sys.modules:
        return

    try:
        __import__("lerobot.utils.utils")
        return
    except Exception:
        pass

    def enter_pressed() -> bool:
        import select

        if not sys.stdin.isatty():
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return False
        sys.stdin.readline()
        return True

    def move_cursor_up(lines: int) -> None:
        if lines > 0:
            print(f"\033[{lines}A", end="")

    utils_stub = types.ModuleType("lerobot.utils.utils")
    utils_stub.enter_pressed = enter_pressed
    utils_stub.move_cursor_up = move_cursor_up
    sys.modules["lerobot.utils.utils"] = utils_stub


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


def _normalize_ports_and_ids(ports: list[str], ids: list[str]) -> tuple[list[str], list[str]]:
    ports = [port for port in ports if port]
    ids = [leader_id for leader_id in ids if leader_id]
    if not 1 <= len(ports) <= 2:
        raise ValueError(f"Expected one or two leader ports, got {ports!r}.")
    if len(ids) < len(ports):
        ids.extend(f"leader_{idx + 1}" for idx in range(len(ids), len(ports)))
    return ports, ids[: len(ports)]


def _leader_calibration_paths(ports: list[str], ids: list[str], calibration_dir: Path) -> list[Path]:
    ports, ids = _normalize_ports_and_ids(ports, ids)
    return [calibration_dir / f"{leader_id}.json" for leader_id in ids]


def _ensure_leader_calibrations(device: torch.device | str) -> None:
    """Run leader calibration before hotkey handling if any calibration file is missing."""

    calibration_paths = _leader_calibration_paths(
        args_cli.leader_ports, args_cli.leader_ids, args_cli.leader_calibration_dir
    )
    missing_paths = [path for path in calibration_paths if not path.is_file()]
    if not missing_paths:
        print("[Leader] Calibration files found.")
        return

    print("[Leader] Missing calibration files:")
    for path in missing_paths:
        print(f"  {path}")

    if not args_cli.leader_calibration:
        missing = "\n  ".join(str(path) for path in missing_paths)
        raise RuntimeError(
            "Leader calibration is disabled, but calibration files are missing:\n"
            f"  {missing}\n"
            "Run without --no-leader-calibration to calibrate first."
        )

    print("[Leader] Running calibration before keyboard control starts.")
    calibration_controller = SO101LeaderJointController(
        ports=args_cli.leader_ports,
        ids=args_cli.leader_ids,
        calibration_dir=args_cli.leader_calibration_dir,
        calibrate=True,
        device=device,
    )
    calibration_controller.close()
    missing_after = [path for path in calibration_paths if not path.is_file()]
    if missing_after:
        missing = "\n  ".join(str(path) for path in missing_after)
        raise RuntimeError(f"Calibration did not create all expected files:\n  {missing}")
    print("[Leader] Calibration complete. Keyboard control is now enabled.")


class SO101LeaderJointController:
    """Reads calibrated SO101 leader joint angles and converts them to simulator joint targets."""

    def __init__(
        self,
        ports: list[str],
        ids: list[str],
        calibration_dir: Path,
        calibrate: bool,
        device: torch.device | str,
    ):
        _install_lerobot_motor_utils_stub()

        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

        ports, ids = _normalize_ports_and_ids(ports, ids)
        self._device = device
        self._buses = []
        self._calibration_paths = []
        self._lower = SO101_JOINT_LOWER.to(device=device)
        self._upper = SO101_JOINT_UPPER.to(device=device)
        calibration_dir.mkdir(parents=True, exist_ok=True)

        for port, leader_id in zip(ports, ids, strict=True):
            calibration_path = calibration_dir / f"{leader_id}.json"
            calibration = self._load_calibration(calibration_path, MotorCalibration)
            bus = FeetechMotorsBus(
                port=port,
                motors={
                    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=calibration,
            )
            print(f"[Leader] Connecting {leader_id} on {port}")
            bus.connect()
            if not bus.is_calibrated:
                if not calibrate:
                    bus.disconnect()
                    raise RuntimeError(
                        f"{leader_id} on {port} is not calibrated. "
                        f"Run without --no-leader-calibration, or provide {calibration_path}."
                    )
                self._calibrate_bus(bus, leader_id, calibration_path, MotorCalibration, OperatingMode)
            self._configure_bus(bus, OperatingMode)
            self._buses.append(bus)
            self._calibration_paths.append(calibration_path)

    @property
    def num_leaders(self) -> int:
        return len(self._buses)

    @staticmethod
    def _load_calibration(calibration_path: Path, calibration_type) -> dict:
        if not calibration_path.is_file():
            return {}
        with open(calibration_path) as f:
            raw = json.load(f)
        return {motor: calibration_type(**values) for motor, values in raw.items()}

    @staticmethod
    def _save_calibration(calibration_path: Path, calibration: dict) -> None:
        with open(calibration_path, "w") as f:
            json.dump({motor: asdict(values) for motor, values in calibration.items()}, f, indent=4)

    @staticmethod
    def _make_hardware_safe_calibration(bus, calibration: dict, calibration_type) -> dict:
        safe_calibration = {}
        for motor, values in calibration.items():
            max_res = bus.model_resolution_table[bus.motors[motor].model] - 1
            range_min = int(max(0, min(max_res, values.range_min)))
            range_max = int(max(0, min(max_res, values.range_max)))
            if range_min >= range_max:
                range_min = 0
                range_max = max_res
            safe_calibration[motor] = calibration_type(
                id=values.id,
                drive_mode=values.drive_mode,
                homing_offset=values.homing_offset,
                range_min=range_min,
                range_max=range_max,
            )
        return safe_calibration

    def _write_calibration(self, bus, calibration: dict, calibration_type) -> None:
        safe_calibration = self._make_hardware_safe_calibration(bus, calibration, calibration_type)
        bus.write_calibration(safe_calibration, cache=False)
        bus.calibration = calibration

    def _configure_bus(self, bus, operating_mode) -> None:
        bus.disable_torque()
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, operating_mode.POSITION.value)

    def _calibrate_bus(self, bus, leader_id: str, calibration_path: Path, calibration_type, operating_mode) -> None:
        if bus.calibration:
            answer = input(
                f"Press ENTER to write existing calibration for {leader_id} to the motors, "
                "or type 'c' and press ENTER to run a new calibration: "
            )
            if answer.strip().lower() != "c":
                self._write_calibration(bus, bus.calibration, calibration_type)
                return

        print(f"\n[Leader] Running calibration for {leader_id}")
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, operating_mode.POSITION.value)

        input(f"Move {leader_id} to the middle of its range of motion and press ENTER...")
        homing_offsets = bus.set_half_turn_homings()

        print(
            "Move all joints, including 'wrist_roll', sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = bus.record_ranges_of_motion()

        calibration = {}
        for motor, motor_cfg in bus.motors.items():
            calibration[motor] = calibration_type(
                id=motor_cfg.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self._write_calibration(bus, calibration, calibration_type)
        self._save_calibration(calibration_path, calibration)
        print(f"[Leader] Calibration saved to {calibration_path}")

    def poll(self) -> torch.Tensor:
        targets = []
        for bus in self._buses:
            action = bus.sync_read("Present_Position")
            joint_values = []
            for joint_name in SO101_ARM_JOINT_NAMES:
                joint_values.append(torch.deg2rad(torch.tensor(float(action[joint_name]))))

            gripper_percent = float(action[SO101_GRIPPER_JOINT])
            gripper_percent = max(0.0, min(100.0, gripper_percent))
            gripper_lower = float(SO101_JOINT_LOWER[-1])
            gripper_upper = float(SO101_JOINT_UPPER[-1])
            gripper_target = gripper_lower + (gripper_percent / 100.0) * (gripper_upper - gripper_lower)
            joint_values.append(torch.tensor(gripper_target))

            targets.append(torch.stack(joint_values).to(device=self._device, dtype=torch.float32))

        return torch.clamp(torch.stack(targets), self._lower, self._upper)

    def close(self) -> None:
        for bus in self._buses:
            try:
                if bus.is_connected:
                    bus.disconnect()
            except Exception as err:
                print(f"[WARNING] Failed to disconnect {bus.port}: {err}")


class TeleopHotkeys:
    """Keyboard hotkeys for SO101 teleoperation and recording."""

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
            "start_teleop": self._pressed(key.T),
            "start_record": self._pressed(key.R),
            "stop": self._pressed(key.P),
            "reset": self._pressed(key.N),
        }


class LeRobotEpisodeRecorder:
    """Small wrapper around LeRobotDataset for one r-to-p episode at a time."""

    def __init__(
        self,
        root_dir: Path,
        dataset_path: Path | None,
        fps: int,
        task: str,
        image_writer_threads: int,
        video_codec: str,
    ):
        missing_deps = [module for module in ("datasets", "pyarrow", "av") if importlib.util.find_spec(module) is None]
        if missing_deps:
            raise RuntimeError(
                "LeRobot recording requires missing Python packages: "
                f"{', '.join(missing_deps)}.\n"
                "Install them in the IsaacLab Python environment, for example:\n"
                "  ./isaaclab.sh -p -m pip install datasets pyarrow av"
            )

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._task = task
        self._episode_frames = 0
        features = {
            RECORD_CAMERA_KEY: {
                "dtype": "video",
                "shape": RECORD_CAMERA_SHAPE,
                "names": ["height", "width", "channels"],
            },
            RECORD_STATE_KEY: {
                "dtype": "float32",
                "shape": (12,),
                "names": [f"robot_1.{name}" for name in SO101_JOINT_NAMES]
                + [f"robot_2.{name}" for name in SO101_JOINT_NAMES],
            },
            RECORD_ACTION_KEY: {
                "dtype": "float32",
                "shape": (12,),
                "names": [f"robot_1.{name}" for name in SO101_JOINT_NAMES]
                + [f"robot_2.{name}" for name in SO101_JOINT_NAMES],
            },
        }

        if dataset_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root = root_dir / timestamp
            suffix = 1
            while root.exists():
                root = root_dir / f"{timestamp}_{suffix:02d}"
                suffix += 1
            root_dir.mkdir(parents=True, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                repo_id=f"dual_so_arm_tel/{root.name}",
                root=root,
                fps=int(fps),
                features=features,
                robot_type="dual_so101",
                use_videos=True,
                image_writer_threads=max(0, int(image_writer_threads)),
                vcodec=video_codec,
            )
            print(f"[Record] Created timestamped dataset: {self.dataset.root}")
            return

        root = dataset_path.expanduser()
        if root.exists() and not root.is_dir():
            raise RuntimeError(f"--record-dataset-path must be a directory, got file: {root}")

        info_path = root / "meta" / "info.json"
        if info_path.is_file():
            self.dataset = LeRobotDataset(
                repo_id=f"dual_so_arm_tel/{root.name}",
                root=root,
                download_videos=False,
                vcodec=video_codec,
            )
            if int(self.dataset.fps) != int(fps):
                raise RuntimeError(
                    f"Existing dataset FPS is {self.dataset.fps}, but --record-fps is {fps}. "
                    "Use the same FPS when appending."
                )
            self._validate_existing_features(features)
            if image_writer_threads > 0:
                self.dataset.start_image_writer(num_threads=int(image_writer_threads))
            print(
                f"[Record] Appending to existing dataset: {self.dataset.root} "
                f"(episodes={self.dataset.meta.total_episodes}, frames={self.dataset.meta.total_frames})"
            )
        else:
            if root.exists() and any(root.iterdir()):
                raise RuntimeError(
                    f"--record-dataset-path exists but is not a LeRobot dataset and is not empty: {root}"
                )
            if root.exists():
                root.rmdir()
            self.dataset = LeRobotDataset.create(
                repo_id=f"dual_so_arm_tel/{root.name}",
                root=root,
                fps=int(fps),
                features=features,
                robot_type="dual_so101",
                use_videos=True,
                image_writer_threads=max(0, int(image_writer_threads)),
                vcodec=video_codec,
            )
            print(f"[Record] Created dataset at requested path: {self.dataset.root}")

        print(f"[Record] Dataset root: {self.dataset.root}")

    def _validate_existing_features(self, expected_features: dict) -> None:
        def normalize(value):
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        for key, expected in expected_features.items():
            if key not in self.dataset.features:
                raise RuntimeError(f"Existing dataset is missing required feature: {key}")
            actual = self.dataset.features[key]
            for field in ("dtype", "shape", "names"):
                if normalize(actual.get(field)) != normalize(expected.get(field)):
                    raise RuntimeError(
                        f"Existing dataset feature {key!r} has incompatible {field}: "
                        f"{actual.get(field)!r} != {expected.get(field)!r}"
                    )

    @property
    def active(self) -> bool:
        return self._episode_frames > 0

    def add_frame(self, image: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
        self.dataset.add_frame(
            {
                RECORD_CAMERA_KEY: image,
                RECORD_STATE_KEY: state.astype(np.float32, copy=False),
                RECORD_ACTION_KEY: action.astype(np.float32, copy=False),
                "task": self._task,
            }
        )
        self._episode_frames += 1

    def save_episode(self) -> None:
        if self._episode_frames == 0:
            print("[Record] No frames captured; skipping save_episode.")
            return
        self.dataset.save_episode()
        print(f"[Record] Saved episode with {self._episode_frames} frames.")
        self._episode_frames = 0

    def close(self) -> None:
        if self._episode_frames > 0:
            self.save_episode()
        finalize = getattr(self.dataset, "finalize", None)
        if finalize is not None:
            finalize()
        image_writer = getattr(self.dataset, "image_writer", None)
        if image_writer is not None:
            self.dataset.stop_image_writer()


def _read_robot_joint_targets(env, robot_name: str) -> torch.Tensor:
    robot = env.unwrapped.scene[robot_name]
    joint_ids, joint_names = robot.find_joints(SO101_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(SO101_JOINT_NAMES):
        raise RuntimeError(f"Expected joints {SO101_JOINT_NAMES} on {robot_name}, found {joint_names}")
    return robot.data.joint_pos[:, joint_ids].clone()


def _read_record_state(env) -> np.ndarray:
    joints_1 = _read_robot_joint_targets(env, "robot_1")[0]
    joints_2 = _read_robot_joint_targets(env, "robot_2")[0]
    return torch.cat((joints_1, joints_2), dim=0).detach().cpu().numpy().astype(np.float32)


def _read_record_action(robot_1_targets: torch.Tensor, robot_2_targets: torch.Tensor) -> np.ndarray:
    return torch.cat((robot_1_targets[0], robot_2_targets[0]), dim=0).detach().cpu().numpy().astype(np.float32)


def _get_camera(env):
    try:
        return env.unwrapped.scene["camera"]
    except KeyError:
        return None


def _read_rgb_image(env) -> np.ndarray | None:
    camera = _get_camera(env)
    if camera is None or "rgb" not in camera.data.output:
        return None
    image = camera.data.output["rgb"][0].detach().cpu().numpy()
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype(np.uint8, copy=False)


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
    cached_xforms: dict[str, list[tuple[str, object]]],
    target_random_x: float,
    target_random_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()
    _restore_cached_xforms(cached_xforms)
    _reset_scene_deformable_objects_to_defaults(env, RESET_DEFORMABLE_OBJECT_NAMES)
    _reset_scene_rigid_objects_to_defaults(env, RESET_RIGID_OBJECT_NAMES)
    _randomize_target_xy(env, target_random_x, target_random_y)
    _reset_robot_to_defaults(env, "robot_1")
    _reset_robot_to_defaults(env, "robot_2")
    return _read_robot_joint_targets(env, "robot_1"), _read_robot_joint_targets(env, "robot_2")


def _write_joint_actions(actions: torch.Tensor, robot_1_targets: torch.Tensor, robot_2_targets: torch.Tensor) -> None:
    actions.zero_()
    if actions.ndim == 2:
        actions[:, 0:6] = robot_1_targets
        actions[:, 6:12] = robot_2_targets
    else:
        actions[0:6] = robot_1_targets[0]
        actions[6:12] = robot_2_targets[0]


def _blend_targets(current: torch.Tensor, new_target: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha >= 1.0:
        return new_target
    if alpha <= 0.0:
        return current
    return current + alpha * (new_target - current)


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
    print("  Control: direct SO101 leader joint-position mapping")
    print(f"  Record dir: {args_cli.record_dir}")
    if args_cli.record_dataset_path is not None:
        print(f"  Record dataset path: {args_cli.record_dataset_path}")
    print(f"  Record video codec: {args_cli.record_video_codec}")
    print(f"  Recording camera: {'enabled' if args_cli.enable_cameras else 'disabled; pass --enable_cameras to record RGB'}")
    print("=" * 80 + "\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    cached_xforms = _cache_initial_xforms(RESET_XFORM_PRIM_NAMES)
    if cached_xforms:
        print(f"[Reset] Cached initial xforms for: {', '.join(sorted(cached_xforms))}")
    else:
        print("[Reset] No DeformableOccluder xforms found to cache.")
    _reset_scene_deformable_objects_to_defaults(env, RESET_DEFORMABLE_OBJECT_NAMES)
    _reset_scene_rigid_objects_to_defaults(env, RESET_RIGID_OBJECT_NAMES)
    _randomize_target_xy(env, args_cli.target_random_x, args_cli.target_random_y)
    task_monitor = TargetPlaneMonitor(env)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    action_dim = actions.shape[-1] if actions.ndim > 1 else actions.shape[0]
    if action_dim != 12:
        raise RuntimeError(f"Expected 12D action space for dual SO101 joint-position control, got {action_dim}.")

    target_joints_1 = _read_robot_joint_targets(env, "robot_1")
    target_joints_2 = _read_robot_joint_targets(env, "robot_2")
    smoothing = max(0.0, min(1.0, float(args_cli.joint_smoothing)))

    leader_controller = None
    hotkeys = None
    teleop_active = False
    recorder = None
    recording_active = False
    task_complete_notified = False
    task_status_print_interval = max(1, int(args_cli.task_status_print_interval))
    if args_cli.headless:
        print("[HEADLESS MODE] Leader-arm control disabled. Holding initial joint poses.")
    else:
        _ensure_leader_calibrations(env.unwrapped.device)
        hotkeys = TeleopHotkeys()
        print("SO101 leader joint control ready.")
        print(f"  Leader ports: {', '.join(args_cli.leader_ports)}")
        print(f"  Joint order: {SO101_JOINT_NAMES}")
        print(f"  Calibration dir: {args_cli.leader_calibration_dir}")
        print("  Hotkeys: T=start teleop, R=start recording, P=stop teleop/recording, N=reset")

    step_count = 0
    try:
        while simulation_app.is_running():
            if hotkeys is not None:
                events = hotkeys.poll()
                if events["start_teleop"] and not teleop_active:
                    leader_controller = SO101LeaderJointController(
                        ports=args_cli.leader_ports,
                        ids=args_cli.leader_ids,
                        calibration_dir=args_cli.leader_calibration_dir,
                        calibrate=False,
                        device=env.unwrapped.device,
                    )
                    teleop_active = True
                    target_joints_1 = _read_robot_joint_targets(env, "robot_1")
                    target_joints_2 = _read_robot_joint_targets(env, "robot_2")
                    print("[Teleop] Started.")

                if events["start_record"]:
                    if not teleop_active:
                        print("[Record] Press T to start teleop before recording.")
                    elif _read_rgb_image(env) is None:
                        print("[Record] RGB camera is unavailable. Start with --enable_cameras and check /Scene/Camera.")
                    elif not recording_active:
                        try:
                            recorder = LeRobotEpisodeRecorder(
                                root_dir=args_cli.record_dir,
                                dataset_path=args_cli.record_dataset_path,
                                fps=args_cli.record_fps,
                                task=args_cli.record_task,
                                image_writer_threads=args_cli.record_image_writer_threads,
                                video_codec=args_cli.record_video_codec,
                            )
                        except RuntimeError as err:
                            print(f"[Record] Cannot start LeRobot recording:\n{err}")
                            recorder = None
                        else:
                            recording_active = True
                            print("[Record] Started.")
                    else:
                        print("[Record] Already recording.")

                if events["stop"]:
                    if recording_active and recorder is not None:
                        recorder.save_episode()
                        recorder.close()
                        recorder = None
                        recording_active = False
                    if leader_controller is not None:
                        leader_controller.close()
                        leader_controller = None
                    teleop_active = False
                    print("[Teleop] Stopped.")

                if events["reset"]:
                    if recording_active and recorder is not None:
                        recorder.save_episode()
                        recorder.close()
                        recorder = None
                        recording_active = False
                    if leader_controller is not None:
                        leader_controller.close()
                        leader_controller = None
                    teleop_active = False
                    target_joints_1, target_joints_2 = _reset_environment(
                        env,
                        cached_xforms,
                        args_cli.target_random_x,
                        args_cli.target_random_y,
                    )
                    task_complete_notified = False
                    print("[Reset] Environment reset to initial scene and robot defaults.")

            if teleop_active and leader_controller is not None:
                leader_targets = leader_controller.poll()
                target_joints_1 = _blend_targets(
                    target_joints_1,
                    leader_targets[0].unsqueeze(0).expand_as(target_joints_1),
                    smoothing,
                )
                if leader_controller.num_leaders > 1:
                    target_joints_2 = _blend_targets(
                        target_joints_2,
                        leader_targets[1].unsqueeze(0).expand_as(target_joints_2),
                        smoothing,
                    )
            elif args_cli.headless and step_count >= args_cli.max_steps_headless:
                print(f"\n[INFO] Headless test completed after {step_count} steps.")
                break

            _write_joint_actions(actions, target_joints_1, target_joints_2)
            _, _, terminated, truncated, _ = env.step(actions)

            task_status = task_monitor.read_status()
            task_inside = bool(task_status["complete"])
            if step_count % task_status_print_interval == 0:
                task_monitor.print_status(step_count, task_status)

            if recording_active and recorder is not None:
                rgb_image = _read_rgb_image(env)
                if rgb_image is None:
                    print("[Record] RGB camera became unavailable; stopping recording.")
                    recorder.close()
                    recorder = None
                    recording_active = False
                else:
                    recorder.add_frame(
                        image=rgb_image,
                        state=_read_record_state(env),
                        action=_read_record_action(target_joints_1, target_joints_2),
                    )

            if task_inside and not task_complete_notified:
                task_complete_notified = True
                print("[Task Complete] Target is fully inside the plane area.", flush=True)
                if leader_controller is not None:
                    leader_controller.close()
                    leader_controller = None
                if teleop_active:
                    teleop_active = False
                    print("[Teleop] Stopped because task is complete.", flush=True)
                if recording_active and recorder is not None:
                    recorder.save_episode()
                    recorder.close()
                    recorder = None
                    recording_active = False
                    print("[Record] Stopped because task is complete.", flush=True)
            elif not task_inside:
                task_complete_notified = False

            done = bool(torch.any(terminated) or torch.any(truncated))
            if done:
                if recording_active and recorder is not None:
                    recorder.save_episode()
                    recorder.close()
                    recorder = None
                    recording_active = False
                target_joints_1, target_joints_2 = _reset_environment(
                    env,
                    cached_xforms,
                        args_cli.target_random_x,
                        args_cli.target_random_y,
                    )
                task_complete_notified = False

            if step_count % 200 == 0 and step_count > 0:
                print(
                    f"step={step_count} "
                    f"robot_1_joints={target_joints_1[0].tolist()} "
                    f"robot_2_joints={target_joints_2[0].tolist()}"
                )

            step_count += 1
    finally:
        if recorder is not None:
            recorder.close()
        if leader_controller is not None:
            leader_controller.close()
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
