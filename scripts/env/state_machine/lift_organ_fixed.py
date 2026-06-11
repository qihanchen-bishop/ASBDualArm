# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Single robot environment test script with keyboard teleoperation
# Robot end-effector pose can be controlled via keyboard
#
# Usage (with GUI - required for keyboard control):
#   ${IsaacLab_PATH}/isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/lift_organ_fixed.py --num_envs 1
#
# Usage (headless - auto test mode):
#   ${IsaacLab_PATH}/isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/lift_organ_fixed.py --num_envs 1 --headless
#
# Keyboard controls (GUI mode only):
#   W/S: Move end-effector along X-axis
#   A/D: Move end-effector along Y-axis
#   Q/E: Move end-effector along Z-axis
#   Z/X: Rotate around X-axis (roll)
#   T/G: Rotate around Y-axis (pitch)
#   C/V: Rotate around Z-axis (yaw)
#   L: Reset environment


"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Single robot environment with keyboard teleoperation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor for keyboard control.")
parser.add_argument(
    "--save-data",
    "--save_data",
    dest="save_data",
    action="store_true",
    default=False,
    help="Save camera images and point clouds.",
)
parser.add_argument(
    "--save-dir",
    "--save_dir",
    dest="save_dir",
    type=str,
    default="/workspace/isaaclab/source/ASBDualArm/scripts/saved",
    help="Base directory for saved images and point clouds.",
)
parser.add_argument(
    "--image-save-interval",
    "--image_save_interval",
    dest="image_save_interval",
    type=int,
    default=50,
    help="Save camera outputs every N simulation steps.",
)
parser.add_argument(
    "--show-helpers",
    action="store_true",
    default=False,
    help="Show auxiliary coordinate frames and command debug visualizations.",
)
parser.add_argument(
    "--semantic-debug-interval",
    "--semantic_debug_interval",
    dest="semantic_debug_interval",
    type=int,
    default=0,
    help="Print semantic runtime diagnostics every N steps (0 disables).",
)
parser.add_argument(
    "--semantic-debug-labels",
    "--semantic_debug_labels",
    dest="semantic_debug_labels",
    type=str,
    default="gripper,vessel,robot",
    help="Comma-separated semantic label keywords to track in diagnostics.",
)
parser.add_argument(
    "--semantic-debug-max-colors",
    "--semantic_debug_max_colors",
    dest="semantic_debug_max_colors",
    type=int,
    default=8,
    help="Maximum number of dominant rendered semantic colors to print.",
)
parser.add_argument(
    "--usd-path",
    "--usd_path",
    dest="usd_path",
    type=str,
    default=None,
    help="Override the scene USD path configured by the environment.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
import os
import re
from datetime import datetime

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.sensors import TiledCamera, save_images_to_file
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import numpy as np

import isaaclab_tasks  # noqa: F401
import msr.tasks  # noqa: F401

# USD/PhysX imports for analyzing scene structure
from pxr import Usd, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage


def analyze_scene_structure(env_prim_path: str):
    """
    Analyze the structure of the loaded scene to verify all components.
    
    Args:
        env_prim_path: Environment prim path, e.g., "/World/envs/env_0"
    """
    stage = get_current_stage()
    
    print("="*80)
    print(f"Analyzing Scene Structure under: {env_prim_path}")
    print("="*80)
    
    # Check main organ prim
    organ_prim_path = f"{env_prim_path}/Organ"
    organ_prim = stage.GetPrimAtPath(organ_prim_path)
    
    if not organ_prim.IsValid():
        print(f"[WARNING] Organ prim not found at: {organ_prim_path}")
        return
    
    print(f"\n[INFO] Found Organ prim at: {organ_prim_path}")
    
    # Look for key prims (paths relative to Organ/)
    key_prims = [
        "msr_psm",           # Robot arm (directly under Organ/)
        "GroundPlane",       # Ground plane (directly under Organ/)
        "table",             # Table (directly under Organ/)
        "upe6",              # upe6 container
        "upe6/Cube",         # Organ mesh (under Organ/upe6/)
        "upe6/Sphere",       # Sphere object (under Organ/upe6/)
        "upe6/Cylinder",     # Cylinder object (under Organ/upe6/)
        "upe6/Cylinder_01",  # Cylinder_01 object (under Organ/upe6/)
    ]
    
    print("\n[Checking Key Prims]")
    for prim_name in key_prims:
        full_path = f"{organ_prim_path}/{prim_name}"
        prim = stage.GetPrimAtPath(full_path)
        if prim.IsValid():
            prim_type = prim.GetTypeName()
            print(f"  ✓ {prim_name}: {prim_type}")
            
            # Check for physics APIs
            has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            has_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            
            if has_rigid_body:
                print(f"      - Has RigidBodyAPI")
            if has_articulation:
                print(f"      - Has ArticulationRootAPI")
        else:
            print(f"  ✗ {prim_name}: NOT FOUND")
    
    # List all immediate children of Organ/upe6
    upe6_path = f"{organ_prim_path}/upe6"
    upe6_prim = stage.GetPrimAtPath(upe6_path)
    
    if upe6_prim.IsValid():
        print(f"\n[Children of {upe6_path}]")
        for child in upe6_prim.GetChildren():
            child_type = child.GetTypeName()
            print(f"  - {child.GetName()}: {child_type}")
    
    print("="*80)


def print_robot_info(robot, name: str):
    """Print robot joint and pose information."""
    print(f"\n[{name} Information]")
    print(f"  Prim path: {robot.cfg.prim_path}")
    print(f"  Number of instances: {robot.num_instances}")
    print(f"  Number of bodies: {robot.num_bodies}")
    print(f"  Number of joints: {robot.num_joints}")
    print(f"  Joint names: {robot.joint_names}")
    print(f"  Body names: {robot.body_names}")
    
    # Get current joint positions
    joint_pos = robot.data.joint_pos
    print(f"\n  Current joint positions:")
    for i, name in enumerate(robot.joint_names):
        print(f"    {name}: {joint_pos[0, i].item():.4f}")
    
    # Get root pose
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    print(f"\n  Root position: {root_pos[0].cpu().numpy()}")
    print(f"  Root orientation (wxyz): {root_quat[0].cpu().numpy()}")


def save_pointcloud_to_ply(file_path: str, points_xyz: torch.Tensor, points_rgb: torch.Tensor):
    """Save a colored point cloud to an ASCII PLY file."""
    points_xyz_np = points_xyz.detach().cpu().numpy()
    points_rgb_np = torch.clamp(points_rgb, 0, 255).to(torch.uint8).detach().cpu().numpy()

    with open(file_path, "w", encoding="ascii") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {points_xyz_np.shape[0]}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for point, color in zip(points_xyz_np, points_rgb_np, strict=False):
            ply_file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def force_gripper_closed(robot, device: torch.device, num_envs: int):
    """Clamp the PSM jaw joints to the closed posture used by this task."""
    gripper_joint_ids, _ = robot.find_joints(["psm_gripper1_Joint", "psm_gripper2_Joint"], preserve_order=True)
    closed_joint_pos = torch.tensor([[0.01, -0.01]], device=device).repeat(num_envs, 1)
    closed_joint_vel = torch.zeros_like(closed_joint_pos)
    robot.write_joint_state_to_sim(closed_joint_pos, closed_joint_vel, joint_ids=gripper_joint_ids)
    robot.set_joint_position_target(closed_joint_pos, joint_ids=gripper_joint_ids)


def configure_helper_visuals(env_cfg, show_helpers: bool):
    """Enable or disable command/debug helper visuals in the environment config."""
    for command_name in ("ee_1_pose", "ee_2_pose", "lift_pose", "object_pose"):
        command_cfg = getattr(env_cfg.commands, command_name, None)
        if command_cfg is not None and hasattr(command_cfg, "debug_vis"):
            command_cfg.debug_vis = show_helpers


def _resolve_usd_path(raw_path: str) -> str:
    """Resolve local USD paths and preserve URI-style asset paths."""
    expanded = os.path.expanduser(raw_path)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", expanded):
        return expanded
    return os.path.abspath(expanded)


def apply_organ_usd_override(env_cfg, usd_path_override: str | None):
    """Override env scene USD path when --usd-path is provided."""
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

    # Keep robot init_state aligned with the active scene USD.
    try:
        from msr.tasks.direct.lift_organ_fixed.msr import joint_pos_env_cfg as lift_organ_fixed_joint_cfg

        lift_organ_fixed_joint_cfg.apply_robot_1_init_state_from_usd(
            env_cfg,
            spawn_cfg.usd_path,
            verbose=True,
        )
    except Exception as err:
        print(f"  [WARNING] Failed to refresh robot init state from overridden USD: {err}")


def _safe_int(value):
    if isinstance(value, str):
        text = value.strip()
        # Handle plain integer strings quickly.
        if text.lstrip("+-").isdigit():
            return int(text)
        # Handle composite keys like "id=7", "<7>", "semanticId: 7".
        match = re.search(r"[-+]?\d+", text)
        if match is not None:
            try:
                return int(match.group(0))
            except ValueError:
                return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_rgba(color):
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r = int(color[0])
            g = int(color[1])
            b = int(color[2])
            a = int(color[3]) if len(color) > 3 else 255
            return (r, g, b, a)
        except (TypeError, ValueError):
            return None
    if isinstance(color, dict):
        keys = {str(k).lower(): v for k, v in color.items()}
        if all(k in keys for k in ("r", "g", "b")):
            try:
                return (int(keys["r"]), int(keys["g"]), int(keys["b"]), int(keys.get("a", 255)))
            except (TypeError, ValueError):
                return None
    return None


def _extract_label_text(meta):
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
        for key in (
            "class",
            "label",
            "name",
            "semanticLabel",
            "semantic_label",
            "className",
            "class_name",
            "data",
        ):
            if key in meta and isinstance(meta[key], str):
                return meta[key]
        parts = [str(v) for v in meta.values() if isinstance(v, str)]
        return "|".join(parts)
    if isinstance(meta, (list, tuple)):
        parts = [str(v) for v in meta if isinstance(v, str)]
        return "|".join(parts)
    return str(meta)


def _extract_id_to_labels(sem_info):
    if not isinstance(sem_info, dict):
        return {}
    id_to_labels = {}
    for key, value in sem_info.items():
        key_lower = str(key).lower()
        if "label" not in key_lower and "semantic" not in key_lower:
            continue

        # Common format: {"idToLabels": {id: meta}}
        if isinstance(value, dict):
            for raw_id, meta in value.items():
                sem_id = _safe_int(raw_id)
                if sem_id is None:
                    continue
                label = _extract_label_text(meta)
                if label:
                    id_to_labels[sem_id] = label
            continue

        # Alternate format: {"idToLabels": [{"id": 1, "label": "..."}, ...]}
        if isinstance(value, (list, tuple)):
            for item in value:
                if not isinstance(item, dict):
                    continue
                sem_id = _safe_int(item.get("id", item.get("semanticId", item.get("semantic_id"))))
                if sem_id is None:
                    continue
                label = _extract_label_text(item)
                if label:
                    id_to_labels[sem_id] = label
    return id_to_labels


def _extract_id_to_colors(sem_info):
    if not isinstance(sem_info, dict):
        return {}
    id_to_colors = {}
    for key, value in sem_info.items():
        if not isinstance(value, dict):
            continue
        if "color" not in str(key).lower():
            continue
        for raw_id, raw_color in value.items():
            sem_id = _safe_int(raw_id)
            if sem_id is None:
                continue
            rgba = _normalize_rgba(raw_color)
            if rgba is not None:
                id_to_colors[sem_id] = rgba
    return id_to_colors


def print_semantic_debug_report(
    step_count: int,
    sem_data: torch.Tensor,
    sem_info,
    tracked_labels: list[str],
    max_colors: int,
    cfg_mapping=None,
):
    """Print a compact semantic diagnostic report for env_0."""
    print("\n[Semantic Debug]" + "=" * 66)
    print(f"  Step: {step_count}")
    print(f"  Tensor shape: {tuple(sem_data.shape)}, dtype: {sem_data.dtype}")

    if sem_data.ndim != 4 or sem_data.shape[0] < 1:
        print("  Invalid semantic tensor shape. Expected [N,H,W,C].")
        print("[Semantic Debug]" + "=" * 66)
        return

    env0 = sem_data[0]
    num_pixels = float(env0.shape[0] * env0.shape[1])
    if num_pixels <= 0:
        print("  Empty semantic frame.")
        print("[Semantic Debug]" + "=" * 66)
        return

    id_to_labels = _extract_id_to_labels(sem_info)
    id_to_colors = _extract_id_to_colors(sem_info)

    if isinstance(sem_info, dict):
        print(f"  sem_info keys: {list(sem_info.keys())}")
    else:
        print(f"  sem_info type: {type(sem_info)}")

    # Build color->labels lookup from camera cfg mapping for direct rendered-color matching.
    cfg_color_to_labels = {}
    if isinstance(cfg_mapping, dict):
        for raw_label, raw_rgba in cfg_mapping.items():
            rgba = _normalize_rgba(raw_rgba)
            if rgba is None:
                continue
            rgb_key = (rgba[0], rgba[1], rgba[2])
            cfg_color_to_labels.setdefault(rgb_key, []).append(str(raw_label))

    if id_to_labels:
        print("  ID -> label -> color (first 20):")
        for sem_id in sorted(id_to_labels.keys())[:20]:
            print(f"    {sem_id}: {id_to_labels[sem_id]} | {id_to_colors.get(sem_id)}")
    else:
        print("  No ID->label mapping found in sem_info.")
        raw_id_to_labels = sem_info.get("idToLabels") if isinstance(sem_info, dict) else None
        if isinstance(raw_id_to_labels, dict):
            print("  Raw idToLabels samples (first 5):")
            for idx, (raw_key, raw_val) in enumerate(raw_id_to_labels.items()):
                if idx >= 5:
                    break
                print(f"    key={raw_key!r} value={raw_val!r}")
        elif isinstance(raw_id_to_labels, (list, tuple)):
            print("  Raw idToLabels samples (first 5 list items):")
            for idx, item in enumerate(raw_id_to_labels[:5]):
                print(f"    item[{idx}]={item!r}")

    if env0.shape[-1] == 1 and env0.dtype in (torch.int16, torch.int32, torch.int64):
        sem_ids = env0[..., 0].to(torch.int64)
        unique_ids, counts = torch.unique(sem_ids, return_counts=True)
        sorted_idx = torch.argsort(counts, descending=True)
        print("  Dominant semantic IDs (env_0):")
        for idx in sorted_idx[:max(1, max_colors)]:
            sem_id = int(unique_ids[idx].item())
            ratio = float(counts[idx].item() / num_pixels)
            print(f"    id={sem_id} ratio={ratio:.4f} label={id_to_labels.get(sem_id, 'UNKNOWN')}")
    elif env0.shape[-1] >= 3:
        rgb = env0[..., :3].to(torch.int64).reshape(-1, 3)
        unique_rgb, counts = torch.unique(rgb, dim=0, return_counts=True)
        sorted_idx = torch.argsort(counts, descending=True)
        print("  Dominant rendered RGB colors (env_0):")
        for idx in sorted_idx[:max(1, max_colors)]:
            color = unique_rgb[idx].tolist()
            ratio = float(counts[idx].item() / num_pixels)
            rgb_key = (int(color[0]), int(color[1]), int(color[2]))
            matched_cfg_labels = cfg_color_to_labels.get(rgb_key, [])
            print(
                f"    rgb=({color[0]},{color[1]},{color[2]}) ratio={ratio:.4f} "
                f"cfg_labels={matched_cfg_labels if matched_cfg_labels else 'UNMAPPED'}"
            )
    else:
        print("  Semantic tensor has unexpected channel layout.")

    if cfg_color_to_labels and env0.shape[-1] >= 3:
        rgb = env0[..., :3].to(torch.int64).reshape(-1, 3)
        unique_rgb, counts = torch.unique(rgb, dim=0, return_counts=True)
        rendered_colors = {tuple(int(v) for v in row.tolist()) for row in unique_rgb}
        unmapped = sorted([color for color in rendered_colors if color not in cfg_color_to_labels])
        if unmapped:
            print("  Unmapped rendered colors (first 10):")
            for color in unmapped[:10]:
                print(f"    {color}")

    if tracked_labels and id_to_labels:
        print("  Tracked label coverage (env_0):")
        is_colorized = env0.shape[-1] >= 3
        if is_colorized:
            rgb = env0[..., :3].to(torch.int64)
            for token in tracked_labels:
                matched_ids = [k for k, v in id_to_labels.items() if token in v.lower()]
                matched_colors = [id_to_colors[k] for k in matched_ids if k in id_to_colors]
                mask = torch.zeros(rgb.shape[:-1], dtype=torch.bool, device=rgb.device)
                for r, g, b, _ in matched_colors:
                    mask |= (rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)
                ratio = float(mask.sum().item() / num_pixels)
                print(f"    token='{token}' ids={matched_ids} colors={matched_colors} ratio={ratio:.4f}")
        else:
            sem_ids = env0[..., 0].to(torch.int64)
            for token in tracked_labels:
                matched_ids = [k for k, v in id_to_labels.items() if token in v.lower()]
                mask = torch.zeros_like(sem_ids, dtype=torch.bool)
                for sem_id in matched_ids:
                    mask |= sem_ids == sem_id
                ratio = float(mask.sum().item() / num_pixels)
                print(f"    token='{token}' ids={matched_ids} ratio={ratio:.4f}")

    print("[Semantic Debug]" + "=" * 66)


def describe_observation(value):
    """Return a compact, human-readable summary for an observation value."""
    if hasattr(value, "shape"):
        return f"tensor shape={tuple(value.shape)}"

    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if hasattr(item, "shape"):
                parts.append(f"{key}: shape={tuple(item.shape)}")
            else:
                parts.append(f"{key}: {type(item).__name__}")
        return "dict{" + ", ".join(parts) + "}"

    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} len={len(value)}"

    return type(value).__name__


def main():
    """Main function with keyboard teleoperation for end-effector control."""
    
    # Parse configuration using the new environment
    env_cfg = parse_env_cfg(
        "Isaac-VesselSemFixed-SingleRobot-IK-RCM-ConnectivityOnly-Play-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_helper_visuals(env_cfg, args_cli.show_helpers)
    apply_organ_usd_override(env_cfg, args_cli.usd_path)
    
    print("\n" + "="*80)
    print("Creating Environment: Isaac-VesselSemFixed-SingleRobot-IK-RCM-ConnectivityOnly-Play-v0")
    print("="*80)
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {args_cli.device}")
    organ_spawn_cfg = getattr(getattr(env_cfg.scene, "organ", None), "spawn", None)
    if organ_spawn_cfg is not None and hasattr(organ_spawn_cfg, "usd_path"):
        print(f"  Organ USD: {organ_spawn_cfg.usd_path}")
    print("="*80 + "\n")
    
    # Create environment
    env = gym.make("Isaac-VesselSemFixed-SingleRobot-IK-RCM-ConnectivityOnly-Play-v0", cfg=env_cfg)
    
    # Reset environment
    obs, info = env.reset()
    
    print("\n[Environment Reset Complete]")
    policy_obs = obs["policy"] if isinstance(obs, dict) and "policy" in obs else None
    print(f"  Observation type: {type(policy_obs).__name__}")
    print(f"  Observation summary: {describe_observation(policy_obs)}")
    print(f"  Number of environments: {env.unwrapped.num_envs}")
    
    # ========================================================================
    # SETUP CAMERA AND IMAGE SAVING
    # ========================================================================
    print("\n" + "="*80)
    print("Setting up Camera and Image Saving")
    print("="*80)
    
    # Create save directory with timestamp
    save_base_path = args_cli.save_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_base_path, timestamp)
    rgb_save_dir = os.path.join(save_dir, "rgb")
    sem_save_dir = os.path.join(save_dir, "semantic")
    depth_save_dir = os.path.join(save_dir, "depth")
    pointcloud_save_dir = os.path.join(save_dir, "pointcloud")
    if args_cli.save_data:
        os.makedirs(rgb_save_dir, exist_ok=True)
        os.makedirs(sem_save_dir, exist_ok=True)
        os.makedirs(depth_save_dir, exist_ok=True)
        os.makedirs(pointcloud_save_dir, exist_ok=True)
        print(f"  Save directory: {save_dir}")
        print(f"  RGB images: {rgb_save_dir}")
        print(f"  Semantic images: {sem_save_dir}")
        print(f"  Depth images: {depth_save_dir}")
        print(f"  Point clouds: {pointcloud_save_dir}")
    else:
        print("  Data saving disabled. Pass --save-data to enable image and point cloud export.")
    
    # Get camera from scene
    camera = None
    try:
        camera: TiledCamera = env.unwrapped.scene["camera"]
        print(f"  Camera found at: {camera.cfg.prim_path}")
        print(f"  Camera resolution: {camera.cfg.width}x{camera.cfg.height}")
        print(f"  Data types: {camera.cfg.data_types}")
    except KeyError:
        print("  [WARNING] Camera not found in scene")
        print("  Make sure the D455 color camera exists at {ENV_REGEX_NS}/Organ/rsd455/RSD455/Camera_OmniVision_OV9782_Color")

    depth_camera = None
    try:
        depth_camera: TiledCamera = env.unwrapped.scene["depth_camera"]
        print(f"  Depth camera found at: {depth_camera.cfg.prim_path}")
        print(f"  Depth camera resolution: {depth_camera.cfg.width}x{depth_camera.cfg.height}")
        print(f"  Depth camera data types: {depth_camera.cfg.data_types}")
    except KeyError:
        print("  [WARNING] Depth camera not found in scene")
        print("  Make sure the D455 depth camera exists at {ENV_REGEX_NS}/Organ/rsd455/RSD455/Camera_Pseudo_Depth")
    
    # Image save counter and interval
    image_save_interval = args_cli.image_save_interval
    image_counter = 0
    semantic_debug_interval = max(0, int(args_cli.semantic_debug_interval))
    semantic_debug_max_colors = max(1, int(args_cli.semantic_debug_max_colors))
    semantic_debug_labels = [
        token.strip().lower()
        for token in args_cli.semantic_debug_labels.split(",")
        if token.strip()
    ]
    print(f"  Image save interval: every {image_save_interval} steps")
    if semantic_debug_interval > 0:
        print(f"  Semantic debug interval: every {semantic_debug_interval} steps")
        print(f"  Semantic tracked labels: {semantic_debug_labels}")
    else:
        print("  Semantic debug interval: disabled (0)")
    print("="*80)
    
    # Analyze scene structure
    print("\n")
    analyze_scene_structure("/World/envs/env_0")
    
    # NOTE: Semantic labels are now added directly in the USD asset file
    # No need to apply them here at runtime
    
    # ========================================================================
    # GET SCENE OBJECTS
    # ========================================================================
    print("\n" + "="*80)
    print("Scene Objects")
    print("="*80)
    
    # Get robot_1
    try:
        robot_1 = env.unwrapped.scene["robot_1"]
        print_robot_info(robot_1, "Robot 1 (PSM)")
    except KeyError:
        print("[WARNING] robot_1 not found in scene")
        robot_1 = None
    
    print("="*80)
    
    # ========================================================================
    # CREATE KEYBOARD TELEOPERATION INTERFACE (GUI mode only)
    # ========================================================================
    print("\n" + "="*80)
    print("Setting up Control Interface")
    print("="*80)
    
    # Check if running in headless mode
    is_headless = args_cli.headless
    teleop_interface = None
    
    if is_headless:
        print("  [HEADLESS MODE] Running automated test (no keyboard control)")
        print("  To use keyboard control, run without --headless flag")
    else:
        try:
            # Create keyboard controller
            keyboard_cfg = Se3KeyboardCfg(
                pos_sensitivity=0.005 * args_cli.sensitivity,  # Position delta per key press
                rot_sensitivity=0.05 * args_cli.sensitivity,   # Rotation delta per key press
                gripper_term=False,
                sim_device=env.unwrapped.device,
            )
            teleop_interface = Se3Keyboard(keyboard_cfg)
            
            # Add reset callback
            teleop_interface.add_callback("L", lambda: env.reset())
            
            # Print keyboard controls
            print(teleop_interface)
        except (AttributeError, RuntimeError) as e:
            print(f"  [WARNING] Keyboard setup failed: {e}")
            print("  Running in automated test mode instead")
            teleop_interface = None
    
    print("="*80)
    
    # ========================================================================
    # CREATE DIFFERENTIAL IK CONTROLLER
    # ========================================================================
    print("\n" + "="*80)
    print("Setting up Differential IK Controller")
    print("="*80)
    
    # Define arm joint names (excluding gripper)
    arm_joint_names = [
        "psm_shoulder_Joint",
        "psm_upper_Joint",
        "psm_fore_Joint",
        "psm_wrist1_Joint",
        "psm_wrist2_Joint",
        "psm_wrist3_Joint",
        "psm_insertion_Joint",
        "psm_roll_Joint",
        "psm_pitch_Joint",
        "psm_yaw_Joint",
    ]
    
    # Get joint indices for arm joints (for action mapping)
    arm_joint_indices_list = []
    for joint_name in arm_joint_names:
        try:
            idx = robot_1.joint_names.index(joint_name)
            arm_joint_indices_list.append(idx)
        except ValueError:
            print(f"[WARNING] Joint {joint_name} not found")
    
    arm_joint_indices = arm_joint_indices_list  # Keep as list for indexing
    print(f"  Arm joint indices: {arm_joint_indices}")
    
    # Create differential IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,  # Use delta pose commands
        ik_method="dls",  # Damped Least Squares for stability
        ik_params={"lambda_val": 0.05},  # Damping factor
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, env.unwrapped.num_envs, env.unwrapped.device)
    
    # Get end-effector body index
    ee_body_name = "psm_tool_tip_Link"
    ee_body_ids, ee_body_names = robot_1.find_bodies(ee_body_name)
    if len(ee_body_ids) != 1:
        raise ValueError(f"Expected one match for body name: {ee_body_name}. Found {len(ee_body_ids)}: {ee_body_names}")
    ee_body_idx = ee_body_ids[0]
    print(f"  End-effector body: {ee_body_name} (index: {ee_body_idx})")
    
    # For jacobian computation:
    # - Fixed-base articulation: jacobian doesn't include base, so body index is (body_idx - 1)
    # - Floating-base articulation: jacobian includes base, so body index is body_idx
    if robot_1.is_fixed_base:
        ee_jacobi_idx = ee_body_idx - 1
        jacobi_joint_ids = arm_joint_indices  # No offset needed for fixed-base
    else:
        ee_jacobi_idx = ee_body_idx
        jacobi_joint_ids = [i + 6 for i in arm_joint_indices]  # Add 6 for floating-base (6 DOF base)
    
    print(f"  Jacobian body index: {ee_jacobi_idx}")
    print(f"  Robot is fixed-base: {robot_1.is_fixed_base}")
    
    print(f"  Differential IK method: {diff_ik_cfg.ik_method}")
    
    # Get default joint positions for arm joints (needed for action mapping)
    # With use_default_offset=True: action = (target_pos - default_pos) / scale
    # Since scale=1.0, we just subtract default positions
    default_arm_joint_pos = robot_1.data.default_joint_pos[:, arm_joint_indices].clone()
    print(f"  Default arm joint positions: {default_arm_joint_pos[0].cpu().tolist()}")
    
    # Convert arm_joint_indices to tensor for use in main loop
    arm_joint_indices_tensor = torch.tensor(arm_joint_indices, device=env.unwrapped.device)
    
    print("="*80)
    
    # ========================================================================
    # CREATE VISUALIZATION MARKERS (GUI mode only)
    # ========================================================================
    ee_frame_marker = None
    target_frame_marker = None
    cube_frame_marker = None
    
    if not is_headless and args_cli.show_helpers:
        print("\n" + "="*80)
        print("Creating Visualization Markers")
        print("="*80)
        
        # Robot end-effector frame marker
        ee_marker_cfg = FRAME_MARKER_CFG.copy()
        ee_marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)  # 1cm frame
        ee_marker_cfg.prim_path = "/Visuals/EndEffectorFrame"
        ee_frame_marker = VisualizationMarkers(ee_marker_cfg)
        print("  ✓ End-effector frame marker (1cm)")
        
        # Target frame marker (shows where the end-effector should go)
        target_marker_cfg = FRAME_MARKER_CFG.copy()
        target_marker_cfg.markers["frame"].scale = (0.008, 0.008, 0.008)  # 0.8cm frame
        target_marker_cfg.prim_path = "/Visuals/TargetFrame"
        target_frame_marker = VisualizationMarkers(target_marker_cfg)
        print("  ✓ Target frame marker (0.8cm)")
        
        print("="*80 + "\n")
    else:
        print("\n[INFO] Helper visualization disabled")
    
    # Initialize action buffer
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    action_scale = float(getattr(getattr(env_cfg.actions, "arm_1_action", None), "scale", 1.0))
    force_gripper_closed(robot_1, env.unwrapped.device, env.unwrapped.num_envs)
    
    # ========================================================================
    # ACCUMULATED TARGET POSE (prevents snapping back when key is released)
    # ========================================================================
    # Initialize target pose to current end-effector pose
    target_pos = robot_1.data.body_pos_w[:, ee_body_idx, :].clone()
    target_quat = robot_1.data.body_quat_w[:, ee_body_idx, :].clone()
    print(f"\n[Initial Target Pose]")
    print(f"  Position: [{target_pos[0, 0]:.4f}, {target_pos[0, 1]:.4f}, {target_pos[0, 2]:.4f}]")
    
    print("\n" + "="*80)
    if teleop_interface is not None:
        print("Starting Keyboard Teleoperation")
        print("="*80)
        print("Controls:")
        print("  W/S: Move along X-axis")
        print("  A/D: Move along Y-axis")
        print("  Q/E: Move along Z-axis")
        print("  Z/X: Rotate around X-axis (roll)")
        print("  T/G: Rotate around Y-axis (pitch)")
        print("  C/V: Rotate around Z-axis (yaw)")
        print("  L: Reset environment")
        print("  Ctrl+C: Exit")
        teleop_interface.reset()
    else:
        print("Starting Automated Test (Headless Mode)")
        print("="*80)
        print("  Simulating circular end-effector motion for testing")
    print("="*80 + "\n")
    
    step_count = 0
    max_steps_headless = 500  # Max steps for headless auto-test
    
    while simulation_app.is_running():
        if teleop_interface is not None:
            # GUI mode: Get keyboard command
            # Returns: [dx, dy, dz, drx, dry, drz]
            keyboard_cmd = teleop_interface.advance()

            # Extract delta pose command
            delta_pose = keyboard_cmd[:6]  # [dx, dy, dz, drx, dry, drz]
        else:
            # Headless mode: Generate test commands (small circular motion)
            import math
            t = step_count * 0.02  # Time parameter
            delta_pose = torch.tensor([
                0.001 * math.sin(t),  # dx - oscillate X
                0.001 * math.cos(t),  # dy - oscillate Y
                0.0,                   # dz - no vertical
                0.0,                   # drx
                0.0,                   # dry
                0.0,                   # drz
            ], device=env.unwrapped.device)
            
            # Exit after max_steps in headless mode
            if step_count >= max_steps_headless:
                print(f"\n[INFO] Headless test completed after {step_count} steps")
                break
        
        # ====================================================================
        # ACCUMULATED TARGET POSE UPDATE
        # ====================================================================
        # Update target position by accumulating delta (prevents snapping back)
        delta_pos = delta_pose[:3]  # [dx, dy, dz]
        delta_rot = delta_pose[3:6]  # [drx, dry, drz] - axis-angle increments
        
        # Accumulate position
        target_pos = target_pos + delta_pos.unsqueeze(0)
        
        # Accumulate rotation (convert axis-angle delta to quaternion and multiply)
        if torch.norm(delta_rot) > 1e-6:
            from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
            # Interpret delta_rot as roll, pitch, yaw increments
            delta_quat = quat_from_euler_xyz(
                delta_rot[0].unsqueeze(0),  # roll (X)
                delta_rot[1].unsqueeze(0),  # pitch (Y)
                delta_rot[2].unsqueeze(0),  # yaw (Z)
            )
            # Apply rotation (new_quat = delta_quat * current_quat)
            target_quat = quat_mul(delta_quat, target_quat)
        
        # Get current end-effector pose
        ee_pos_curr = robot_1.data.body_pos_w[:, ee_body_idx, :].clone()
        ee_quat_curr = robot_1.data.body_quat_w[:, ee_body_idx, :].clone()  # (w, x, y, z)
        
        # ====================================================================
        # COMPUTE IK TOWARDS ACCUMULATED TARGET (not using relative mode)
        # ====================================================================
        # Compute position and orientation error
        pos_error = target_pos - ee_pos_curr
        
        # For orientation error, use axis-angle representation
        from isaaclab.utils.math import compute_pose_error
        _, axis_angle_error = compute_pose_error(
            ee_pos_curr, ee_quat_curr, 
            target_pos, target_quat, 
            rot_error_type="axis_angle"
        )
        
        # Combine into pose error [pos_error, rot_error]
        pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)
        
        # Get Jacobian from robot
        # jacobian shape: (num_envs, num_bodies, 6, num_joints)
        jacobian = robot_1.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jacobi_joint_ids]
        
        # Get current joint positions for arm joints
        joint_pos_curr = robot_1.data.joint_pos[:, arm_joint_indices]
        
        # Compute delta joint positions using damped least squares
        # J^T * (J * J^T + lambda^2 * I)^-1 * error
        lambda_val = 0.05
        jacobian_T = jacobian.transpose(1, 2)  # (num_envs, num_joints, 6)
        lambda_matrix = (lambda_val ** 2) * torch.eye(6, device=env.unwrapped.device)
        
        # (J * J^T + lambda^2 * I)
        JJT_damped = torch.bmm(jacobian, jacobian_T) + lambda_matrix.unsqueeze(0)
        
        # J^T * (J * J^T + lambda^2 * I)^-1 * error
        delta_joint_pos = torch.bmm(
            jacobian_T,
            torch.linalg.solve(JJT_damped, pose_error.unsqueeze(-1))
        ).squeeze(-1)
        
        # Compute desired joint positions
        joint_pos_des = joint_pos_curr + delta_joint_pos
        
        # The current environment expects 6D pose actions.
        # Convert the keyboard delta from physical units to the raw action space.
        normalized_delta_pose = torch.as_tensor(
            delta_pose, device=env.unwrapped.device, dtype=actions.dtype
        ) / action_scale
        if actions.ndim == 2:
            actions[:] = normalized_delta_pose.unsqueeze(0).expand(actions.shape[0], -1)
        else:
            actions[:] = normalized_delta_pose
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Print script-level semantic diagnostics using the same data path as saved images.
        if (
            semantic_debug_interval > 0
            and step_count % semantic_debug_interval == 0
            and camera is not None
            and "semantic_segmentation" in camera.cfg.data_types
        ):
            sem_info = camera.data.info.get("semantic_segmentation", {})
            print_semantic_debug_report(
                step_count=step_count,
                sem_data=camera.data.output["semantic_segmentation"],
                sem_info=sem_info,
                tracked_labels=semantic_debug_labels,
                max_colors=semantic_debug_max_colors,
                cfg_mapping=getattr(camera.cfg, "semantic_segmentation_mapping", {}),
            )
        
        # Update visualization markers every 5 steps (GUI mode only)
        if step_count % 5 == 0 and not is_headless:
            # Update end-effector marker
            if ee_frame_marker is not None:
                ee_pos = robot_1.data.body_pos_w[:, ee_body_idx, :]
                ee_quat = robot_1.data.body_quat_w[:, ee_body_idx, :]
                ee_frame_marker.visualize(ee_pos, ee_quat)
            
            # Update target marker (show accumulated target pose)
            if target_frame_marker is not None:
                target_frame_marker.visualize(target_pos, target_quat)
            

        # ====================================================================
        # CAPTURE AND SAVE CAMERA IMAGES
        # ====================================================================
        if args_cli.save_data and step_count % image_save_interval == 0 and (camera is not None or depth_camera is not None):
            try:
                # Save RGB images
                if camera is not None and "rgb" in camera.cfg.data_types:
                    rgb_data = camera.data.output["rgb"].clone()
                    # Camera data is in [0, 255] float range, need to normalize to [0, 1] for saving
                    # Also handle RGBA case - take only first 3 channels
                    if rgb_data.shape[-1] == 4:
                        rgb_data = rgb_data[..., :3]  # Remove alpha channel
                    # Normalize to [0, 1] and clamp
                    rgb_data = torch.clamp(rgb_data / 255.0, 0.0, 1.0)
                    # Save images to rgb subfolder
                    rgb_image_path = os.path.join(rgb_save_dir, f"rgb_{image_counter:06d}.png")
                    save_images_to_file(rgb_data, rgb_image_path)
                    print(f"  [Camera] Saved RGB image: rgb_{image_counter:06d}.png")
                
                # Save semantic segmentation images
                if camera is not None and "semantic_segmentation" in camera.cfg.data_types:
                    sem_data = camera.data.output["semantic_segmentation"].clone()
                    # Semantic data is colorized uint8 RGBA [0, 255]
                    # Take only first 3 channels (RGB) for saving
                    if sem_data.shape[-1] == 4:
                        sem_data = sem_data[..., :3]
                    # Normalize to [0, 1] for save_images_to_file
                    sem_data = torch.clamp(sem_data.float() / 255.0, 0.0, 1.0)
                    # Save images to semantic subfolder
                    sem_image_path = os.path.join(sem_save_dir, f"sem_{image_counter:06d}.png")
                    save_images_to_file(sem_data, sem_image_path)
                    print(f"  [Camera] Saved semantic image: sem_{image_counter:06d}.png")

                if depth_camera is not None and "depth" in depth_camera.cfg.data_types:
                    depth_data = depth_camera.data.output["depth"].clone()
                    finite_mask = torch.isfinite(depth_data)
                    if finite_mask.any():
                        finite_depth = depth_data[finite_mask]
                        depth_min = finite_depth.min()
                        depth_max = finite_depth.max()
                        if (depth_max - depth_min) > 1e-6:
                            depth_vis = (depth_data - depth_min) / (depth_max - depth_min)
                        else:
                            depth_vis = torch.zeros_like(depth_data)
                        depth_vis = torch.where(finite_mask, depth_vis, torch.zeros_like(depth_vis))
                        depth_vis = torch.clamp(depth_vis, 0.0, 1.0)
                        depth_image_path = os.path.join(depth_save_dir, f"depth_{image_counter:06d}.png")
                        save_images_to_file(depth_vis, depth_image_path)
                        print(
                            f"  [Camera] Saved depth image: depth_{image_counter:06d}.png "
                            f"(range: {depth_min.item():.4f}m to {depth_max.item():.4f}m)"
                        )
                    else:
                        print("  [Camera] Depth frame contains no finite values, skipping save")

                if (
                    camera is not None
                    and depth_camera is not None
                    and "semantic_segmentation" in camera.cfg.data_types
                    and "depth" in depth_camera.cfg.data_types
                ):
                    sem_data = camera.data.output["semantic_segmentation"].clone()
                    depth_data = depth_camera.data.output["depth"].clone()

                    for env_id in range(env.unwrapped.num_envs):
                        semantic_rgb = sem_data[env_id, ..., :3].to(torch.float32)
                        depth_image = depth_data[env_id, ..., 0]
                        finite_depth_mask = torch.isfinite(depth_image)

                        if not finite_depth_mask.any():
                            print(f"  [PointCloud] Env {env_id}: no finite depth values, skipping point cloud")
                            continue

                        points_xyz, points_rgb = create_pointcloud_from_rgbd(
                            intrinsic_matrix=depth_camera.data.intrinsic_matrices[env_id],
                            depth=depth_image,
                            rgb=semantic_rgb,
                            normalize_rgb=False,
                            position=depth_camera.data.pos_w[env_id],
                            orientation=depth_camera.data.quat_w_world[env_id],
                            device=env.unwrapped.device,
                        )

                        if points_xyz.shape[0] == 0:
                            print(f"  [PointCloud] Env {env_id}: empty point cloud after filtering, skipping")
                            continue

                        pointcloud_path = os.path.join(
                            pointcloud_save_dir, f"pc_sem_env{env_id}_{image_counter:06d}.ply"
                        )
                        save_pointcloud_to_ply(pointcloud_path, points_xyz, points_rgb)
                        print(
                            f"  [PointCloud] Saved semantic-colored point cloud: "
                            f"pc_sem_env{env_id}_{image_counter:06d}.ply ({points_xyz.shape[0]} points)"
                        )
                
                image_counter += 1
            except Exception as e:
                print(f"  [Camera] Error saving image: {e}")
        
        # Handle environment reset (reset target pose when episode ends)
        if terminated.any() or truncated.any():
            # Reset target pose to current end-effector pose
            target_pos = robot_1.data.body_pos_w[:, ee_body_idx, :].clone()
            target_quat = robot_1.data.body_quat_w[:, ee_body_idx, :].clone()
            force_gripper_closed(robot_1, env.unwrapped.device, env.unwrapped.num_envs)
            print(f"\n[Environment Reset] Target pose reset to current EE pose")
        
        # Print status every 200 steps
        if step_count % 200 == 0 and step_count > 0:
            ee_pos = robot_1.data.body_pos_w[0, ee_body_idx, :]
            tgt_pos = target_pos[0]
            print(f"Step: {step_count}")
            print(f"  Target pos: [{tgt_pos[0]:.4f}, {tgt_pos[1]:.4f}, {tgt_pos[2]:.4f}]")
            print(f"  EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
        
        step_count += 1
    
    # Clean up
    env.close()
    print("\n[Environment Closed]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
