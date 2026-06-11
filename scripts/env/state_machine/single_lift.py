# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Single robot environment test script with keyboard teleoperation
# Robot end-effector pose can be controlled via keyboard
#
# Usage (with GUI - required for keyboard control):
#   ${IsaacLab_PATH}/isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/single_lift.py --num_envs 1
#
# Usage (headless - auto test mode):
#   ${IsaacLab_PATH}/isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/single_lift.py --num_envs 1 --headless
#
# Keyboard controls (GUI mode only):
#   W/S: Move end-effector along X-axis
#   A/D: Move end-effector along Y-axis
#   Q/E: Move end-effector along Z-axis
#   Z/X: Rotate around X-axis (roll)
#   T/G: Rotate around Y-axis (pitch)
#   C/V: Rotate around Z-axis (yaw)
#   K: Toggle gripper open/close
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
from datetime import datetime

from isaaclab.assets.rigid_object import RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.sensors import TiledCamera, save_images_to_file
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import numpy as np

import isaaclab_tasks  # noqa: F401
import asb_dual_arm.tasks  # noqa: F401

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


def main():
    """Main function with keyboard teleoperation for end-effector control."""
    
    # Parse configuration using the new environment
    env_cfg = parse_env_cfg(
        "Isaac-LiftOrgan-Upe6-SingleRobot-JointPos-Play-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    print("\n" + "="*80)
    print("Creating Environment: Isaac-LiftOrgan-Upe6-SingleRobot-JointPos-Play-v0")
    print("="*80)
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Device: {args_cli.device}")
    print("="*80 + "\n")
    
    # Create environment
    env = gym.make("Isaac-LiftOrgan-Upe6-SingleRobot-JointPos-Play-v0", cfg=env_cfg)
    
    # Reset environment
    obs, info = env.reset()
    
    print("\n[Environment Reset Complete]")
    print(f"  Observation shape: {obs['policy'].shape}")
    print(f"  Number of environments: {env.unwrapped.num_envs}")
    
    # ========================================================================
    # SETUP CAMERA AND IMAGE SAVING
    # ========================================================================
    print("\n" + "="*80)
    print("Setting up Camera and Image Saving")
    print("="*80)
    
    # Create save directory with timestamp
    save_base_path = "/workspace/isaaclab/source/ASBDualArm/scripts/saved"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_base_path, timestamp)
    rgb_save_dir = os.path.join(save_dir, "rgb")
    sem_save_dir = os.path.join(save_dir, "semantic")
    depth_save_dir = os.path.join(save_dir, "depth")
    pointcloud_save_dir = os.path.join(save_dir, "pointcloud")
    os.makedirs(rgb_save_dir, exist_ok=True)
    os.makedirs(sem_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(pointcloud_save_dir, exist_ok=True)
    print(f"  Save directory: {save_dir}")
    print(f"  RGB images: {rgb_save_dir}")
    print(f"  Semantic images: {sem_save_dir}")
    print(f"  Depth images: {depth_save_dir}")
    print(f"  Point clouds: {pointcloud_save_dir}")
    
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
    image_save_interval = 50  # Save image every N steps
    image_counter = 0
    print(f"  Image save interval: every {image_save_interval} steps")
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
    
    # Check for cube_02
    try:
        cube_02: RigidObject = env.unwrapped.scene["cube_02"]
        print(f"\n[Cube_02 Information]")
        print(f"  Prim path: {cube_02.cfg.prim_path}")
        print(f"  Number of instances: {cube_02.num_instances}")
        print(f"  Position: {cube_02.data.root_pos_w[0].cpu().numpy()}")
        print(f"  Orientation (wxyz): {cube_02.data.root_quat_w[0].cpu().numpy()}")
    except KeyError:
        print("[WARNING] cube_02 not found in scene")
        cube_02 = None
    
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
                gripper_term=True,
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
    
    if not is_headless:
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
        
        # Cube frame marker (if cube exists)
        if cube_02 is not None:
            cube_marker_cfg = FRAME_MARKER_CFG.copy()
            cube_marker_cfg.markers["frame"].scale = (0.005, 0.005, 0.005)  # 5mm frame
            cube_marker_cfg.prim_path = "/Visuals/CubeFrame"
            cube_frame_marker = VisualizationMarkers(cube_marker_cfg)
            print("  ✓ Cube_02 frame marker (5mm)")
        
        print("="*80 + "\n")
    else:
        print("\n[HEADLESS MODE] Skipping visualization markers")
    
    # Initialize action buffer
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    
    # Track gripper state
    gripper_open = True  # Start with gripper open
    
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
        print("  K: Toggle gripper open/close")
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
            # Returns: [dx, dy, dz, drx, dry, drz, gripper_command]
            keyboard_cmd = teleop_interface.advance()
            
            # Extract delta pose (first 6 elements) and gripper command
            delta_pose = keyboard_cmd[:6]  # [dx, dy, dz, drx, dry, drz]
            gripper_cmd = keyboard_cmd[6] if len(keyboard_cmd) > 6 else 1.0  # +1 open, -1 close
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
            gripper_cmd = 1.0  # Keep gripper open
            
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
        
        # Build action tensor: arm joints + gripper
        # Action space: 10 arm joints + 1 gripper binary command
        # With use_default_offset=True: processed = raw * scale + default_pos
        # So raw_action = (target_pos - default_pos) / scale
        # Since scale=1.0, raw_action = target_pos - default_pos
        arm_actions = joint_pos_des - default_arm_joint_pos
        
        # Map to full action space
        # The action space is: arm_1_action (10 joints) + gripper_1_action (1 binary)
        actions[:, :10] = arm_actions
        
        # Gripper action: positive = open, negative = close
        actions[:, 10] = gripper_cmd
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
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
            
            # Update cube marker
            if cube_02 is not None and cube_frame_marker is not None:
                cube_pos = cube_02.data.root_pos_w
                cube_quat = cube_02.data.root_quat_w
                cube_frame_marker.visualize(cube_pos, cube_quat)
        
        # ====================================================================
        # CAPTURE AND SAVE CAMERA IMAGES
        # ====================================================================
        if step_count % image_save_interval == 0 and (camera is not None or depth_camera is not None):
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
                    sem_data_colored = camera.data.output["semantic_segmentation"].clone()
                    depth_data = depth_camera.data.output["depth"].clone()

                    for env_id in range(env.unwrapped.num_envs):
                        semantic_rgb = sem_data_colored[env_id, ..., :3].to(torch.float32)
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
            print(f"\n[Environment Reset] Target pose reset to current EE pose")
        
        # Print status every 200 steps
        if step_count % 200 == 0 and step_count > 0:
            ee_pos = robot_1.data.body_pos_w[0, ee_body_idx, :]
            tgt_pos = target_pos[0]
            print(f"Step: {step_count}")
            print(f"  Target pos: [{tgt_pos[0]:.4f}, {tgt_pos[1]:.4f}, {tgt_pos[2]:.4f}]")
            print(f"  EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
            print(f"  Gripper: {'OPEN' if gripper_cmd > 0 else 'CLOSED'}")
        
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
