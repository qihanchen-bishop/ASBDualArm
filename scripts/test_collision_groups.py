# Quick test script for collision group setup
# Runs just the analysis without full simulation

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test collision groups")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema
from isaacsim.core.utils.stage import get_current_stage
import traceback

import isaaclab_tasks  # noqa: F401
import msr.tasks  # noqa: F401

def analyze_organ_structure(env_prim_path: str):
    """Analyze the USD structure under Organ prim"""
    stage = get_current_stage()
    organ_prim_path = f"{env_prim_path}/Organ"
    organ_prim = stage.GetPrimAtPath(organ_prim_path)
    
    print("\n" + "="*80)
    print(f"Analyzing USD Structure under: {organ_prim_path}")
    print("="*80)
    
    if not organ_prim.IsValid():
        print(f"ERROR: Organ prim not found at {organ_prim_path}")
        return []
    
    physics_prims = []
    for prim in Usd.PrimRange(organ_prim):
        prim_path = str(prim.GetPath())
        prim_type = prim.GetTypeName()
        
        # Check physics APIs
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_deformable_body = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
        has_deformable_surface = prim.HasAPI(PhysxSchema.PhysxDeformableSurfaceAPI)
        has_soft_body = prim.HasAPI(PhysxSchema.PhysxDeformableSoftBodyAPI) if hasattr(PhysxSchema, 'PhysxDeformableSoftBodyAPI') else False
        
        apis = []
        if has_rigid:
            apis.append("RigidBody")
        if has_collision:
            apis.append("Collision")
        if has_deformable_body:
            apis.append("DeformableBody")
        if has_deformable_surface:
            apis.append("DeformableSurface")
        if has_soft_body:
            apis.append("SoftBody")
        
        if apis or prim_type == "Mesh":
            api_str = ", ".join(apis) if apis else "None"
            print(f"  {prim_path}")
            print(f"    Type: {prim_type}, APIs: [{api_str}]")
            if apis:
                physics_prims.append((prim_path, prim_type, apis))
    
    print("="*80)
    return physics_prims


def setup_collision_groups(env_prim_path: str):
    """Setup collision groups"""
    stage = get_current_stage()
    
    robot_cube_group_path = f"{env_prim_path}/RobotCubeCollisionGroup"
    deformable_group_path = f"{env_prim_path}/DeformableCollisionGroup"
    
    try:
        # 1. Create collision group for robot arms and Cube_02
        robot_cube_group = UsdPhysics.CollisionGroup.Define(stage, robot_cube_group_path)
        print(f"[Collision] Created: {robot_cube_group_path}")
        
        # 2. Create collision group for deformable objects
        deformable_group = UsdPhysics.CollisionGroup.Define(stage, deformable_group_path)
        print(f"[Collision] Created: {deformable_group_path}")
        
        # 3. Add robot arms and Cube_02 to RobotCubeCollisionGroup
        robot_cube_collection = Usd.CollectionAPI.Apply(robot_cube_group.GetPrim(), "colliders")
        robot_cube_includes = robot_cube_collection.CreateIncludesRel()
        
        robot_1_path = f"{env_prim_path}/Robot_1"
        robot_2_path = f"{env_prim_path}/Robot_2"
        cube_02_path = f"{env_prim_path}/Organ/Cube_02"
        
        robot_cube_includes.AddTarget(robot_1_path)
        robot_cube_includes.AddTarget(robot_2_path)
        robot_cube_includes.AddTarget(cube_02_path)
        
        print(f"[Collision] Added to RobotCubeCollisionGroup: Robot_1, Robot_2, Cube_02")
        
        # 4. Find deformable/mesh prims under Organ
        organ_prim_path = f"{env_prim_path}/Organ"
        organ_prim = stage.GetPrimAtPath(organ_prim_path)
        
        deformable_collection = Usd.CollectionAPI.Apply(deformable_group.GetPrim(), "colliders")
        deformable_includes = deformable_collection.CreateIncludesRel()
        
        deformable_prims = []
        if organ_prim.IsValid():
            for prim in Usd.PrimRange(organ_prim):
                prim_path = str(prim.GetPath())
                
                # Skip Cube_02 (it's the grasp target)
                if "Cube_02" in prim_path:
                    continue
                
                # Only add prims with DeformableBody or DeformableSurface API
                # (not just any mesh with collision - we want deformable objects only)
                has_deformable = (
                    prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI) or
                    prim.HasAPI(PhysxSchema.PhysxDeformableSurfaceAPI)
                )
                
                if has_deformable:
                    deformable_includes.AddTarget(prim_path)
                    deformable_prims.append(prim_path)
        
        print(f"[Collision] Added {len(deformable_prims)} prims to DeformableCollisionGroup")
        for p in deformable_prims:
            print(f"  - {p}")
        
        # 5. Setup filtering
        robot_cube_filtered = robot_cube_group.CreateFilteredGroupsRel()
        robot_cube_filtered.AddTarget(deformable_group_path)
        
        deformable_filtered = deformable_group.CreateFilteredGroupsRel()
        deformable_filtered.AddTarget(robot_cube_group_path)
        
        print(f"[Collision] Filtering set: RobotCubeCollisionGroup <-> DeformableCollisionGroup")
        
        return True
        
    except Exception as e:
        print(f"[Collision] Error: {e}")
        traceback.print_exc()
        return False


def main():
    # Import config
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    
    env_cfg = parse_env_cfg(
        "Isaac-LiftOrgan-Needle_with_Rope-MSRPSM-IK-Rel-Play-v0",
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )
    
    # Create environment
    print("\n[Main] Creating environment...")
    env = gym.make("Isaac-LiftOrgan-Needle_with_Rope-MSRPSM-IK-Rel-Play-v0", cfg=env_cfg)
    
    # Reset to populate the scene
    print("[Main] Resetting environment...")
    env.reset()
    
    # Analyze and setup collision groups
    print("\n[Main] Analyzing organ structure...")
    physics_prims = analyze_organ_structure("/World/envs/env_0")
    
    print("\n[Main] Setting up collision groups...")
    success = setup_collision_groups("/World/envs/env_0")
    
    if success:
        print("\n[SUCCESS] Collision groups setup complete!")
        print("Robot arms will collide with Cube_02 but NOT with deformable mesh")
    else:
        print("\n[FAILURE] Could not setup collision groups")
    
    # Run a few simulation steps to verify
    print("\n[Main] Running a few simulation steps...")
    for i in range(10):
        env.step(env.action_space.sample() * 0)
    
    print("[Main] Test complete!")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
