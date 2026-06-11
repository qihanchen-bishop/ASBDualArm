#!/usr/bin/env python3
"""
Script to set up collision filtering in the USD asset file.

This script modifies the organ USD file to add collision filtering that:
1. Allows Robot_2 to grasp Cube_02 (rigid body)
2. Prevents Robot_2 from colliding with deformable organs (Sphere, Cube, Cylinder, etc.)

The collision filtering is set up using UsdPhysics.FilteredPairsAPI, which specifies
pairs of objects that should NOT collide.

Usage:
    python setup_usd_collision_filtering.py --input /path/to/upe2.usd --output /path/to/upe2_filtered.usd

Note: This script should be run ONCE to create a modified USD file, 
      then update the env config to use the filtered USD file.
"""

import argparse
import os
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema


def find_physics_prims(stage):
    """Find all prims with physics APIs."""
    rigid_bodies = []
    deformable_bodies = []
    collisions = []
    
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim_path)
            
        if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
            deformable_bodies.append(prim_path)
            
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(prim_path)
    
    return rigid_bodies, deformable_bodies, collisions


def setup_filtered_pairs(stage, source_prim_path, target_prim_paths, debug=True):
    """
    Set up FilteredPairsAPI to disable collisions between source and targets.
    
    FilteredPairsAPI specifies pairs of objects that should NOT collide.
    
    Args:
        stage: USD stage
        source_prim_path: Path of the source prim (e.g., Cube_02)
        target_prim_paths: List of target prim paths that should NOT collide with source
        debug: Print debug information
    """
    source_prim = stage.GetPrimAtPath(source_prim_path)
    
    if not source_prim.IsValid():
        print(f"[WARNING] Source prim not found: {source_prim_path}")
        return False
    
    # Apply FilteredPairsAPI to source prim
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
    filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()
    
    for target_path in target_prim_paths:
        target_prim = stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            filtered_pairs_rel.AddTarget(target_path)
            if debug:
                print(f"  Added filtered pair: {source_prim_path} <-> {target_path}")
        else:
            if debug:
                print(f"  [SKIP] Target prim not found: {target_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Set up collision filtering in USD asset file")
    parser.add_argument("--input", type=str, required=True, help="Input USD file path")
    parser.add_argument("--output", type=str, default=None, help="Output USD file path (default: overwrite input)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't modify")
    args = parser.parse_args()
    
    # Open the USD stage
    print(f"\nOpening USD file: {args.input}")
    stage = Usd.Stage.Open(args.input)
    
    if not stage:
        print(f"ERROR: Failed to open USD file: {args.input}")
        return 1
    
    # Analyze the stage
    print("\n" + "="*80)
    print("Analyzing USD Structure")
    print("="*80)
    
    rigid_bodies, deformable_bodies, collisions = find_physics_prims(stage)
    
    print(f"\nRigid Bodies ({len(rigid_bodies)}):")
    for path in rigid_bodies:
        print(f"  {path}")
    
    print(f"\nDeformable Bodies ({len(deformable_bodies)}):")
    for path in deformable_bodies:
        print(f"  {path}")
    
    print(f"\nCollision Prims ({len(collisions)}):")
    for path in collisions[:20]:  # Show first 20
        print(f"  {path}")
    if len(collisions) > 20:
        print(f"  ... and {len(collisions) - 20} more")
    
    if args.analyze_only:
        print("\n[Analyze only mode - not modifying file]")
        return 0
    
    # Set up collision filtering
    print("\n" + "="*80)
    print("Setting up Collision Filtering")
    print("="*80)
    
    # Find Cube_02 (the rigid body to grasp)
    cube_02_path = None
    for path in rigid_bodies:
        if "Cube_02" in path:
            cube_02_path = path
            break
    
    if not cube_02_path:
        print("[WARNING] Cube_02 not found in USD file")
        # Try to find it in all prims
        for prim in stage.Traverse():
            if "Cube_02" in str(prim.GetPath()):
                cube_02_path = str(prim.GetPath())
                print(f"Found Cube_02 at: {cube_02_path}")
                break
    
    # Find deformable objects that should NOT collide with the robot gripper
    deformable_to_filter = []
    keywords = ["Sphere", "Cube/", "Cylinder", "Kidney", "Liver", "Tissue"]
    
    for path in deformable_bodies + collisions:
        # Skip Cube_02 (we want robot to collide with it)
        if "Cube_02" in path:
            continue
        # Check if this is a deformable organ
        for keyword in keywords:
            if keyword in path:
                deformable_to_filter.append(path)
                break
    
    # Remove duplicates
    deformable_to_filter = list(set(deformable_to_filter))
    
    print(f"\nObjects to filter (will NOT collide with robot gripper):")
    for path in deformable_to_filter[:20]:
        print(f"  {path}")
    if len(deformable_to_filter) > 20:
        print(f"  ... and {len(deformable_to_filter) - 20} more")
    
    # Note: We can't set up robot-to-deformable filtering in the organ USD
    # because the robot is not part of this USD file.
    # Instead, we need to do this in the environment configuration or at runtime.
    
    print("\n" + "="*80)
    print("IMPORTANT NOTE")
    print("="*80)
    print("""
The collision filtering between Robot_2 and deformable organs cannot be set up
in the organ USD file because the robot is spawned separately.

Recommended solutions:
1. Modify the upe2.usd file to add a collision group marker (e.g., add a custom attribute)
2. In the environment config, set up collision filtering after all assets are spawned
3. Use PhysX contact callbacks to implement custom collision response

For now, this script only analyzes the USD structure.
""")
    
    # Save the modified stage
    output_path = args.output if args.output else args.input
    if output_path != args.input:
        print(f"\nSaving to: {output_path}")
        stage.Export(output_path)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
