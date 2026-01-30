#!/usr/bin/env python3
"""
Extract Forward Kinematics parameters from USD file using Isaac Lab.
This script loads the G1 robot from USD, computes FK at zero pose, and generates
C++ code with the correct kinematic parameters.

Usage:
    python scripts/extract_fk_from_usd.py
"""

import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from isaaclab.app import AppLauncher

# Launch app first
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

def extract_fk_parameters():
    """Extract FK parameters by loading robot and computing body positions."""
    
    # Initialize simulation
    sim = SimulationContext(SimulationCfg(dt=0.01))
    
    # Create robot
    robot_cfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Set all joints to zero
    joint_pos = torch.zeros(robot.num_joints, device=sim.device)
    robot.write_joint_position_to_sim(joint_pos)
    sim.step()
    
    # Get body positions in world frame
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()  # Shape: (num_bodies, 3)
    root_pos_w = robot.data.root_pos_w[0].cpu().numpy()  # Shape: (3,)
    root_quat_w = robot.data.root_quat_w[0].cpu().numpy()  # Shape: (4,) wxyz
    
    # Get body names
    body_names = robot.body_names
    
    # Key bodies we need
    key_bodies = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    ]
    
    # Convert root quat to rotation matrix
    from scipy.spatial.transform import Rotation as R
    root_rot = R.from_quat([root_quat_w[1], root_quat_w[2], root_quat_w[3], root_quat_w[0]])  # xyzw
    root_rotm = root_rot.as_matrix()
    
    # Transform body positions to base frame
    key_body_pos_b = {}
    for body_name in key_bodies:
        if body_name not in body_names:
            print(f"WARNING: {body_name} not found in body_names!")
            continue
        
        body_idx = body_names.index(body_name)
        body_pos_world = body_pos_w[body_idx]
        
        # Transform to base frame: p_b = R^T * (p_w - root_pos_w)
        body_pos_base = root_rotm.T @ (body_pos_world - root_pos_w)
        key_body_pos_b[body_name] = body_pos_base
    
    # Print results
    print("=" * 80)
    print("FK PARAMETERS EXTRACTED FROM USD (zero pose)")
    print("=" * 80)
    print("\nKey body positions in base (pelvis) frame:")
    for body_name, pos in key_body_pos_b.items():
        print(f"  {body_name:30s}: [{pos[0]:10.6f}, {pos[1]:10.6f}, {pos[2]:10.6f}]")
    
    # Also get joint information
    print("\n" + "=" * 80)
    print("JOINT INFORMATION")
    print("=" * 80)
    joint_names = robot.joint_names
    print(f"\nTotal joints: {len(joint_names)}")
    print("\nJoint names (in order):")
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name}")
    
    # Get joint positions at zero pose
    joint_pos_zero = robot.data.joint_pos[0].cpu().numpy()
    print(f"\nJoint positions at zero pose:")
    for i, (name, pos) in enumerate(zip(joint_names, joint_pos_zero)):
        print(f"  {name:30s}: {pos:10.6f}")
    
    # Generate C++ code snippet
    print("\n" + "=" * 80)
    print("C++ CODE SNIPPET (for verification)")
    print("=" * 80)
    print("\n// Expected positions at zero pose (from USD):")
    for body_name, pos in key_body_pos_b.items():
        print(f'// {body_name}: [{pos[0]:.6f}f, {pos[1]:.6f}f, {pos[2]:.6f}f]')
    
    return key_body_pos_b, body_names, joint_names

if __name__ == "__main__":
    try:
        key_body_pos_b, body_names, joint_names = extract_fk_parameters()
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print("\nNote: This extracts positions at zero pose.")
        print("For full FK, you need joint transforms which require USD parsing.")
        print("The current hardcoded FK implementation should match these values at zero pose.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
