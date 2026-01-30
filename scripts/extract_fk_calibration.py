#!/usr/bin/env python3
"""
Extract and calibrate FK parameters by comparing Isaac Lab FK with C++ implementation.
This script:
1. Loads robot from USD
2. Tests various joint configurations
3. Compares Isaac Lab's body_pos_w with expected C++ FK results
4. Generates calibration data

Usage:
    python scripts/extract_fk_calibration.py
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
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from scipy.spatial.transform import Rotation as R

def test_fk_at_config(joint_positions, robot, sim):
    """Test FK at a specific joint configuration."""
    # Set joint positions
    joint_pos_tensor = torch.tensor(joint_positions, device=sim.device, dtype=torch.float32)
    robot.write_joint_position_to_sim(joint_pos_tensor.unsqueeze(0))
    sim.step()
    
    # Get body positions
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
    root_pos_w = robot.data.root_pos_w[0].cpu().numpy()
    root_quat_w = robot.data.root_quat_w[0].cpu().numpy()  # wxyz
    
    # Convert to base frame
    root_rot = R.from_quat([root_quat_w[1], root_quat_w[2], root_quat_w[3], root_quat_w[0]])  # xyzw
    root_rotm = root_rot.as_matrix()
    
    key_bodies = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    ]
    
    results = {}
    for body_name in key_bodies:
        if body_name not in robot.body_names:
            continue
        body_idx = robot.body_names.index(body_name)
        body_pos_world = body_pos_w[body_idx]
        body_pos_base = root_rotm.T @ (body_pos_world - root_pos_w)
        results[body_name] = body_pos_base
    
    return results

def main():
    """Main extraction function."""
    # Initialize simulation
    sim = SimulationContext(SimulationCfg(dt=0.01))
    
    # Create robot
    robot_cfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Get joint names in Lab order
    lab_joint_names = robot.joint_names
    print("Lab joint order:")
    for i, name in enumerate(lab_joint_names):
        print(f"  [{i:2d}] {name}")
    
    # Get SDK joint order from config
    sdk_joint_names = UNITREE_G1_29DOF_CFG.joint_sdk_names
    print("\nSDK joint order:")
    for i, name in enumerate(sdk_joint_names):
        print(f"  [{i:2d}] {name}")
    
    # Test configurations
    test_configs = [
        ("zero_pose", [0.0] * 29),
        ("standing", [
            -0.1, -0.1, 0.0, 0.0, 0.0, 0.0,  # left leg
            -0.1, -0.1, 0.0, 0.0, 0.0, 0.0,  # right leg
            0.0, 0.0, 0.0,  # waist
            0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,  # left arm
            0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,  # right arm
        ]),
    ]
    
    print("\n" + "=" * 80)
    print("FK CALIBRATION DATA FROM USD")
    print("=" * 80)
    
    all_results = {}
    for config_name, joint_pos in test_configs:
        print(f"\n--- Configuration: {config_name} ---")
        results = test_fk_at_config(joint_pos, robot, sim)
        all_results[config_name] = results
        
        for body_name, pos in results.items():
            print(f"  {body_name:30s}: [{pos[0]:10.6f}, {pos[1]:10.6f}, {pos[2]:10.6f}]")
    
    # Generate C++ verification code
    print("\n" + "=" * 80)
    print("C++ VERIFICATION CODE")
    print("=" * 80)
    print("\n// Add this to your C++ FK test:")
    print("// Expected positions at zero pose (from USD):")
    zero_results = all_results.get("zero_pose", {})
    for body_name, pos in zero_results.items():
        print(f'// assert(std::abs(pos.x() - {pos[0]:.6f}f) < 0.01f);')
        print(f'// assert(std::abs(pos.y() - {pos[1]:.6f}f) < 0.01f);')
        print(f'// assert(std::abs(pos.z() - {pos[2]:.6f}f) < 0.01f);')
    
    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
