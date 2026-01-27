#!/usr/bin/env python3
"""
Python test script to compare FK results with simulation.
This helps diagnose if the C++ FK matches Python/simulation FK.
"""

import numpy as np
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

try:
    from isaaclab.app import AppLauncher
    import gymnasium as gym
    import legged_lab.tasks  # noqa: F401
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure you're running this from the Isaac Lab environment")
    sys.exit(1)


def test_fk_vs_simulation(task_name: str, num_tests: int = 5):
    """Compare FK results from C++ (via observation) with simulation."""
    
    # Create app launcher (headless)
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    # Import after app launch
    from isaaclab.envs import ManagerBasedRLEnv
    
    # Create environment
    env_cfg_class = None
    for name in dir(gym.envs.registry.all()):
        if task_name in name:
            env_cfg_class = gym.envs.registry[name].entry_point
            break
    
    if env_cfg_class is None:
        print(f"[ERROR] Task {task_name} not found")
        return False
    
    env_cfg = gym.make(task_name).unwrapped.cfg
    env_cfg.scene.num_envs = 1  # Single env for testing
    
    env = gym.make(task_name, cfg=env_cfg)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    
    if not isinstance(base_env, ManagerBasedRLEnv):
        print("[ERROR] Environment is not ManagerBasedRLEnv")
        return False
    
    # Get observation manager
    obs_manager = base_env.observation_manager
    
    # Key body names
    key_bodies = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link"
    ]
    
    print("=" * 80)
    print("FK Diagnostic: Python Simulation vs C++ Implementation")
    print("=" * 80)
    
    # Reset environment
    obs, _ = env.reset()
    
    for test_idx in range(num_tests):
        print(f"\n{'=' * 80}")
        print(f"TEST {test_idx + 1}/{num_tests}")
        print(f"{'=' * 80}")
        
        # Get current joint positions
        robot = base_env.scene["robot"]
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        
        print(f"\nJoint positions (first 6): {joint_pos[:6]}")
        
        # Get key_body_pos_b observation (this uses C++ FK if deployed, or Python FK in sim)
        try:
            # Try to get the observation directly
            obs_dict = obs_manager.compute()
            
            if "key_body_pos_b" in obs_dict:
                key_body_obs = obs_dict["key_body_pos_b"]
                print(f"\nkey_body_pos_b observation shape: {key_body_obs.shape}")
                
                # Reshape to (num_bodies, 3)
                if len(key_body_obs.shape) == 2:
                    key_body_obs = key_body_obs[0]  # Take first env
                
                num_bodies = len(key_bodies)
                if len(key_body_obs) == num_bodies * 3:
                    positions = key_body_obs.reshape(num_bodies, 3)
                    
                    print("\nKey body positions from observation:")
                    for i, body_name in enumerate(key_bodies):
                        pos = positions[i]
                        print(f"  {body_name:30s}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
                        
                        # Check for invalid values
                        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                            print(f"    WARNING: Invalid values detected!")
                        if np.linalg.norm(pos) > 2.0:
                            print(f"    WARNING: Distance from origin ({np.linalg.norm(pos):.4f}) seems large!")
                else:
                    print(f"WARNING: Expected {num_bodies * 3} values, got {len(key_body_obs)}")
            else:
                print("WARNING: key_body_pos_b not found in observations")
                print(f"Available observations: {list(obs_dict.keys())}")
                
        except Exception as e:
            print(f"ERROR computing key_body_pos_b: {e}")
            import traceback
            traceback.print_exc()
        
        # Step environment to get new random state
        if test_idx < num_tests - 1:
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
    
    env.close()
    simulation_app.close()
    
    print("\n" + "=" * 80)
    print("Diagnostic complete!")
    print("=" * 80)
    print("\nTo compare with C++ FK:")
    print("1. Run the C++ test: cd deploy/robots/g1_29dof/build && ./test_fk")
    print("2. Compare the positions with the Python output above")
    print("3. They should match (within numerical precision)")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test FK implementation")
    parser.add_argument("--task", type=str, default="Isaac-G1-Amp-v0", 
                       help="Task name to test")
    parser.add_argument("--num_tests", type=int, default=5,
                       help="Number of test configurations")
    
    args = parser.parse_args()
    
    test_fk_vs_simulation(args.task, args.num_tests)
