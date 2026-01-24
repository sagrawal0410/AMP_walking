#!/usr/bin/env python3
"""
Dump AMP policy observations from Python environment.

This script creates a ground truth observation dump for comparison with C++ deploy code.
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

try:
    import torch
    from isaaclab.app import AppLauncher
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure you're running this from the Isaac Lab environment")
    sys.exit(1)


def dump_amp_obs(task_name: str, output_path: str, num_steps: int = 1):
    """Dump AMP observations from environment."""
    from isaaclab.envs import ManagerBasedRLEnv
    
    # Create app launcher (headless)
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    # Import after app launch
    import gymnasium as gym
    from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1AmpEnvCfg
    
    # Create environment
    env_cfg = G1AmpEnvCfg()
    env_cfg.scene.num_envs = 1  # Single env for debugging
    env = gym.make(task_name, cfg=env_cfg)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get observation manager
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if not isinstance(base_env, ManagerBasedRLEnv):
        print("[ERROR] Environment is not ManagerBasedRLEnv")
        return False
    
    obs_manager = base_env.observation_manager
    
    # Dump observations for specified number of steps
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Dumping observations to {output_dir}")
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Steps: {num_steps}")
    
    all_obs_vectors = []
    
    for step in range(num_steps):
        # Get policy observations
        policy_obs = obs_manager.compute()["policy"]
        
        # Convert to numpy if tensor
        if isinstance(policy_obs, torch.Tensor):
            obs_vector = policy_obs[0].detach().cpu().numpy()  # First env
        else:
            obs_vector = np.array(policy_obs[0]) if isinstance(policy_obs, (list, tuple)) else np.array(policy_obs)
        
        all_obs_vectors.append(obs_vector)
        
        # Dump individual term observations
        if step == 0:  # Only dump first step details
            print("\n[INFO] Observation breakdown (first step):")
            print("=" * 80)
            
            obs_names = obs_manager.active_terms["policy"]
            obs_cfgs = obs_manager._group_obs_term_cfgs["policy"]
            
            total_dim = 0
            for obs_name, obs_cfg in zip(obs_names, obs_cfgs):
                # Compute observation
                obs_term = obs_cfg.func(base_env, **obs_cfg.params)
                if isinstance(obs_term, torch.Tensor):
                    obs_term_np = obs_term[0].detach().cpu().numpy()
                else:
                    obs_term_np = np.array(obs_term[0]) if isinstance(obs_term, (list, tuple)) else np.array(obs_term)
                
                # Get flattened history
                history_length = obs_cfg.history_length
                dim_per_step = obs_term_np.size // history_length if history_length > 1 else obs_term_np.size
                total_term_dim = obs_term_np.size
                
                print(f"  {obs_name:30s} -> {dim_per_step:3d} * {history_length} = {total_term_dim:4d} dims")
                
                # Save individual term
                term_file = output_dir / f"obs_term_{obs_name}_step{step}.npy"
                np.save(term_file, obs_term_np)
                
                total_dim += total_term_dim
            
            print(f"\n  {'TOTAL':30s} -> {total_dim:4d} dims")
            print("=" * 80)
        
        # Step environment with zero action
        action = np.zeros(base_env.action_space.shape[0])
        obs, _, _, _, _ = env.step(action)
    
    # Save full observation vectors
    all_obs_array = np.array(all_obs_vectors)
    full_obs_file = output_dir / "obs_full.npy"
    np.save(full_obs_file, all_obs_array)
    
    print(f"\n[SUCCESS] Dumped {len(all_obs_vectors)} observation vectors")
    print(f"  Full obs shape: {all_obs_array.shape}")
    print(f"  Saved to: {full_obs_file}")
    
    # Save metadata
    metadata = {
        "task_name": task_name,
        "num_steps": num_steps,
        "obs_dim": all_obs_array.shape[1] if len(all_obs_array.shape) > 1 else all_obs_array.shape[0],
        "expected_dim": 585,
    }
    
    import json
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved to: {metadata_file}")
    
    env.close()
    simulation_app.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Dump AMP observations from Python environment")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Locomotion-AMP-G1-v0",
        help="Task name (default: Isaac-Locomotion-AMP-G1-v0)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dump_amp_obs",
        help="Output directory (default: dump_amp_obs)",
    )
    parser.add_argument(
        "-n", "--num-steps",
        type=int,
        default=1,
        help="Number of steps to dump (default: 1)",
    )
    args = parser.parse_args()
    
    success = dump_amp_obs(args.task, args.output, args.num_steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
