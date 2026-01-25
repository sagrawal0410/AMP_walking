#!/usr/bin/env python3
"""Script to regenerate deploy.yaml for an existing policy."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Regenerate deploy.yaml for an existing policy.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--log_dir", type=str, required=True, help="Path to the log directory containing the policy.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab.envs import (
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401
from legged_lab.utils import export_deploy_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # handle multi-agent case
    if hasattr(env.unwrapped, "num_agents") and env.unwrapped.num_agents > 1:
        env = multi_agent_to_single_agent(env)
    
    if isinstance(env.unwrapped, ManagerBasedRLEnv):
        # Reset environment once to ensure all observations are properly initialized
        # This is critical for AMP policies with key_body_pos_b and root_local_rot_tan_norm
        print("[INFO] Resetting environment to initialize observations...")
        env.reset()
        
        # Export deploy config after reset to ensure all observation terms are available
        log_dir = args_cli.log_dir
        print(f"[INFO] Exporting deploy.yaml to: {os.path.join(log_dir, 'params', 'deploy.yaml')}")
        export_deploy_cfg.export_deploy_cfg(env.unwrapped, log_dir)
        print("[INFO] deploy.yaml exported successfully!")
    else:
        print("[ERROR] Environment is not a ManagerBasedRLEnv. Cannot export deploy config.")
        sys.exit(1)
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
