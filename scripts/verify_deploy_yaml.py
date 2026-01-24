#!/usr/bin/env python3
"""
Verify deploy.yaml correctness for AMP policies.

Checks:
- obs_order matches required list exactly
- All observations have history_length == 5
- Total observation dimension == 585
- key_body_pos_b has correct body_names
"""

import argparse
import yaml
import sys
from pathlib import Path

# Ground truth observation order for AMP policies
REQUIRED_OBS_ORDER = [
    "base_ang_vel",
    "root_local_rot_tan_norm",
    "keyboard_velocity_commands",
    "joint_pos",
    "joint_vel",
    "last_action",
    "key_body_pos_b",
]

# Expected dimensions per timestep (before history)
EXPECTED_DIMS_PER_STEP = {
    "base_ang_vel": 3,
    "root_local_rot_tan_norm": 6,
    "keyboard_velocity_commands": 3,
    "joint_pos": 29,
    "joint_vel": 29,
    "last_action": 29,
    "key_body_pos_b": 18,  # 6 bodies * 3 (xyz)
}

# Expected key body names (must match g1_amp_env_cfg.py)
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

EXPECTED_HISTORY_LENGTH = 5
EXPECTED_TOTAL_DIM = 585


def verify_deploy_yaml(deploy_yaml_path: str, verbose: bool = True):
    """Verify deploy.yaml correctness."""
    deploy_path = Path(deploy_yaml_path)
    if not deploy_path.exists():
        print(f"[ERROR] deploy.yaml not found: {deploy_path}")
        return False
    
    with open(deploy_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    errors = []
    warnings = []
    
    # Check obs_order exists
    if "obs_order" not in cfg:
        errors.append("Missing 'obs_order' field in deploy.yaml")
        obs_order = list(cfg.get("observations", {}).keys())
        warnings.append(f"Using observation keys as order: {obs_order}")
    else:
        obs_order = cfg["obs_order"]
    
    # Check obs_order matches required order exactly
    if obs_order != REQUIRED_OBS_ORDER:
        errors.append(
            f"obs_order mismatch!\n"
            f"  Expected: {REQUIRED_OBS_ORDER}\n"
            f"  Got:      {obs_order}"
        )
    
    # Check all required observations exist
    if "observations" not in cfg:
        errors.append("Missing 'observations' field in deploy.yaml")
        return False
    
    obs_cfg = cfg["observations"]
    missing_obs = set(REQUIRED_OBS_ORDER) - set(obs_cfg.keys())
    if missing_obs:
        errors.append(f"Missing observations: {missing_obs}")
    
    # Verify each observation
    total_dim = 0
    for obs_name in REQUIRED_OBS_ORDER:
        if obs_name not in obs_cfg:
            continue
        
        obs_term = obs_cfg[obs_name]
        
        # Check history_length
        history_length = obs_term.get("history_length", 1)
        if history_length != EXPECTED_HISTORY_LENGTH:
            errors.append(
                f"Observation '{obs_name}': history_length={history_length}, "
                f"expected {EXPECTED_HISTORY_LENGTH}"
            )
        
        # Check dimensions
        expected_dim_per_step = EXPECTED_DIMS_PER_STEP.get(obs_name)
        if expected_dim_per_step is None:
            warnings.append(f"Unknown observation '{obs_name}', cannot verify dims")
        else:
            # Try to infer dim_per_step from scale or params
            dim_per_step = None
            if "scale" in obs_term and isinstance(obs_term["scale"], list):
                dim_per_step = len(obs_term["scale"])
            elif obs_name == "key_body_pos_b":
                # For key_body_pos_b, check body_names
                params = obs_term.get("params", {})
                asset_cfg = params.get("asset_cfg", {})
                body_names = asset_cfg.get("body_names", [])
                if isinstance(body_names, list):
                    dim_per_step = len(body_names) * 3
                    # Verify body names match
                    if body_names != KEY_BODY_NAMES:
                        errors.append(
                            f"key_body_pos_b body_names mismatch!\n"
                            f"  Expected: {KEY_BODY_NAMES}\n"
                            f"  Got:      {body_names}"
                        )
            elif obs_name in ["joint_pos", "joint_vel", "last_action"]:
                # These should be 29 for G1 29DOF
                dim_per_step = 29
            
            if dim_per_step is None:
                dim_per_step = expected_dim_per_step
                warnings.append(
                    f"Could not infer dim_per_step for '{obs_name}', "
                    f"assuming {expected_dim_per_step}"
                )
            
            if dim_per_step != expected_dim_per_step:
                errors.append(
                    f"Observation '{obs_name}': dim_per_step={dim_per_step}, "
                    f"expected {expected_dim_per_step}"
                )
            
            term_total_dim = dim_per_step * history_length
            total_dim += term_total_dim
            
            if verbose:
                print(
                    f"  {obs_name:30s} -> {dim_per_step:3d} * {history_length} = "
                    f"{term_total_dim:4d} dims"
                )
    
    # Check total dimension
    if total_dim != EXPECTED_TOTAL_DIM:
        errors.append(
            f"Total observation dimension mismatch!\n"
            f"  Expected: {EXPECTED_TOTAL_DIM}\n"
            f"  Got:      {total_dim}"
        )
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print(f"Total observation dimension: {total_dim}")
        print("=" * 80)
    
    # Report results
    if warnings and verbose:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print(f"  - {e}")
        return False
    
    if verbose:
        print("\n[SUCCESS] deploy.yaml verification passed!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify deploy.yaml for AMP policies")
    parser.add_argument(
        "deploy_yaml",
        type=str,
        help="Path to deploy.yaml file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    args = parser.parse_args()
    
    success = verify_deploy_yaml(args.deploy_yaml, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
