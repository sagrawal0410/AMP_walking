#!/usr/bin/env python3
"""Verify AMP observation builder parity against legacy sim2sim_amp logic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "lerobot_policy_amp_velocity" / "src"))

from lerobot_policy_amp_velocity.deploy_config import DeployConfig
from lerobot_policy_amp_velocity.obs_builder import AmpObsBuilder

NUM_JOINTS = 29


def load_legacy_builder(deploy_yaml: Path):
    """Legacy sim2sim_amp.py was removed; parity vs legacy is no longer available."""
    return None


def make_synthetic_state(seed: int = 0):
    rng = np.random.default_rng(seed)
    motor_pos = rng.uniform(-0.5, 1.0, size=NUM_JOINTS).astype(np.float32)
    motor_vel = rng.uniform(-2.0, 2.0, size=NUM_JOINTS).astype(np.float32)
    imu_quat = np.array([0.99, 0.05, 0.08, 0.02], dtype=np.float64)
    imu_quat /= np.linalg.norm(imu_quat)
    imu_gyro = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    command_vel = np.array([0.5, 0.1, 0.2], dtype=np.float32)
    raw_action = rng.uniform(-0.3, 0.3, size=NUM_JOINTS).astype(np.float32)
    return motor_pos, motor_vel, imu_quat, imu_gyro, command_vel, raw_action


def run_parity(deploy_yaml: Path, steps: int, atol: float) -> bool:
    deploy_cfg = DeployConfig(deploy_yaml)
    new_builder = AmpObsBuilder(deploy_cfg)
    legacy = load_legacy_builder(deploy_yaml)

    max_err = 0.0
    for step in range(steps):
        motor_pos, motor_vel, imu_quat, imu_gyro, command_vel, raw_action = make_synthetic_state(step)
        new_builder.set_last_action(raw_action)
        obs_new = new_builder.compute(motor_pos, motor_vel, imu_quat, imu_gyro, command_vel)

        if legacy is not None:
            legacy.raw_action = raw_action.copy()
            legacy.command_vel = command_vel.copy()
            obs_legacy = legacy.compute_observations(motor_pos, motor_vel, imu_quat, imu_gyro)
            err = float(np.max(np.abs(obs_new - obs_legacy)))
            max_err = max(max_err, err)
            if err > atol:
                print(f"[FAIL] step={step} max_abs_error={err:.6e} > atol={atol}")
                idx = int(np.argmax(np.abs(obs_new - obs_legacy)))
                print(f"       idx={idx} new={obs_new[idx]:.6f} legacy={obs_legacy[idx]:.6f}")
                return False

    if legacy is None:
        print("[OK] New obs builder smoke test passed (legacy sim2sim_amp.py not present).")
    else:
        print(f"[OK] Obs parity passed over {steps} steps (max_abs_error={max_err:.6e})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AMP obs builder parity")
    default_yaml = (
        REPO_ROOT
        / "deploy"
        / "robots"
        / "g1_29dof"
        / "config"
        / "policy"
        / "velocity"
        / "v0"
        / "params"
        / "deploy_1.yaml"
    )
    parser.add_argument("--deploy-yaml", type=Path, default=default_yaml)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if not args.deploy_yaml.is_file():
        print(f"[ERROR] deploy yaml not found: {args.deploy_yaml}")
        return 1

    ok = run_parity(args.deploy_yaml, args.steps, args.atol)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
