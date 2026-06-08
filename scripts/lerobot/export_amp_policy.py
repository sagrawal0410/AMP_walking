#!/usr/bin/env python3
"""Export Isaac Lab AMP ONNX artifacts to a LeRobot pretrained_model directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = (
    REPO_ROOT
    / "deploy"
    / "robots"
    / "g1_29dof"
    / "config"
    / "policy"
    / "velocity"
    / "v0"
)


def find_default_paths() -> tuple[Path, Path]:
    onnx_path = DEFAULT_POLICY_DIR / "exported" / "policy.onnx"
    deploy_yaml = DEFAULT_POLICY_DIR / "params" / "deploy_1.yaml"
    if not deploy_yaml.is_file():
        deploy_yaml = DEFAULT_POLICY_DIR / "params" / "deploy.yaml"
    return onnx_path, deploy_yaml


def build_config_dict() -> dict:
    return {
        "type": "amp_velocity",
        "device": "cpu",
        "deploy_yaml": "deploy.yaml",
        "onnx_filename": "policy.onnx",
        "horizon": 1,
        "n_action_steps": 1,
        "n_obs_steps": 1,
        "history_length": 5,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "IDENTITY",
            "ACTION": "IDENTITY",
        },
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [585]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [29]},
        },
        "step_dt": 0.02,
    }


def write_processor_configs(output_dir: Path) -> bool:
    """Generate policy_preprocessor.json / policy_postprocessor.json via the real
    lerobot pipeline. Returns False (with a warning) if lerobot/the plugin can't be
    imported (e.g. when run from the Isaac Lab env instead of the deploy env)."""
    try:
        from lerobot.configs import PreTrainedConfig
        from lerobot.utils.constants import (
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        # Importing the plugin registers the "amp_velocity" config subclass so that
        # PreTrainedConfig.from_pretrained can dispatch on the JSON "type" field.
        import lerobot_policy_amp_velocity  # noqa: F401
        from lerobot_policy_amp_velocity.processor_amp_velocity import (
            make_amp_velocity_pre_post_processors,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "[WARN] Could not import lerobot + lerobot_policy_amp_velocity "
            f"({exc}).\n"
            "       Skipping processor config generation. Re-run this script in the "
            "lerobot deploy env to produce policy_preprocessor.json / "
            "policy_postprocessor.json."
        )
        return False

    # Use the BASE class so draccus consumes the "type" discriminator in config.json.
    config = PreTrainedConfig.from_pretrained(output_dir)
    # from_pretrained does not populate pretrained_path; the processor factory needs it
    # to resolve the deploy.yaml inside the model directory.
    config.pretrained_path = output_dir
    pre, post = make_amp_velocity_pre_post_processors(config)
    pre.save_pretrained(output_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    post.save_pretrained(output_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
    return True


def export_amp_policy(onnx_path: Path, deploy_yaml: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, output_dir / "policy.onnx")
    shutil.copy2(deploy_yaml, output_dir / "deploy.yaml")

    config = build_config_dict()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Remove any stale, wrongly-named processor files from earlier exports.
    for stale in ("preprocessor_config.json", "postprocessor_config.json"):
        (output_dir / stale).unlink(missing_ok=True)

    write_processor_configs(output_dir)

    readme = f"""---
library_name: lerobot
tags:
- robotics
- amp_velocity
- unitree_g1
---

# AMP Velocity Locomotion Policy

Exported from Isaac Lab AMP training for Unitree G1 29-DOF.

- Observation: 585-dim proprioceptive vector (5-step history)
- Action: 29-dim joint position targets
- Control rate: 50 Hz

Deploy with:

```bash
amp-rollout \\
  --strategy.type=base \\
  --policy.path={output_dir} \\
  --robot.type=amp_g1 \\
  --robot.is_simulation=true \\
  --fps=50
```
"""
    (output_dir / "README.md").write_text(readme)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AMP policy to LeRobot pretrained_model format")
    default_onnx, default_yaml = find_default_paths()
    parser.add_argument("--onnx", type=Path, default=default_onnx, help="Path to policy.onnx")
    parser.add_argument("--deploy-yaml", type=Path, default=default_yaml, help="Path to deploy yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_POLICY_DIR / "exported" / "pretrained_model",
        help="Output pretrained_model directory",
    )
    args = parser.parse_args()

    for label, path in [("ONNX", args.onnx), ("deploy yaml", args.deploy_yaml)]:
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}")
            return 1

    out = export_amp_policy(args.onnx, args.deploy_yaml, args.output)
    print(f"[OK] Exported LeRobot model to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
