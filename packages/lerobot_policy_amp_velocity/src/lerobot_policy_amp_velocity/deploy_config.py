"""Load and parse Isaac Lab deploy.yaml for AMP velocity policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .constants import HISTORY_LEN, KEY_BODY_NAMES, NUM_JOINTS


class DeployConfig:
    """Runtime deploy configuration matching Isaac Lab export format."""

    def __init__(self, yaml_path: str | Path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.yaml_path = str(yaml_path)
        self.joint_ids_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        if len(self.joint_ids_map) != NUM_JOINTS:
            raise ValueError(
                f"joint_ids_map has {len(self.joint_ids_map)} entries, expected {NUM_JOINTS}"
            )

        self.sdk_to_policy = np.zeros(NUM_JOINTS, dtype=np.int32)
        for policy_idx, sdk_idx in enumerate(self.joint_ids_map):
            self.sdk_to_policy[sdk_idx] = policy_idx

        self.step_dt = float(cfg["step_dt"])
        self.stiffness_sdk = np.array(cfg["stiffness"], dtype=np.float32)
        self.damping_sdk = np.array(cfg["damping"], dtype=np.float32)
        self.default_joint_pos_sdk = np.array(cfg["default_joint_pos"], dtype=np.float32)

        act_cfg = cfg["actions"]["JointPositionAction"]
        self.action_scale = np.array(act_cfg["scale"], dtype=np.float32)
        self.action_offset = np.array(act_cfg["offset"], dtype=np.float32)

        obs_cfg = cfg.get("observations", {})
        self.obs_order = obs_cfg.get(
            "obs_order",
            [
                "base_ang_vel",
                "root_local_rot_tan_norm",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "last_action",
                "key_body_pos_b",
            ],
        )
        self.use_gym_history = obs_cfg.get("use_gym_history", False)

        self.obs_term_cfgs: dict[str, dict] = {}
        for term_name in self.obs_order:
            term_cfg = obs_cfg.get(term_name, {}) or {}
            scale = term_cfg.get("scale", [1.0])
            self.obs_term_cfgs[term_name] = {
                "history_length": int(term_cfg.get("history_length", HISTORY_LEN) or HISTORY_LEN),
                "scale": np.array(scale, dtype=np.float32),
            }

        kb_cfg = obs_cfg.get("key_body_pos_b", {}) or {}
        kb_params = kb_cfg.get("params", {}) or {}
        kb_asset = kb_params.get("asset_cfg", {}) or {}
        self.key_body_names = kb_asset.get("body_names", KEY_BODY_NAMES) or KEY_BODY_NAMES

    @property
    def obs_term_dims(self) -> dict[str, int]:
        return {
            "base_ang_vel": 3,
            "root_local_rot_tan_norm": 6,
            "velocity_commands": 3,
            "keyboard_velocity_commands": 3,
            "joint_pos": NUM_JOINTS,
            "joint_vel": NUM_JOINTS,
            "last_action": NUM_JOINTS,
            "key_body_pos_b": len(self.key_body_names) * 3,
        }
