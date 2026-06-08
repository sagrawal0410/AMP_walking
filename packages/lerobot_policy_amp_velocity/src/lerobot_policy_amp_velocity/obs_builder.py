"""585-dim AMP observation builder with per-term history."""

from __future__ import annotations

from collections import deque

import numpy as np

from .amp_kinematics import compute_key_body_pos_b, compute_root_local_rot_tan_norm
from .constants import HISTORY_LEN, NUM_JOINTS, TOTAL_OBS
from .deploy_config import DeployConfig


class ObsTermBuffer:
    def __init__(self, dim: int, history_length: int, scale: np.ndarray):
        self.dim = dim
        self.history_length = history_length
        self.scale = scale if len(scale) == dim else np.ones(dim, dtype=np.float32)
        self.buffer: deque[np.ndarray] = deque(maxlen=history_length)
        self._initialized = False

    def add(self, obs: np.ndarray) -> None:
        if obs.shape != (self.dim,):
            raise ValueError(f"Expected ({self.dim},), got {obs.shape}")
        scaled = obs * self.scale
        if not self._initialized:
            for _ in range(self.history_length):
                self.buffer.append(scaled.copy())
            self._initialized = True
        else:
            self.buffer.append(scaled.copy())

    def get_flat(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=0)

    def reset(self) -> None:
        self.buffer.clear()
        self._initialized = False


class AmpObsBuilder:
    """Builds the flat 585-dim AMP observation vector."""

    def __init__(self, deploy_cfg: DeployConfig):
        self.cfg = deploy_cfg
        self.raw_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.obs_buffers: dict[str, ObsTermBuffer] = {}
        for term_name in self.cfg.obs_order:
            dim = self.cfg.obs_term_dims[term_name]
            term_cfg = self.cfg.obs_term_cfgs.get(term_name, {})
            history_len = term_cfg.get("history_length", HISTORY_LEN)
            scale = term_cfg.get("scale", np.ones(dim, dtype=np.float32))
            self.obs_buffers[term_name] = ObsTermBuffer(dim, history_len, scale)

    def reset(self) -> None:
        self.raw_action[:] = 0.0
        for buf in self.obs_buffers.values():
            buf.reset()

    def set_last_action(self, raw_action: np.ndarray) -> None:
        self.raw_action = np.asarray(raw_action, dtype=np.float32).copy()

    def compute(
        self,
        motor_pos_sdk: np.ndarray,
        motor_vel_sdk: np.ndarray,
        imu_quat_wxyz: np.ndarray,
        imu_gyro: np.ndarray,
        command_vel: np.ndarray,
    ) -> np.ndarray:
        joint_pos_policy = motor_pos_sdk[self.cfg.joint_ids_map]
        joint_vel_policy = motor_vel_sdk[self.cfg.joint_ids_map]

        obs_terms = {
            "base_ang_vel": np.asarray(imu_gyro, dtype=np.float32).copy(),
            "root_local_rot_tan_norm": compute_root_local_rot_tan_norm(imu_quat_wxyz),
            "velocity_commands": np.asarray(command_vel, dtype=np.float32).copy(),
            "keyboard_velocity_commands": np.asarray(command_vel, dtype=np.float32).copy(),
            "joint_pos": joint_pos_policy.astype(np.float32),
            "joint_vel": joint_vel_policy.astype(np.float32),
            "last_action": self.raw_action.copy(),
            "key_body_pos_b": compute_key_body_pos_b(motor_pos_sdk, self.cfg.key_body_names),
        }

        for term_name in self.cfg.obs_order:
            self.obs_buffers[term_name].add(obs_terms[term_name])

        if self.cfg.use_gym_history:
            parts = []
            history_len = HISTORY_LEN
            for t in range(history_len):
                for term_name in self.cfg.obs_order:
                    parts.append(list(self.obs_buffers[term_name].buffer)[t])
            obs = np.concatenate(parts)
        else:
            obs = np.concatenate([self.obs_buffers[name].get_flat() for name in self.cfg.obs_order])

        if obs.shape != (TOTAL_OBS,):
            raise ValueError(f"Obs shape {obs.shape}, expected ({TOTAL_OBS},)")
        return obs.astype(np.float32)

    def policy_to_sdk_targets(self, raw_action: np.ndarray) -> np.ndarray:
        """Convert raw policy output to SDK-order joint position targets."""
        target_policy = raw_action * self.cfg.action_scale + self.cfg.action_offset
        target_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        for policy_idx in range(NUM_JOINTS):
            target_sdk[self.cfg.joint_ids_map[policy_idx]] = target_policy[policy_idx]
        return target_sdk
