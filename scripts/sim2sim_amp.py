#!/usr/bin/env python3
"""
AMP Policy Controller for Unitree G1 29-DOF.

Communicates via Unitree SDK2 DDS protocol — pairs with unitree_mujoco (sim2sim)
or a real Unitree G1 robot (sim2real). Exact Python equivalent of the C++ g1_ctrl.

Observation structure (per step = 117 dims, × 5 history = 585 total):
  base_ang_vel              :  3
  root_local_rot_tan_norm   :  6
  velocity_commands         :  3
  joint_pos                 : 29  (absolute, NOT relative)
  joint_vel                 : 29
  last_action               : 29
  key_body_pos_b            : 18  (6 key bodies × 3 xyz)

History stacking: use_gym_history=False → per-term concatenation
  [term1_t0, term1_t1, ..., term1_t4, term2_t0, ..., term7_t4]

Sim2Sim usage (pair with unitree_mujoco):
  Terminal 1: cd unitree_mujoco/simulate_python && python unitree_mujoco.py
  Terminal 2: python sim2sim_amp.py --network lo

Sim2Real usage (connect to real G1):
  python sim2sim_amp.py --network eth0

Controls (keyboard):
  ↑  : PASSIVE → FIXSTAND
  →  : FIXSTAND → VELOCITY (activates policy)
  ↓  : Any → PASSIVE
  W/S/A/D/Q/E : velocity commands (only in VELOCITY mode)
  Space : zero velocity
"""

import argparse
import os
import sys
import time
import threading
from collections import deque
from enum import Enum

import numpy as np
import yaml

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HGLowState
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as HGLowCmdDefault
    UNITREE_SDK_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] unitree_sdk2py import failed: {e}")
    UNITREE_SDK_AVAILABLE = False

try:
    from pynput import keyboard as pynput_keyboard
    from pynput.keyboard import Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUM_JOINTS = 29
HISTORY_LEN = 5
OBS_PER_STEP = 3 + 6 + 3 + 29 + 29 + 29 + 18  # 117
TOTAL_OBS = OBS_PER_STEP * HISTORY_LEN  # 585

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint",  "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",       "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",      "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class FSMState(Enum):
    PASSIVE = 1
    FIXSTAND = 2
    VELOCITY = 3


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers (matching Isaac Lab conventions exactly)
# ──────────────────────────────────────────────────────────────────────────────

def quat_conjugate(q):
    """Conjugate of quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q):
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def yaw_quat(q):
    """Extract yaw-only quaternion from (w, x, y, z)."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (w, x, y, z)."""
    q_vec = q[1:4]
    a = v * (2.0 * q[0] ** 2 - 1.0)
    b = np.cross(q_vec, v) * q[0] * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# ──────────────────────────────────────────────────────────────────────────────
# Forward Kinematics for key body positions
# ──────────────────────────────────────────────────────────────────────────────
# Since we communicate via DDS (no direct MuJoCo access), we compute
# key body positions from joint angles using the kinematic chain from the XML.

def _axis_angle_to_rotmat(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _make_transform(pos, quat=None):
    T = np.eye(4)
    T[:3, 3] = pos
    if quat is not None:
        T[:3, :3] = quat_to_rotmat(quat)
    return T


def _joint_transform(axis, angle):
    T = np.eye(4)
    T[:3, :3] = _axis_angle_to_rotmat(np.array(axis, dtype=np.float64), angle)
    return T


def compute_key_body_positions_g1(joint_pos_sdk):
    """
    Compute key body positions in pelvis (base) frame using forward kinematics.
    Joint positions must be in SDK order (same as MuJoCo XML order).
    Returns dict: body_name → position (3,) in base frame.
    """
    q = joint_pos_sdk.astype(np.float64)

    # ── Left leg chain: pelvis → left_ankle_roll_link ──
    T = np.eye(4)
    T = T @ _make_transform([0, 0.064452, -0.1027])
    T = T @ _joint_transform([0, 1, 0], q[0])   # left_hip_pitch
    T = T @ _make_transform([0, 0.052, -0.030465], quat=[0.996179, 0, -0.0873386, 0])
    T = T @ _joint_transform([1, 0, 0], q[1])   # left_hip_roll
    T = T @ _make_transform([0.025001, 0, -0.12412])
    T = T @ _joint_transform([0, 0, 1], q[2])   # left_hip_yaw
    T = T @ _make_transform([-0.078273, 0.0021489, -0.17734], quat=[0.996179, 0, 0.0873386, 0])
    T = T @ _joint_transform([0, 1, 0], q[3])   # left_knee
    T = T @ _make_transform([0, -9.4445e-05, -0.30001])
    T = T @ _joint_transform([0, 1, 0], q[4])   # left_ankle_pitch
    T = T @ _make_transform([0, 0, -0.017558])
    T = T @ _joint_transform([1, 0, 0], q[5])   # left_ankle_roll
    left_ankle_roll_pos = T[:3, 3].copy()

    # ── Right leg chain: pelvis → right_ankle_roll_link ──
    T = np.eye(4)
    T = T @ _make_transform([0, -0.064452, -0.1027])
    T = T @ _joint_transform([0, 1, 0], q[6])   # right_hip_pitch
    T = T @ _make_transform([0, -0.052, -0.030465], quat=[0.996179, 0, -0.0873386, 0])
    T = T @ _joint_transform([1, 0, 0], q[7])   # right_hip_roll
    T = T @ _make_transform([0.025001, 0, -0.12412])
    T = T @ _joint_transform([0, 0, 1], q[8])   # right_hip_yaw
    T = T @ _make_transform([-0.078273, -0.0021489, -0.17734], quat=[0.996179, 0, 0.0873386, 0])
    T = T @ _joint_transform([0, 1, 0], q[9])   # right_knee
    T = T @ _make_transform([0, 9.4445e-05, -0.30001])
    T = T @ _joint_transform([0, 1, 0], q[10])  # right_ankle_pitch
    T = T @ _make_transform([0, 0, -0.017558])
    T = T @ _joint_transform([1, 0, 0], q[11])  # right_ankle_roll
    right_ankle_roll_pos = T[:3, 3].copy()

    # ── Waist → torso chain ──
    T_waist = np.eye(4)
    T_waist = T_waist @ _joint_transform([0, 0, 1], q[12])  # waist_yaw
    T_waist = T_waist @ _make_transform([-0.0039635, 0, 0.035])
    T_waist = T_waist @ _joint_transform([1, 0, 0], q[13])  # waist_roll
    T_waist = T_waist @ _make_transform([0, 0, 0.019])
    T_waist = T_waist @ _joint_transform([0, 1, 0], q[14])  # waist_pitch

    # ── Left arm chain: torso → left_shoulder_roll_link, left_wrist_yaw_link ──
    T = T_waist.copy()
    T = T @ _make_transform([0.0039563, 0.10022, 0.23778],
                             quat=[0.990264, 0.139201, 1.38722e-05, -9.86868e-05])
    T = T @ _joint_transform([0, 1, 0], q[15])  # left_shoulder_pitch
    T_lsr = T.copy()
    T_lsr = T_lsr @ _make_transform([0, 0.038, -0.013831], quat=[0.990268, -0.139172, 0, 0])
    T_lsr = T_lsr @ _joint_transform([1, 0, 0], q[16])  # left_shoulder_roll
    left_shoulder_roll_pos = T_lsr[:3, 3].copy()

    T = T_lsr.copy()
    T = T @ _make_transform([0, 0.00624, -0.1032])
    T = T @ _joint_transform([0, 0, 1], q[17])  # left_shoulder_yaw
    T = T @ _make_transform([0.015783, 0, -0.080518])
    T = T @ _joint_transform([0, 1, 0], q[18])  # left_elbow
    T = T @ _make_transform([0.1, 0.00188791, -0.01])
    T = T @ _joint_transform([1, 0, 0], q[19])  # left_wrist_roll
    T = T @ _make_transform([0.038, 0, 0])
    T = T @ _joint_transform([0, 1, 0], q[20])  # left_wrist_pitch
    T = T @ _make_transform([0.046, 0, 0])
    T = T @ _joint_transform([0, 0, 1], q[21])  # left_wrist_yaw
    left_wrist_yaw_pos = T[:3, 3].copy()

    # ── Right arm chain: torso → right_shoulder_roll_link, right_wrist_yaw_link ──
    T = T_waist.copy()
    T = T @ _make_transform([0.0039563, -0.10021, 0.23778],
                             quat=[0.990264, -0.139201, 1.38722e-05, 9.86868e-05])
    T = T @ _joint_transform([0, 1, 0], q[22])  # right_shoulder_pitch
    T_rsr = T.copy()
    T_rsr = T_rsr @ _make_transform([0, -0.038, -0.013831], quat=[0.990268, 0.139172, 0, 0])
    T_rsr = T_rsr @ _joint_transform([1, 0, 0], q[23])  # right_shoulder_roll
    right_shoulder_roll_pos = T_rsr[:3, 3].copy()

    T = T_rsr.copy()
    T = T @ _make_transform([0, -0.00624, -0.1032])
    T = T @ _joint_transform([0, 0, 1], q[24])  # right_shoulder_yaw
    T = T @ _make_transform([0.015783, 0, -0.080518])
    T = T @ _joint_transform([0, 1, 0], q[25])  # right_elbow
    T = T @ _make_transform([0.1, -0.00188791, -0.01])
    T = T @ _joint_transform([1, 0, 0], q[26])  # right_wrist_roll
    T = T @ _make_transform([0.038, 0, 0])
    T = T @ _joint_transform([0, 1, 0], q[27])  # right_wrist_pitch
    T = T @ _make_transform([0.046, 0, 0])
    T = T @ _joint_transform([0, 0, 1], q[28])  # right_wrist_yaw
    right_wrist_yaw_pos = T[:3, 3].copy()

    return {
        "left_ankle_roll_link": left_ankle_roll_pos,
        "right_ankle_roll_link": right_ankle_roll_pos,
        "left_wrist_yaw_link": left_wrist_yaw_pos,
        "right_wrist_yaw_link": right_wrist_yaw_pos,
        "left_shoulder_roll_link": left_shoulder_roll_pos,
        "right_shoulder_roll_link": right_shoulder_roll_pos,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Deploy config loader
# ──────────────────────────────────────────────────────────────────────────────

class DeployConfig:
    """Parses deploy.yaml and provides all configuration needed for deployment."""

    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.joint_ids_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        assert len(self.joint_ids_map) == NUM_JOINTS, (
            f"joint_ids_map has {len(self.joint_ids_map)} entries, expected {NUM_JOINTS}"
        )

        self.sdk_to_policy = np.zeros(NUM_JOINTS, dtype=np.int32)
        for pi, si in enumerate(self.joint_ids_map):
            self.sdk_to_policy[si] = pi

        self.step_dt = float(cfg["step_dt"])
        self.stiffness_sdk = np.array(cfg["stiffness"], dtype=np.float32)
        self.damping_sdk = np.array(cfg["damping"], dtype=np.float32)
        self.default_joint_pos_sdk = np.array(cfg["default_joint_pos"], dtype=np.float32)

        act_cfg = cfg["actions"]["JointPositionAction"]
        self.action_scale = np.array(act_cfg["scale"], dtype=np.float32)
        self.action_offset = np.array(act_cfg["offset"], dtype=np.float32)

        obs_cfg = cfg.get("observations", {})
        self.obs_order = obs_cfg.get("obs_order", [
            "base_ang_vel", "root_local_rot_tan_norm", "velocity_commands",
            "joint_pos", "joint_vel", "last_action", "key_body_pos_b",
        ])
        self.use_gym_history = obs_cfg.get("use_gym_history", False)

        self.obs_term_cfgs = {}
        for term_name in self.obs_order:
            tcfg = obs_cfg.get(term_name, {})
            if tcfg is None:
                tcfg = {}
            self.obs_term_cfgs[term_name] = {
                "history_length": int(tcfg.get("history_length", HISTORY_LEN) or HISTORY_LEN),
                "scale": np.array(tcfg.get("scale", [1.0]), dtype=np.float32),
            }

        kb_cfg = obs_cfg.get("key_body_pos_b", {}) or {}
        kb_params = kb_cfg.get("params", {}) or {}
        kb_asset = kb_params.get("asset_cfg", {}) or {}
        self.key_body_names = kb_asset.get("body_names", KEY_BODY_NAMES) or KEY_BODY_NAMES

        print(f"[CONFIG] step_dt={self.step_dt}")
        print(f"[CONFIG] obs_order={self.obs_order}")
        print(f"[CONFIG] key_bodies={self.key_body_names}")
        print(f"[CONFIG] use_gym_history={self.use_gym_history}")


# ──────────────────────────────────────────────────────────────────────────────
# Observation history buffer
# ──────────────────────────────────────────────────────────────────────────────

class ObsTermBuffer:
    """Per-term FIFO history buffer matching Isaac Lab's ObservationTermCfg."""

    def __init__(self, dim: int, history_length: int, scale: np.ndarray):
        self.dim = dim
        self.history_length = history_length
        self.scale = scale if len(scale) == dim else np.ones(dim, dtype=np.float32)
        self.buffer: deque[np.ndarray] = deque(maxlen=history_length)
        self._initialized = False

    def add(self, obs: np.ndarray):
        assert obs.shape == (self.dim,), f"Expected ({self.dim},), got {obs.shape}"
        scaled = obs * self.scale
        if not self._initialized:
            for _ in range(self.history_length):
                self.buffer.append(scaled.copy())
            self._initialized = True
        else:
            self.buffer.append(scaled.copy())

    def get_flat(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=0)

    def reset(self):
        self.buffer.clear()
        self._initialized = False


# ──────────────────────────────────────────────────────────────────────────────
# AMP Policy Controller (DDS-based)
# ──────────────────────────────────────────────────────────────────────────────

class AmpController:
    """
    AMP policy controller for Unitree G1 via DDS protocol.

    Subscribes to rt/lowstate for sensor data, computes observations matching
    Isaac Lab training exactly, runs ONNX policy, and publishes rt/lowcmd.

    Works identically for sim2sim (with unitree_mujoco) and sim2real (with real G1).
    """

    def __init__(self, policy_path: str, deploy_cfg: DeployConfig):
        self.cfg = deploy_cfg

        # ── Load ONNX policy ──
        if ort is None:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        self.session = ort.InferenceSession(policy_path)
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        print(f"[POLICY] Input: {inp.name} shape={inp.shape}")
        print(f"[POLICY] Output: {out.name} shape={out.shape}")
        self.input_name = inp.name

        # ── State variables ──
        self.raw_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.command_vel = np.zeros(3, dtype=np.float32)
        self.fsm_state = FSMState.PASSIVE
        self.fsm_start_time = time.time()
        self.fixstand_start_pos = None

        # ── Sensor data (updated by DDS callback) ──
        self.imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.imu_gyro = np.zeros(3, dtype=np.float32)
        self.motor_pos_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.motor_vel_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.state_lock = threading.Lock()
        self.state_received = False

        # ── Observation term dims ──
        self.obs_term_dims = {
            "base_ang_vel": 3,
            "root_local_rot_tan_norm": 6,
            "velocity_commands": 3,
            "keyboard_velocity_commands": 3,
            "joint_pos": NUM_JOINTS,
            "joint_vel": NUM_JOINTS,
            "last_action": NUM_JOINTS,
            "key_body_pos_b": len(self.cfg.key_body_names) * 3,
        }

        # ── Observation buffers ──
        self.obs_buffers: dict[str, ObsTermBuffer] = {}
        for term_name in self.cfg.obs_order:
            dim = self.obs_term_dims[term_name]
            tcfg = self.cfg.obs_term_cfgs.get(term_name, {})
            hl = tcfg.get("history_length", HISTORY_LEN)
            scale = tcfg.get("scale", np.ones(dim, dtype=np.float32))
            self.obs_buffers[term_name] = ObsTermBuffer(dim, hl, scale)

    # ──────────────────────────────────────────────────────────────────────
    # DDS callback
    # ──────────────────────────────────────────────────────────────────────

    def _lowstate_callback(self, msg):
        """Called by DDS subscriber when new LowState arrives."""
        with self.state_lock:
            for i in range(NUM_JOINTS):
                self.motor_pos_sdk[i] = msg.motor_state[i].q
                self.motor_vel_sdk[i] = msg.motor_state[i].dq
            # IMU quaternion (w, x, y, z) — same convention as Isaac Lab
            self.imu_quat_wxyz[0] = msg.imu_state.quaternion[0]
            self.imu_quat_wxyz[1] = msg.imu_state.quaternion[1]
            self.imu_quat_wxyz[2] = msg.imu_state.quaternion[2]
            self.imu_quat_wxyz[3] = msg.imu_state.quaternion[3]
            # IMU gyroscope (angular velocity in body frame)
            self.imu_gyro[0] = msg.imu_state.gyroscope[0]
            self.imu_gyro[1] = msg.imu_state.gyroscope[1]
            self.imu_gyro[2] = msg.imu_state.gyroscope[2]
            self.state_received = True

    # ──────────────────────────────────────────────────────────────────────
    # Observation computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_root_local_rot_tan_norm(self, imu_quat):
        """Yaw-removed rotation: columns 0 and 2 of rotation matrix."""
        yaw_q = yaw_quat(imu_quat)
        local_q = quat_mul(quat_conjugate(yaw_q), imu_quat)
        R = quat_to_rotmat(local_q)
        tan_vec = R[:, 0]
        norm_vec = R[:, 2]
        return np.concatenate([tan_vec, norm_vec]).astype(np.float32)

    def _compute_key_body_pos_b(self, motor_pos):
        """Key body positions in base frame via forward kinematics."""
        positions = compute_key_body_positions_g1(motor_pos)
        result = []
        for name in self.cfg.key_body_names:
            result.append(positions[name])
        return np.concatenate(result).astype(np.float32)

    def compute_observations(self, motor_pos, motor_vel, imu_quat, imu_gyro):
        """Compute the full 585-dim observation vector."""
        # Convert to policy order: joint_ids_map[policy_idx] = sdk_idx
        joint_pos_policy = motor_pos[self.cfg.joint_ids_map]
        joint_vel_policy = motor_vel[self.cfg.joint_ids_map]

        obs_terms = {
            "base_ang_vel": imu_gyro.copy(),
            "root_local_rot_tan_norm": self._compute_root_local_rot_tan_norm(imu_quat),
            "velocity_commands": self.command_vel.copy(),
            "keyboard_velocity_commands": self.command_vel.copy(),
            "joint_pos": joint_pos_policy.astype(np.float32),
            "joint_vel": joint_vel_policy.astype(np.float32),
            "last_action": self.raw_action.copy(),
            "key_body_pos_b": self._compute_key_body_pos_b(motor_pos),
        }

        for term_name in self.cfg.obs_order:
            self.obs_buffers[term_name].add(obs_terms[term_name])

        if self.cfg.use_gym_history:
            obs_parts = []
            for t in range(HISTORY_LEN):
                for term_name in self.cfg.obs_order:
                    obs_parts.append(list(self.obs_buffers[term_name].buffer)[t])
            obs = np.concatenate(obs_parts)
        else:
            obs = np.concatenate([
                self.obs_buffers[name].get_flat() for name in self.cfg.obs_order
            ])

        assert obs.shape == (TOTAL_OBS,), f"Obs shape {obs.shape}, expected ({TOTAL_OBS},)"
        return obs.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Policy inference
    # ──────────────────────────────────────────────────────────────────────

    def run_policy(self, obs):
        """Run ONNX policy inference. Returns raw action (29 dims, policy order)."""
        obs_input = obs.reshape(1, -1).astype(np.float32)
        result = self.session.run(None, {self.input_name: obs_input})
        return result[0][0, :NUM_JOINTS].astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Safety checks
    # ──────────────────────────────────────────────────────────────────────

    def check_orientation(self, imu_quat):
        """Check if robot orientation is within safe limits (60° from upright)."""
        R = quat_to_rotmat(imu_quat)
        cos_angle = R[2, 2]  # z-axis of body dotted with world z
        return cos_angle > np.cos(np.radians(60.0))

    # ──────────────────────────────────────────────────────────────────────
    # Main control loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self, network: str = "lo", domain_id: int = 0):
        if not UNITREE_SDK_AVAILABLE:
            print("[ERROR] unitree_sdk2py not installed.")
            print("  Install: pip install unitree_sdk2py")
            sys.exit(1)

        # ── Initialize DDS ──
        ChannelFactoryInitialize(domain_id, network)
        print(f"[DDS] Initialized: domain_id={domain_id}, network={network}")

        # ── Subscribe to lowstate ──
        sub = ChannelSubscriber("rt/lowstate", HGLowState)
        sub.Init(self._lowstate_callback, 10)
        print("[DDS] Subscribed to rt/lowstate")

        # ── Publisher for lowcmd ──
        pub = ChannelPublisher("rt/lowcmd", HGLowCmd)
        pub.Init()
        print("[DDS] Publisher ready on rt/lowcmd")

        # ── Wait for first state ──
        print("[CTRL] Waiting for robot state...")
        timeout = time.time() + 30.0
        while not self.state_received:
            if time.time() > timeout:
                print("[ERROR] Timeout waiting for robot state. Is unitree_mujoco running?")
                sys.exit(1)
            time.sleep(0.01)
        print("[CTRL] Robot state received!")

        # ── Setup keyboard (hold-to-move with smooth decay) ──
        transition_request = [None]
        held_keys = set()  # track which movement keys are currently held

        # ── Velocity targets when key is held ──
        KEY_VELOCITIES = {
            'w': np.array([ 0.6,  0.0,  0.0]),   # forward
            's': np.array([-0.6,  0.0,  0.0]),   # backward
            'a': np.array([ 0.0,  0.2, 0.0]),   # strafe left
            'd': np.array([ 0.0, -0.2, 0.0]),   # strafe right
            'q': np.array([ 0.0,  0.0,  0.2]),  # turn left
            'e': np.array([ 0.0,  0.0, -0.2]),  # turn right
        }
        # Smoothing factor (matches C++ controller's 0.15 exponential smoothing).
        # At 50Hz with 0.15: reaches ~50% in 200ms, ~95% in 600ms.
        # Keep this LOW — the policy was trained with commands held constant
        # for long periods, so rapid changes destabilize it.
        SMOOTHING = 0.15

        if PYNPUT_AVAILABLE:
            def on_key_press(key):
                try:
                    if key == Key.up:
                        if self.fsm_state == FSMState.PASSIVE:
                            transition_request[0] = FSMState.FIXSTAND
                    elif key == Key.down:
                        transition_request[0] = FSMState.PASSIVE
                    elif key == Key.right:
                        if self.fsm_state == FSMState.FIXSTAND:
                            transition_request[0] = FSMState.VELOCITY
                    elif hasattr(key, 'char') and key.char:
                        c = key.char.lower()
                        if c in KEY_VELOCITIES:
                            held_keys.add(c)
                        elif c == ' ':
                            self.command_vel[:] = 0.0
                            held_keys.clear()
                            print("[CMD] Velocity zeroed")
                except Exception:
                    pass

            def on_key_release(key):
                try:
                    if hasattr(key, 'char') and key.char:
                        held_keys.discard(key.char.lower())
                except Exception:
                    pass

            listener = pynput_keyboard.Listener(
                on_press=on_key_press,
                on_release=on_key_release,
            )
            listener.start()
        else:
            print("[WARNING] pynput not installed — no keyboard control")
            print("  Install: pip install pynput")

        # ── Print controls ──
        print("\n" + "=" * 60)
        print("  AMP Policy Controller — Unitree G1 29-DOF")
        print("=" * 60)
        print(f"  State: {self.fsm_state.name}")
        print(f"  Network: {network} (domain {domain_id})")
        print()
        print("  Controls:")
        print("    ↑    : PASSIVE → FIXSTAND")
        print("    →    : FIXSTAND → VELOCITY (activates policy)")
        print("    ↓    : Any → PASSIVE")
        print("    W/S  : Forward/backward velocity")
        print("    A/D  : Left/right strafe")
        print("    Q/E  : Turn left/right")
        print("    Space: Zero velocity command")
        print("    Ctrl+C: Emergency stop & exit")
        print("=" * 60 + "\n")

        # ── Control loop ──
        step_count = 0
        try:
            while True:
                loop_start = time.time()

                # ── Read latest sensor data ──
                with self.state_lock:
                    motor_pos = self.motor_pos_sdk.copy()
                    motor_vel = self.motor_vel_sdk.copy()
                    imu_quat = self.imu_quat_wxyz.copy()
                    imu_gyro = self.imu_gyro.copy()

                # ── Handle FSM transitions ──
                if transition_request[0] is not None:
                    new_state = transition_request[0]
                    transition_request[0] = None
                    old_name = self.fsm_state.name
                    self.fsm_state = new_state
                    self.fsm_start_time = time.time()
                    if new_state == FSMState.FIXSTAND:
                        self.fixstand_start_pos = None
                    if new_state == FSMState.VELOCITY:
                        # Reset obs buffers on entering velocity
                        for buf in self.obs_buffers.values():
                            buf.reset()
                        self.raw_action[:] = 0.0
                    print(f"[FSM] {old_name} → {new_state.name}")

                # ── Smooth velocity: exponential smoothing toward target ──
                # Same approach as C++ controller: smoothly interpolate toward
                # target (key-held) or zero (no key), using a single rate.
                if self.fsm_state == FSMState.VELOCITY:
                    target = np.zeros(3)
                    if held_keys:
                        for k in held_keys:
                            if k in KEY_VELOCITIES:
                                target += KEY_VELOCITIES[k]
                    # Smooth interpolation: same rate for attack AND decay
                    self.command_vel += (target - self.command_vel) * SMOOTHING
                    # Deadzone: snap to zero when very small
                    mask = np.abs(self.command_vel) < 0.01
                    self.command_vel[mask] = 0.0

                # ── Safety: orientation check in VELOCITY ──
                if self.fsm_state == FSMState.VELOCITY:
                    if not self.check_orientation(imu_quat):
                        print("[SAFETY] Bad orientation! → PASSIVE")
                        self.fsm_state = FSMState.PASSIVE

                # ── Compute observations ──
                obs = self.compute_observations(motor_pos, motor_vel, imu_quat, imu_gyro)

                # ── Run policy (only in VELOCITY) ──
                if self.fsm_state == FSMState.VELOCITY:
                    self.raw_action = self.run_policy(obs)

                # ── Build motor command ──
                cmd = HGLowCmdDefault()
                cmd.mode_machine = 5  # 29-DOF mode

                if self.fsm_state == FSMState.PASSIVE:
                    for i in range(NUM_JOINTS):
                        cmd.motor_cmd[i].mode = 1
                        cmd.motor_cmd[i].q = 0.0
                        cmd.motor_cmd[i].kp = 0.0
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = 3.0
                        cmd.motor_cmd[i].tau = 0.0

                elif self.fsm_state == FSMState.FIXSTAND:
                    elapsed = time.time() - self.fsm_start_time
                    ramp_time = 3.0
                    if self.fixstand_start_pos is None:
                        self.fixstand_start_pos = motor_pos.copy()
                    alpha = min(elapsed / ramp_time, 1.0)
                    target = (self.fixstand_start_pos * (1 - alpha)
                              + self.cfg.default_joint_pos_sdk * alpha)

                    for i in range(NUM_JOINTS):
                        cmd.motor_cmd[i].mode = 1
                        cmd.motor_cmd[i].q = float(target[i])
                        cmd.motor_cmd[i].kp = float(self.cfg.stiffness_sdk[i])
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = float(self.cfg.damping_sdk[i])
                        cmd.motor_cmd[i].tau = 0.0

                elif self.fsm_state == FSMState.VELOCITY:
                    # Policy action → target joint positions (policy order)
                    target_policy = (self.raw_action * self.cfg.action_scale
                                     + self.cfg.action_offset)
                    # Convert to SDK order
                    target_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
                    for pi in range(NUM_JOINTS):
                        target_sdk[self.cfg.joint_ids_map[pi]] = target_policy[pi]

                    for i in range(NUM_JOINTS):
                        val = float(target_sdk[i])
                        if not np.isfinite(val):
                            val = float(motor_pos[i])
                            print(f"[SAFETY] NaN/Inf action for joint {i}")
                        cmd.motor_cmd[i].mode = 1
                        cmd.motor_cmd[i].q = val
                        cmd.motor_cmd[i].kp = float(self.cfg.stiffness_sdk[i])
                        cmd.motor_cmd[i].dq = 0.0
                        cmd.motor_cmd[i].kd = float(self.cfg.damping_sdk[i])
                        cmd.motor_cmd[i].tau = 0.0

                # ── Publish command ──
                pub.Write(cmd)

                # ── Diagnostics ──
                step_count += 1
                if step_count % 100 == 0:
                    R = quat_to_rotmat(imu_quat)
                    cos_tilt = R[2, 2]
                    print(f"[DIAG] step={step_count}  state={self.fsm_state.name}  "
                          f"cmd=[{self.command_vel[0]:.2f},{self.command_vel[1]:.2f},"
                          f"{self.command_vel[2]:.2f}]  "
                          f"cos_tilt={cos_tilt:.3f}  "
                          f"action_max={np.max(np.abs(self.raw_action)):.3f}  "
                          f"jpos=[{motor_pos.min():.2f},{motor_pos.max():.2f}]")

                # ── Maintain control rate ──
                elapsed = time.time() - loop_start
                sleep_time = self.cfg.step_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > self.cfg.step_dt * 2.0:
                    print(f"[WARNING] Loop overrun: {elapsed*1000:.1f}ms")

        except KeyboardInterrupt:
            print("\n[CTRL] Shutting down...")
            # Send passive command
            cmd = HGLowCmdDefault()
            cmd.mode_machine = 5
            for i in range(NUM_JOINTS):
                cmd.motor_cmd[i].mode = 1
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 3.0
                cmd.motor_cmd[i].tau = 0.0
            pub.Write(cmd)
            print("[CTRL] Robot set to passive. Goodbye!")

        finally:
            if PYNPUT_AVAILABLE:
                try:
                    listener.stop()
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def find_default_paths():
    """Find default policy and deploy.yaml paths relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    policy_dir = os.path.join(
        root, "deploy", "robots", "g1_29dof", "config", "policy", "velocity", "v0"
    )
    policy_onnx = os.path.join(policy_dir, "exported", "policy.onnx")

    deploy_yaml = os.path.join(policy_dir, "params", "deploy.yaml")
    if not os.path.isfile(deploy_yaml):
        deploy_yaml = os.path.join(policy_dir, "params", "deploy_1.yaml")

    return policy_onnx, deploy_yaml


def main():
    parser = argparse.ArgumentParser(
        description="AMP Policy Controller for Unitree G1 29-DOF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Sim2Sim (pair with unitree_mujoco):
    Terminal 1: cd unitree_mujoco/simulate_python && python unitree_mujoco.py
    Terminal 2: python sim2sim_amp.py --network lo

  Sim2Real (connect to real G1):
    python sim2sim_amp.py --network eth0
""",
    )
    parser.add_argument("--network", "-n", type=str, default="lo",
                        help="DDS network interface (default: lo)")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain ID (default: 0, unitree_mujoco default: 1)")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy.onnx")
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml")
    args = parser.parse_args()

    # Resolve defaults
    default_policy, default_yaml = find_default_paths()
    policy_path = args.policy or default_policy
    yaml_path = args.deploy_yaml or default_yaml

    # Validate files
    for name, path in [("Policy", policy_path), ("deploy.yaml", yaml_path)]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
        print(f"[INFO] {name}: {path}")

    # Load config and run
    cfg = DeployConfig(yaml_path)
    controller = AmpController(policy_path=policy_path, deploy_cfg=cfg)
    controller.run(network=args.network, domain_id=args.domain_id)


if __name__ == "__main__":
    main()
