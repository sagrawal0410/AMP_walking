#!/usr/bin/env python3
"""
Sim2Real deployment of AMP walking policy on Unitree G1 29-DOF.

Uses the same observation computation and action processing as sim2sim_amp.py,
but reads sensor data from the real Unitree G1 via unitree_sdk2py and sends
motor commands back.

Observation structure (per step = 117 dims, × 5 history = 585 total):
  base_ang_vel              :  3
  root_local_rot_tan_norm   :  6
  velocity_commands         :  3
  joint_pos                 : 29  (absolute, NOT relative)
  joint_vel                 : 29
  last_action               : 29
  key_body_pos_b            : 18  (6 key bodies × 3 xyz)

Usage:
  python sim2real_amp.py --policy /path/to/policy.onnx --deploy-yaml /path/to/deploy.yaml --network eth0
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
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # noqa: F401
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState
    from unitree_sdk2py.g1.low_level.g1_low_level import G1LowLevel
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    UNITREE_SDK_AVAILABLE = False

try:
    from pynput import keyboard
    from pynput.keyboard import Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Constants (shared with sim2sim_amp.py)
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
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class FSMState(Enum):
    PASSIVE = 1
    FIXSTAND = 2
    VELOCITY = 3


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers (identical to sim2sim_amp.py)
# ──────────────────────────────────────────────────────────────────────────────

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])

def quat_rotate_inverse(q, v):
    q_vec = q[1:4]
    a = v * (2.0 * q[0]**2 - 1.0)
    b = np.cross(q_vec, v) * q[0] * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# ──────────────────────────────────────────────────────────────────────────────
# Forward Kinematics for key body positions
# ──────────────────────────────────────────────────────────────────────────────
# On the real robot, we don't have MuJoCo's FK. We must compute key body
# positions from joint angles using the URDF/USD kinematic chain.
# This uses the same kinematic chain as the MuJoCo XML.

def _axis_angle_to_rotmat(axis, angle):
    """Rodrigues formula: axis (unit vector) + angle → 3x3 rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])

def _make_transform(pos, quat=None, rotmat=None):
    """Create 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, 3] = pos
    if quat is not None:
        T[:3, :3] = quat_to_rotmat(quat)
    elif rotmat is not None:
        T[:3, :3] = rotmat
    return T

def _joint_transform(axis, angle):
    """Joint rotation as 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = _axis_angle_to_rotmat(np.array(axis), angle)
    return T


def compute_key_body_positions_g1(joint_pos_sdk: np.ndarray) -> dict:
    """
    Compute key body positions in the pelvis (base) frame using forward kinematics.

    Uses the kinematic chain from the MuJoCo XML (g1_29dof.xml).
    Joint positions must be in SDK order (same as MuJoCo XML order).

    Returns dict: body_name → position (3,) in base frame
    """
    q = joint_pos_sdk

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
    # Now at torso_link frame

    # ── Left arm chain: torso → left_wrist_yaw_link ──
    T = T_waist.copy()
    T = T @ _make_transform([0.0039563, 0.10022, 0.23778], quat=[0.990264, 0.139201, 1.38722e-05, -9.86868e-05])
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

    # ── Right arm chain: torso → right_wrist_yaw_link ──
    T = T_waist.copy()
    T = T @ _make_transform([0.0039563, -0.10021, 0.23778], quat=[0.990264, -0.139201, 1.38722e-05, 9.86868e-05])
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
# Deploy config loader (same as sim2sim_amp.py)
# ──────────────────────────────────────────────────────────────────────────────

class DeployConfig:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.joint_ids_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        assert len(self.joint_ids_map) == NUM_JOINTS

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
            self.obs_term_cfgs[term_name] = {
                "history_length": int(tcfg.get("history_length", HISTORY_LEN)),
                "scale": np.array(tcfg.get("scale", [1.0]), dtype=np.float32),
            }

        kb_cfg = obs_cfg.get("key_body_pos_b", {})
        kb_params = kb_cfg.get("params", {})
        kb_asset = kb_params.get("asset_cfg", {})
        self.key_body_names = kb_asset.get("body_names", KEY_BODY_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# Observation buffer (same as sim2sim_amp.py)
# ──────────────────────────────────────────────────────────────────────────────

class ObsTermBuffer:
    def __init__(self, dim: int, history_length: int, scale: np.ndarray):
        self.dim = dim
        self.history_length = history_length
        self.scale = scale if len(scale) == dim else np.ones(dim, dtype=np.float32)
        self.buffer: deque[np.ndarray] = deque(maxlen=history_length)
        self._initialized = False

    def add(self, obs: np.ndarray):
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
# Real Robot Runner
# ──────────────────────────────────────────────────────────────────────────────

class AmpRealRunner:
    """
    Sim2Real runner for AMP policy on Unitree G1.

    Reads sensor data from Unitree SDK, computes observations matching training,
    runs ONNX policy, and sends motor commands.

    Safety features:
    - NaN/Inf detection on all actions
    - Joint position limits checked before sending
    - Gradual ramp-up from current position in FixStand
    - Emergency stop on bad orientation
    """

    def __init__(self, policy_path: str, deploy_cfg: DeployConfig):
        self.cfg = deploy_cfg

        # ── Load ONNX policy ──
        if ort is None:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        self.session = ort.InferenceSession(policy_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        print(f"[POLICY] Input: {inp.name} shape={inp.shape}")

        # ── State ──
        self.raw_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.command_vel = np.zeros(3, dtype=np.float32)
        self.fsm_state = FSMState.PASSIVE
        self.fsm_start_time = 0.0
        self.fixstand_start_pos = None
        self.current_time = 0.0

        # ── Sensor data (updated by SDK callback) ──
        self.imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.imu_gyro = np.zeros(3, dtype=np.float32)
        self.motor_pos_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.motor_vel_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)

        # ── Observation buffers ──
        self.obs_term_dims = {
            "base_ang_vel": 3,
            "root_local_rot_tan_norm": 6,
            "velocity_commands": 3,
            "joint_pos": NUM_JOINTS,
            "joint_vel": NUM_JOINTS,
            "last_action": NUM_JOINTS,
            "key_body_pos_b": len(self.cfg.key_body_names) * 3,
        }
        self.obs_buffers: dict[str, ObsTermBuffer] = {}
        for term_name in self.cfg.obs_order:
            dim = self.obs_term_dims[term_name]
            tcfg = self.cfg.obs_term_cfgs.get(term_name, {})
            hl = tcfg.get("history_length", HISTORY_LEN)
            scale = tcfg.get("scale", np.ones(dim, dtype=np.float32))
            self.obs_buffers[term_name] = ObsTermBuffer(dim, hl, scale)

    # ──────────────────────────────────────────────────────────────────────
    # Sensor reading from Unitree SDK
    # ──────────────────────────────────────────────────────────────────────

    def update_sensor_data(self, lowstate):
        """Update sensor data from Unitree SDK LowState message."""
        # IMU quaternion (Unitree SDK: w, x, y, z)
        q = lowstate.imu_state().quaternion()
        self.imu_quat_wxyz = np.array([q[0], q[1], q[2], q[3]], dtype=np.float64)

        # IMU gyroscope (angular velocity in body frame)
        g = lowstate.imu_state().gyroscope()
        self.imu_gyro = np.array([g[0], g[1], g[2]], dtype=np.float32)

        # Motor positions and velocities (SDK order)
        for i in range(NUM_JOINTS):
            self.motor_pos_sdk[i] = lowstate.motor_state()[i].q()
            self.motor_vel_sdk[i] = lowstate.motor_state()[i].dq()

    # ──────────────────────────────────────────────────────────────────────
    # Observation computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_root_local_rot_tan_norm(self) -> np.ndarray:
        root_quat = self.imu_quat_wxyz
        yaw_q = yaw_quat(root_quat)
        local_q = quat_mul(quat_conjugate(yaw_q), root_quat)
        R = quat_to_rotmat(local_q)
        tan_vec = R[:, 0]
        norm_vec = R[:, 2]
        return np.concatenate([tan_vec, norm_vec]).astype(np.float32)

    def _compute_key_body_pos_b(self) -> np.ndarray:
        """Compute key body positions in base frame using forward kinematics."""
        positions = compute_key_body_positions_g1(self.motor_pos_sdk)
        result = []
        for name in self.cfg.key_body_names:
            result.append(positions[name])
        return np.concatenate(result).astype(np.float32)

    def compute_observations(self) -> np.ndarray:
        # Convert to policy order
        joint_pos_policy = self.motor_pos_sdk[self.cfg.joint_ids_map]
        joint_vel_policy = self.motor_vel_sdk[self.cfg.joint_ids_map]

        obs_terms = {
            "base_ang_vel": self.imu_gyro.copy(),
            "root_local_rot_tan_norm": self._compute_root_local_rot_tan_norm(),
            "velocity_commands": self.command_vel.copy(),
            "joint_pos": joint_pos_policy,
            "joint_vel": joint_vel_policy,
            "last_action": self.raw_action.copy(),
            "key_body_pos_b": self._compute_key_body_pos_b(),
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

        return obs.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Policy inference
    # ──────────────────────────────────────────────────────────────────────

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        obs_input = obs.reshape(1, -1).astype(np.float32)
        result = self.session.run(None, {self.input_name: obs_input})
        return result[0][0, :NUM_JOINTS].astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Action → motor commands
    # ──────────────────────────────────────────────────────────────────────

    def compute_motor_commands(self, lowcmd):
        """
        Set motor commands on lowcmd based on current FSM state.
        Uses position control mode (kp * (target - pos) + kd * (0 - vel)).
        """
        if self.fsm_state == FSMState.PASSIVE:
            # Damping only
            for i in range(NUM_JOINTS):
                lowcmd.motor_cmd()[i].q(0.0)
                lowcmd.motor_cmd()[i].dq(0.0)
                lowcmd.motor_cmd()[i].kp(0.0)
                lowcmd.motor_cmd()[i].kd(3.0)
                lowcmd.motor_cmd()[i].tau(0.0)

        elif self.fsm_state == FSMState.FIXSTAND:
            elapsed = self.current_time - self.fsm_start_time
            ramp_time = 3.0

            if self.fixstand_start_pos is None:
                self.fixstand_start_pos = self.motor_pos_sdk.copy()

            alpha = min(elapsed / ramp_time, 1.0)
            target = self.fixstand_start_pos * (1 - alpha) + self.cfg.default_joint_pos_sdk * alpha

            for i in range(NUM_JOINTS):
                lowcmd.motor_cmd()[i].q(float(target[i]))
                lowcmd.motor_cmd()[i].dq(0.0)
                lowcmd.motor_cmd()[i].kp(float(self.cfg.stiffness_sdk[i]))
                lowcmd.motor_cmd()[i].kd(float(self.cfg.damping_sdk[i]))
                lowcmd.motor_cmd()[i].tau(0.0)

        elif self.fsm_state == FSMState.VELOCITY:
            # Policy action → target joint positions (policy order)
            target_policy = self.raw_action * self.cfg.action_scale + self.cfg.action_offset

            # Convert to SDK order
            target_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
            for pi in range(NUM_JOINTS):
                target_sdk[self.cfg.joint_ids_map[pi]] = target_policy[pi]

            for i in range(NUM_JOINTS):
                val = float(target_sdk[i])
                # Safety: check for NaN/Inf
                if not np.isfinite(val):
                    val = float(self.motor_pos_sdk[i])
                    print(f"[SAFETY] NaN/Inf in action for joint {i}, using current pos")

                lowcmd.motor_cmd()[i].q(val)
                lowcmd.motor_cmd()[i].dq(0.0)
                lowcmd.motor_cmd()[i].kp(float(self.cfg.stiffness_sdk[i]))
                lowcmd.motor_cmd()[i].kd(float(self.cfg.damping_sdk[i]))
                lowcmd.motor_cmd()[i].tau(0.0)

    # ──────────────────────────────────────────────────────────────────────
    # Safety checks
    # ──────────────────────────────────────────────────────────────────────

    def check_orientation(self) -> bool:
        """Check if robot orientation is within safe limits (60° from upright)."""
        R = quat_to_rotmat(self.imu_quat_wxyz)
        z_body_in_world = R[:, 2]  # z-axis of body in world frame
        cos_angle = z_body_in_world[2]  # dot product with world z-axis
        return cos_angle > np.cos(np.radians(60.0))

    # ──────────────────────────────────────────────────────────────────────
    # Main control loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self, network: str = "eth0"):
        """
        Main control loop for real robot deployment.

        Args:
            network: Network interface for Unitree SDK (e.g., "eth0", "lo")
        """
        if not UNITREE_SDK_AVAILABLE:
            print("[ERROR] unitree_sdk2py not installed. Install it to run on real hardware.")
            print("[INFO] For sim2sim testing, use sim2sim_amp.py instead.")
            sys.exit(1)

        # Initialize Unitree SDK
        ChannelFactoryInitialize(0, network)
        robot = G1LowLevel()
        print(f"[SDK] Initialized on network: {network}")
        print("[SDK] Waiting for robot connection...")
        robot.wait_for_connection()
        print("[SDK] Robot connected!")

        # Setup keyboard listener
        transition_request = [None]

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
                        if self.fsm_state == FSMState.VELOCITY:
                            if key.char == 'w':
                                self.command_vel[0] = np.clip(self.command_vel[0] + 0.1, -0.5, 1.0)
                            elif key.char == 's':
                                self.command_vel[0] = np.clip(self.command_vel[0] - 0.1, -0.5, 1.0)
                            elif key.char == 'a':
                                self.command_vel[1] = np.clip(self.command_vel[1] + 0.1, -0.5, 0.5)
                            elif key.char == 'd':
                                self.command_vel[1] = np.clip(self.command_vel[1] - 0.1, -0.5, 0.5)
                            elif key.char == 'q':
                                self.command_vel[2] = np.clip(self.command_vel[2] + 0.1, -1.0, 1.0)
                            elif key.char == 'e':
                                self.command_vel[2] = np.clip(self.command_vel[2] - 0.1, -1.0, 1.0)
                            elif key.char == ' ':
                                self.command_vel[:] = 0.0
                except Exception:
                    pass

            listener = keyboard.Listener(on_press=on_key_press)
            listener.start()

        print("\n" + "=" * 60)
        print("AMP Sim2Real Controller")
        print("=" * 60)
        print("State: PASSIVE")
        print("Controls: ↑=FixStand, →=Velocity, ↓=Passive")
        print("  WASD+QE = velocity commands (in Velocity mode)")
        print("=" * 60 + "\n")

        step_count = 0
        try:
            while True:
                loop_start = time.time()

                # Update sensor data
                lowstate = robot.get_low_state()
                self.update_sensor_data(lowstate)
                self.current_time = time.time()

                # Handle FSM transitions
                if transition_request[0] is not None:
                    new_state = transition_request[0]
                    transition_request[0] = None
                    old_name = self.fsm_state.name
                    self.fsm_state = new_state
                    self.fsm_start_time = self.current_time
                    if new_state == FSMState.FIXSTAND:
                        self.fixstand_start_pos = None
                    print(f"[FSM] {old_name} → {new_state.name}")

                # Safety: check orientation
                if self.fsm_state == FSMState.VELOCITY and not self.check_orientation():
                    print("[SAFETY] Bad orientation detected! Switching to PASSIVE")
                    self.fsm_state = FSMState.PASSIVE

                # Compute observations and run policy
                obs = self.compute_observations()
                if self.fsm_state == FSMState.VELOCITY:
                    self.raw_action = self.run_policy(obs)

                # Send motor commands
                lowcmd = robot.get_low_cmd()
                lowcmd.mode_machine(5)  # 29-DOF mode
                self.compute_motor_commands(lowcmd)
                robot.send_low_cmd(lowcmd)

                step_count += 1
                if step_count % 50 == 0:
                    height_proxy = quat_to_rotmat(self.imu_quat_wxyz)[:, 2][2]
                    print(f"[DIAG] t={self.current_time:.1f}  state={self.fsm_state.name}  "
                          f"cmd=[{self.command_vel[0]:.1f},{self.command_vel[1]:.1f},{self.command_vel[2]:.1f}]  "
                          f"cos_tilt={height_proxy:.3f}  "
                          f"action_max={np.max(np.abs(self.raw_action)):.3f}")

                # Maintain control rate
                elapsed = time.time() - loop_start
                sleep_time = self.cfg.step_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > self.cfg.step_dt * 1.5:
                    print(f"[WARNING] Control loop overrun: {elapsed*1000:.1f}ms > {self.cfg.step_dt*1000:.1f}ms")

        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            # Go to passive mode
            lowcmd = robot.get_low_cmd()
            lowcmd.mode_machine(5)
            for i in range(NUM_JOINTS):
                lowcmd.motor_cmd()[i].q(0.0)
                lowcmd.motor_cmd()[i].dq(0.0)
                lowcmd.motor_cmd()[i].kp(0.0)
                lowcmd.motor_cmd()[i].kd(3.0)
                lowcmd.motor_cmd()[i].tau(0.0)
            robot.send_low_cmd(lowcmd)
            print("[INFO] Robot set to passive mode. Goodbye!")

        finally:
            if PYNPUT_AVAILABLE:
                listener.stop()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sim2Real deployment of AMP walking policy on Unitree G1.",
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to policy.onnx")
    parser.add_argument("--deploy-yaml", type=str, required=True,
                        help="Path to deploy.yaml")
    parser.add_argument("--network", type=str, default="eth0",
                        help="Network interface for Unitree SDK (default: eth0)")
    args = parser.parse_args()

    for name, path in [("Policy", args.policy), ("deploy.yaml", args.deploy_yaml)]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)

    cfg = DeployConfig(args.deploy_yaml)
    runner = AmpRealRunner(policy_path=args.policy, deploy_cfg=cfg)
    runner.run(network=args.network)


if __name__ == "__main__":
    main()
