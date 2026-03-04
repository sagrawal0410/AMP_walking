#!/usr/bin/env python3
"""
Sim2Sim MuJoCo runner for AMP walking policy on Unitree G1 29-DOF.

Deploys an ONNX-exported AMP policy in MuJoCo, faithfully replicating the
Isaac Lab training environment's observation computation, action processing,
history stacking, and PD control.

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

Usage:
  python sim2sim_amp.py --policy /path/to/policy.onnx --deploy-yaml /path/to/deploy.yaml
"""

import argparse
import os
import re
import sys
import time
from collections import deque
from enum import Enum

import mujoco
import mujoco.viewer
import numpy as np
import yaml

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUM_JOINTS = 29
HISTORY_LEN = 5
OBS_PER_STEP = 3 + 6 + 3 + 29 + 29 + 29 + 18  # 117
TOTAL_OBS = OBS_PER_STEP * HISTORY_LEN  # 585

# Key body names (must match training config exactly — order matters)
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

# SDK / MuJoCo joint order (from unitree.py joint_sdk_names)
SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class FSMState(Enum):
    PASSIVE = 1
    FIXSTAND = 2
    VELOCITY = 3


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers (matching Isaac Lab conventions exactly)
# ──────────────────────────────────────────────────────────────────────────────

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from (w, x, y, z)."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q (w, x, y, z)."""
    q_vec = q[1:4]
    a = v * (2.0 * q[0] ** 2 - 1.0)
    b = np.cross(q_vec, v) * q[0] * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# ──────────────────────────────────────────────────────────────────────────────
# Deploy config loader
# ──────────────────────────────────────────────────────────────────────────────

class DeployConfig:
    """Parses deploy.yaml and provides all configuration needed for deployment."""

    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Joint mapping: joint_ids_map[policy_idx] = sdk_idx
        self.joint_ids_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        assert len(self.joint_ids_map) == NUM_JOINTS, (
            f"joint_ids_map has {len(self.joint_ids_map)} entries, expected {NUM_JOINTS}"
        )

        # Inverse: sdk_to_policy[sdk_idx] = policy_idx
        self.sdk_to_policy = np.zeros(NUM_JOINTS, dtype=np.int32)
        for policy_idx, sdk_idx in enumerate(self.joint_ids_map):
            self.sdk_to_policy[sdk_idx] = policy_idx

        # Control timing
        self.step_dt = float(cfg["step_dt"])  # 0.02
        self.sim_dt = 0.005  # MuJoCo physics timestep
        self.decimation = int(round(self.step_dt / self.sim_dt))  # 4

        # PD gains (SDK order in deploy.yaml)
        self.stiffness_sdk = np.array(cfg["stiffness"], dtype=np.float32)
        self.damping_sdk = np.array(cfg["damping"], dtype=np.float32)

        # Default joint positions (SDK order in deploy.yaml)
        self.default_joint_pos_sdk = np.array(cfg["default_joint_pos"], dtype=np.float32)

        # Action config
        act_cfg = cfg["actions"]["JointPositionAction"]
        self.action_scale = np.array(act_cfg["scale"], dtype=np.float32)  # policy order
        self.action_offset = np.array(act_cfg["offset"], dtype=np.float32)  # policy order

        # Observation config
        obs_cfg = cfg.get("observations", {})
        self.obs_order = obs_cfg.get("obs_order", [
            "base_ang_vel", "root_local_rot_tan_norm", "velocity_commands",
            "joint_pos", "joint_vel", "last_action", "key_body_pos_b",
        ])
        self.use_gym_history = obs_cfg.get("use_gym_history", False)

        # Per-term history lengths and scales
        self.obs_term_cfgs = {}
        for term_name in self.obs_order:
            tcfg = obs_cfg.get(term_name, {})
            self.obs_term_cfgs[term_name] = {
                "history_length": int(tcfg.get("history_length", HISTORY_LEN)),
                "scale": np.array(tcfg.get("scale", [1.0]), dtype=np.float32),
            }

        # Key body names from config (or default)
        kb_cfg = obs_cfg.get("key_body_pos_b", {})
        kb_params = kb_cfg.get("params", {})
        kb_asset = kb_params.get("asset_cfg", {})
        self.key_body_names = kb_asset.get("body_names", KEY_BODY_NAMES)

        print(f"[CONFIG] step_dt={self.step_dt}, decimation={self.decimation}")
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
        """Add a new observation frame. On first call, warm-fill the buffer."""
        assert obs.shape == (self.dim,), f"Expected ({self.dim},), got {obs.shape}"
        scaled = obs * self.scale
        if not self._initialized:
            # Warm-fill: repeat first frame to avoid zero padding
            for _ in range(self.history_length):
                self.buffer.append(scaled.copy())
            self._initialized = True
        else:
            self.buffer.append(scaled.copy())

    def get_flat(self) -> np.ndarray:
        """Return flattened history [oldest, ..., newest], shape (dim * history_length,)."""
        return np.concatenate(list(self.buffer), axis=0)

    def reset(self):
        self.buffer.clear()
        self._initialized = False


# ──────────────────────────────────────────────────────────────────────────────
# MuJoCo AMP Runner
# ──────────────────────────────────────────────────────────────────────────────

class AmpMujocoRunner:
    """
    High-fidelity sim2sim runner for AMP policy on Unitree G1 in MuJoCo.

    Key design:
    - All observations computed to match Isaac Lab training exactly
    - Joint order handled via joint_ids_map from deploy.yaml
    - key_body_pos_b uses MuJoCo's built-in FK (body xpos), no manual FK
    - PD torque control applied per-joint in SDK/MuJoCo order
    - Per-term history stacking with warm-fill initialization
    """

    def __init__(self, policy_path: str, model_path: str, deploy_cfg: DeployConfig,
                 duration: float = 100.0, render: bool = True):
        self.cfg = deploy_cfg
        self.duration = duration
        self.do_render = render

        # ── Load MuJoCo model ──
        model_path_abs = os.path.abspath(model_path)
        model_dir = os.path.dirname(model_path_abs)
        mesh_dir = os.path.join(model_dir, "meshes")
        with open(model_path_abs, "r") as f:
            xml = f.read()
        xml = re.sub(r'meshdir="meshes"', f'meshdir="{mesh_dir}"', xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)

        # ── Resolve MuJoCo body IDs for key bodies ──
        self.key_body_ids = []
        for name in self.cfg.key_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Key body '{name}' not found in MuJoCo model")
            self.key_body_ids.append(bid)
        print(f"[MUJOCO] Key body IDs: {dict(zip(self.cfg.key_body_names, self.key_body_ids))}")

        # ── Resolve pelvis body ID (root) ──
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id < 0:
            raise ValueError("Pelvis body not found in MuJoCo model")

        # ── Resolve MuJoCo joint qpos addresses ──
        # MuJoCo joint order should match SDK order. Verify by name.
        self.joint_qpos_adr = np.zeros(NUM_JOINTS, dtype=np.int32)
        self.joint_dof_adr = np.zeros(NUM_JOINTS, dtype=np.int32)
        for sdk_idx, jname in enumerate(SDK_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
            self.joint_qpos_adr[sdk_idx] = self.model.jnt_qposadr[jid]
            self.joint_dof_adr[sdk_idx] = self.model.jnt_dofadr[jid]

        # ── Resolve actuator IDs (actuators are in SDK order in the XML) ──
        self.actuator_ids = np.zeros(NUM_JOINTS, dtype=np.int32)
        actuator_names = [n.replace("_joint", "") for n in SDK_JOINT_NAMES]
        for sdk_idx, aname in enumerate(actuator_names):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(f"Actuator '{aname}' not found in MuJoCo model")
            self.actuator_ids[sdk_idx] = aid

        # ── Load ONNX policy ──
        if ort is None:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        self.session = ort.InferenceSession(policy_path)
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        print(f"[POLICY] Input: {inp.name} shape={inp.shape}, Output: {out.name} shape={out.shape}")
        expected_dim = inp.shape[1] if len(inp.shape) == 2 else None
        if expected_dim and expected_dim != TOTAL_OBS:
            print(f"[WARNING] Policy expects {expected_dim} obs dims, computed {TOTAL_OBS}")
        self.input_name = inp.name

        # ── State variables ──
        self.raw_action = np.zeros(NUM_JOINTS, dtype=np.float32)  # policy output (policy order)
        self.command_vel = np.zeros(3, dtype=np.float32)  # vx, vy, yaw_rate
        self.fsm_state = FSMState.FIXSTAND
        self.fsm_start_time = 0.0
        self.fixstand_start_pos = None  # SDK order, captured on entry

        # ── Observation term buffers ──
        self.obs_term_dims = {
            "base_ang_vel": 3,
            "root_local_rot_tan_norm": 6,
            "velocity_commands": 3,
            "keyboard_velocity_commands": 3,  # alias used by some deploy.yaml variants
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
    # Sensor reading helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_joint_pos_sdk(self) -> np.ndarray:
        """Joint positions in SDK order from MuJoCo qpos."""
        return np.array([self.data.qpos[a] for a in self.joint_qpos_adr], dtype=np.float32)

    def _get_joint_vel_sdk(self) -> np.ndarray:
        """Joint velocities in SDK order from MuJoCo qvel."""
        return np.array([self.data.qvel[a] for a in self.joint_dof_adr], dtype=np.float32)

    def _get_root_quat_wxyz(self) -> np.ndarray:
        """Root orientation as quaternion (w, x, y, z) from MuJoCo."""
        # MuJoCo stores free joint quat as (w, x, y, z) in qpos[3:7]
        return self.data.qpos[3:7].copy().astype(np.float64)

    def _get_root_pos(self) -> np.ndarray:
        """Root position (x, y, z) from MuJoCo."""
        return self.data.qpos[0:3].copy().astype(np.float64)

    def _get_base_ang_vel(self) -> np.ndarray:
        """Base angular velocity in body frame from MuJoCo IMU gyro."""
        try:
            return self.data.sensor("imu_gyro").data.astype(np.float32).copy()
        except KeyError:
            # Fallback: rotate world angular velocity to body frame
            root_quat = self._get_root_quat_wxyz()
            world_ang_vel = self.data.qvel[3:6].astype(np.float32)
            return quat_rotate_inverse(root_quat, world_ang_vel).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Observation computation (matches Isaac Lab training exactly)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_root_local_rot_tan_norm(self) -> np.ndarray:
        """
        Compute root_local_rot_tan_norm: 6-dim observation.
        Removes yaw from root quaternion, extracts columns 0 and 2 of rotation matrix.
        Matches: legged_lab/tasks/locomotion/amp/mdp/observations.py::root_local_rot_tan_norm
        """
        root_quat = self._get_root_quat_wxyz()
        yaw_q = yaw_quat(root_quat)
        local_q = quat_mul(quat_conjugate(yaw_q), root_quat)
        R = quat_to_rotmat(local_q)
        tan_vec = R[:, 0]   # first column
        norm_vec = R[:, 2]  # third column
        return np.concatenate([tan_vec, norm_vec]).astype(np.float32)

    def _compute_key_body_pos_b(self) -> np.ndarray:
        """
        Compute key_body_pos_b: key body positions in base frame.
        Uses MuJoCo's built-in FK (body xpos) — no manual FK needed.
        Matches: legged_lab/tasks/locomotion/deepmimic/mdp/observations.py::key_body_pos_b
        """
        root_pos = self._get_root_pos()
        root_quat = self._get_root_quat_wxyz()

        positions = []
        for bid in self.key_body_ids:
            body_pos_w = self.data.xpos[bid].copy()  # world position
            rel_pos = body_pos_w - root_pos
            pos_b = quat_rotate_inverse(root_quat, rel_pos)
            positions.append(pos_b)

        return np.concatenate(positions).astype(np.float32)

    def compute_observations(self) -> np.ndarray:
        """
        Compute the full 585-dim observation vector.

        Order: [term1_history, term2_history, ..., term7_history]
        Each term's history: [oldest_frame, ..., newest_frame] (flattened)
        """
        # Get sensor data
        joint_pos_sdk = self._get_joint_pos_sdk()
        joint_vel_sdk = self._get_joint_vel_sdk()
        base_ang_vel = self._get_base_ang_vel()
        root_rot = self._compute_root_local_rot_tan_norm()
        key_body_pos = self._compute_key_body_pos_b()

        # Convert to policy order using joint_ids_map
        # joint_ids_map[policy_idx] = sdk_idx → policy_pos[policy_idx] = sdk_pos[sdk_idx]
        joint_pos_policy = joint_pos_sdk[self.cfg.joint_ids_map]
        joint_vel_policy = joint_vel_sdk[self.cfg.joint_ids_map]

        # Compute observation terms
        obs_terms = {
            "base_ang_vel": base_ang_vel,
            "root_local_rot_tan_norm": root_rot,
            "velocity_commands": self.command_vel.copy(),
            "keyboard_velocity_commands": self.command_vel.copy(),  # alias
            "joint_pos": joint_pos_policy,
            "joint_vel": joint_vel_policy,
            "last_action": self.raw_action.copy(),
            "key_body_pos_b": key_body_pos,
        }

        # Add to history buffers
        for term_name in self.cfg.obs_order:
            self.obs_buffers[term_name].add(obs_terms[term_name])

        # Concatenate all terms' histories
        if self.cfg.use_gym_history:
            # Interleaved: [term1_t0, term2_t0, ..., term7_t0, term1_t1, ...]
            obs_parts = []
            for t in range(HISTORY_LEN):
                for term_name in self.cfg.obs_order:
                    buf = self.obs_buffers[term_name]
                    obs_parts.append(list(buf.buffer)[t])
            obs = np.concatenate(obs_parts)
        else:
            # Per-term: [term1_t0..t4, term2_t0..t4, ...]
            obs = np.concatenate([
                self.obs_buffers[name].get_flat() for name in self.cfg.obs_order
            ])

        assert obs.shape == (TOTAL_OBS,), f"Obs shape {obs.shape}, expected ({TOTAL_OBS},)"
        return obs.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Policy inference
    # ──────────────────────────────────────────────────────────────────────

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        """Run ONNX policy inference. Returns raw action (29 dims, policy order)."""
        obs_input = obs.reshape(1, -1).astype(np.float32)
        result = self.session.run(None, {self.input_name: obs_input})
        action = result[0][0, :NUM_JOINTS].astype(np.float32)
        return action

    # ──────────────────────────────────────────────────────────────────────
    # Action processing → PD torques
    # ──────────────────────────────────────────────────────────────────────

    def compute_pd_torques(self) -> np.ndarray:
        """
        Compute PD torques in SDK/MuJoCo order for VELOCITY state.

        Processed action (policy order) = raw_action * scale + offset
        Then convert to SDK order for PD: τ = kp * (target - current) + kd * (0 - vel)
        """
        # Processed action = target joint positions in policy order
        target_policy = self.raw_action * self.cfg.action_scale + self.cfg.action_offset

        # Convert to SDK order
        target_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        for pi in range(NUM_JOINTS):
            target_sdk[self.cfg.joint_ids_map[pi]] = target_policy[pi]

        # Current state (SDK order)
        cur_pos = self._get_joint_pos_sdk()
        cur_vel = self._get_joint_vel_sdk()

        # PD control (SDK order, matching deploy.yaml stiffness/damping which are in SDK order)
        torques = self.cfg.stiffness_sdk * (target_sdk - cur_pos) - self.cfg.damping_sdk * cur_vel
        return torques

    def compute_fixstand_torques(self) -> np.ndarray:
        """PD torques for FixStand: interpolate from current to default position."""
        elapsed = self.data.time - self.fsm_start_time
        ramp_time = 3.0  # seconds to reach default

        if self.fixstand_start_pos is None:
            self.fixstand_start_pos = self._get_joint_pos_sdk()

        alpha = min(elapsed / ramp_time, 1.0)
        target = self.fixstand_start_pos * (1.0 - alpha) + self.cfg.default_joint_pos_sdk * alpha

        cur_pos = self._get_joint_pos_sdk()
        cur_vel = self._get_joint_vel_sdk()

        torques = self.cfg.stiffness_sdk * (target - cur_pos) - self.cfg.damping_sdk * cur_vel
        return torques

    def compute_passive_torques(self) -> np.ndarray:
        """Passive: damping only."""
        cur_vel = self._get_joint_vel_sdk()
        kd_passive = np.full(NUM_JOINTS, 3.0, dtype=np.float32)
        return -kd_passive * cur_vel

    # ──────────────────────────────────────────────────────────────────────
    # Apply torques to MuJoCo actuators
    # ──────────────────────────────────────────────────────────────────────

    def apply_torques(self, torques_sdk: np.ndarray):
        """Set MuJoCo actuator controls (torques in SDK order → actuator order)."""
        for sdk_idx in range(NUM_JOINTS):
            aid = self.actuator_ids[sdk_idx]
            self.data.ctrl[aid] = torques_sdk[sdk_idx]

    # ──────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset simulation to initial standing pose."""
        mujoco.mj_resetData(self.model, self.data)

        # Set root position and orientation
        self.data.qpos[0:3] = [0.0, 0.0, 0.793]  # standing height from XML
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (upright)

        # Set default joint positions (SDK order)
        for sdk_idx in range(NUM_JOINTS):
            self.data.qpos[self.joint_qpos_adr[sdk_idx]] = self.cfg.default_joint_pos_sdk[sdk_idx]

        # Zero velocities
        self.data.qvel[:] = 0.0

        # Forward to update derived quantities (body positions, sensor data)
        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self.raw_action[:] = 0.0
        self.command_vel[:] = 0.0
        self.fsm_state = FSMState.FIXSTAND
        self.fsm_start_time = 0.0
        self.fixstand_start_pos = None

        # Reset observation buffers
        for buf in self.obs_buffers.values():
            buf.reset()

        print("[RESET] Simulation reset to default standing pose")

    # ──────────────────────────────────────────────────────────────────────
    # Main simulation loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        """Run the sim2sim loop with FSM control and keyboard input."""
        self.reset()

        # Keyboard state
        transition_request = [None]
        vel_adjust = [None]

        def key_callback(keycode):
            import glfw
            if keycode == glfw.KEY_UP:
                if self.fsm_state == FSMState.PASSIVE:
                    transition_request[0] = FSMState.FIXSTAND
            elif keycode == glfw.KEY_DOWN:
                if self.fsm_state in (FSMState.FIXSTAND, FSMState.VELOCITY):
                    transition_request[0] = FSMState.PASSIVE
            elif keycode == glfw.KEY_RIGHT:
                if self.fsm_state == FSMState.FIXSTAND:
                    transition_request[0] = FSMState.VELOCITY
            elif keycode == glfw.KEY_KP_8 or keycode == glfw.KEY_W:
                vel_adjust[0] = (0, 0.2)
            elif keycode == glfw.KEY_KP_2 or keycode == glfw.KEY_S:
                vel_adjust[0] = (0, -0.2)
            elif keycode == glfw.KEY_KP_4 or keycode == glfw.KEY_A:
                vel_adjust[0] = (1, 0.2)
            elif keycode == glfw.KEY_KP_6 or keycode == glfw.KEY_D:
                vel_adjust[0] = (1, -0.2)
            elif keycode == glfw.KEY_KP_7 or keycode == glfw.KEY_Q:
                vel_adjust[0] = (2, 0.2)
            elif keycode == glfw.KEY_KP_9 or keycode == glfw.KEY_E:
                vel_adjust[0] = (2, -0.2)
            elif keycode == glfw.KEY_SPACE:
                self.command_vel[:] = 0.0
                print("[CMD] Velocity zeroed")

        print("\n" + "=" * 70)
        print("AMP Sim2Sim Controller")
        print("=" * 70)
        print("State: FIXSTAND (standing up...)")
        print("\nControls:")
        print("  ↑    : PASSIVE → FIXSTAND")
        print("  ↓    : Any → PASSIVE")
        print("  →    : FIXSTAND → VELOCITY (activates policy)")
        print("  W/S  : Forward/backward velocity")
        print("  A/D  : Left/right strafe")
        print("  Q/E  : Turn left/right")
        print("  Space: Zero velocity command")
        print("=" * 70 + "\n")

        step_count = 0
        with mujoco.viewer.launch_passive(self.model, self.data,
                                           key_callback=key_callback) as viewer:
            while viewer.is_running() and self.data.time < self.duration:
                step_start = time.time()

                # ── Handle FSM transitions ──
                if transition_request[0] is not None:
                    new_state = transition_request[0]
                    transition_request[0] = None
                    old_name = self.fsm_state.name
                    if self._valid_transition(self.fsm_state, new_state):
                        self.fsm_state = new_state
                        self.fsm_start_time = self.data.time
                        if new_state == FSMState.FIXSTAND:
                            self.fixstand_start_pos = None
                        print(f"[FSM] {old_name} → {new_state.name}")

                # ── Handle velocity adjustments ──
                if vel_adjust[0] is not None:
                    idx, delta = vel_adjust[0]
                    vel_adjust[0] = None
                    if self.fsm_state == FSMState.VELOCITY:
                        self.command_vel[idx] = np.clip(self.command_vel[idx] + delta, -1.0, 3.0)
                        print(f"[CMD] vel=[{self.command_vel[0]:.1f}, {self.command_vel[1]:.1f}, {self.command_vel[2]:.1f}]")

                # ── Compute observations ──
                obs = self.compute_observations()

                # ── Run policy (only in VELOCITY state) ──
                if self.fsm_state == FSMState.VELOCITY:
                    self.raw_action = self.run_policy(obs)

                # ── Physics sub-steps ──
                for _ in range(self.cfg.decimation):
                    if self.fsm_state == FSMState.PASSIVE:
                        torques = self.compute_passive_torques()
                    elif self.fsm_state == FSMState.FIXSTAND:
                        torques = self.compute_fixstand_torques()
                    elif self.fsm_state == FSMState.VELOCITY:
                        torques = self.compute_pd_torques()
                    else:
                        torques = np.zeros(NUM_JOINTS, dtype=np.float32)

                    self.apply_torques(torques)
                    mujoco.mj_step(self.model, self.data)

                viewer.sync()

                step_count += 1
                if step_count % 250 == 0:
                    self._print_diagnostics()

                # Real-time sync
                elapsed = time.time() - step_start
                sleep_time = self.cfg.step_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _valid_transition(self, old: FSMState, new: FSMState) -> bool:
        valid = {
            (FSMState.PASSIVE, FSMState.FIXSTAND),
            (FSMState.FIXSTAND, FSMState.PASSIVE),
            (FSMState.FIXSTAND, FSMState.VELOCITY),
            (FSMState.VELOCITY, FSMState.PASSIVE),
        }
        return (old, new) in valid

    def _print_diagnostics(self):
        """Print periodic diagnostic info."""
        root_pos = self._get_root_pos()
        root_quat = self._get_root_quat_wxyz()
        jp = self._get_joint_pos_sdk()
        print(f"[DIAG] t={self.data.time:.2f}s  state={self.fsm_state.name}  "
              f"pos=[{root_pos[0]:.2f},{root_pos[1]:.2f},{root_pos[2]:.2f}]  "
              f"quat=[{root_quat[0]:.3f},{root_quat[1]:.3f},{root_quat[2]:.3f},{root_quat[3]:.3f}]  "
              f"cmd=[{self.command_vel[0]:.1f},{self.command_vel[1]:.1f},{self.command_vel[2]:.1f}]  "
              f"action_max={np.max(np.abs(self.raw_action)):.3f}  "
              f"joint_pos_range=[{jp.min():.2f},{jp.max():.2f}]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def find_default_paths(robot: str = "g1_29dof"):
    """Find default policy and model paths relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)

    policy_dir = os.path.join(root, "deploy", "robots", robot, "config", "policy", "velocity", "v0")
    policy_onnx = os.path.join(policy_dir, "exported", "policy.onnx")
    deploy_yaml = os.path.join(policy_dir, "params", "deploy.yaml")
    # Try deploy_1.yaml if deploy.yaml doesn't exist
    if not os.path.isfile(deploy_yaml):
        deploy_yaml = os.path.join(policy_dir, "params", "deploy_1.yaml")

    model_xml = os.path.join(root, "deploy", "robots", robot, f"{robot}.xml")

    return policy_onnx, deploy_yaml, model_xml


def main():
    parser = argparse.ArgumentParser(
        description="Sim2Sim MuJoCo runner for AMP walking policy on Unitree G1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy.onnx")
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to MuJoCo XML model")
    parser.add_argument("--robot", type=str, default="g1_29dof",
                        help="Robot variant (default: g1_29dof)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    args = parser.parse_args()

    # Resolve defaults
    default_policy, default_yaml, default_model = find_default_paths(args.robot)
    policy_path = args.policy or default_policy
    yaml_path = args.deploy_yaml or default_yaml
    model_path = args.model or default_model

    # Validate files exist
    for name, path in [("Policy", policy_path), ("deploy.yaml", yaml_path), ("MuJoCo XML", model_path)]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
        print(f"[INFO] {name}: {path}")

    # Load config and run
    cfg = DeployConfig(yaml_path)
    runner = AmpMujocoRunner(
        policy_path=policy_path,
        model_path=model_path,
        deploy_cfg=cfg,
        duration=args.duration,
        render=not args.no_render,
    )
    runner.run()


if __name__ == "__main__":
    main()
