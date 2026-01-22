# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os
import sys
from enum import Enum

import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
from pynput.keyboard import Key
import time
import yaml

# Try to import onnxruntime for ONNX model support
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class FSMState(Enum):
    """Finite State Machine states matching config.yaml"""
    PASSIVE = 1
    FIXSTAND = 2
    VELOCITY = 3

class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Matches the training configuration from velocity_env_cfg.py exactly.
    Observation structure:
    - base_lin_vel: 3 dims
    - base_ang_vel: 3 dims  
    - projected_gravity: 3 dims
    - velocity_commands: 3 dims
    - joint_pos_rel: 29 dims
    - joint_vel_rel: 29 dims
    - last_action: 29 dims
    Total: 99 dims per step
    
    With history_length=5: 99 * 5 = 495 dims total
    If policy expects different size, it will be auto-detected and padded.
    """

    class sim:
        sim_duration = 100.0
        num_action = 29  # G1 29 DOF
        # Observation structure from training config (velocity_env_cfg.py)
        # Each term has its own history, concatenated in order
        obs_term_dims = [3, 3, 3, 3, 29, 29, 29]  # base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action
        obs_per_step = sum([3, 3, 3, 3, 29, 29, 29])  # 99 dims per step
        actor_obs_history_length = 5  # Standard history length for velocity policies
        # Total expected: 99 * 5 = 495 dims (will be auto-adjusted if policy expects different)
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25  # Default, will be overridden by deploy.yaml if available


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path, deploy_yaml_path=None):
        self.cfg = cfg
        network_path = policy_path
        
        # Load deploy.yaml configuration
        if deploy_yaml_path is None:
            # Try to find deploy.yaml relative to policy path
            policy_dir = os.path.dirname(os.path.abspath(policy_path))
            # Look for deploy.yaml in common locations
            possible_paths = [
                os.path.join(policy_dir, "params", "deploy.yaml"),
                os.path.join(policy_dir, "deploy.yaml"),
                os.path.join(os.path.dirname(policy_dir), "params", "deploy.yaml"),
            ]
            deploy_yaml_path = None
            for path in possible_paths:
                if os.path.isfile(path):
                    deploy_yaml_path = path
                    break
        
        if deploy_yaml_path is None or not os.path.isfile(deploy_yaml_path):
            raise FileNotFoundError(
                f"deploy.yaml not found. Please specify --deploy-yaml or ensure it exists at:\n"
                f"  - {os.path.join(os.path.dirname(policy_path), 'params', 'deploy.yaml')}\n"
                f"  - {os.path.join(os.path.dirname(policy_path), 'deploy.yaml')}"
            )
        
        self.deploy_config = self._load_deploy_yaml(deploy_yaml_path)
        print(f"[INFO] Loaded deploy.yaml from: {deploy_yaml_path}")
        
        # Load MuJoCo model with proper mesh path resolution
        # MuJoCo resolves meshdir relative to current working directory, not XML file location
        # So we need to modify the XML to use absolute paths for meshes
        model_path_abs = os.path.abspath(model_path)
        model_dir = os.path.dirname(model_path_abs)
        mesh_dir = os.path.join(model_dir, "meshes")
        
        # Read XML file and replace relative meshdir with absolute path
        with open(model_path_abs, 'r') as f:
            xml_content = f.read()
        
        # Replace relative meshdir with absolute path
        # Pattern: meshdir="meshes" -> meshdir="/absolute/path/to/meshes"
        import re
        xml_content = re.sub(
            r'meshdir="meshes"',
            f'meshdir="{mesh_dir}"',
            xml_content
        )
        
        # Load model from modified XML string
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.model.opt.timestep = self.cfg.sim.dt

        # Load policy - support both TorchScript (.pt) and ONNX (.onnx) formats
        if network_path.endswith('.onnx'):
            if not ONNXRUNTIME_AVAILABLE:
                raise ImportError("onnxruntime is required for ONNX models. Install with: pip install onnxruntime")
            self.policy_session = ort.InferenceSession(network_path)
            self.policy_type = 'onnx'
            # Get expected input size from ONNX model
            input_shape = self.policy_session.get_inputs()[0].shape
            if len(input_shape) == 2:  # [batch, features]
                self.policy_obs_size = input_shape[1]
            else:
                self.policy_obs_size = None  # Will be auto-detected
            print(f"[INFO] Loaded ONNX policy: {network_path}")
            if self.policy_obs_size:
                print(f"[INFO] Policy expects observation size: {self.policy_obs_size}")
        else:
            self.policy = torch.jit.load(network_path)
            self.policy_type = 'torchscript'
            # Try to infer input size from TorchScript model (may not always work)
            try:
                # Create a dummy input to check the expected size
                dummy_input = torch.zeros(1, self.cfg.sim.obs_per_step * self.cfg.sim.actor_obs_history_length)
                with torch.no_grad():
                    _ = self.policy(dummy_input)
                self.policy_obs_size = None  # Will be auto-detected on first real inference
            except:
                self.policy_obs_size = None  # Will be auto-detected
            print(f"[INFO] Loaded TorchScript policy: {network_path}")
        
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()
        self.reset()
    
    def _load_deploy_yaml(self, deploy_yaml_path: str) -> dict:
        """
        Load and parse deploy.yaml configuration file.
        
        Args:
            deploy_yaml_path: Path to deploy.yaml file
            
        Returns:
            dict: Parsed YAML configuration
        """
        with open(deploy_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        # First, define joint mappings (needed before loading deploy.yaml values)
        # G1 joint mapping: MuJoCo order (GMR order) to Isaac Lab order
        # Mapping from MuJoCo indices to Isaac Lab indices
        # MuJoCo order (same as SDK order from unitree.py): [left_hip_pitch(0), left_hip_roll(1), left_hip_yaw(2), left_knee(3), left_ankle_pitch(4), left_ankle_roll(5),
        #          right_hip_pitch(6), right_hip_roll(7), right_hip_yaw(8), right_knee(9), right_ankle_pitch(10), right_ankle_roll(11),
        #          waist_yaw(12), waist_roll(13), waist_pitch(14),
        #          left_shoulder_pitch(15), left_shoulder_roll(16), left_shoulder_yaw(17), left_elbow(18), left_wrist_roll(19), left_wrist_pitch(20), left_wrist_yaw(21),
        #          right_shoulder_pitch(22), right_shoulder_roll(23), right_shoulder_yaw(24), right_elbow(25), right_wrist_roll(26), right_wrist_pitch(27), right_wrist_yaw(28)]
        # Isaac Lab order (from retarget config): [left_hip_pitch(0), right_hip_pitch(1), waist_yaw(2), left_hip_roll(3), right_hip_roll(4), waist_roll(5),
        #             left_hip_yaw(6), right_hip_yaw(7), waist_pitch(8), left_knee(9), right_knee(10),
        #             left_shoulder_pitch(11), right_shoulder_pitch(12), left_ankle_pitch(13), right_ankle_pitch(14),
        #             left_shoulder_roll(15), right_shoulder_roll(16), left_ankle_roll(17), right_ankle_roll(18),
        #             left_shoulder_yaw(19), right_shoulder_yaw(20), left_elbow(21), right_elbow(22),
        #             left_wrist_roll(23), right_wrist_roll(24), left_wrist_pitch(25), right_wrist_pitch(26),
        #             left_wrist_yaw(27), right_wrist_yaw(28)]
        self.mujoco_to_isaac_idx = [
            0,   # mujoco 0 (left_hip_pitch) -> isaac 0
            3,   # mujoco 1 (left_hip_roll) -> isaac 3
            6,   # mujoco 2 (left_hip_yaw) -> isaac 6
            9,   # mujoco 3 (left_knee) -> isaac 9
            13,  # mujoco 4 (left_ankle_pitch) -> isaac 13
            17,  # mujoco 5 (left_ankle_roll) -> isaac 17
            1,   # mujoco 6 (right_hip_pitch) -> isaac 1
            4,   # mujoco 7 (right_hip_roll) -> isaac 4
            7,   # mujoco 8 (right_hip_yaw) -> isaac 7
            10,  # mujoco 9 (right_knee) -> isaac 10
            14,  # mujoco 10 (right_ankle_pitch) -> isaac 14
            18,  # mujoco 11 (right_ankle_roll) -> isaac 18
            2,   # mujoco 12 (waist_yaw) -> isaac 2
            5,   # mujoco 13 (waist_roll) -> isaac 5
            8,   # mujoco 14 (waist_pitch) -> isaac 8
            11,  # mujoco 15 (left_shoulder_pitch) -> isaac 11
            15,  # mujoco 16 (left_shoulder_roll) -> isaac 15
            19,  # mujoco 17 (left_shoulder_yaw) -> isaac 19
            21,  # mujoco 18 (left_elbow) -> isaac 21
            23,  # mujoco 19 (left_wrist_roll) -> isaac 23
            25,  # mujoco 20 (left_wrist_pitch) -> isaac 25
            27,  # mujoco 21 (left_wrist_yaw) -> isaac 27
            12,  # mujoco 22 (right_shoulder_pitch) -> isaac 12
            16,  # mujoco 23 (right_shoulder_roll) -> isaac 16
            20,  # mujoco 24 (right_shoulder_yaw) -> isaac 20
            22,  # mujoco 25 (right_elbow) -> isaac 22
            24,  # mujoco 26 (right_wrist_roll) -> isaac 24
            26,  # mujoco 27 (right_wrist_pitch) -> isaac 26
            28,  # mujoco 28 (right_wrist_yaw) -> isaac 28
        ]
        
        # Create inverse mapping: Isaac Lab order to MuJoCo order
        self.isaac_to_mujoco_idx = [0] * 29
        for mujoco_idx, isaac_idx in enumerate(self.mujoco_to_isaac_idx):
            self.isaac_to_mujoco_idx[isaac_idx] = mujoco_idx
        
        # Load default joint positions from deploy.yaml
        # deploy.yaml stores default_joint_pos in Isaac Lab order
        if 'default_joint_pos' not in self.deploy_config:
            raise ValueError("deploy.yaml must contain 'default_joint_pos' field")
        
        default_joint_pos_isaac = np.array(self.deploy_config['default_joint_pos'], dtype=np.float32)
        if len(default_joint_pos_isaac) != self.cfg.sim.num_action:
            raise ValueError(
                f"default_joint_pos in deploy.yaml has {len(default_joint_pos_isaac)} elements, "
                f"but expected {self.cfg.sim.num_action} (num_action)"
            )
        
        # Convert from Isaac Lab order to SDK/MuJoCo order
        default_dof_pos_sdk = default_joint_pos_isaac[self.isaac_to_mujoco_idx]
        
        # Store default positions in SDK/MuJoCo order (needed for reset)
        self.default_dof_pos_sdk = default_dof_pos_sdk.copy()
        
        # Convert default positions from SDK order (same as MuJoCo order) to Isaac Lab order
        # This is needed because observations use Isaac Lab order
        self.default_dof_pos = default_dof_pos_sdk[self.mujoco_to_isaac_idx]
        
        # Load PD control gains from deploy.yaml (in Isaac Lab order)
        if 'stiffness' not in self.deploy_config:
            raise ValueError("deploy.yaml must contain 'stiffness' field")
        if 'damping' not in self.deploy_config:
            raise ValueError("deploy.yaml must contain 'damping' field")
        
        stiffness_isaac = np.array(self.deploy_config['stiffness'], dtype=np.float32)
        damping_isaac = np.array(self.deploy_config['damping'], dtype=np.float32)
        
        if len(stiffness_isaac) != self.cfg.sim.num_action:
            raise ValueError(
                f"stiffness in deploy.yaml has {len(stiffness_isaac)} elements, "
                f"but expected {self.cfg.sim.num_action} (num_action)"
            )
        if len(damping_isaac) != self.cfg.sim.num_action:
            raise ValueError(
                f"damping in deploy.yaml has {len(damping_isaac)} elements, "
                f"but expected {self.cfg.sim.num_action} (num_action)"
            )
        
        # Convert to MuJoCo order for control
        self.stiffness_mujoco = stiffness_isaac[self.isaac_to_mujoco_idx]
        self.damping_mujoco = damping_isaac[self.isaac_to_mujoco_idx]
        
        # Load action scale from deploy.yaml
        if 'actions' in self.deploy_config and 'JointPositionAction' in self.deploy_config['actions']:
            joint_action_cfg = self.deploy_config['actions']['JointPositionAction']
            if 'scale' in joint_action_cfg:
                # If scale is a list, use the first value (assuming uniform scaling)
                scale = joint_action_cfg['scale']
                if isinstance(scale, list):
                    # Use first value (should be uniform)
                    self.cfg.sim.action_scale = float(scale[0])
                else:
                    self.cfg.sim.action_scale = float(scale)
                print(f"[INFO] Loaded action_scale from deploy.yaml: {self.cfg.sim.action_scale}")
        
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        
        # Episode tracking
        self.episode_length_buf = 0
        
        # FSM state management - Start in FIXSTAND mode (active stand)
        self.fsm_state = FSMState.FIXSTAND
        self.fsm_state_start_time = 0.0
        
        # Passive state: damping only (mode=1 means damping only)
        # kd values: 3.0 for all joints (hardcoded, matching typical config.yaml values)
        self.passive_kd_isaac = np.array([3.0] * self.cfg.sim.num_action, dtype=np.float32)
        self.passive_kd_mujoco = self.passive_kd_isaac[self.isaac_to_mujoco_idx]
        
        # FixStand state: Use stiffness and damping from deploy.yaml (same as Velocity state)
        # FixStand kp = stiffness from deploy.yaml (already loaded)
        # FixStand kd = damping from deploy.yaml (already loaded)
        self.fixstand_kp_mujoco = self.stiffness_mujoco.copy()
        self.fixstand_kd_mujoco = self.damping_mujoco.copy()
        
        # FixStand trajectory: Simple trajectory that goes to default joint positions
        # ts: [0, 3] seconds (start immediately, reach target in 3 seconds)
        self.fixstand_ts = [0.0, 3.0]
        
        # qs: [[], default_joint_positions]
        # Empty first entry means use current position as starting point
        # Second entry is target position (default joint positions in Isaac Lab order)
        fixstand_target_isaac = self.default_dof_pos.copy()  # Already in Isaac Lab order
        # Convert to SDK/MuJoCo order for trajectory
        fixstand_target_sdk = fixstand_target_isaac[self.isaac_to_mujoco_idx]
        
        # Store trajectory points in SDK/MuJoCo order
        self.fixstand_qs_sdk = [
            np.array([]),  # Empty = use current position on entry
            fixstand_target_sdk.copy()  # Target = default joint positions
        ]
        
        self.fixstand_start_qs_sdk = None  # Will be set when entering FixStand state
        
        # Keyboard state tracking for transitions
        self.pressed_keys = set()
        self.key_transition_requested = None
        
        # Observation history buffers for each term (matching training config structure)
        # Each term maintains its own history buffer, matching Isaac Lab's observation manager
        # Order matches velocity_env_cfg.py: base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action
        self.obs_buffers = {
            'base_lin_vel': np.zeros((self.cfg.sim.actor_obs_history_length, 3), dtype=np.float32),
            'base_ang_vel': np.zeros((self.cfg.sim.actor_obs_history_length, 3), dtype=np.float32),
            'projected_gravity': np.zeros((self.cfg.sim.actor_obs_history_length, 3), dtype=np.float32),
            'velocity_commands': np.zeros((self.cfg.sim.actor_obs_history_length, 3), dtype=np.float32),
            'joint_pos_rel': np.zeros((self.cfg.sim.actor_obs_history_length, 29), dtype=np.float32),
            'joint_vel_rel': np.zeros((self.cfg.sim.actor_obs_history_length, 29), dtype=np.float32),
            'last_action': np.zeros((self.cfg.sim.actor_obs_history_length, 29), dtype=np.float32),
        }
        
        # Expected observation size from training config: 99 * 5 = 495 dims
        # Will be auto-detected from policy if different
        self.expected_obs_size = self.cfg.sim.obs_per_step * self.cfg.sim.actor_obs_history_length
        self.obs_history = np.zeros(self.expected_obs_size, dtype=np.float32)
        
        # Policy expected observation size is set in __init__ before init_variables is called
        # If it wasn't set (e.g., for TorchScript), it will be None and auto-detected on first inference

    def reset(self) -> None:
        """
        Reset the simulation to initial state.
        Sets joint positions, velocities, and initializes observation buffers.
        """
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions in MuJoCo (in MuJoCo order)
        # self.default_dof_pos_sdk is already in MuJoCo order (SDK order = MuJoCo order)
        # Set joint positions in MuJoCo (qpos for joints starts after root body pose)
        # Root body has 7 DOF (3 pos + 4 quat), then joints follow
        num_root_dof = 7  # 3 position + 4 quaternion
        self.data.qpos[num_root_dof:num_root_dof + self.cfg.sim.num_action] = self.default_dof_pos_sdk
        
        # Set root body position - start on ground, standing height
        # G1 robot base height ~0.95m when standing on ground
        self.data.qpos[2] = 0.95  # z position (height) - standing on ground
        
        # Set root body orientation to upright (quaternion: w, x, y, z)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (upright)
        
        # Set all velocities to zero
        self.data.qvel[:] = 0.0
        
        # Forward kinematics to update sensor data
        mujoco.mj_forward(self.model, self.data)
        
        # Reset action to zeros
        self.action[:] = 0.0
        
        # Initialize observation buffers with current state (not zeros)
        # Fill all history slots with the initial observation to avoid zero-padding
        # Get initial observation values
        self.dof_pos = self.data.sensordata[0:self.cfg.sim.num_action]
        self.dof_vel = self.data.sensordata[self.cfg.sim.num_action:2*self.cfg.sim.num_action]
        
        try:
            quat = self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
        except KeyError:
            quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)
        
        world_lin_vel = self.data.qvel[0:3].astype(np.float32)
        base_lin_vel = self.quat_rotate_inverse(quat, world_lin_vel).astype(np.float32)
        
        try:
            base_ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32)
        except KeyError:
            base_ang_vel = self.data.sensor("imu_gyro").data.astype(np.float32)
        
        projected_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1])).astype(np.float32)
        velocity_commands = self.command_vel.astype(np.float32)
        dof_pos_isaac = self.dof_pos[self.mujoco_to_isaac_idx] - self.default_dof_pos
        dof_pos_isaac = dof_pos_isaac.astype(np.float32)
        dof_vel_isaac = self.dof_vel[self.mujoco_to_isaac_idx].astype(np.float32)
        last_action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions).astype(np.float32)
        
        # Fill all history slots with initial values
        for i in range(self.cfg.sim.actor_obs_history_length):
            self.obs_buffers['base_lin_vel'][i] = base_lin_vel
            self.obs_buffers['base_ang_vel'][i] = base_ang_vel
            self.obs_buffers['projected_gravity'][i] = projected_gravity
            self.obs_buffers['velocity_commands'][i] = velocity_commands
            self.obs_buffers['joint_pos_rel'][i] = dof_pos_isaac
            self.obs_buffers['joint_vel_rel'][i] = dof_vel_isaac
            self.obs_buffers['last_action'][i] = last_action
        
        # Reset command velocity
        self.command_vel = np.array([0.0, 0.0, 0.0])
        
        # Reset episode tracking
        self.episode_length_buf = 0
        
        print("[INFO] Simulation reset to initial state")
        # Reset FSM state to FIXSTAND (active stand mode)
        self.fsm_state = FSMState.FIXSTAND
        self.fsm_state_start_time = 0.0
        self.fixstand_start_qs_sdk = None  # Will capture current position when FixStand starts

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.
        
        Matches training config (velocity_env_cfg.py) exactly:
        - base_lin_vel: 3 dims (base linear velocity in base frame)
        - base_ang_vel: 3 dims
        - projected_gravity: 3 dims  
        - velocity_commands: 3 dims (lin_vel_x, lin_vel_y, ang_vel_z)
        - joint_pos_rel: 29 dims (relative to default positions, in Isaac Lab order)
        - joint_vel_rel: 29 dims (in Isaac Lab order)
        - last_action: 29 dims (in Isaac Lab order)
        
        Each term maintains its own history buffer (history_length=5).
        When concatenating, each term's full history is flattened and concatenated in order.
        Total: (3+3+3+3+29+29+29) * 5 = 99 * 5 = 495 dims
        
        If policy expects different size, will be auto-padded/truncated.

        Returns:
            np.ndarray: Observation history matching training config structure.
        """
        # Get joint positions and velocities from MuJoCo sensors (in MuJoCo order)
        self.dof_pos = self.data.sensordata[0:self.cfg.sim.num_action]
        self.dof_vel = self.data.sensordata[self.cfg.sim.num_action:2*self.cfg.sim.num_action]

        # Get base linear velocity (3 dims) - in base frame
        # Need to compute from world velocity and orientation
        try:
            quat = self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
        except KeyError:
            quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)
        
        # Get world linear velocity (from qvel of root body)
        world_lin_vel = self.data.qvel[0:3].astype(np.float32)
        # Rotate to base frame
        base_lin_vel = self.quat_rotate_inverse(quat, world_lin_vel).astype(np.float32)
        
        # Get base angular velocity (3 dims)
        try:
            base_ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32)
        except KeyError:
            base_ang_vel = self.data.sensor("imu_gyro").data.astype(np.float32)
        
        # Get projected gravity (3 dims) - gravity vector in base frame
        projected_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1])).astype(np.float32)
        
        # Velocity commands (3 dims: lin_vel_x, lin_vel_y, ang_vel_z)
        velocity_commands = self.command_vel.astype(np.float32)
        
        # Convert joint positions from MuJoCo order to Isaac Lab order, then compute relative to default
        dof_pos_isaac = self.dof_pos[self.mujoco_to_isaac_idx] - self.default_dof_pos
        dof_pos_isaac = dof_pos_isaac.astype(np.float32)
        
        # Joint velocities (convert from MuJoCo order to Isaac Lab order)
        dof_vel_isaac = self.dof_vel[self.mujoco_to_isaac_idx].astype(np.float32)
        
        # Last action (in Isaac Lab order)
        last_action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions).astype(np.float32)

        # Update observation buffers (FIFO for each term)
        # Matching deploy code: each term maintains its own history
        self.obs_buffers['base_lin_vel'] = np.roll(self.obs_buffers['base_lin_vel'], shift=-1, axis=0)
        self.obs_buffers['base_lin_vel'][-1] = base_lin_vel
        
        self.obs_buffers['base_ang_vel'] = np.roll(self.obs_buffers['base_ang_vel'], shift=-1, axis=0)
        self.obs_buffers['base_ang_vel'][-1] = base_ang_vel
        
        self.obs_buffers['projected_gravity'] = np.roll(self.obs_buffers['projected_gravity'], shift=-1, axis=0)
        self.obs_buffers['projected_gravity'][-1] = projected_gravity
        
        self.obs_buffers['velocity_commands'] = np.roll(self.obs_buffers['velocity_commands'], shift=-1, axis=0)
        self.obs_buffers['velocity_commands'][-1] = velocity_commands
        
        self.obs_buffers['joint_pos_rel'] = np.roll(self.obs_buffers['joint_pos_rel'], shift=-1, axis=0)
        self.obs_buffers['joint_pos_rel'][-1] = dof_pos_isaac
        
        self.obs_buffers['joint_vel_rel'] = np.roll(self.obs_buffers['joint_vel_rel'], shift=-1, axis=0)
        self.obs_buffers['joint_vel_rel'][-1] = dof_vel_isaac
        
        self.obs_buffers['last_action'] = np.roll(self.obs_buffers['last_action'], shift=-1, axis=0)
        self.obs_buffers['last_action'][-1] = last_action

        # Concatenate observations matching training config (velocity_env_cfg.py) exactly
        # Order: base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action
        # Each term's history is flattened and concatenated (matching Isaac Lab's observation manager when use_gym_history=False)
        obs_list = [
            self.obs_buffers['base_lin_vel'].flatten(),      # 3 * 5 = 15 dims
            self.obs_buffers['base_ang_vel'].flatten(),      # 3 * 5 = 15 dims
            self.obs_buffers['projected_gravity'].flatten(), # 3 * 5 = 15 dims
            self.obs_buffers['velocity_commands'].flatten(), # 3 * 5 = 15 dims
            self.obs_buffers['joint_pos_rel'].flatten(),     # 29 * 5 = 145 dims
            self.obs_buffers['joint_vel_rel'].flatten(),     # 29 * 5 = 145 dims
            self.obs_buffers['last_action'].flatten(),       # 29 * 5 = 145 dims
        ]
        
        obs = np.concatenate(obs_list, axis=0).astype(np.float32)
        
        # Adjust observation size to match policy expectations (if policy size is known)
        # If policy expects different size, pad with zeros or truncate
        if self.policy_obs_size is not None and len(obs) != self.policy_obs_size:
            if len(obs) < self.policy_obs_size:
                # Pad with zeros if observation is smaller than expected
                padding = np.zeros(self.policy_obs_size - len(obs), dtype=np.float32)
                obs = np.concatenate([obs, padding], axis=0)
            else:
                # Truncate if observation is larger than expected
                obs = obs[:self.policy_obs_size]

        return np.clip(obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def compute_control(self) -> np.ndarray:
        """
        Compute control torques based on current FSM state.
        
        Returns:
            np.ndarray: Joint torques in MuJoCo order.
        """
        current_time = self.data.time
        
        if self.fsm_state == FSMState.PASSIVE:
            # Passive: damping only (mode=1)
            # Torque = -kd * velocity (damping only, no position control)
            current_velocities = self.dof_vel  # Already in MuJoCo order
            torques = -self.passive_kd_mujoco * current_velocities
            return torques
            
        elif self.fsm_state == FSMState.FIXSTAND:
            # FixStand: PD control with trajectory interpolation (from config.yaml)
            elapsed_time = current_time - self.fsm_state_start_time
            
            # Ensure starting position is set (capture current position on first call)
            if self.fixstand_start_qs_sdk is None:
                self.fixstand_start_qs_sdk = self.dof_pos.copy()  # Current position in MuJoCo order (same as SDK order)
                # Set first trajectory point to current position if it's empty
                if len(self.fixstand_qs_sdk[0]) == 0:
                    self.fixstand_qs_sdk[0] = self.fixstand_start_qs_sdk.copy()
            
            # Interpolate target position based on trajectory from config.yaml
            target_positions = None
            if elapsed_time <= self.fixstand_ts[0]:
                # Use first trajectory point
                if len(self.fixstand_qs_sdk[0]) > 0:
                    target_positions = self.fixstand_qs_sdk[0].copy()
                else:
                    target_positions = self.fixstand_start_qs_sdk.copy()
            elif elapsed_time >= self.fixstand_ts[-1]:
                # Use final trajectory point
                target_positions = self.fixstand_qs_sdk[-1].copy()
            else:
                # Linear interpolation between trajectory points
                for i in range(len(self.fixstand_ts) - 1):
                    if self.fixstand_ts[i] <= elapsed_time < self.fixstand_ts[i + 1]:
                        t0, t1 = self.fixstand_ts[i], self.fixstand_ts[i + 1]
                        q0 = self.fixstand_qs_sdk[i] if len(self.fixstand_qs_sdk[i]) > 0 else self.fixstand_start_qs_sdk
                        q1 = self.fixstand_qs_sdk[i + 1]
                        alpha = (elapsed_time - t0) / (t1 - t0)
                        target_positions = q0 + alpha * (q1 - q0)
                        break
            
            # Get current positions and velocities (in MuJoCo order)
            current_positions = self.dof_pos
            current_velocities = self.dof_vel
            
            # PD control using FixStand gains from config.yaml
            # torque = kp * (target_pos - current_pos) + kd * (0 - current_vel)
            position_error = target_positions - current_positions
            velocity_error = 0.0 - current_velocities
            
            torques = self.fixstand_kp_mujoco * position_error + self.fixstand_kd_mujoco * velocity_error
            return torques
            
        elif self.fsm_state == FSMState.VELOCITY:
            # Velocity: RL policy control (PD control with policy actions)
            # Actions are in Isaac Lab order and relative to default positions
            actions_mujoco = self.action[self.isaac_to_mujoco_idx]
            # Scale and add default positions to get target positions
            actions_scaled = actions_mujoco * self.cfg.sim.action_scale
            default_pos_mujoco = self.default_dof_pos[self.isaac_to_mujoco_idx]
            target_positions = actions_scaled + default_pos_mujoco
            
            # Get current positions and velocities (in MuJoCo order)
            current_positions = self.dof_pos
            current_velocities = self.dof_vel
            
            # PD control: torque = kp * (target_pos - current_pos) + kd * (0 - current_vel)
            position_error = target_positions - current_positions
            velocity_error = 0.0 - current_velocities
            
            torques = self.stiffness_mujoco * position_error + self.damping_mujoco * velocity_error
            return torques
        
        else:
            # Fallback: zero torques
            return np.zeros(self.cfg.sim.num_action, dtype=np.float32)
    
    def handle_fsm_transitions(self) -> None:
        """Handle FSM state transitions based on keyboard input."""
        # Check for transition requests
        if self.key_transition_requested is not None:
            new_state = self.key_transition_requested
            self.key_transition_requested = None
            
            # Validate transition
            if self.fsm_state == FSMState.PASSIVE and new_state == FSMState.FIXSTAND:
                # Passive -> FixStand
                self.fsm_state = FSMState.FIXSTAND
                self.fsm_state_start_time = self.data.time
                self.fixstand_start_qs_sdk = None  # Reset to capture current position
                print("[FSM] Transition: PASSIVE -> FIXSTAND")
            elif self.fsm_state == FSMState.FIXSTAND and new_state == FSMState.PASSIVE:
                # FixStand -> Passive
                self.fsm_state = FSMState.PASSIVE
                self.fsm_state_start_time = self.data.time
                print("[FSM] Transition: FIXSTAND -> PASSIVE")
            elif self.fsm_state == FSMState.FIXSTAND and new_state == FSMState.VELOCITY:
                # FixStand -> Velocity
                self.fsm_state = FSMState.VELOCITY
                self.fsm_state_start_time = self.data.time
                print("[FSM] Transition: FIXSTAND -> VELOCITY")
            elif self.fsm_state == FSMState.VELOCITY and new_state == FSMState.PASSIVE:
                # Velocity -> Passive
                self.fsm_state = FSMState.PASSIVE
                self.fsm_state_start_time = self.data.time
                print("[FSM] Transition: VELOCITY -> PASSIVE")

    def run(self) -> None:
        """
        Run the simulation loop with FSM-based control and keyboard transitions.
        """
        self.setup_keyboard_listener()
        self.listener.start()
        
        print("[INFO] FSM Control System Started")
        print("[INFO] Current State: FIXSTAND (Active Stand Mode)")
        print("[INFO] Robot starting in active stand mode, standing stable on ground")
        print("[INFO] Keyboard Controls:")
        print("  - Up Arrow: PASSIVE -> FIXSTAND")
        print("  - Down Arrow: FIXSTAND/VELOCITY -> PASSIVE")
        print("  - Right Arrow: FIXSTAND -> VELOCITY")
        print("  - NumPad 8/2: Increase/Decrease forward velocity (VELOCITY state)")
        print("  - NumPad 4/6: Strafe left/right (VELOCITY state)")
        print("  - NumPad 7/9: Turn left/right (VELOCITY state)")

        while self.data.time < self.cfg.sim.sim_duration:
            # Handle FSM state transitions
            self.handle_fsm_transitions()
            
            # Update observations (needed for Velocity state)
            self.obs_history = self.get_obs()
            
            # Run policy inference only in Velocity state
            if self.fsm_state == FSMState.VELOCITY:
                # Auto-detect policy observation size on first inference (for TorchScript)
                if self.policy_type == 'torchscript' and self.policy_obs_size is None:
                    try:
                        # Try inference with current observation size
                        obs_tensor = torch.tensor(self.obs_history, dtype=torch.float32).unsqueeze(0)
                        with torch.inference_mode():
                            _ = self.policy(obs_tensor)
                        self.policy_obs_size = len(self.obs_history)
                        print(f"[INFO] Policy observation size matches computed size: {self.policy_obs_size} dims")
                    except RuntimeError as e:
                        if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                            import re
                            match = re.search(r'\(1x\d+\s+and\s+(\d+)x', str(e))
                            if not match:
                                match = re.search(r'and\s+(\d+)x\d+', str(e))
                            if match:
                                detected_size = int(match.group(1))
                                computed_size = len(self.obs_history)
                                print(f"[WARNING] Observation size mismatch detected!")
                                print(f"[INFO] Computed observation size: {computed_size} dims")
                                print(f"[INFO] Policy expects: {detected_size} dims")
                                print(f"[INFO] Auto-adjusting observation size to match policy (padding with zeros)...")
                                self.policy_obs_size = detected_size
                                self.obs_history = self.get_obs()
                            else:
                                raise RuntimeError(f"Could not detect policy observation size from error: {e}")
                        else:
                            raise
                
                # Run policy inference
                obs_tensor = torch.tensor(self.obs_history, dtype=torch.float32).unsqueeze(0)
                
                if self.policy_type == 'onnx':
                    obs_input = self.obs_history.reshape(1, -1).astype(np.float32)
                    action_output = self.policy_session.run(None, {self.policy_session.get_inputs()[0].name: obs_input})[0]
                    self.action[:] = action_output[0, :self.cfg.sim.num_action]
                else:
                    with torch.inference_mode():
                        action_tensor = self.policy(obs_tensor)
                        self.action[:] = action_tensor.squeeze(0).detach().numpy()[:self.cfg.sim.num_action]
                
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
            else:
                # In Passive or FixStand states, set action to zeros
                self.action[:] = 0.0

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                # Reset applied forces before each step
                self.data.xfrc_applied[:] = 0.0

                # Compute control based on current FSM state
                torques = self.compute_control()
                self.data.ctrl = torques
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1

        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for FSM transitions and velocity commands.
        """

        def on_press(key):
            try:
                # FSM state transitions (matching config.yaml)
                if key == Key.up:  # Up arrow: PASSIVE -> FIXSTAND
                    if self.fsm_state == FSMState.PASSIVE:
                        self.key_transition_requested = FSMState.FIXSTAND
                elif key == Key.down:  # Down arrow: FIXSTAND/VELOCITY -> PASSIVE
                    if self.fsm_state in [FSMState.FIXSTAND, FSMState.VELOCITY]:
                        self.key_transition_requested = FSMState.PASSIVE
                elif key == Key.right:  # Right arrow: FIXSTAND -> VELOCITY
                    if self.fsm_state == FSMState.FIXSTAND:
                        self.key_transition_requested = FSMState.VELOCITY
                elif hasattr(key, 'char') and key.char:
                    # Velocity commands (only active in Velocity state)
                    if self.fsm_state == FSMState.VELOCITY:
                        if key.char == "8":  # NumPad 8      x += 0.2
                            self.adjust_command_vel(0, 0.2)
                        elif key.char == "2":  # NumPad 2      x -= 0.2
                            self.adjust_command_vel(0, -0.2)
                        elif key.char == "4":  # NumPad 4      y -= 0.2
                            self.adjust_command_vel(1, -0.2)
                        elif key.char == "6":  # NumPad 6      y += 0.2
                            self.adjust_command_vel(1, 0.2)
                        elif key.char == "7":  # NumPad 7      yaw += 0.2
                            self.adjust_command_vel(2, -0.2)
                        elif key.char == "9":  # NumPad 9      yaw -= 0.2
                            self.adjust_command_vel(2, 0.2)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller for Unitree G1.")
    parser.add_argument(
        "--robot",
        type=str,
        default="g1_29dof",
        choices=["g1_29dof", "g1_23dof"],
        help="Robot variant: 'g1_29dof' or 'g1_23dof'",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt or policy.onnx. If not specified, will use default G1 velocity policy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to MuJoCo XML model file (.xml). If not specified, will use default from unitree_mujoco repository.\n"
             "The model must have 29 DOF matching G1 robot structure with proper sensors and actuators.",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument(
        "--deploy-yaml",
        type=str,
        default=None,
        help="Path to deploy.yaml configuration file. If not specified, will try to find it relative to policy path.",
    )
    args = parser.parse_args()

    # Set default policy path if not provided
    if args.policy is None:
        policy_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "deploy", "robots", args.robot, "config", "policy", "velocity", "v0", "exported"
        )
        # Try ONNX first (deployed format), then PT (TorchScript)
        onnx_path = os.path.join(policy_dir, "policy.onnx")
        pt_path = os.path.join(policy_dir, "policy.pt")
        if os.path.isfile(onnx_path):
            args.policy = onnx_path
        elif os.path.isfile(pt_path):
            args.policy = pt_path
        else:
            print(f"[ERROR] Default policy not found. Please specify --policy")
            print(f"[INFO] Expected locations: {onnx_path} or {pt_path}")
            sys.exit(1)

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    
    # Set default model path from unitree_mujoco repository if not provided
    if args.model is None:
        model_filename = f"{args.robot}.xml"
        default_model_path = os.path.join(
            LEGGED_LAB_ROOT_DIR, "deploy", "unitree_mujoco", "unitree_robots", "g1", model_filename
        )
        if os.path.isfile(default_model_path):
            args.model = default_model_path
            print(f"[INFO] Using default model from unitree_mujoco: {args.model}")
        else:
            print(f"[ERROR] Default MuJoCo model file not found: {default_model_path}")
            print(f"[INFO] Please specify --model with path to a valid G1 MuJoCo XML model file.")
            sys.exit(1)
    
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Robot: {args.robot.upper()}")
    print(f"[INFO] Policy: {args.policy}")
    print(f"[INFO] Model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
        deploy_yaml_path=args.deploy_yaml,
    )
    runner.run()