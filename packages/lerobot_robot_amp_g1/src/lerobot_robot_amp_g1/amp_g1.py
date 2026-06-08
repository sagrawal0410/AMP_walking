"""AMP velocity deploy robot for Unitree G1 via DDS."""

from __future__ import annotations

import logging
import threading
import time
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from lerobot_policy_amp_velocity.amp_kinematics import check_orientation_safe
from lerobot_policy_amp_velocity.constants import G1_JOINT_NAMES, NUM_JOINTS, joint_q_key, raw_obs_keys
from lerobot_policy_amp_velocity.deploy_config import DeployConfig

from .config_amp_g1 import AmpG1Config
from .fsm import FIXSTAND_KD, FIXSTAND_KP, FSMState, KEY_VELOCITIES, PASSIVE_KD, VELOCITY_SMOOTHING

if _unitree_sdk_available:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize as _SDKChannelFactoryInitialize,
        ChannelPublisher as _SDKChannelPublisher,
        ChannelSubscriber as _SDKChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as HGLowCmdDefault
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
    from unitree_sdk2py.utils.crc import CRC
else:
    _SDKChannelFactoryInitialize = None
    _SDKChannelPublisher = None
    _SDKChannelSubscriber = None
    HGLowCmdDefault = None
    hg_LowCmd = None
    hg_LowState = None
    CRC = None

try:
    from pynput import keyboard as pynput_keyboard
    from pynput.keyboard import Key

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    from lerobot.robots.robot import Robot
except ImportError:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)

kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"


class AmpG1(Robot):
    """Unitree G1 robot with AMP FSM, DDS I/O, and velocity command interface."""

    config_class = AmpG1Config
    name = "amp_g1"

    def __init__(self, config: AmpG1Config):
        require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
        super().__init__(config)
        self.config = config
        self.control_dt = config.policy_step_dt

        if not config.deploy_yaml:
            raise ValueError("AmpG1Config.deploy_yaml must point to a valid deploy yaml.")
        self.deploy_cfg = DeployConfig(config.deploy_yaml)

        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._subscribe_thread: threading.Thread | None = None
        self._publisher_thread: threading.Thread | None = None
        self._keyboard_listener = None

        self.fsm_state = FSMState.PASSIVE
        self.fsm_start_time = time.time()
        self.fixstand_start_pos: np.ndarray | None = None
        self.command_vel = np.zeros(3, dtype=np.float32)
        self._held_keys: set[str] = set()
        self._transition_request: FSMState | None = None

        self.motor_pos_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.motor_vel_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.imu_gyro = np.zeros(3, dtype=np.float32)

        self._cmd_lock = threading.Lock()
        self._live_cmd = None
        self._publisher_running = False
        self._connected = False

        self.sim_env = None
        self._env_wrapper = None
        self.crc = None
        self.msg = None
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None

        if config.is_simulation:
            self._ChannelFactoryInitialize = _SDKChannelFactoryInitialize
            self._ChannelPublisher = _SDKChannelPublisher
            self._ChannelSubscriber = _SDKChannelSubscriber
        else:
            from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )

            self._ChannelFactoryInitialize = ChannelFactoryInitialize
            self._ChannelPublisher = ChannelPublisher
            self._ChannelSubscriber = ChannelSubscriber

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # The .pos-suffixed raw keys are concatenated by lerobot into
        # `observation.state` (consumed by AmpObsBuilderProcessorStep). fsm_state
        # and policy_enabled deliberately lack the .pos suffix so they are kept
        # out of the policy tensor (diagnostics only).
        features: dict[str, type | tuple] = {key: float for key in raw_obs_keys()}
        features["fsm_state"] = float
        features["policy_enabled"] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {joint_q_key(name): float for name in G1_JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _subscribe_lowstate(self) -> None:
        rate_dt = 1.0 / 250.0
        while not self._shutdown_event.is_set():
            start = time.time()
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                with self._lowstate_lock:
                    for i in range(NUM_JOINTS):
                        self.motor_pos_sdk[i] = msg.motor_state[i].q
                        self.motor_vel_sdk[i] = msg.motor_state[i].dq
                    self.imu_quat_wxyz[:] = msg.imu_state.quaternion
                    self.imu_gyro[:] = msg.imu_state.gyroscope
                    self._lowstate = msg

            elapsed = time.time() - start
            time.sleep(max(0.0, rate_dt - elapsed))

    def _publish_cmd(self, cmd) -> None:
        cmd.mode_pr = 2
        if self.crc is not None:
            cmd.crc = self.crc.Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

    def _cmd_publisher_thread(self) -> None:
        cmd_dt = 1.0 / self.config.cmd_publish_hz
        while self._publisher_running and not self._shutdown_event.is_set():
            with self._cmd_lock:
                if self._live_cmd is not None:
                    if self.fsm_state == FSMState.PASSIVE:
                        with self._lowstate_lock:
                            pos = self.motor_pos_sdk.copy()
                        for i in range(NUM_JOINTS):
                            self._live_cmd.motor_cmd[i].q = float(pos[i])
                    elif self.fsm_state == FSMState.FIXSTAND:
                        if self.fixstand_start_pos is not None:
                            elapsed = time.time() - self.fsm_start_time
                            alpha = min(elapsed / self.config.fixstand_duration_s, 1.0)
                            for i in range(NUM_JOINTS):
                                target = (
                                    self.fixstand_start_pos[i] * (1 - alpha)
                                    + self.deploy_cfg.default_joint_pos_sdk[i] * alpha
                                )
                                self._live_cmd.motor_cmd[i].q = float(target)
                    self._publish_cmd(self._live_cmd)
            time.sleep(cmd_dt)

    def _apply_fsm_transition(self, new_state: FSMState) -> None:
        with self._lowstate_lock:
            motor_pos = self.motor_pos_sdk.copy()
            imu_quat = self.imu_quat_wxyz.copy()

        with self._cmd_lock:
            if self._live_cmd is None:
                return
            if new_state == FSMState.PASSIVE:
                for i in range(NUM_JOINTS):
                    self._live_cmd.motor_cmd[i].mode = 1
                    self._live_cmd.motor_cmd[i].q = float(motor_pos[i])
                    self._live_cmd.motor_cmd[i].kp = 0.0
                    self._live_cmd.motor_cmd[i].kd = float(PASSIVE_KD[i])
                    self._live_cmd.motor_cmd[i].dq = 0.0
                    self._live_cmd.motor_cmd[i].tau = 0.0
            elif new_state == FSMState.FIXSTAND:
                self.fixstand_start_pos = motor_pos.copy()
                for i in range(NUM_JOINTS):
                    self._live_cmd.motor_cmd[i].mode = 1
                    self._live_cmd.motor_cmd[i].q = float(motor_pos[i])
                    self._live_cmd.motor_cmd[i].kp = float(FIXSTAND_KP[i])
                    self._live_cmd.motor_cmd[i].kd = float(FIXSTAND_KD[i])
                    self._live_cmd.motor_cmd[i].dq = 0.0
                    self._live_cmd.motor_cmd[i].tau = 0.0
            elif new_state == FSMState.VELOCITY:
                for i in range(NUM_JOINTS):
                    self._live_cmd.motor_cmd[i].mode = 1
                    self._live_cmd.motor_cmd[i].q = float(motor_pos[i])
                    self._live_cmd.motor_cmd[i].kp = float(self.deploy_cfg.stiffness_sdk[i])
                    self._live_cmd.motor_cmd[i].kd = float(self.deploy_cfg.damping_sdk[i])
                    self._live_cmd.motor_cmd[i].dq = 0.0
                    self._live_cmd.motor_cmd[i].tau = 0.0

        old = self.fsm_state.name
        self.fsm_state = new_state
        self.fsm_start_time = time.time()
        if new_state == FSMState.VELOCITY:
            self.command_vel[:] = 0.0
        logger.info("FSM transition: %s -> %s", old, new_state.name)

    def _update_velocity_command(self) -> None:
        if self.fsm_state != FSMState.VELOCITY:
            return
        target = np.zeros(3, dtype=np.float32)
        for key in self._held_keys:
            if key in KEY_VELOCITIES:
                target += np.array(KEY_VELOCITIES[key], dtype=np.float32)
        self.command_vel += (target - self.command_vel) * VELOCITY_SMOOTHING
        mask = np.abs(self.command_vel) < 0.01
        self.command_vel[mask] = 0.0

    def _check_safety(self) -> None:
        if self.fsm_state != FSMState.VELOCITY:
            return
        with self._lowstate_lock:
            imu_quat = self.imu_quat_wxyz.copy()
        if not check_orientation_safe(imu_quat):
            logger.warning("Orientation unsafe — transitioning to PASSIVE")
            self._apply_fsm_transition(FSMState.PASSIVE)

    def _process_fsm_updates(self) -> None:
        if self._transition_request is not None:
            new_state = self._transition_request
            self._transition_request = None
            self._apply_fsm_transition(new_state)
        self._update_velocity_command()
        self._check_safety()

    def _start_keyboard_listener(self) -> None:
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not installed — keyboard FSM/velocity control disabled")
            return

        def on_press(key):
            try:
                if key == Key.up and self.fsm_state == FSMState.PASSIVE:
                    self._transition_request = FSMState.FIXSTAND
                elif key == Key.down:
                    self._transition_request = FSMState.PASSIVE
                elif key == Key.right and self.fsm_state == FSMState.FIXSTAND:
                    self._transition_request = FSMState.VELOCITY
                elif hasattr(key, "char") and key.char:
                    c = key.char.lower()
                    if c in KEY_VELOCITIES:
                        self._held_keys.add(c)
                    elif c == " ":
                        self.command_vel[:] = 0.0
                        self._held_keys.clear()
            except Exception:
                pass

        def on_release(key):
            try:
                if hasattr(key, "char") and key.char:
                    self._held_keys.discard(key.char.lower())
            except Exception:
                pass

        self._keyboard_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._keyboard_listener.start()

    def connect(self, calibrate: bool = True) -> None:
        if self.config.is_simulation:
            from lerobot.envs import make_env

            self._ChannelFactoryInitialize(self.config.domain_id, self.config.network)
            self._env_wrapper = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        else:
            self._ChannelFactoryInitialize(self.config.domain_id, config=self.config)

        self.lowcmd_publisher = self._ChannelPublisher(kTopicLowCommand, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = self._ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        self._shutdown_event.clear()
        self._subscribe_thread = threading.Thread(target=self._subscribe_lowstate, daemon=True)
        self._subscribe_thread.start()

        deadline = time.time() + 30.0
        while self._lowstate is None:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for robot lowstate (30s)")
            time.sleep(0.01)

        self.crc = CRC()
        self.msg = HGLowCmdDefault()
        self.msg.mode_machine = 5

        with self._cmd_lock:
            self._live_cmd = HGLowCmdDefault()
            self._live_cmd.mode_machine = 5
            with self._lowstate_lock:
                init_pos = self.motor_pos_sdk.copy()
            for i in range(NUM_JOINTS):
                self._live_cmd.motor_cmd[i].mode = 1
                self._live_cmd.motor_cmd[i].q = float(init_pos[i])
                self._live_cmd.motor_cmd[i].kp = 0.0
                self._live_cmd.motor_cmd[i].kd = float(PASSIVE_KD[i])
                self._live_cmd.motor_cmd[i].dq = 0.0
                self._live_cmd.motor_cmd[i].tau = 0.0

        self._publisher_running = True
        self._publisher_thread = threading.Thread(target=self._cmd_publisher_thread, daemon=True)
        self._publisher_thread.start()
        self._start_keyboard_listener()
        self._connected = True
        logger.info("AmpG1 connected (sim=%s, network=%s)", self.config.is_simulation, self.config.network)

    def get_observation(self) -> RobotObservation:
        self._process_fsm_updates()
        with self._lowstate_lock:
            pos = self.motor_pos_sdk.copy()
            vel = self.motor_vel_sdk.copy()
            imu_quat = self.imu_quat_wxyz.copy()
            imu_gyro = self.imu_gyro.copy()

        # Order MUST match constants.raw_obs_keys() so the concatenated
        # observation.state vector slices correctly in AmpObsBuilderProcessorStep.
        raw_vals = (
            [float(pos[idx]) for idx in range(NUM_JOINTS)]
            + [float(vel[idx]) for idx in range(NUM_JOINTS)]
            + [float(imu_quat[0]), float(imu_quat[1]), float(imu_quat[2]), float(imu_quat[3])]
            + [float(imu_gyro[0]), float(imu_gyro[1]), float(imu_gyro[2])]
            + [float(self.command_vel[0]), float(self.command_vel[1]), float(self.command_vel[2])]
        )
        obs: RobotObservation = dict(zip(raw_obs_keys(), raw_vals))
        obs["fsm_state"] = float(self.fsm_state.value)
        obs["policy_enabled"] = float(self.fsm_state == FSMState.VELOCITY)
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if self.fsm_state != FSMState.VELOCITY:
            return action

        with self._cmd_lock:
            if self._live_cmd is None:
                return action
            for idx, name in enumerate(G1_JOINT_NAMES):
                key = joint_q_key(name)
                if key not in action:
                    continue
                val = float(action[key])
                if not np.isfinite(val):
                    with self._lowstate_lock:
                        val = float(self.motor_pos_sdk[idx])
                self._live_cmd.motor_cmd[idx].q = val
                self._live_cmd.motor_cmd[idx].kp = float(self.deploy_cfg.stiffness_sdk[idx])
                self._live_cmd.motor_cmd[idx].kd = float(self.deploy_cfg.damping_sdk[idx])
                self._live_cmd.motor_cmd[idx].tau = 0.0
        return action

    @property
    def policy_enabled(self) -> bool:
        return self.fsm_state == FSMState.VELOCITY

    def _send_zero_torque(self) -> None:
        try:
            with self._lowstate_lock:
                pos = self.motor_pos_sdk.copy()
            with self._cmd_lock:
                if self._live_cmd is None:
                    return
                for i in range(NUM_JOINTS):
                    self._live_cmd.motor_cmd[i].q = float(pos[i])
                    self._live_cmd.motor_cmd[i].kp = 0.0
                    self._live_cmd.motor_cmd[i].kd = 0.0
                    self._live_cmd.motor_cmd[i].tau = 0.0
                self._publish_cmd(self._live_cmd)
        except Exception as exc:
            logger.warning("Failed to send zero-torque on disconnect: %s", exc)

    def disconnect(self) -> None:
        if not self.config.is_simulation:
            self._send_zero_torque()

        self._shutdown_event.set()
        self._publisher_running = False

        if self._subscribe_thread is not None:
            self._subscribe_thread.join(timeout=2.0)
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=2.0)
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass

        if self.config.is_simulation and self.sim_env is not None:
            try:
                self.sim_env.close()
            except Exception as exc:
                logger.warning("Error closing sim env: %s", exc)
            self.sim_env = None
            self._env_wrapper = None

        self._connected = False
        logger.info("AmpG1 disconnected")
