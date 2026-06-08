"""Shared constants for AMP velocity deploy."""

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

# G1 SDK joint order (matches Unitree motor indices 0-28)
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

# LeRobot G1_29_JointIndex enum names (SDK order)
G1_JOINT_NAMES = [
    "kLeftHipPitch",
    "kLeftHipRoll",
    "kLeftHipYaw",
    "kLeftKnee",
    "kLeftAnklePitch",
    "kLeftAnkleRoll",
    "kRightHipPitch",
    "kRightHipRoll",
    "kRightHipYaw",
    "kRightKnee",
    "kRightAnklePitch",
    "kRightAnkleRoll",
    "kWaistYaw",
    "kWaistRoll",
    "kWaistPitch",
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
]

# lerobot's rollout pipeline only routes float observation/action features whose
# key ends in ".pos" into the policy-facing `observation.state` / `action`
# tensors (see lerobot.rollout.context.build_rollout_context). Every key we want
# the framework to forward must therefore end in ".pos".
def joint_q_key(name: str) -> str:
    """Joint position key (also the action target key)."""
    return f"{name}.pos"


def joint_dq_key(name: str) -> str:
    """Joint velocity key. The ".pos" suffix is required so lerobot forwards it
    into `observation.state`; it does not imply a position channel."""
    return f"{name}.vel.pos"


# Raw, per-step quantities the AmpObsBuilder needs, in the exact order they are
# concatenated into `observation.state` by lerobot's `build_dataset_frame`. The
# robot declares features in this order and AmpObsBuilderProcessorStep slices the
# resulting vector with the same layout.
_IMU_QUAT_KEYS = ["imu.quat.w.pos", "imu.quat.x.pos", "imu.quat.y.pos", "imu.quat.z.pos"]
_IMU_GYRO_KEYS = ["imu.gyro.x.pos", "imu.gyro.y.pos", "imu.gyro.z.pos"]
_CMD_KEYS = ["velocity_commands.0.pos", "velocity_commands.1.pos", "velocity_commands.2.pos"]

RAW_OBS_DIM = 2 * NUM_JOINTS + len(_IMU_QUAT_KEYS) + len(_IMU_GYRO_KEYS) + len(_CMD_KEYS)  # 68


def raw_obs_keys() -> list[str]:
    """Ordered raw observation feature keys backing `observation.state` (len 68)."""
    keys = [joint_q_key(name) for name in G1_JOINT_NAMES]
    keys += [joint_dq_key(name) for name in G1_JOINT_NAMES]
    keys += _IMU_QUAT_KEYS + _IMU_GYRO_KEYS + _CMD_KEYS
    return keys
