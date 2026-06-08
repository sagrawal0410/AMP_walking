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

def joint_q_key(name: str) -> str:
    return f"{name}.q"

def joint_dq_key(name: str) -> str:
    return f"{name}.dq"
