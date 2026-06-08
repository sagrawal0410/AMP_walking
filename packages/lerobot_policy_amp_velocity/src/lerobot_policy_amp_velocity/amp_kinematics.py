"""Forward kinematics and observation math for AMP velocity policies."""

from __future__ import annotations

import numpy as np

from .constants import KEY_BODY_NAMES


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=q.dtype)


def _axis_angle_to_rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _make_transform(pos, quat=None) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = pos
    if quat is not None:
        t[:3, :3] = quat_to_rotmat(quat)
    return t


def _joint_transform(axis, angle: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _axis_angle_to_rotmat(np.array(axis, dtype=np.float64), angle)
    return t


def compute_key_body_positions_g1(joint_pos_sdk: np.ndarray) -> dict[str, np.ndarray]:
    """Analytic FK for the six AMP key bodies (base frame, SDK joint order)."""
    q = joint_pos_sdk.astype(np.float64)

    t = np.eye(4)
    t = t @ _make_transform([0, 0.064452, -0.1027])
    t = t @ _joint_transform([0, 1, 0], q[0])
    t = t @ _make_transform([0, 0.052, -0.030465], quat=[0.996179, 0, -0.0873386, 0])
    t = t @ _joint_transform([1, 0, 0], q[1])
    t = t @ _make_transform([0.025001, 0, -0.12412])
    t = t @ _joint_transform([0, 0, 1], q[2])
    t = t @ _make_transform([-0.078273, 0.0021489, -0.17734], quat=[0.996179, 0, 0.0873386, 0])
    t = t @ _joint_transform([0, 1, 0], q[3])
    t = t @ _make_transform([0, -9.4445e-05, -0.30001])
    t = t @ _joint_transform([0, 1, 0], q[4])
    t = t @ _make_transform([0, 0, -0.017558])
    t = t @ _joint_transform([1, 0, 0], q[5])
    left_ankle_roll_pos = t[:3, 3].copy()

    t = np.eye(4)
    t = t @ _make_transform([0, -0.064452, -0.1027])
    t = t @ _joint_transform([0, 1, 0], q[6])
    t = t @ _make_transform([0, -0.052, -0.030465], quat=[0.996179, 0, -0.0873386, 0])
    t = t @ _joint_transform([1, 0, 0], q[7])
    t = t @ _make_transform([0.025001, 0, -0.12412])
    t = t @ _joint_transform([0, 0, 1], q[8])
    t = t @ _make_transform([-0.078273, -0.0021489, -0.17734], quat=[0.996179, 0, 0.0873386, 0])
    t = t @ _joint_transform([0, 1, 0], q[9])
    t = t @ _make_transform([0, 9.4445e-05, -0.30001])
    t = t @ _joint_transform([0, 1, 0], q[10])
    t = t @ _make_transform([0, 0, -0.017558])
    t = t @ _joint_transform([1, 0, 0], q[11])
    right_ankle_roll_pos = t[:3, 3].copy()

    t_waist = np.eye(4)
    t_waist = t_waist @ _joint_transform([0, 0, 1], q[12])
    t_waist = t_waist @ _make_transform([-0.0039635, 0, 0.035])
    t_waist = t_waist @ _joint_transform([1, 0, 0], q[13])
    t_waist = t_waist @ _make_transform([0, 0, 0.019])
    t_waist = t_waist @ _joint_transform([0, 1, 0], q[14])

    t = t_waist.copy()
    t = t @ _make_transform(
        [0.0039563, 0.10022, 0.23778],
        quat=[0.990264, 0.139201, 1.38722e-05, -9.86868e-05],
    )
    t = t @ _joint_transform([0, 1, 0], q[15])
    t_lsr = t.copy()
    t_lsr = t_lsr @ _make_transform([0, 0.038, -0.013831], quat=[0.990268, -0.139172, 0, 0])
    t_lsr = t_lsr @ _joint_transform([1, 0, 0], q[16])
    left_shoulder_roll_pos = t_lsr[:3, 3].copy()

    t = t_lsr.copy()
    t = t @ _make_transform([0, 0.00624, -0.1032])
    t = t @ _joint_transform([0, 0, 1], q[17])
    t = t @ _make_transform([0.015783, 0, -0.080518])
    t = t @ _joint_transform([0, 1, 0], q[18])
    t = t @ _make_transform([0.1, 0.00188791, -0.01])
    t = t @ _joint_transform([1, 0, 0], q[19])
    t = t @ _make_transform([0.038, 0, 0])
    t = t @ _joint_transform([0, 1, 0], q[20])
    t = t @ _make_transform([0.046, 0, 0])
    t = t @ _joint_transform([0, 0, 1], q[21])
    left_wrist_yaw_pos = t[:3, 3].copy()

    t = t_waist.copy()
    t = t @ _make_transform(
        [0.0039563, -0.10021, 0.23778],
        quat=[0.990264, -0.139201, 1.38722e-05, 9.86868e-05],
    )
    t = t @ _joint_transform([0, 1, 0], q[22])
    t_rsr = t.copy()
    t_rsr = t_rsr @ _make_transform([0, -0.038, -0.013831], quat=[0.990268, 0.139172, 0, 0])
    t_rsr = t_rsr @ _joint_transform([1, 0, 0], q[23])
    right_shoulder_roll_pos = t_rsr[:3, 3].copy()

    t = t_rsr.copy()
    t = t @ _make_transform([0, -0.00624, -0.1032])
    t = t @ _joint_transform([0, 0, 1], q[24])
    t = t @ _make_transform([0.015783, 0, -0.080518])
    t = t @ _joint_transform([0, 1, 0], q[25])
    t = t @ _make_transform([0.1, -0.00188791, -0.01])
    t = t @ _joint_transform([1, 0, 0], q[26])
    t = t @ _make_transform([0.038, 0, 0])
    t = t @ _joint_transform([0, 1, 0], q[27])
    t = t @ _make_transform([0.046, 0, 0])
    t = t @ _joint_transform([0, 0, 1], q[28])
    right_wrist_yaw_pos = t[:3, 3].copy()

    return {
        "left_ankle_roll_link": left_ankle_roll_pos,
        "right_ankle_roll_link": right_ankle_roll_pos,
        "left_wrist_yaw_link": left_wrist_yaw_pos,
        "right_wrist_yaw_link": right_wrist_yaw_pos,
        "left_shoulder_roll_link": left_shoulder_roll_pos,
        "right_shoulder_roll_link": right_shoulder_roll_pos,
    }


def compute_root_local_rot_tan_norm(imu_quat: np.ndarray) -> np.ndarray:
    """Yaw-removed orientation: columns 0 and 2 of the rotation matrix."""
    yaw_q = yaw_quat(imu_quat)
    local_q = quat_mul(quat_conjugate(yaw_q), imu_quat)
    rot = quat_to_rotmat(local_q)
    return np.concatenate([rot[:, 0], rot[:, 2]]).astype(np.float32)


def compute_key_body_pos_b(
    joint_pos_sdk: np.ndarray,
    key_body_names: list[str] | None = None,
) -> np.ndarray:
    names = key_body_names or KEY_BODY_NAMES
    positions = compute_key_body_positions_g1(joint_pos_sdk)
    return np.concatenate([positions[name] for name in names]).astype(np.float32)


def check_orientation_safe(imu_quat: np.ndarray, max_tilt_deg: float = 60.0) -> bool:
    rot = quat_to_rotmat(imu_quat)
    cos_angle = rot[2, 2]
    return cos_angle > np.cos(np.radians(max_tilt_deg))
