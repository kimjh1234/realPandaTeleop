import numpy as np
import torch
import math


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Compute the quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def mat_to_quat(m):
    w = torch.sqrt(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]) / 2.0
    w4 = (4.0 * w)
    x = (m[:, 2, 1] - m[:, 1, 2]) / w4
    y = (m[:, 0, 2] - m[:, 2, 0]) / w4
    z = (m[:, 1, 0] - m[:, 0, 1]) / w4
    return torch.stack([w, x, y, z], dim=-1).to(m.device)


def deg2rad(deg):
    return deg * (np.pi / 180.0)


def euler_to_mat3d(roll, pitch, yaw):    # radian input
    rx = torch.tensor([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = torch.tensor([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = torch.tensor([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return torch.matmul(rx, torch.matmul(ry, rz))


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = (qq - ww + (z1 - y1) * (y2 - z2))
    x = (qq - xx + (x1 + w1) * (x2 + w2))
    y = (qq - yy + (w1 - x1) * (y2 + z2))
    z = (qq - zz + (z1 + y1) * (w2 - x2))

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def conjugate(quat):
    w, x, y, z = quat
    quat = w, -x, -y, -z
    return torch.tensor(quat)


def rotation_quaternion(theta, axis):
    quat = np.cos(theta*0.5), np.sin(theta*0.5)*axis[0], np.sin(theta*0.5)*axis[1], np.sin(theta*0.5)*axis[2]
    return torch.tensor(quat)


def align_quaternion(quat):
    # Convert to the same as the robot root coordinate system
    con_quat_ori = quat_mul(rotation_quaternion(-np.pi / 2, [0, 1, 0]), rotation_quaternion(np.pi / 4, [1, 0, 0]))
    quat = quat_mul(quat_mul(con_quat_ori, quat), conjugate(con_quat_ori))
    return torch.tensor(quat)


def diff_quaternion(name, quat):
    # if name == "ee":
    #     # reset reference value
    #     quat_2_conj = 0.0006451968220062554, -0.9993628859519958, -0.035676825791597366, -0.0007577423239126801
    #     quat = quat_mul(quat, torch.tensor(quat_2_conj))
    if name == "con":
        # Convert to the same as the robot root coordinate system
        con_quat_ori = quat_mul(rotation_quaternion(-np.pi / 2, [1, 0, 0]), rotation_quaternion(np.pi * 2 / 3, [0, 1, 0]))
        quat = quat_mul(quat_mul(con_quat_ori, quat), conjugate(con_quat_ori))
        # reset reference value
        quat_2_conj = 0.26446104, -0.03798511, 0.02586818, 0.96329206
        quat = quat_mul(quat, torch.tensor(quat_2_conj))
    elif name == "ee":
        con_quat_ori = quat_mul(rotation_quaternion(np.pi, [0, 0, 1]), rotation_quaternion(- np.pi / 2, [0, 1, 0]))
        quat = quat_mul(quat_mul(con_quat_ori, quat), conjugate(con_quat_ori))

    else:
        quat = np.zeros(4)
    return quat


def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def apply_transformations(point, pitch, roll):
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Compute rotation matrices
    pitch_matrix = rotation_matrix_y(pitch_rad)
    roll_matrix = rotation_matrix_x(roll_rad)

    # Apply pitch rotation
    rotated_point = np.dot(pitch_matrix, point)

    # Apply roll rotation
    transformed_point = np.dot(roll_matrix, rotated_point)

    return transformed_point


def apply_transformations1(point, yaw):
    # Convert angles from degrees to radians
    yaw_rad = np.radians(yaw)

    # Compute rotation matrices
    yaw_matrix = rotation_matrix_y(yaw_rad)

    # Apply pitch rotation
    transformed_point = np.dot(yaw_matrix, point)

    return transformed_point


def quaternion_to_euler_angles(quaternion):
    w, x, y, z = quaternion
    # normalization
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.clamp(sinp, -1, 1),  # 아크사인 함수에 입력 범위를 보장
        sinp
    )
    pitch = torch.asin(pitch)
    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    euler = torch.stack([roll, pitch, yaw], dim=-1)

    return euler