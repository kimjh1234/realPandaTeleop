import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import ctypes
import pyRobotiqGripper
import time
import triad_openvr
import math

import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from sympy import *
from utilities import *


Q = np.zeros(7)
X, Y, Z, ROLL, PITCH, YAW = 0.3, 0.0, 0.45, 180, 0.0, 45
target_joint_positions = []
new_order = [0, 1, 2, 5, 6, 3, 4]
interval_gripper_command = 1.5
interval_reset_command = 1.5
start_gripper = time.time()
start_reset = time.time()
gripper_switch = 1
reset_button = 0


class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        qos_profile = QoSProfile(depth=10)
        self.subscription = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.listener_callback,
            qos_profile)
        self.timer = self.create_timer(0.0001, self.update_vr_data)
        self.timer = self.create_timer(0.0001, self.update_gripper_data)

    def update_vr_data(self):
        global d
        d = teleop_interface.devices["controller_1"].get_controller_inputs()

    def update_gripper_data(self):
        global gripper
        gripper = pyRobotiqGripper.RobotiqGripper(portname="/dev/ttyUSB0")

    def listener_callback(self, msg):
        # print("listner_callback@@@")
        global Q
        global X, Y, Z, ROLL, PITCH, YAW
        global target_joint_positions
        global gripper_switch
        global start_gripper
        global start_reset
        global reset_button
        reset_button = 0

        if d['grip_button']:
            if interval_gripper_command - (time.time() - start_gripper) > 0:
                pass
            else:
                start_gripper = time.time()
                if gripper_switch == 1:
                    # gripper closing
                    gripper.close()
                    gripper_switch = 0
                else:
                    # gripper opening
                    gripper.open()
                    gripper_switch = 1

        # print('Position: ')
        for i in range(len(msg.position.tolist())):
            # print(msg.position.tolist()[i])
            Q[i] = msg.position.tolist()[i]
        # print(Q)
        # print('Velocity: ')
        # for i in range(len(msg.velocity.tolist())):
        #     print(msg.velocity.tolist()[i])
        #
        # print('Effort: ')
        # for i in range(len(msg.effort.tolist())):
        #     print(msg.effort.tolist()[i])
        # positions = msg.position[1]

        # Q = [round(num, 2) for num in [Q[i] for i in new_order]]
        Q = [Q[i] for i in new_order]
        if d['trigger']:
            # position
            pos_ee, rot_quat = jointState(Q)
            lv = np.array([v for v in teleop_interface.devices["controller_1"].get_velocity()]) * 0.5  # m/s
            pos_ee += apply_transformations(lv, 120, 90)  # align position
            # orientation
            quat_euler_raw = torch.tensor(teleop_interface.devices["controller_1"].get_pose_euler()[3:])
            roll = np.deg2rad(90)
            pitch = np.deg2rad(0)
            yaw = np.deg2rad(180)
            rotation = R.from_euler('XYZ', [roll, pitch, yaw])
            rotated_vector = rotation.apply(quat_euler_raw)
            inverse_euler_angles = [2.913758, 176.08311462, 254.30770874]
            rot_euler = rotated_vector + inverse_euler_angles
            rot_euler = np.array([normalize_angle(angle) for angle in rot_euler])

            X, Y, Z = pos_ee
            ROLL, PITCH, YAW = rot_euler

            pandaik_lib = ctypes.CDLL('/usr/local/diana/lib/libPanda-IK.so')
            pandaik_lib.compute_inverse_kinematics_void.argtypes = [ctypes.POINTER(ctypes.c_double),
                                                                    ctypes.POINTER(ctypes.c_double),
                                                                    ctypes.POINTER(ctypes.c_double)]
            pandaik_lib.compute_inverse_kinematics_void.restype = None

            xyzrpy = (ctypes.c_double * 6)(X, Y, Z, ROLL, PITCH, YAW)
            q_actual = (ctypes.c_double * 7)(Q[0], Q[1], Q[2], Q[3], Q[4], Q[5], Q[6])
            output = (ctypes.c_double * 7)()

            pandaik_lib.compute_inverse_kinematics_void(xyzrpy, q_actual, output)

            target_joint_positions = list(output)
        elif d['menu_button']:
            if interval_reset_command - (time.time() - start_reset) > 0:
                pass
            else:
                gripper.goTo(128)
                target_joint_positions = [0.001, -0.785398, 0.001, -2.35619, 0.001, 1.55622, 0.000]
                start_reset = time.time()
                reset_button = 1
        else:
            target_joint_positions = Q

        node_p = JointTrajectoryPublisher()
        rclpy.spin_once(node_p)
        node_p.destroy_node()


class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher')
        self.publisher_ = self.create_publisher(JointTrajectory, '/panda_arm_controller/joint_trajectory', 10)
        self.timer = self.create_timer(0.001, self.publish_trajectory)

    def publish_trajectory(self):
        global Q
        global X, Y, Z, ROLL, PITCH, YAW
        global target_joint_positions
        global reset_button

        msg = JointTrajectory()
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5',
                           'panda_joint6', 'panda_joint7']

        # Define trajectory points
        point1 = JointTrajectoryPoint()
        # point1.positions = [0.0163373, -0.793209, -0.00298236, -2.36398, -0.0143024, 1.55622, 0.796992]
        # point1.positions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # point1.positions = [0.001, -0.785398, 0.001, -2.35619, 0.001, 1.55622, 0.000]
        point1.positions = target_joint_positions


        point1.time_from_start.sec = 1


        msg.points = [point1]

        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing JointTrajectory message')


def main(args=None):
    # openvr init
    global teleop_interface
    teleop_interface = triad_openvr.triad_openvr()
    # node activate
    rclpy.init(args=args)
    node_s = JointStateSubscriber()
    rclpy.spin(node_s)
    node_s.destroy_node()
    rclpy.shutdown()


def normalize_angle(angle):
    return (angle + 180) % 360 - 180


def dh_params(joint_variable):

    joint_var = joint_variable
    M_PI = math.pi

    # Create DH parameters (data given by maker franka-emika)
    dh = [[0, 0, 0.333, joint_var[0]],
               [-M_PI / 2, 0, 0, joint_var[1]],
               [M_PI / 2, 0, 0.316, joint_var[2]],
               [M_PI / 2, 0.0825, 0, joint_var[3]],
               [-M_PI / 2, -0.0825, 0.384, joint_var[4]],
               [M_PI / 2, 0, 0, joint_var[5]],
               [M_PI / 2, 0.088, 0.107, joint_var[6]]]

    return dh

def TF_matrix(i, dh):
    # Define Transformation matrix based on DH params
    alpha = dh[i][0]
    a = dh[i][1]
    d = dh[i][2]
    q = dh[i][3]

    TF = Matrix([[cos(q), -sin(q), 0, a],
                 [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                 [sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha), cos(alpha) * d],
                 [0, 0, 0, 1]])
    return TF

def jointState(arg):
    joint_var = []
    for i in range(0, 7):
        joint_var.append((arg[i]))

    dh_parameters = dh_params(joint_var)

    T_01 = TF_matrix(0, dh_parameters)
    T_12 = TF_matrix(1, dh_parameters)
    T_23 = TF_matrix(2, dh_parameters)
    T_34 = TF_matrix(3, dh_parameters)
    T_45 = TF_matrix(4, dh_parameters)
    T_56 = TF_matrix(5, dh_parameters)
    T_67 = TF_matrix(6, dh_parameters)

    T_07 = T_01 * T_12 * T_23 * T_34 * T_45 * T_56 * T_67

    quaternions = R.from_matrix(T_07[:3, :3]).as_quat()
    pos_ee = [T_07[:3, 3][0], T_07[:3, 3][1], T_07[:3, 3][2] - 0.1]
    quat_ee = diff_quaternion("ee", quaternions)
    # Writing data to the Pose message for publishing
    # print(pos_ee)
    # print(quat_ee)
    return pos_ee, quat_ee


if __name__ == '__main__':
    main()
