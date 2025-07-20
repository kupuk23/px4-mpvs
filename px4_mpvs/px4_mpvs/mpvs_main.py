#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Pedro Roque, Jaeyoung Lim"
__contact__ = "padr@kth.se, jalim@ethz.ch"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from std_msgs.msg import Float32MultiArray, Int8
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
from px4_mpvs.utils.ros_utils import lookup_transform
from tf2_geometry_msgs import do_transform_pose

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import VehicleTorqueSetpoint
from px4_msgs.msg import VehicleThrustSetpoint
from vs_msgs.msg import ServoPoses

from px4_mpvs.models.spacecraft_vs_model import SpacecraftVSModel
from px4_mpvs.controllers.spacecraft_mpvs_controller import SpacecraftVSMPC


from mpc_msgs.srv import SetPose
from vs_msgs.srv import SetHomePose

from px4_mpvs.docking_state_machine import docking_state_machine

from time import perf_counter


class SpacecraftIBMPVS(Node):

    def __init__(self):
        super().__init__("spacecraft_mpvs")

        self.build = True  # Set to False after the first run to avoid rebuilding
        self.sitl = False

        self.aligning_threshold = 0.2

        self.camera_frame_id = self.declare_parameter(
            "camera_frame_id", "camera_link"
        ).value

        # flattened 2d coordinates of the desired points (4x2)
        self.desired_points = np.array(
            [[82, 123], [563, 123], [176, 337], [505, 218]]
        ).flatten()

        

        # Get namespace
        self.namespace = self.declare_parameter("namespace", "").value
        self.namespace_prefix = f"/{self.namespace}" if self.namespace else ""

        # Get setpoint from rviz (true/false)
        self.setpoint_from_rviz = self.declare_parameter(
            "setpoint_from_rviz", True
        ).value

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        # Setup publishers and subscribers
        self.set_publishers_subscribers(qos_profile_pub, qos_profile_sub)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        # self.setpoint_position = np.array([0.0, 0.0, 0.0])
        # self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])

        # first setpoint #
        self.setpoint_position = np.array([2.5, 0.0, 0.0])  # inverted z and y axis
        self.setpoint_attitude = np.array([0.0, 0.0, 0.0, 1.0])  # invered z and y axis, default = np.array([1.0, 0.0, 0.0, 0.0]) 

        self.p_obj = np.array([-100.0, 0.0, 0.0])  # object position in map
        self.p_markers = np.array([100, 100, 400, 100, 100, 300, 400, 300])
        self.Z = np.array([1.0, 1.0, 1.0, 1.0])  # Z coordinates of the markers
        self.statistics = {
            "recorded_features": [],
            "recorded_wp": [],
            "recorded_ws": [],
            "features_error": [],
            "desired_points": self.desired_points,
            "Vp_dot": [],
            "Vs_dot": [],
            "hybrid_duration": 0.0,  # duration of the hybrid control in seconds
            "full_docking_duration": 0.0,  # duration of the full docking in seconds
        }

        
        # TIMER RELATED #
        self.start_recording = False  # flag to start recording the statistics
        self.pre_dock_timer = perf_counter()  # Timer for docking
        self.start_full_docking_time = perf_counter()  # Timer for docking start
        self.hybrid_start_time = 0.0  # duration of the hybrid control in seconds
        self.pre_docked_time = 0  # Timer for pre-docking
        self.pre_docked_time_threshold = 2  # time to stabilize the robot before docking (seconds)
       

        self.aligning = False
        self.aligned = False  # flag to check if the robot is aligned
        self.markers_detected = False
        self.pre_docked = False
        self.docked = False
        self.aligned = False  # True if the robot is aligned with the object
        self.model = SpacecraftVSModel()
        self.mpc = SpacecraftVSMPC(self.model, build = self.build)
        self.mode = 0  # 0: PBVS, 1: hybrid, 2: IBVS
        self.hybrid_mode = "discrete" # "softmax" or "discrete" or "ratio"
        self.ibvs_e_threshold = 20
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def set_publishers_subscribers(self, qos_profile_pub, qos_profile_sub):


        self.mode_pub = self.create_publisher(Int8, f"{self.namespace_prefix}/servoing_mode", 10) 

        self.markers_sub = self.create_subscription(
            Float32MultiArray, "/detected_markers", self.marker_callback, 10
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            f"{self.namespace_prefix}/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile_sub,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            f"{self.namespace_prefix}/fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            qos_profile_sub,
        )
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            f"{self.namespace_prefix}/fmu/out/vehicle_angular_velocity",
            self.vehicle_angular_velocity_callback,
            qos_profile_sub,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f"{self.namespace_prefix}/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )

        self.set_servo_srv = self.create_service(
            SetHomePose,
            f"{self.namespace_prefix}/set_servo_pose",
            self.servo_srv_callback,
        )

        if self.setpoint_from_rviz:
            self.set_pose_srv = self.create_service(
                SetPose, f"{self.namespace_prefix}/set_pose", self.add_set_pos_callback
            )

        self.setpoint_pose_sub = self.create_subscription(
            ServoPoses,
            f"{self.namespace_prefix}/pbvs_pose",
            self.get_setpoint_pose_callback,
            0,
        )

        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            f"{self.namespace_prefix}/fmu/in/offboard_control_mode",
            qos_profile_pub,
        )
        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors,
            f"{self.namespace_prefix}/fmu/in/actuator_motors",
            qos_profile_pub,
        )

        self.predicted_path_pub = self.create_publisher(
            Path, f"{self.namespace_prefix}/px4_mpc/predicted_path", 10
        )
        self.reference_pub = self.create_publisher(
            Marker, f"{self.namespace_prefix}/px4_mpc/reference", 10
        )

        self.obj_pub = self.create_publisher(
            PoseStamped, f"{self.namespace_prefix}/pose_debug", 10
        )
        if self.sitl:
            self.odom_pub = self.create_publisher(
                Odometry, f"{self.namespace_prefix}/odom", qos_profile_pub
            )
        return

    def marker_callback(self, msg):
        # Receive the detected markers, unflatten the array into x y z

        # convert to int16 for the x and y coordinates
        points_3d = np.array(msg.data).reshape(-1, 3)
        self.p_markers = points_3d[:, 0:2].astype(np.int16).flatten()
        self.Z = points_3d[:, 2].astype(np.float16)

        self.markers_detected = True

    def vehicle_attitude_callback(self, msg):
        # NED-> ENU transformation
        # Receives quaternion in NED frame as (qw, qx, qy, qz)
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)

    def vehicle_local_position_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_direct_actuator_setpoint(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # NOTE:
        # Output is float[16]
        # u1 needs to be divided between 1 and 2
        # u2 needs to be divided between 3 and 4
        # u3 needs to be divided between 5 and 6
        # u4 needs to be divided between 7 and 8
        # positve component goes for the first, the negative for the second
        thrust = u_pred[0, :] / self.model.max_thrust  # normalizes w.r.t. max thrust
        # print("Thrust rates: ", thrust[0:4])

        thrust_command = np.zeros(12, dtype=np.float32)
        thrust_command[0] = 0.0 if thrust[0] <= 0.0 else thrust[0]
        thrust_command[1] = 0.0 if thrust[0] >= 0.0 else -thrust[0]

        thrust_command[2] = 0.0 if thrust[1] <= 0.0 else thrust[1]
        thrust_command[3] = 0.0 if thrust[1] >= 0.0 else -thrust[1]

        thrust_command[4] = 0.0 if thrust[2] <= 0.0 else thrust[2]
        thrust_command[5] = 0.0 if thrust[2] >= 0.0 else -thrust[2]

        thrust_command[6] = 0.0 if thrust[3] <= 0.0 else thrust[3]
        thrust_command[7] = 0.0 if thrust[3] >= 0.0 else -thrust[3]

        actuator_outputs_msg.control = thrust_command.flatten()
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def publish_sitl_odometry(self):
        msg = Odometry()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.header.stamp = Clock().now().to_msg()
        msg.pose.pose.position.x = self.vehicle_local_position[0]
        msg.pose.pose.position.y = self.vehicle_local_position[1]
        msg.pose.pose.position.z = self.vehicle_local_position[2]
        msg.pose.pose.orientation.w = self.vehicle_attitude[0]
        msg.pose.pose.orientation.x = self.vehicle_attitude[1]
        msg.pose.pose.orientation.y = self.vehicle_attitude[2]
        msg.pose.pose.orientation.z = self.vehicle_attitude[3]
        msg.twist.twist.linear.x = self.vehicle_local_velocity[0]
        msg.twist.twist.linear.y = self.vehicle_local_velocity[1]
        msg.twist.twist.linear.z = self.vehicle_local_velocity[2]
        msg.twist.twist.angular.x = self.vehicle_angular_velocity[0]
        msg.twist.twist.angular.y = self.vehicle_angular_velocity[1]
        msg.twist.twist.angular.z = self.vehicle_angular_velocity[2]
        self.odom_pub.publish(msg)
        return

    def cmdloop_callback(self):
        docking_state_machine(self)
        
        mode = Int8()
        mode.data = self.mode
        self.mode_pub.publish(mode)

    def add_set_pos_callback(self, request, response):
        self.update_setpoint(request)
        print("Setpoint from RVIZ: ", self.setpoint_position, self.setpoint_attitude)
        return response

    def get_setpoint_pose_callback(self, msg: ServoPoses):
        self.update_setpoint(msg.goal_pose_stamped)

        self.p_obj = np.array(
            [msg.obj_pose.position.x, msg.obj_pose.position.y, msg.obj_pose.position.z]
        )

        obj_pose = PoseStamped()
        obj_pose.header.frame_id = "map"
        obj_pose.header.stamp = Clock().now().to_msg()
        obj_pose.pose.position.x = msg.obj_pose.position.x
        obj_pose.pose.position.y = msg.obj_pose.position.y
        obj_pose.pose.position.z = msg.obj_pose.position.z
        obj_pose.pose.orientation.w = msg.obj_pose.orientation.w
        obj_pose.pose.orientation.x = msg.obj_pose.orientation.x
        obj_pose.pose.orientation.y = msg.obj_pose.orientation.y
        obj_pose.pose.orientation.z = msg.obj_pose.orientation.z
        self.obj_pub.publish(obj_pose)

    def update_setpoint(self, msg):
        self.setpoint_position[0] = msg.pose.position.x
        self.setpoint_position[1] = msg.pose.position.y
        self.setpoint_position[2] = msg.pose.position.z
        self.setpoint_attitude[0] = msg.pose.orientation.w
        self.setpoint_attitude[1] = msg.pose.orientation.x
        self.setpoint_attitude[2] = msg.pose.orientation.y
        self.setpoint_attitude[3] = msg.pose.orientation.z

    def servo_srv_callback(self, request: SetHomePose, response: SetHomePose.Response):
        if request.align_mode:
            self.start_docking_time = perf_counter()
            self.start_recording = True
            self.update_setpoint(request)
            self.aligning = True
            self.get_logger().info("Starting homing mode")
        else:
            self.hybrid_start_time = perf_counter()
            self.aligned = request.aligned
            self.aligning = False
            self.mode = 1
            
            self.get_logger().info("Robot is aligned, stopping homing mode")
            # update setpoint to be somewhere between the robot and object

            self.get_logger().info(
            f"OLD Setpoint position: {self.setpoint_position}"
        )

            self.setpoint_position = np.array(
                [
                    (self.vehicle_local_position[0] + self.p_obj[0]) / 2,
                    (self.vehicle_local_position[1] + self.p_obj[1]) / 2,
                    (self.vehicle_local_position[2] + self.p_obj[2]) / 2,
                ]
            )
        self.get_logger().info(
            f"NEW Setpoint position: {self.setpoint_position}"
        )
        self.mpc.update_constraints(self.aligning)

        self.p_obj = np.array([-100.0, 0, 0])

        return response


def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpvs = SpacecraftIBMPVS()

    rclpy.spin(spacecraft_mpvs)

    spacecraft_mpvs.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
