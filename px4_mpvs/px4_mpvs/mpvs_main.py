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

from std_msgs.msg import Float32MultiArray
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


from std_srvs.srv import SetBool
from mpc_msgs.srv import SetPose
from vs_msgs.srv import SetHomePose


from px4_mpvs.ibvs_controller import handle_ibvs_control
from px4_mpvs.pbvs_controller import handle_pbvs_control


class SpacecraftIBMPVS(Node):

    def __init__(self):
        super().__init__("spacecraft_ib_mpvs")

        self.servo_mode = "ibvs"  # pbvs or ibvs
        self.aligning_threshold = 0.2

        # self.srv = self.create_service(
        #     SetBool, "run_docking", self.docking_callback_enabled
        # )

        self.camera_frame_id = self.declare_parameter(
            "camera_frame_id", "camera_link"
        ).value

        # flattened 2d coordinates of the desired points (4x2)
        self.desired_points = np.array(
            [
                [99, 186],
                [535, 187],
                [190, 394],
                [481, 277],
            ]
        ).flatten()
        # Get mode; rate, wrench, direct_allocation
        self.mode = self.declare_parameter("mode", "direct_allocation").value
        self.sitl = True

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
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        # self.setpoint_position = np.array([0.0, 0.0, 0.0])
        # self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])

        # first setpoint #
        self.setpoint_position = np.array([0.0, 0.0, 0.0])  # inverted z and y axis
        self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])  # invered z and y axis

        self.p_obj = np.array([0.0, 0.0, 0.0])  # object position in map
        self.p_markers = np.array([0.0, 0.0])  # object position in camera frame
        self.Z = np.zeros(4)
        self.recorded_markers = np.zeros(8)  # for recording markers
        self.recorded_p_error = np.zeros(3)  # for recording position error

        self.aligning = False
        self.markers_detected = False
        self.pre_docked = False
        self.switch_mode(self.servo_mode)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def switch_mode(self, mode):
        if mode == "pbvs":
            self.get_logger().info("Loading PBVS controller")
            self.model = SpacecraftVSModel(mode="pbvs")
            self.mpc = SpacecraftVSMPC(self.model, mode="pbvs")
            self.servo_mode = "pbvs"
        else:
            self.get_logger().info("Loading IBVS controller")
            self.model = SpacecraftVSModel(mode="ibvs")
            self.mpc = SpacecraftVSMPC(self.model, Z=self.Z, mode="ibvs")
            self.servo_mode = "ibvs"

    def set_publishers_subscribers(self, qos_profile_pub, qos_profile_sub):

        # self.object_pose_sub = self.create_subscription(
        #     PoseStamped,
        #     '/pose/icp_result',  # Topic for object pose
        #     self.object_pose_callback,
        #     10
        # )

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
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # TODO: handle NED->ENU transformation
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

        # TODO: change wall into 4 big points and test the normal IBVS//
        if self.servo_mode == "ibvs" and self.markers_detected:
            handle_ibvs_control(self)
        elif self.servo_mode == "pbvs":
            handle_pbvs_control(self)

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
            self.update_setpoint(request)
            self.aligning = True
            self.get_logger().info("Starting homing mode")
        else:
            self.aligned = request.aligned
            self.aligning = False
            self.get_logger().info("Stopping homing mode")

        self.mpc.update_constraints(self.aligning)
        return response


def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpvs = SpacecraftIBMPVS()

    rclpy.spin(spacecraft_mpvs)

    spacecraft_mpvs.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
