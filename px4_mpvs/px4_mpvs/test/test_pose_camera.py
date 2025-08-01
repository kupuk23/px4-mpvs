import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from vs_msgs.srv import SetHomePose
from vs_msgs.msg import ServoPoses
from mpc_msgs.srv import SetPose
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import px4_mpvs.utils.ros_utils as ros_utils
from tf2_geometry_msgs import do_transform_pose
import tf_transformations as tft
import numpy as np
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import ParameterType, Parameter, SetParametersResult
from rclpy.time import Time
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
import px4_mpvs.utils.math_utils as math_utils


class VisualServo(Node):
    def __init__(self):
        super().__init__("visual_servo")

        self.namespace = self.declare_parameter("namespace", "pop").value
        self.namespace_prefix = f"/{self.namespace}" if self.namespace else ""

        # Create a subscription to the pose topic
        self.object_pose_sub = self.create_subscription(
            PoseStamped, "/pose/icp_result", self.pose_callback, 10
        )
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            f"{self.namespace_prefix}/fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            qos_profile_sub,
        )

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f"{self.namespace_prefix}/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )

        self.goal_pub = self.create_publisher(ServoPoses, f"{self.namespace_prefix}/pbvs_pose", 10)
        self.goal_posestamped_pub = self.create_publisher(
            PoseStamped, f"{self.namespace_prefix}/goal_pose_offset", 10)

        # Create a client for the set_pose service
        self.client_servo = self.create_client(
            SetHomePose, f"{self.namespace_prefix}/set_servo_pose"
        )

        self.client_set_pose = self.create_client(
            SetPose, f"{self.namespace_prefix}/set_pose"
        )
        # Wait for service to become available
        while not self.client_servo.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /set_servo_pose not available, waiting...")

        while not self.client_set_pose.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /set_pose not available, waiting...")

        # Robot pose in map
        self.vehicle_local_position = np.zeros(3)  # robot position in map
        self.vehicle_local_velocity = np.zeros(3)
        self.vehicle_attitude = np.zeros(4)  # robot attitude in map

        # Add pose history buffer for consistency checking
        self.goal_pose_history = []
        self.obj_pose_history = []
        self.history_size = 5
        self.goal_pose = Pose()
        self.position_threshold = 0.5
        self.orientation_threshold = 15.0
        self.is_pose_consistent = False
        self.last_consistent_goal_pose = None  # Last consistent pose
        self.last_consistent_obj_pose = None  # Last consistent pose

        # State variable for docking
        self.success_start_time = None
        self.success_duration_required = 4.0  # seconds
        self.docking_running = False
        self.docking_enabled = False
        self.x_offset = 0.7
        self.latest_time = self.get_clock().now()
        self.pose_obtained = False

        # Add TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create service for docking control
        self.srv = self.create_service(
            SetBool, "run_docking", self.docking_callback_enabled
        )

        self.get_logger().info(
            "Pose forwarder initialized. Use docking_enabled service to enable/disable."
        )

        # # spawn pose #1
        self.init_pos = np.array(
            [1.36114323, -0.23419029, 0.0]
        )  # inverted z and y axis
        self.init_att = np.array(
            [6.68084145e-01, 8.91955807e-08, 6.40660858e-10, 7.44085729e-01]
        )

        # [ 0.11974171 -1.50361025  0.35518408] [ 8.64499688e-01  7.21904883e-08 -3.61660879e-09  5.02633214e-01]

        # Spawn Pose #2
        # self.init_pos = np.array([0.11974171, -1.50361025, 0])
        # self.init_att = np.array(
        #     [8.64499688e-01, 7.21904883e-08, -3.61660879e-09, 5.02633214e-01]
        # )

        # Spawn pose Reversed
        # self.init_pos = np.array([1.56462193e-07, 1.405, 0])
        # self.init_att = np.array(
        #     [2.17068925e-01, 1.25027753e-07, -2.44279164e-07, 9.76156294e-01]
        # )

        self.param_client = self.create_client(
            SetParameters, "pose_estimation_pcl/set_parameters"
        )

        # Wait for service (add timeout for robustness)
        if not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                "pose_estimation_pcl parameter service not available"
            )
            return

        self.move_robot(self.init_pos, self.init_att)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.aligning_callback)

    def aligning_callback(self):
        current_time = self.get_clock().now()
        # time_diff = (
        #     current_time - self.latest_time
        # ).nanoseconds / 1e9  # Convert to seconds

        if self.docking_enabled and self.docking_running:
            # change params in pose_estimation_pcl using dynamic reconfigure
            self.check_docking_status()
            map_goal_pose = ros_utils.generate_pose_stamped(
                self.last_consistent_goal_pose, clock=self.get_clock()
            )
            msg = ServoPoses()
            msg.goal_pose_stamped = map_goal_pose
            msg.obj_pose = self.last_consistent_obj_pose

            self.goal_pub.publish(msg)

            # make  goal posestamped object and publish to the topic
            goal_posestamped = map_goal_pose
            self.goal_posestamped_pub.publish(goal_posestamped)

            # if time_diff > 1:
            #     self.enable_goicp(False)
            #     self.reset_all_variables()

    def check_docking_status(self):
        # Check if the robot is already arrived at the goal pose by comparing self.consistent_goal with robot pose
        goal_position = np.array(
            [
                self.last_consistent_goal_pose.position.x,
                self.last_consistent_goal_pose.position.y,
                self.last_consistent_goal_pose.position.z,
            ]
        )
        position_err = math_utils.calc_position_error(
            self.vehicle_local_position, goal_position
        )

        goal_orientation = np.array(
            [
                self.last_consistent_goal_pose.orientation.w,
                self.last_consistent_goal_pose.orientation.x,
                self.last_consistent_goal_pose.orientation.y,
                self.last_consistent_goal_pose.orientation.z,
            ]
        )

        orientation_err = math_utils.calculate_orientation_error(
            self.vehicle_attitude, goal_orientation
        )

        #TODO : FIX ORIENTATION ERROR CALCULATION BASED ON QUATERNIONS
        print(f"position error: {position_err:.2f} m")
        print(f"orientation error: {orientation_err:.2f} degrees")

        # Check if we're within thresholds
        if (
            position_err < self.position_threshold
            and orientation_err < self.orientation_threshold
        ):
            current_time = self.get_clock().now()

            # Start the timer if not already started
            if self.success_start_time is None:
                self.success_start_time = current_time
                self.get_logger().info(
                    "Position and orientation within thresholds, starting timer"
                )

            # Check if we've been within thresholds long enough
            elapsed_time = (current_time - self.success_start_time).nanoseconds / 1e9
            if elapsed_time >= self.success_duration_required:
                self.get_logger().info(
                    f"Docking completed successfully! "
                    + f"Position error: {position_err:.2f} m, "
                    + f"Orientation error: {orientation_err:.2f} degrees"
                )
                self.reset_all_variables()
        else:
            # Reset timer if we go outside thresholds
            if self.success_start_time is not None:
                self.get_logger().info(
                    "Position or orientation outside thresholds, resetting timer"
                )
                self.success_start_time = None

    def reset_all_variables(self):
        self.docking_running = False
        self.docking_enabled = False
        self.enable_goicp(False)
        self.pose_obtained = False
        self.goal_pose_history = []
        self.obj_pose_history = []
        # self.last_consistent_goal_pose = None

        self.stop_aligning(aligned=True)

    def stop_aligning(self, aligned=False):
        req = SetHomePose.Request()
        req.pose = Pose()
        req.align_mode = False
        req.aligned = aligned

        future = self.client_servo.call_async(req)
        future.add_done_callback(self.service_callback)

    def enable_goicp(self, enable):
        # Create parameter objects
        # Create request
        request = SetParameters.Request()
        param = Parameter()
        param.name = "go_icp.use_goicp"
        param.value.type = ParameterType.PARAMETER_BOOL
        param.value.bool_value = enable
        request.parameters = [param]

        # Send the request
        future = self.param_client.call_async(request)
        future.add_done_callback(self.param_callback)

    def param_callback(self, future):
        try:
            response = future.result()
            for result in response.results:
                if result.successful:
                    self.get_logger().info(f"Parameter set successfully")
                else:
                    self.get_logger().warn(f"Failed to set parameter: {result.reason}")
        except Exception as e:
            self.get_logger().error(f"Parameter setting failed: {e}")

    def move_robot(self, position, orientation):
        init_poseStamped = PoseStamped()
        init_poseStamped.header.frame_id = "map"
        init_poseStamped.header.stamp = self.get_clock().now().to_msg()
        init_poseStamped.pose.position.x = position[0]
        init_poseStamped.pose.position.y = position[1]
        init_poseStamped.pose.position.z = position[2]
        init_poseStamped.pose.orientation.w = orientation[0]
        init_poseStamped.pose.orientation.x = orientation[1]
        init_poseStamped.pose.orientation.y = orientation[2]
        init_poseStamped.pose.orientation.z = orientation[3]

        request = SetPose.Request()
        request.pose = init_poseStamped.pose

        # Send the request asynchronously
        future = self.client_set_pose.call_async(request)
        future.add_done_callback(self.service_callback)

    def docking_callback_enabled(self, request, response):
        """Service callback to enable/disable pose forwarding"""
        self.docking_enabled = request.data
        response.success = True
        if self.docking_enabled:
            self.enable_goicp(True)  # Enable GOICP
            response.message = "Docking mode enabled"
            self.get_logger().info("Docking mode enabled")
        else:
            response.message = "Docking mode disabled"
            self.get_logger().info("Docking mode disabled")
        return response

    def pose_callback(self, msg):
        # Get current ROS time
        self.latest_time = self.get_clock().now()

        map_goal_pose, map_obj_pose = ros_utils.generate_goal_from_object_pose(
            msg.pose,
            self.tf_buffer,
            self.x_offset,
            self.get_clock(),
        )
        if map_goal_pose is None:
            self.get_logger().warn("Failed to transform pose, skipping")
            return

        # Store the pose in history for consistency check
        self.obj_pose_history.append(map_obj_pose)
        self.goal_pose_history.append(map_goal_pose.pose)
        if len(self.goal_pose_history) > self.history_size:
            self.goal_pose_history.pop(0)
            self.obj_pose_history.pop(0)

        self.is_pose_consistent = self.check_pose_consistency()

        # Only forward the pose if it's consistent and docking_enabled is True
        if (
            self.docking_enabled
            and self.is_pose_consistent
            and not self.docking_running
        ):
            # self.docking_enabled = False
            self.get_logger().info("Pose consistent, forwarding to align service")

            if map_goal_pose is None:
                self.get_logger().warn("Failed to transform pose, skipping")
                return

            request = SetHomePose.Request()
            request.pose = map_goal_pose.pose
            request.align_mode = True

            # Send the request asynchronously
            future = self.client_servo.call_async(request)
            future.add_done_callback(self.service_callback)
            self.docking_running = True

        elif not self.docking_enabled:
            self.get_logger().debug("docking_enabled is False, not forwarding pose")
        elif not self.is_pose_consistent:
            self.get_logger().debug("Pose not consistent yet, waiting for stability")

    def check_pose_consistency(self):
        """Check if the recent poses are consistent within threshold"""

        consistent_pos_tresh = 0.1
        consistent_orient_tresh = 3.0
        if len(self.goal_pose_history) < self.history_size:
            return False

        # Check the last self.history_size poses
        reference_pose = self.goal_pose_history[-1]
        for i in range(0, -self.history_size, -1):

            # if (
            #     i == -self.history_size
            # ):  # add a check for last_consistent_pose on the last iteration
            #     if self.last_consistent_goal_pose is None:
            #         break
            #     pose = self.last_consistent_goal_pose

            # else:
            pose = self.goal_pose_history[i]

            position_err = math_utils.calc_position_error(
                [
                    reference_pose.position.x,
                    reference_pose.position.y,
                    reference_pose.position.z,
                ],
                [pose.position.x, pose.position.y, pose.position.z],
            )
            orientation_err = math_utils.calculate_orientation_error(
                [
                    reference_pose.orientation.w,
                    reference_pose.orientation.x,
                    reference_pose.orientation.y,
                    reference_pose.orientation.z,
                ],
                [
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ],
            )

            if (
                position_err > consistent_pos_tresh
                or orientation_err > consistent_orient_tresh
            ):
                print(
                    f"Pose error > tresh: {position_err:.2f} > {consistent_pos_tresh:.2f},"
                )
                print(
                    f"Orientation error > tresh : {orientation_err:.2f} > {consistent_orient_tresh:.2f}"
                )
                self.pose_obtained = False
                return False

        self.last_consistent_goal_pose = self.goal_pose_history[-1]
        self.last_consistent_obj_pose = self.obj_pose_history[-1]

        self.pose_obtained = True
        return True

    def service_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Service call succeeded. Response: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

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


def main():
    rclpy.init()
    node = VisualServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Pose forwarder stopped by user")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
