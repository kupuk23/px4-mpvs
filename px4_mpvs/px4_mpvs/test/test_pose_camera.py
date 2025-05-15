import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import rclpy.parameter
import rclpy.parameter_client
from vs_msgs.srv import SetServoPose
from mpc_msgs.srv import SetPose
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import px4_mpc.utils.ros_utils as ros_utils
from tf2_geometry_msgs import do_transform_pose
import tf_transformations as tft
import numpy as np
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import ParameterType, Parameter, SetParametersResult



class VisualServo(Node):
    def __init__(self):
        super().__init__("visual_servo")

        self.namespace = self.declare_parameter("namespace", "").value
        self.namespace_prefix = f"/{self.namespace}" if self.namespace else ""

        # Create a subscription to the pose topic
        self.object_pose_sub = self.create_subscription(
            PoseStamped, "/pose/icp_result", self.pose_callback, 10
        )

        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Create a client for the set_pose service
        self.client_servo = self.create_client(
            SetServoPose, f"{self.namespace_prefix}/set_servo_pose"
        )

        self.client_set_pose = self.create_client(
            SetPose, f"{self.namespace_prefix}/set_pose"
        )
        # Wait for service to become available
        while not self.client_servo.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /set_servo_pose not available, waiting...")

        while not self.client_set_pose.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /set_pose not available, waiting...")

        # Add pose history buffer for consistency checking
        self.pose_history = []
        self.history_size = 5
        self.position_threshold = 0.05

        # State variable for docking
        self.run_docking = False
        self.x_offset = 0.8

        # Add TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create service for docking control
        self.srv = self.create_service(
            SetBool, "run_docking", self.run_docking_callback
        )

        self.get_logger().info(
            "Pose forwarder initialized. Use run_docking service to enable/disable."
        )

        # # spawn pose #1
        # self.init_pos = np.array(
        #     [-0.01404785, 1.36248684, 0.0]
        # )  # inverted z and y axis
        # self.init_att = np.array(
        #     [8.92433882e-01, -5.40154197e-08, 4.97020096e-08, -4.51177984e-01]
        # )  # invered z and y axis

        # Spawn Pose #2
        self.init_pos = np.array([0.10324634, -1.0295148, 0])
        self.init_att = np.array(
            [9.52709079e-01, 2.49832297e-08, 2.80198997e-09, 3.03883404e-01]
        )

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
        self.timer = self.create_timer(timer_period, self.servoing_callback)

    def servoing_callback(self):
        if self.run_docking:
            # change params in pose_estimation_pcl using dynamic reconfigure
            self.enable_goicp(True)

        else:
            pass

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
            response  = future.result()
            for result in response.results:
                if result.successful:
                    self.get_logger().info(f"Parameter set successfully")
                else:
                    self.get_logger().warn(
                        f"Failed to set parameter: {result.reason}"
                    )
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

        self.goal_pose_pub.publish(init_poseStamped)
        request = SetPose.Request()
        request.pose = init_poseStamped.pose

        # Send the request asynchronously
        future = self.client_set_pose.call_async(request)
        future.add_done_callback(self.service_callback)

    def run_docking_callback(self, request, response):
        """Service callback to enable/disable pose forwarding"""
        self.run_docking = request.data
        response.success = True
        if self.run_docking:
            response.message = "Docking mode enabled"
            self.get_logger().info("Docking mode enabled")
        else:
            response.message = "Docking mode disabled"
            self.get_logger().info("Docking mode disabled")
        return response

    def pose_callback(self, msg):
        # transformed_pose_stamped = None
        # Store the pose in history for consistency check
        self.pose_history.append(msg.pose)

        # Keep only the most recent poses
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)

        # Only forward the pose if it's consistent and run_docking is True
        if self.run_docking and self.is_pose_consistent():
            # self.run_docking = False
            self.get_logger().info("Pose consistent, transforming to map frame")

            T_cam_obj = ros_utils.pose_to_matrix(msg.pose)
            # Transform the pose to map frame and apply offset/rotation
            T_cam_goal = ros_utils.offset_in_front(T_cam_obj, self.x_offset)

            # transform = self.tf_buffer.lookup_transform(
            #         "map",  # target frame
            #         msg.header.frame_id,  # source frame
            #         rclpy.time.Time(),  # get the latest transform
            #         rclpy.duration.Duration(seconds=1.0),  # timeout
            #     )

            transform = ros_utils.lookup_transform(
                self.tf_buffer, "map", msg.header.frame_id
            )
            if transform is None:
                self.get_logger().error("Failed to get transform from camera to map")
                return

            transformed_pose_stamped = ros_utils.matrix_to_posestamped(
                T_cam_goal, transform, "map", self.get_clock().now().to_msg()
            )

            self.goal_pose_pub.publish(transformed_pose_stamped)
            request = SetServoPose.Request()
            request.pose = transformed_pose_stamped.pose
            request.servoing_mode = True

            # Send the request asynchronously
            future = self.client_servo.call_async(request)
            future.add_done_callback(self.service_callback)

        elif not self.run_docking:
            self.get_logger().debug("run_docking is False, not forwarding pose")
        elif not self.is_pose_consistent():
            # self.run_docking = False
            # self.enable_goicp(False)
            self.get_logger().debug("Pose not consistent yet, waiting for stability")

    def is_pose_consistent(self):
        """Check if the recent poses are consistent within threshold"""
        if len(self.pose_history) < self.history_size:
            return False

        # Check the last self.history_size poses
        reference_pose = self.pose_history[-1]
        for i in range(-2, -self.history_size - 1, -1):
            pose = self.pose_history[i]

            # Calculate Euclidean distance between positions
            dx = reference_pose.position.x - pose.position.x
            dy = reference_pose.position.y - pose.position.y
            dz = reference_pose.position.z - pose.position.z
            distance = (dx**2 + dy**2 + dz**2) ** 0.5

            if distance > self.position_threshold:
                print(
                    f"Pose inconsistency detected: {distance:.2f} > {self.position_threshold:.2f}"
                )

                return False

        return True

    def service_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Service call succeeded. Response: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


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
