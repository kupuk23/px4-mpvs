import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.linalg import pinv
from ibvs_testing.detect_features import detect_circle_features, detect_lines
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from px4_mpvs.marker_detector_blob import CircleFeatureDetector


class MarkerDetectorNode(Node):
    def __init__(self):
        super().__init__("marker_detector_node")
        self.bridge = CvBridge()

        # Get parameters

        self.debug = self.declare_parameter("debug", True).value
        self.visualize = self.declare_parameter("visualize", True).value

        # Debug window
        # if self.visualize:
        #     self.window_name = "IBVS Controller"
        #     cv2.namedWindow(self.window_name)

        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, "camera/depth_image", self.depth_callback, 10
        )

        self.srv = self.create_service(
            SetBool, "enable_ibvs", self.enable_ibvs_callback
        )

        # Create publisher for detected markers
        self.marker_pub = self.create_publisher(Image, "/detected_markers", 10)

        # Initialization for IBVS
        self.depth_image = None
        self.ibvs_enabled = False
        self.marker_pos = None

        # Initialize detector
        # self.get_logger().info(f'Initializing marker detector with reference image: {reference_image_}')
        # self.detector = CircleMarkerDetector(reference_image_, expected_markers=4, debug=self.debug, vis_debug=self.debug )
        self.detector = CircleFeatureDetector(
            min_circle_radius=5,
            max_circle_radius=100,
            circularity_threshold=0.9,
            match_threshold=5.0,
            visualize=self.visualize,
            debug=self.debug,
        )

        self.target_points = np.array(
            [
                [210, 153],  # top-left
                [440, 155],  # top-right
                [210, 407],  # bottom-left
                [440, 404],  # bottom-right
            ],
            dtype=np.float32,
        )

        # Set target points in the detector
        self.detector.set_target_points(self.target_points)

    #     if self.visualize:
    #         self.create_timer(0.05, self.update_gui)

    # def update_gui(self):
    #     try:
    #         cv2.waitKey(1)
    #     except Exception as e:
    #         self.get_logger().error(f"Error in update_gui: {e}")

    def enable_ibvs_callback(self, request, response):
        """Service callback to enable/disable IBVS control"""
        self.ibvs_enabled = request.data

        if self.ibvs_enabled:
            self.get_logger().info("IBVS control enabled")
            response.message = "IBVS control enabled"
        else:
            self.get_logger().info("IBVS control disabled")
            # Send zero velocity to stop the robot
            self.stop_robot()
            response.message = "IBVS control disabled"

        response.success = True
        return response

    def depth_callback(self, msg: Image):
        # Convert the depth image to a numpy array
        # Convert ROS Image message to OpenCV image, with encoding "32FC1"
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="32FC1"
        )  # Float format -> in meters

    def image_callback(self, msg: CompressedImage):
        # Process the image and detect markers
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            # Detect circles
            markers = self.detector.detect(image)
            if markers is not None and len(markers) == 4:
                self.get_logger().debug(f"Detected 4 markers")

                # Draw circles on image for visualization
                viz_img = image.copy()
                for i, (x, y) in enumerate(markers):
                    cv2.circle(viz_img, (int(x), int(y)), 10, (0, 0, 255), -1)
                    cv2.putText(
                        viz_img,
                        str(i),
                        (int(x) + 15, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Publish visualization
                self.marker_pub.publish(self.bridge.cv2_to_imgmsg(viz_img, "bgr8"))
            else:
                self.get_logger().warn("Failed to detect all 4 markers")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def display_visualization(self, image):
        """Display visualization of the IBVS controller."""
        # Create a copy for visualization
        viz_img = image.copy()

        # Draw the current points
        for i, point in enumerate(self.marker_pos):
            cv2.circle(viz_img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.putText(
                viz_img,
                f"{i+1}",
                (int(point[0]) + 10, int(point[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Show error information
        # error = np.linalg.norm(self.marker_pos.flatten() - self.p_desired.flatten())
        # cv2.putText(
        #     viz_img,
        #     f"Error: {error:.2f} px",
        #     (20, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 0, 0),
        #     2,
        # )

        # Display the image
        cv2.imshow(self.window_name, viz_img)
        # cv2.imshow("depth image",self.depth_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
