import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int8
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CompressedImage  
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)
from rclpy.duration import Duration

from px4_mpvs.marker_detector_blob import CircleFeatureDetector

# import matplotlib

# matplotlib.use("Agg")  # Use TkAgg backend for matplotlib


class MarkerDetectorNode(Node):
    def __init__(self):

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        super().__init__("marker_detector_node")
        self.bridge = CvBridge()

        self.save_image = False
        # Get parameters
    
        self.debug = self.declare_parameter("debug", True).value
        self.visualize = self.declare_parameter("visualize", True).value
        self.namespace = self.declare_parameter("namespace", "").value
        self.namespace_prefix = f"/{self.namespace}" if self.namespace else ""

        # Debug window
        # if self.visualize:
        #     self.window_name = "IBVS Controller"
        #     cv2.namedWindow(self.window_name)

        self.mode_sub = self.create_subscription(
            Int8, f"{self.namespace_prefix}/servoing_mode", self.mode_callback, 10
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            # f"{self.namespace_prefix}/camera/image/compressed",
            "/zed/zed_node/rgb/image_rect_color/compressed",
            self.image_callback,
            qos)

        self.depth_sub = self.create_subscription(
            Image,
            # f"{self.namespace_prefix}/camera/depth_image",
            "/zed/zed_node/depth/depth_registered",
            self.depth_callback,
            10
        )

        # self.srv = self.create_service(
        #     SetBool, f"{self.namespace_prefix}/enable_ibvs", self.enable_ibvs_callback
        # )

        self.markers_pub = self.create_publisher(
            Float32MultiArray, f"{self.namespace_prefix}/detected_markers", 10
        )

        # Initialization for IBVS
        self.depth_image = None
        self.ibvs_enabled = False
        self.marker_pos = None
        self.mode = "PBVS"
        self.mode_color = (255, 0, 0)  # Green for PBVS

        # Initialize detector
        # self.get_logger().info(f'Initializing marker detector with reference image: {reference_image_}')
        # self.detector = CircleMarkerDetector(reference_image_, expected_markers=4, debug=self.debug, vis_debug=self.debug )
        self.detector = CircleFeatureDetector(
            min_circle_radius=10,
            max_circle_radius=100,
            circularity_threshold=0.80,
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
            dtype=np.int16,
        )

        # Set target points in the detector
        self.detector.set_target_points(self.target_points)

        print(f"MarkerDetectorNode initialized with namespace: {self.namespace}")

    #     if self.visualize:
    #         self.create_timer(0.05, self.update_gui)

    # def update_gui(self):
    #     try:
    #         cv2.waitKey(1)
    #     except Exception as e:
    #         self.get_logger().error(f"Error in update_gui: {e}")

    def mode_callback(self, msg: Int8):
        """Callback to handle mode changes"""
        if msg.data == 0:
            # self.get_logger().info("MPC running point tracking mode (PBVS)")
            self.mode = "PBVS"
            self.mode_color = (0, 0, 0)  # Blue for PBVS
        elif msg.data == 1:
            # self.get_logger().info("MPC running Hybrid mode")
            self.mode = "Hybrid"
            self.mode_color = (0, 0, 0)  # Green for Hybrid
        elif msg.data == 2:
            # self.get_logger().info("MPC running IBVS mode")
            self.mode = "IBVS"
            self.mode_color = (255, 0, 0)  # Red for IBVS

    # def enable_ibvs_callback(self, request, response):
    #     """Service callback to enable/disable IBVS control"""
    #     self.ibvs_enabled = request.data

    #     if self.ibvs_enabled:
    #         self.get_logger().info("IBVS control enabled")
    #         response.message = "IBVS control enabled"
    #     else:
    #         self.get_logger().info("IBVS control disabled")
    #         # Send zero velocity to stop the robot
    #         self.stop_robot()
    #         response.message = "IBVS control disabled"

    #     response.success = True
    #     return response

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

            if self.save_image:
                cv2.imwrite("/home/discower/tafarrel_ws/src/px4-mpvs/px4_mpvs/resource/test_image.jpg", image)

            # DISPLAY IMAGE FOR DEBUGGING
            

            # Detect circles
            markers, viz_img = self.detector.detect(image)

            if markers is not None and len(markers) == 4:
                # self.get_logger().info(f"Detected {len(markers)} markers")
                Z = np.array(
                    [
                        self.depth_image[int(point[1]), int(point[0])]
                        for point in markers
                    ]
                )

                for i, center in enumerate(markers):
                    offset = 65 if i % 2 == 0 else 0
                    cv2.putText(
                        viz_img,
                        "("
                        + str(center[0])
                        + ","
                        + str(center[1])
                        + ","
                        + "{:.2f}".format(Z[i])
                        + ")",
                        (int(center[0]) - offset, int(center[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

                    # add the self.mode name to the image
                    cv2.putText(
                        viz_img,
                        f"Mode: {self.mode}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        self.mode_color,
                        2,
                    )

                    cv2.imshow("Detected Markers", viz_img)
                    cv2.waitKey(1)

                # append the Z values to the markers
                markers = np.hstack((markers, Z.reshape(-1, 1)))

                # send the markers and depth to the IBVS controller
                markers_msg = Float32MultiArray()
                markers_msg.data = markers.flatten()

                self.markers_pub.publish(markers_msg)
            else:
                # display the raw image if no markers are detected

                cv2.putText(
                    image,
                    f"Mode: {self.mode}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                cv2.imshow("Detected Markers", image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
