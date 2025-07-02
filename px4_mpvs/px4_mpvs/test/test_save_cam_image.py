import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CompressedImage, Image
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge


filename = "docked_image_v4"


class SaveImage(Node):
    def __init__(self):
        super().__init__("save_image_node")

        # Create subscriber to the image topic
        self.subscription = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )
        
        

        # Initialize depth image
        self.depth_image = None

        # Initialize the OpenCV bridge
        self.bridge = CvBridge()

    def image_callback(self, msg=CompressedImage):
        try:
            # Convert ROS Image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite(
                f"/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/resource/{filename}.jpg",
                cv_image,
            )
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return


def main():
    rclpy.init()
    node = SaveImage()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Pose forwarder stopped by user")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
