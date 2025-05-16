
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

class IBVSNode(Node):
    def __init__(self):
        super().__init__('ibvs_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, "camera/depth_image", self.depth_callback, 10
        )
        
        self.srv = self.create_service(
            SetBool, "enable_ibvs", self.enable_ibvs_callback
        )

        # Initialize depth image
        self.depth_image = None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.srv = self.create_service(SetBool, 'start_ibvs', self.start_ibvs_callback)
        self.is_running = False

        # IBVS control enabled by default
        self.ibvs_enabled = False