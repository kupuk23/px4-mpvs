
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
        self.declare_parameter('use_compressed', False)
        self.use_compressed = self.get_parameter('use_compressed').value
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage if self.use_compressed else Image,
            '/camera/image_raw/compressed' if self.use_compressed else '/camera/image_raw',
            self.image_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.srv = self.create_service(SetBool, 'start_ibvs', self.start_ibvs_callback)
        self.is_running = False