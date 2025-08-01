import rclpy
import numpy as np
from rclpy.node import Node
from px4_msgs.msg import VehicleAttitude, VehicleAngularVelocity, VehicleLocalPosition
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, TransformException

qos_profile_sub = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)




class TF_Tester(Node):
    def __init__(self):
        super().__init__("tf_tester")
        self.get_logger().info("TF Tester node has been started.")

        self.namespace = self.declare_parameter("namespace", "").value
        self.namespace_prefix = f"/{self.namespace}" if self.namespace else ""

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

         # Set up the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.check_tf)

    def check_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "camera_optical_frame",
                "map",
                rclpy.time.Time())
            print (f"Transform from camera_link to map: {t.transform.translation.x}, {t.transform.translation.y}, {t.transform.translation.z}")
            print (f"Rotation: {t.transform.rotation.x}, {t.transform.rotation.y}, {t.transform.rotation.z}, {t.transform.rotation.w}")
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform camera_optical_frame to map: {ex}')
            return


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
    node = TF_Tester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Pose forwarder stopped by user")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
