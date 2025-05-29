import numpy as np
from px4_mpvs.utils.ros_utils import vector2PoseMsg

from rclpy.clock import Clock
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import OffboardControlMode
from nav_msgs.msg import Path, Odometry


def handle_pbvs_control(node):

    # Publish odometry for SITL
    if node.sitl:
        node.publish_sitl_odometry()

    # Publish offboard control modes
    offboard_msg = OffboardControlMode()
    offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
    offboard_msg.position = False
    offboard_msg.velocity = False
    offboard_msg.acceleration = False
    offboard_msg.attitude = False
    offboard_msg.body_rate = False
    offboard_msg.direct_actuator = True
    node.publisher_offboard_mode.publish(offboard_msg)

    # Set state and references for each MPC

    x0 = np.array(
        [
            node.vehicle_local_position[0],
            node.vehicle_local_position[1],
            node.vehicle_local_position[2],
            node.vehicle_local_velocity[0],
            node.vehicle_local_velocity[1],
            node.vehicle_local_velocity[2],
            node.vehicle_attitude[0],
            node.vehicle_attitude[1],
            node.vehicle_attitude[2],
            node.vehicle_attitude[3],
            node.vehicle_angular_velocity[0],
            node.vehicle_angular_velocity[1],
            node.vehicle_angular_velocity[2],
        ]
    ).reshape(13, 1)

    ref = np.concatenate(
        (
            node.setpoint_position,  # position
            np.zeros(3),  # velocity
            node.setpoint_attitude,  # attitude
            np.zeros(3),  # angular velocity
            np.zeros(4),  # inputs reference (u1, ..., u4) for 2D platform
        ),
        axis=0,
    )

    ref = np.repeat(ref.reshape((-1, 1)), node.pbvs_mpc.N + 1, axis=1)

    # Solve MPC
    if not node.aligning:
        u_pred, x_pred = node.pbvs_mpc.solve(x0, ref=ref)
    elif np.any(node.p_obj):
        u_pred, x_pred = node.pbvs_mpc.solve(x0, ref=ref, p_obj=node.p_obj)
    else:
        return
    # Colect data
    idx = 0
    predicted_path_msg = Path()
    for predicted_state in x_pred:
        idx = idx + 1
        # Publish time history of the vehicle path
        predicted_pose_msg = vector2PoseMsg(
            "map", predicted_state[0:3], node.setpoint_attitude
        )
        predicted_path_msg.header = predicted_pose_msg.header
        predicted_path_msg.poses.append(predicted_pose_msg)
    node.predicted_path_pub.publish(predicted_path_msg)
    node.publish_reference(node.reference_pub, node.setpoint_position)

    if node.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:

        node.publish_direct_actuator_setpoint(u_pred)
