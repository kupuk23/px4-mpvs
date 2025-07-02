import numpy as np
from px4_mpvs.utils.ros_utils import vector2PoseMsg

from rclpy.clock import Clock
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import OffboardControlMode
from nav_msgs.msg import Path, Odometry


from px4_mpvs.utils.plot_utils import plot_stats
from px4_mpvs.utils import math_utils
from time import perf_counter


def handle_hybrid_control(node):

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
    s_d = node.desired_points.flatten()
    p_markers = node.p_markers

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
            *p_markers,
        ]
    ).reshape(21, 1)

    ref = np.concatenate(
        (
            node.setpoint_position,  # position
            np.zeros(3),  # velocity
            node.setpoint_attitude,  # attitude
            np.zeros(3),  # angular velocity
            s_d,  # desired points
            np.zeros(4),  # inputs reference (u1, ..., u4) for 2D platform
        ),
        axis=0,
    )

    ref = np.repeat(ref.reshape((-1, 1)), node.mpc.N + 1, axis=1)

    t_start = perf_counter()

    if node.docked:
        if node.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            u_pred = np.zeros((node.mpc.N + 1, 4))
            node.publish_direct_actuator_setpoint(u_pred)
        return
    
    # Solve MPC
    if not node.aligned:
        u_pred, x_pred, w_p,w_s = node.mpc.solve(
            x0, verbose=True, ref=ref, p_obj=node.p_obj, Z=node.Z
        )

    elif node.aligned and not node.pre_docked:
        u_pred, x_pred,w_p,w_s = node.mpc.solve(
            x0, verbose=True, ref=ref, p_obj=node.p_obj, Z=node.Z, hybrid_mode=1.0
        )  # TODO: add hybrid flag to use dynamic weight

        # debug reference and current image state
        feature_current = x0[13:21].flatten()  # Current features
        feature_desired = ref[13:21, 0].flatten()  # Desired features
        error = np.linalg.norm(feature_current - feature_desired)
        node.statistics["recorded_features"].append(feature_current)
        node.statistics["features_error"].append(error)
        # print(f"Feature errors: {error}")

        if error < node.ibvs_e_threshold:
            current_time = perf_counter()
            if not node.pre_docked_time:
                node.pre_docked_time = current_time
                
            elapsed_time = current_time - node.pre_docked_time
            node.get_logger().info(
                    f"features are close enough, stabilizing... {int(elapsed_time)}"
                )
            if elapsed_time > node.pre_docked_time_threshold:
                print("Features are close enough, stopping servoing")
                node.pre_docked = True
                node.pre_docked_time = 0
                node.dock_timer = perf_counter()  # Start the dock timer
        else:
            node.pre_docked_time = 0


    if not node.pre_docked:
        w_p = float(w_p)
        w_s = float(w_s)
        # TODO: Add vs_dot and vp_dot to statistics
        node.statistics["recorded_wp"].append(w_p)
        node.statistics["recorded_ws"].append(w_s)

    t_stop = perf_counter()
        

    mpc_time = t_stop - t_start
    # node.get_logger().info(f"MPC update freq = {(1 / mpc_time):.2f} Hz")

    if node.pre_docked and not node.docked:
        # run this for n seconds to ensure the spacecraft is docked
        current_time = perf_counter()
        if current_time - node.dock_timer > 3:
            node.docked = True
            print("Docking complete")
            # plot_stats(node.statistics)


        
        u_pred = np.zeros((node.mpc.N + 1, 4))
        u_pred[:, 0] = -0.1
        u_pred[:, 1] = -0.1
        x_pred = x0.reshape(1, -1).repeat(node.mpc.N + 1, axis=0)

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

