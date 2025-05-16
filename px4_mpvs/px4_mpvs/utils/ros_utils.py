import numpy as np
import tf_transformations as tft
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import rclpy
import tf_transformations as tft

def generate_pose_stamped(pose, clock, frame_id = 'map' ):
    """
    Generate a PoseStamped message from a pose and frame ID.
    
    Parameters:
    -----------
    pose : Pose
        The pose to be converted
    frame_id : str
        The frame ID for the pose
    stamp : Time, optional
        The timestamp for the pose, if None uses the current time
        
    Returns:
    --------
    PoseStamped
        The generated PoseStamped message
    """
    msg = PoseStamped()
    msg.header.stamp = clock.now().to_msg()
    msg.header.frame_id = frame_id
    msg.pose = pose

    return msg

def generate_goal_from_object_pose(object_pose,tf_buffer, x_offset, clock):
    """
    Generate a goal pose from an object's pose by applying an offset and transforming to map frame.
    
    Parameters:
    -----------
    object_pose : PoseStamped
        The pose of the object in camera frame
    tf_buffer : tf2_ros.Buffer
        TF buffer for coordinate transformations
    x_offset : float
        How far in front of the object to place the goal (meters)
    clock : rclpy.clock.Clock, optional
        ROS clock for timestamp, if None uses the timestamp from object_pose
        
    Returns:
    --------
    PoseStamped
        The goal pose in map frame, or None if transformation failed
    """
    frame_id = "camera_link"  # Replace with the actual frame ID if needed
    T_cam_obj = pose_to_matrix(object_pose)
    # Transform the pose to map frame and apply offset/rotation
    T_cam_goal = offset_in_front(T_cam_obj, x_offset)

    transform = lookup_transform(
        tf_buffer, "map", frame_id
    )
    if transform is None:
        return None

    # Use provided clock or get timestamp from the original pose
    
    transformed_pose_stamped = matrix_to_posestamped(
        T_cam_goal, transform, "map", clock.now().to_msg()
    )

    return transformed_pose_stamped

def lookup_transform(tf_buffer, target_frame, source_frame):
    
    try:
        if not tf_buffer.can_transform(
                        target_frame, source_frame, rclpy.time.Time()
                    ):
            transformed_pose_stamped = None
        else:
            transform = tf_buffer.lookup_transform(
                target_frame,  # target frame
                source_frame,  # source frame
                rclpy.time.Time(),  # get the latest transform
                rclpy.duration.Duration(seconds=1.0),  # timeout
            )
            return transform
    except Exception as e:
        print(f"Transform error: {e}")
        return None

def pose_to_matrix(pose):
    """Convert geometry_msgs/Pose to 4×4 homogeneous matrix."""
    q = pose.orientation
    p = pose.position
    T = np.eye(4)
    T[:3, :3] = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def matrix_to_posestamped(T, lookup_transform, target_frame, stamp):
    """Convert 4×4 homogeneous matrix `T` to geometry_msgs/PoseStamped."""

    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = target_frame

    # convert the matrix to Pose message
    q = tft.quaternion_from_matrix(T)  # (x,y,z,w)
    p = T[:3, 3]
    q_z180 = tft.quaternion_from_euler(0, 0, np.pi)  # 180 degrees around Z axis
    q_goal = tft.quaternion_multiply(q, q_z180)

    msg.pose.position.x = p[0]
    msg.pose.position.y = p[1]
    msg.pose.position.z = p[2]
    msg.pose.orientation.x = q_goal[0]
    msg.pose.orientation.y = q_goal[1]
    msg.pose.orientation.z = q_goal[2]
    msg.pose.orientation.w = q_goal[3]

    # Apply the transform to get pose in map frame
    msg.pose = do_transform_pose(msg.pose, lookup_transform)

    return msg


def offset_in_front(T_cam_obj, offset):
    """
    Return T_cam_goal = T_cam_obj  ·  T_offset(+X).

    Parameters
    ----------
    T_cam_obj : (4,4) ndarray – camera ➜ object transform
    offset    : float        – metres to move along +X of the object
    """
    T_offset = np.eye(4)
    T_offset[0, 3] = offset  # +X in object frame
    return T_cam_obj @ T_offset  # matrix product
