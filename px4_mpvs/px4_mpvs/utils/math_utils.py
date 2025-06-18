import numpy as np
import casadi as cs

def quat_error(q, q2):
    conj_q = cs.vertcat(q[0], -q[1:])
    q_err = quat_mul_cs(conj_q, q2)
    att_err = 2 * q_err[1:]          # 3×1
    return att_err

def quat_mul_cs(q, q2):
    w = q[0]*q2[0] - q[1]*q2[1] - q[2]*q2[2] - q[3]*q2[3]
    x = q[0]*q2[1] + q[1]*q2[0] + q[2]*q2[3] - q[3]*q2[2]
    y = q[0]*q2[2] - q[1]*q2[3] + q[2]*q2[0] + q[3]*q2[1]
    z = q[0]*q2[3] + q[1]*q2[2] - q[2]*q2[1] + q[3]*q2[0]
    return cs.vertcat(w, x, y, z)

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def calc_position_error(pos1,pos2):
    """
    Calculate position error between two positions
    
    Args:
        pose1: First pos
        pose2: Second pos
        
    Returns:
        tuple: (position_error, orientation_error_degrees)
    """
    # Position error (Euclidean distance)
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    position_error = np.linalg.norm([dx, dy, dz])
    return position_error

def calculate_orientation_error(q1, q2):
    
    # Normalize quaternions to ensure unit length
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    q2_inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]])

    q_diff = quaternion_multiply(q1, q2_inv)
    
    angle = 2 * np.arccos(q_diff[0])  # angle in radians
    # Handle the antipodal case (q and -q represent the same orientation)
    orientation_error_degrees = np.rad2deg(angle)
    
    return orientation_error_degrees

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