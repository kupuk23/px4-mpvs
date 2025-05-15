import numpy as np

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

def calculate_pose_error(q1, q2):
    
    # Normalize quaternions to ensure unit length
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    q2_inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]])

    q_diff = quaternion_multiply(q1, q2_inv)
    
    angle = 2 * np.arccos(q_diff[0])  # angle in radians
    # Handle the antipodal case (q and -q represent the same orientation)
    orientation_error_degrees = np.rad2deg(angle)
    
    return orientation_error_degrees