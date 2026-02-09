import numpy as np
from geometry_msgs.msg import Pose, PoseArray
from scipy.spatial.transform import Rotation as R


def pose_to_matrix(pose_msg):
    """Convert geometry_msgs/Pose to 4x4 transformation matrix."""
    # Translation
    t = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    
    # Rotation (quaternion to rotation matrix)
    q = [pose_msg.orientation.w, pose_msg.orientation.x, 
            pose_msg.orientation.y, pose_msg.orientation.z]
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
    R_matrix = r.as_matrix()
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = t
    
    return T
    
def matrix_to_pose(T):
    """Convert 4x4 transformation matrix to geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    
    # Rotation matrix to quaternion
    r = R.from_matrix(T[:3, :3])
    q = r.as_quat()  # [x, y, z, w]
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    
    return pose