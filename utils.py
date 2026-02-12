import math
import cv2
import numpy as np


def translation_matrix(dx, dy, dz):
        T = np.eye(4)
        T[:3, 3] = [dx, dy, dz]
        return T

def rotation_matrix(axis, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    if axis == "x":
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == "y":
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == "z":
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    T = np.eye(4)
    T[:3, :3] = R
    return T


def load_pose_from_yaml(yaml_path):
    """
    Load body2world_pose from OpenCV YAML file.
    
    Args:
        yaml_path: Path to YAML file containing body2world_pose
        
    Returns:
        numpy array (4, 4) pose matrix, or None if failed
    """
    try:
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"Error: Could not open YAML file {yaml_path}")
            return None
        
        # Read the matrix
        pose_matrix = fs.getNode("body2world_pose").mat()
        fs.release()
        
        if pose_matrix is None or pose_matrix.shape != (4, 4):
            print(f"Error: Invalid pose matrix in YAML file {yaml_path}")
            return None
        
        # Convert to numpy array
        pose_np = np.array(pose_matrix, dtype=np.float64)
        return pose_np
        
    except Exception as e:
        print(f"Error loading pose from YAML {yaml_path}: {e}")
        return None
        
        
def save_pose_yaml(save_path, pose):
        
    # OpenCV FileStorage must be in WRITE mode for saving
    fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print(f"Error: Could not open YAML file for writing: {save_path}")
        return

    # write the 4x4 pose matrix
    fs.write("body2world_pose", pose.astype(np.float64))
    fs.release()
    print(f"Pose saved → {save_path}")


def configure_pose(key, pose):

    # step sizes
    trans_step = 0.005      # 5 mm
    rot_step = math.radians(2.0)
    delta = np.eye(4)

    # --- Translation ---
    if key == ord('w'):
        delta = translation_matrix(0, 0, trans_step)

    elif key == ord('s'):
        delta = translation_matrix(0, 0, -trans_step)

    elif key == ord('a'):
        delta = translation_matrix(-trans_step, 0, 0)

    elif key == ord('d'):
        delta = translation_matrix(trans_step, 0, 0)

    elif key == ord('r'):
        delta = translation_matrix(0, trans_step, 0)

    elif key == ord('f'):
        delta = translation_matrix(0, -trans_step, 0)

    # --- Rotation ---
    elif key == ord('q'):
        delta = rotation_matrix("z", rot_step)

    elif key == ord('e'):
        delta = rotation_matrix("z", -rot_step)

    elif key == ord('t'):
        delta = rotation_matrix("x", rot_step)

    elif key == ord('g'):
        delta = rotation_matrix("x", -rot_step)

    elif key == ord('z'):
        delta = rotation_matrix("y", rot_step)

    elif key == ord('c'):
        delta = rotation_matrix("y", -rot_step)

    else:
        return pose

    # Apply incremental transform
    pose = pose @ delta

    # Debug print
    #pos = pose[:3, 3]
    #print(f"Pose updated → x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    return pose
