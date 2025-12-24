#!/root/miniconda3/envs/articulation/bin/python
"""
ROS node that simulates a ZED camera by reading frames from a data directory.
Publishes to the same topics as the real ZED camera for compatibility.
"""

import rospy
import cv2
import numpy as np
import os
import glob
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Header
import yaml

class ZEDSimulatorNode:
    def __init__(self):
        rospy.init_node('zed_simulator_node', anonymous=True)
        
        # Parameters
        self.zed_type = rospy.get_param('~zed_type', 'zed')
        self.data_root = rospy.get_param('~data_root', os.path.join('rosbags', 'data', 'scene0010_00'))
        self.color_subdir = rospy.get_param('~color_subdir', 'color')
        self.depth_subdir = rospy.get_param('~depth_subdir', 'depth')
        self.intrinsic_path = rospy.get_param('~intrinsic_path', None)
        self.depth_scale = rospy.get_param('~depth_scale', 1000.0)
        self.fps = rospy.get_param('~fps', 30.0)
        self.loop = rospy.get_param('~loop', True)
        
        # If intrinsic_path not provided -> default location
        if self.intrinsic_path is None:
            default_intrinsic = os.path.join(self.data_root, 'intrinsics', 'intrinsic_color.txt')
            if os.path.exists(default_intrinsic):
                self.intrinsic_path = default_intrinsic
        
        # Publishers
        self.image_pub = rospy.Publisher(f'/{self.zed_type}/zed_node/left/image_rect_color', Image, queue_size=1)
        self.depth_pub = rospy.Publisher(f'/{self.zed_type}/zed_node/depth/depth_registered', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher(f'/{self.zed_type}/zed_node/left/camera_info', CameraInfo, queue_size=1)
        self.depth_camera_info_pub = rospy.Publisher(f'/{self.zed_type}/zed_node/depth/camera_info', CameraInfo, queue_size=1)
        
        self.bridge = CvBridge()
        
        # Load frames
        self._load_frames()
        
        # Load camera info
        self._load_camera_info()
        
        rospy.loginfo(f"ZED Simulator initialized with {len(self.color_paths)} frames")
        rospy.loginfo(f"Publishing at {self.fps} Hz")
        
    def _load_frames(self):
        """Load frame paths from data directory."""
        color_dir = os.path.join(self.data_root, self.color_subdir)
        depth_dir = os.path.join(self.data_root, self.depth_subdir)
        
        if not os.path.isdir(color_dir):
            rospy.logerr(f"Color directory not found: {color_dir}")
            raise RuntimeError(f"Color directory not found: {color_dir}")
        
        if not os.path.isdir(depth_dir):
            rospy.logerr(f"Depth directory not found: {depth_dir}")
            raise RuntimeError(f"Depth directory not found: {depth_dir}")
        
        # Collect files
        color_files = self._collect_files(color_dir, ('.jpg', '.png'))
        depth_files = self._collect_files(depth_dir, ('.png', '.npy'))
        
        # Create maps by frame index
        color_map = {self._stem_to_int(path): path for path in color_files}
        depth_map = {self._stem_to_int(path): path for path in depth_files}
        
        # Find shared indices
        shared_indices = sorted(set(color_map.keys()) & set(depth_map.keys()))
        if not shared_indices:
            raise RuntimeError(f"No overlapping frame ids between {color_dir} and {depth_dir}")
        
        self.frame_indices = shared_indices
        self.color_paths = [color_map[idx] for idx in self.frame_indices]
        self.depth_paths = [depth_map[idx] for idx in self.frame_indices]
        self.frame_ptr = 0
        
        # Get image dimensions from first frame
        sample_image = cv2.imread(self.color_paths[0], cv2.IMREAD_COLOR)
        if sample_image is None:
            raise RuntimeError(f"Failed to read sample color frame from {self.color_paths[0]}")
        
        self.width = sample_image.shape[1]
        self.height = sample_image.shape[0]
        
        rospy.loginfo(f"Loaded {len(self.frame_indices)} frames, resolution: {self.width}x{self.height}")
    
    def _collect_files(self, directory, extensions):
        """Collect files with given extensions from directory."""
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        if not files:
            raise RuntimeError(f"No frames with extensions {extensions} found in {directory}")
        return sorted(files)
    
    def _stem_to_int(self, path):
        """Extract integer frame index from filename."""
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem)
    
    def _load_camera_info(self):
        """Load camera intrinsics and create CameraInfo messages."""
        if self.intrinsic_path is None or not os.path.exists(self.intrinsic_path):
            rospy.logwarn(f"Intrinsic file not found: {self.intrinsic_path}, using defaults")
            # Use default intrinsics (will need to be adjusted)
            fx = fy = self.width * 0.7  # Rough estimate
            cx = self.width / 2.0
            cy = self.height / 2.0
        else:
            # Load from file
            with open(self.intrinsic_path, 'r') as f:
                rows = [list(map(float, line.strip().split())) for line in f if line.strip()]
            matrix = np.array(rows, dtype=np.float32)
            if matrix.shape[0] < 3 or matrix.shape[1] < 3:
                raise ValueError(f"Intrinsic file {self.intrinsic_path} does not contain a valid matrix.")
            K = matrix[:3, :3]
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        
        # Create CameraInfo for color camera
        self.camera_info = CameraInfo()
        self.camera_info.width = self.width
        self.camera_info.height = self.height
        self.camera_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.camera_info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        # Depth camera info (same for now)
        self.depth_camera_info = CameraInfo()
        self.depth_camera_info.width = self.width
        self.depth_camera_info.height = self.height
        self.depth_camera_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.depth_camera_info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.depth_camera_info.distortion_model = 'plumb_bob'
        self.depth_camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.depth_camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def run(self):
        """Main loop to publish frames"""
        rate = rospy.Rate(self.fps)

        while not rospy.is_shutdown():
            
            # Restart from beginning if we reached the end
            if self.frame_ptr >= len(self.frame_indices):

                if self.loop:
                    rospy.loginfo("Reached end of frames, restarting from beginning")
                    self.frame_ptr = 0
                else:
                    rospy.loginfo("Reached end of frames, stopping simulation")
                    break

            # Read frames
            color_path = self.color_paths[self.frame_ptr]
            depth_path = self.depth_paths[self.frame_ptr]
            frame_idx = self.frame_indices[self.frame_ptr]

            # Read color image
            color = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color is None:
                rospy.logwarn(f"Failed to read color frame {frame_idx}")
                self.frame_ptr += 1
                continue

            # Read depth image
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                rospy.logwarn(f"Failed to read depth frame {frame_idx}")
                self.frame_ptr += 1
                continue

            # Convert depth to meters
            if depth_raw.dtype != np.float32:
                depth = depth_raw.astype(np.float32)
                if self.depth_scale and self.depth_scale != 0:
                    depth /= self.depth_scale
            else:
                depth = depth_raw

            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.seq = frame_idx

            # Publish color image
            color_msg = self.bridge.cv2_to_imgmsg(color, 'bgr8')
            color_msg.header = header
            self.image_pub.publish(color_msg)

            self.camera_info.header = header
            self.camera_info_pub.publish(self.camera_info)

            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(depth, '32FC1')
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)

            self.depth_camera_info.header = header
            self.depth_camera_info_pub.publish(self.depth_camera_info)

            self.frame_ptr += 1
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ZEDSimulatorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

