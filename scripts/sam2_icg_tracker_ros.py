#!/root/miniconda3/envs/articulation/bin/python3

"""
SAM2 + ICG Tracker Integration using ROS

Pipeline:
1. Subscribe to RGB-D images from ROS topics
2. Apply SAM2 segmentation
3. Create point cloud from segmented area
4. Globally match point cloud against CAD model to find initial pose
5. Send initial pose to ICG tracker via ROS service
6. Receive poses from ICG tracker via ROS topic
"""

import sys
import os
import argparse
import cv2
import numpy as np
import torch
import time
import copy
import open3d as o3d
import rospy
import threading
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
import message_filters
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sam2_video_predictor'))
from sam2.build_sam import build_sam2_camera_predictor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from point_cloud_utils import *
from visualization_utils import draw_pose_axes, overlay_mask, overlay_rgba_on_bgr
from pose_estimation.srv import SetInitialPose


class SAM2ICGTrackerROS:
    """Main tracker combining SAM2 detection with ICG tracking via ROS."""
    
    def __init__(self, args):
        """Initialize tracker."""
        self.args = args
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = message_filters.Subscriber(f'/{args.zed_type}/zed_node/left/image_rect_color', Image)
        self.depth_sub = message_filters.Subscriber(f'/{args.zed_type}/zed_node/depth/depth_registered', Image)
        self.camera_info_sub = message_filters.Subscriber(f'/{args.zed_type}/zed_node/left/camera_info', CameraInfo)
        
        # Synchronize subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.camera_info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        
        # Subscribe to poses from pose-estimation
        self.pose_sub = rospy.Subscriber('/pose_estimation/poses', PoseArray, self.pose_callback)
        
        # Service client for setting initial pose
        rospy.wait_for_service('/pose_estimation/set_initial_pose')
        self.set_initial_pose_client = rospy.ServiceProxy('/pose_estimation/set_initial_pose', SetInitialPose)
        
        # Current frame data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.camera_intrinsic = None
        self.frame_lock = threading.Lock()
        
        # Initialize SAM2
        print("Initializing SAM2...")
        self.sam2_predictor = build_sam2_camera_predictor(
            config_file=args.model_cfg,
            ckpt_path=args.checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load CAD model
        print(f"Loading CAD model from {args.cad_model_path}...")
        self.cad_mesh, self.cad_pcd = load_cad_model(args.cad_model_path, args.cad_scale)
        print(f"CAD model loaded: {len(self.cad_pcd.points)} points")
        
        # Preprocess CAD model
        cad_pcd_copy = copy.deepcopy(self.cad_pcd)
        cad_center = cad_pcd_copy.get_center()
        print(f"CAD model centered at origin (offset: {cad_center})")
        
        self.cad_pcd_processed = preprocess_point_cloud(
            cad_pcd_copy,
            voxel_size=args.voxel_size,
            estimate_normals=True
        )
        
        # Tracking state
        self.frame_count = 0
        self.tracking_active = False
        self.current_pose = None
        self.last_pose_msg = None
        self.initialized = False
        
        # Progress tracking
        self.progress_mode = args.progress_mode
        self.goal_rotation = args.goal_rotation
        self.goal_translation = args.goal_translation
        self.goal_distance = args.goal_distance
        self.progress_axis = args.progress_axis
        self.track_progress = args.progress_mode is not None
        self.reference_frame_index = 10
        self.reference_pose = None
        self.last_detected_axis = None
        
        # Mouse callback for bounding box
        self.mouse_data = {
            'drawing': False,
            'start_point': None,
            'end_point': None,
            'bbox': None,
            'new_bbox': False
        }
        
        # Initialize model renderer
        if self.args.render_model:
            # Will be initialized when we know image size
            self.renderer = None
            self.scene = None
            self.material = None
        
        self.save_frames = args.save_frames
        self.output_dir = args.output_dir
        
        if self.save_frames:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Frames will be saved to: {self.output_dir}")
    
    def image_callback(self, rgb_msg, depth_msg, camera_info_msg):
        """Callback for synchronized image messages."""
        try:
            # Convert ROS messages to OpenCV
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            
            # Extract camera intrinsic from CameraInfo
            K = np.array(camera_info_msg.K).reshape(3, 3)
            
            with self.frame_lock:
                self.latest_rgb = rgb
                self.latest_depth = depth
                self.latest_camera_info = camera_info_msg
                self.camera_intrinsic = K
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def pose_callback(self, pose_array_msg):
        """Callback for pose updates from pose-estimation."""
        if len(pose_array_msg.poses) > 0:
            # Get first pose (assuming single object tracking)
            pose_msg = pose_array_msg.poses[0]
            
            # Convert to 4x4 transformation matrix
            pose_matrix = self.pose_to_matrix(pose_msg)
            self.last_pose_msg = pose_matrix
            
            if self.tracking_active:
                self.current_pose = pose_matrix
                
                # Set reference pose at reference frame for progress tracking
                if self.track_progress and self.frame_count == self.reference_frame_index:
                    self.reference_pose = self.current_pose.copy()
                    print(f"Reference pose set at frame {self.frame_count + 1} for progress tracking")
    
    def pose_to_matrix(self, pose_msg):
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
    
    def matrix_to_pose(self, T):
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
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing bounding box."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_data['drawing'] = True
            self.mouse_data['start_point'] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_data['drawing']:
                self.mouse_data['end_point'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_data['drawing'] = False
            self.mouse_data['end_point'] = (x, y)
            if self.mouse_data['start_point'] and self.mouse_data['end_point']:
                self.mouse_data['bbox'] = (
                    self.mouse_data['start_point'],
                    self.mouse_data['end_point']
                )
                self.mouse_data['new_bbox'] = True
    
    def _estimate_initial_pose(self, object_pcd):
        """
        Estimate initial pose by matching object point cloud to CAD model.
        
        Args:
            object_pcd: open3d.geometry.PointCloud - segmented object point cloud (in camera frame)
        
        Returns:
            numpy array (4, 4) - T_model_to_camera (body2world_pose)
        """
        # Preprocess object point cloud (camera frame)
        object_pcd_processed = preprocess_point_cloud(
            object_pcd,
            voxel_size=self.args.voxel_size,
            estimate_normals=True
        )
        
        if len(object_pcd_processed.points) == 0:
            print("Error: No valid points in processed point cloud")
            return None
        
        print(f"Object point cloud: {len(object_pcd_processed.points)} points")
        
        # Check if RANSAC is needed
        if requires_ransac(object_pcd_processed, self.cad_pcd_processed):
            print("Using RANSAC for global registration...")
            # Extract features
            object_down, object_fpfh = downsample_and_extract_features(
                object_pcd_processed, self.args.voxel_size
            )
            cad_down, cad_fpfh = downsample_and_extract_features(
                self.cad_pcd_processed, self.args.voxel_size
            )
            
            # Global registration
            result_ransac = global_ransac_registration(
                object_down,
                cad_down,
                object_fpfh, 
                cad_fpfh, 
                self.args.voxel_size
            )
            
            if result_ransac.fitness < 0.1:
                print(f"RANSAC registration failed (fitness: {result_ransac.fitness:.3f})")
                return None
            
            print(f"RANSAC fitness: {result_ransac.fitness:.3f}")
            T_model_to_camera = np.linalg.inv(result_ransac.transformation)
            print(f"RANSAC transform translation: {T_model_to_camera[:3, 3]}")
        else:
            print("Using ICP directly (good initial alignment)...")
            T_model_to_camera = np.eye(4)
        
        return T_model_to_camera
    
    def initialize_tracking(self, rgb_frame, depth_frame):
        """Initialize tracking with first frame."""
        print("\n=== Initialization Phase ===")
        print("Draw a bounding box around the object and press 's' to start tracking")
        
        cv2.namedWindow('Initialization - Draw Bounding Box')
        cv2.setMouseCallback('Initialization - Draw Bounding Box', self.mouse_callback)
        
        while not rospy.is_shutdown():
            display_frame = rgb_frame.copy()
            
            if self.mouse_data['bbox']:
                pt1, pt2 = self.mouse_data['bbox']
                cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
            
            if self.mouse_data['drawing'] and self.mouse_data['start_point'] and self.mouse_data['end_point']:
                cv2.rectangle(
                    display_frame,
                    self.mouse_data['start_point'],
                    self.mouse_data['end_point'],
                    (0, 255, 0),
                    2
                )
            
            cv2.putText(
                display_frame,
                "Draw bounding box and press 's' to start",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Initialization - Draw Bounding Box', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('s') and self.mouse_data['bbox'] is not None:
                break
        
        cv2.destroyWindow('Initialization - Draw Bounding Box')
        
        # Get bounding box
        bbox = self.mouse_data['bbox']
        bbox_array = np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]], dtype=np.float32)
        
        # Run SAM2
        print("Running SAM2 on first frame...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam2_predictor.load_first_frame(rgb_frame)
            frame_idx = 0
            obj_id = 1
            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_prompt(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox=bbox_array
            )
        
        # Get mask
        mask_logits = out_mask_logits[0]
        if mask_logits.dim() == 3:
            mask_logits = mask_logits.squeeze(0)
        mask = (mask_logits > 0.0).cpu().numpy()
        
        if mask.shape != depth_frame.shape:
            mask = cv2.resize(mask.astype(np.uint8), 
                            (depth_frame.shape[1], depth_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        print(f"Mask area: {mask.sum()} pixels")
        
        if mask.sum() == 0:
            print("Error: No mask generated")
            return False
        
        # Extract object point cloud
        print("Extracting object point cloud...")
        object_pcd = depth_to_point_cloud(
            depth_frame,
            self.camera_intrinsic,
            mask=mask
        )
        
        if len(object_pcd.points) == 0:
            print("Error: No valid points in point cloud")
            return False
        
        # Estimate initial pose
        print("Estimating initial pose by matching to CAD model...")
        initial_pose = self._estimate_initial_pose(object_pcd)
        
        if initial_pose is None:
            print("Error: Failed to estimate initial pose")
            return False
        
        print(f"Initial pose estimated:")
        print(initial_pose)
        
        # Send initial pose to pose-estimation via service
        print("Sending initial pose to pose-estimation...")
        try:
            initial_pose_msg = self.matrix_to_pose(initial_pose)
            resp = self.set_initial_pose_client(
                initial_pose_msg,
                "body"
            )

            if resp.success:
                print("Initial pose set successfully in pose-estimation")
                self.current_pose = initial_pose
                self.tracking_active = True
                self.initialized = True
            else:
                print(f"Failed to set initial pose: {resp.message}")
                return False
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        
        return True
    
    def _detect_progress_axis(self, reference_pose, current_pose):
        """Automatically detect the axis with the most rotation/translation change."""
        if self.progress_mode == 'rotation':
            R_ref = reference_pose[:3, :3]
            R_curr = current_pose[:3, :3]
            R_relative = R_curr @ R_ref.T
            
            yaw = abs(np.degrees(np.arctan2(R_relative[1, 0], R_relative[0, 0])))
            pitch = abs(np.degrees(np.arcsin(np.clip(-R_relative[2, 0], -1, 1))))
            roll = abs(np.degrees(np.arctan2(R_relative[2, 1], R_relative[2, 2])))
            
            rotations = {'x': roll, 'y': pitch, 'z': yaw}
            max_axis = max(rotations, key=rotations.get)
            return max_axis, rotations
            
        elif self.progress_mode == 'translation':
            t_ref = reference_pose[:3, 3]
            t_curr = current_pose[:3, 3]
            t_diff = t_curr - t_ref
            
            translations = {'x': abs(t_diff[0]), 'y': abs(t_diff[1]), 'z': abs(t_diff[2])}
            max_axis = max(translations, key=translations.get)
            return max_axis, translations
        
        return 'z', {}
    
    def _calculate_progress(self):
        """Calculate progress percentage (0-100%) based on current pose vs reference pose."""
        if not self.track_progress or self.reference_pose is None or self.current_pose is None:
            return None
        
        if self.progress_mode == 'distance':
            t_initial = self.reference_pose[:3, 3]
            t_current = self.current_pose[:3, 3]
            euclidean_distance = np.linalg.norm(t_current - t_initial)
            
            if self.goal_distance == 0:
                return 0.0
            progress = (euclidean_distance / abs(self.goal_distance)) * 100.0
            progress = np.clip(progress, 0, 100)
            return progress
        
        if self.progress_axis.lower() == 'auto':
            detected_axis, axis_values = self._detect_progress_axis(self.reference_pose, self.current_pose)
            
            if detected_axis != self.last_detected_axis:
                self.last_detected_axis = detected_axis
                if self.args.verbose:
                    if self.progress_mode == 'rotation':
                        print(f"Auto-detected rotation axis: {detected_axis.upper()}")
                    else:
                        print(f"Auto-detected translation axis: {detected_axis.upper()}")
            
            current_axis = detected_axis
        else:
            current_axis = self.progress_axis.lower()
        
        if self.progress_mode == 'rotation':
            R_initial = self.reference_pose[:3, :3]
            R_current = self.current_pose[:3, :3]
            R_relative = R_current @ R_initial.T
            
            trace = np.trace(R_relative)
            angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            R_skew = (R_relative - R_relative.T) / (2 * np.sin(angle_rad) + 1e-8)
            axis_vec = np.array([R_skew[2, 1], R_skew[0, 2], R_skew[1, 0]])
            axis_vec_normalized = axis_vec / (np.linalg.norm(axis_vec) + 1e-8)
            
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[current_axis]
            axis_direction = np.array([1.0 if i == axis_idx else 0.0 for i in range(3)])
            axis_alignment = np.dot(axis_vec_normalized, axis_direction)
            
            if abs(axis_alignment) > 0.7:
                angle_rad_signed = angle_rad * np.sign(axis_alignment)
            else:
                if current_axis == 'z':
                    angle_rad_signed = np.arctan2(R_relative[1, 0], R_relative[0, 0])
                elif current_axis == 'y':
                    angle_rad_signed = np.arcsin(np.clip(-R_relative[2, 0], -1, 1))
                else:
                    angle_rad_signed = np.arctan2(R_relative[2, 1], R_relative[2, 2])
            
            angle_deg = np.degrees(angle_rad_signed)
            
            if self.goal_rotation == 0:
                return 0.0
            progress = (abs(angle_deg) / abs(self.goal_rotation)) * 100.0
            progress = np.clip(progress, 0, 100)
            return progress
            
        elif self.progress_mode == 'translation':
            t_initial = self.reference_pose[:3, 3]
            t_current = self.current_pose[:3, 3]
            t_diff = t_current - t_initial
            
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[current_axis]
            translation_distance = t_diff[axis_idx]
            
            if self.goal_translation == 0:
                return 0.0
            
            if self.goal_translation < 0:
                progress = (translation_distance / self.goal_translation) * 100.0
            else:
                progress = (abs(translation_distance) / abs(self.goal_translation)) * 100.0
            progress = np.clip(progress, 0, 100)
            return progress
        
        return None
    
    def _render_model_overlay(self, mesh, T_model_to_camera, K, width, height):
        """Render model overlay on image."""
        if self.renderer is None:
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            self.scene = self.renderer.scene
            self.scene.set_background([0, 0, 0, 0])
            self.material = o3d.visualization.rendering.MaterialRecord()
            self.material.shader = 'defaultLit'
            self.material.base_color = [0, 0, 255, 1]
        
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.transform(T_model_to_camera.copy())
        
        self.scene.clear_geometry()
        self.scene.add_geometry("model", mesh_copy, self.material)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        
        T_camera_to_world = np.eye(4, dtype=np.float32)
        self.renderer.setup_camera(intrinsic, T_camera_to_world)
        
        img = self.renderer.render_to_image()
        img = np.asarray(img)
        
        if img.shape[2] == 3:
            alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
            img = np.concatenate([img, alpha], axis=2)
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            mask_non_black = np.any(rgb >= 10, axis=2, keepdims=True)
            alpha[:] = 0.8 * mask_non_black.astype(np.float32)
            img[:, :, 3] = (alpha[:, :, 0] * 255).astype(np.uint8)
        
        return img
    
    def run(self):
        """Main tracking loop."""
        print("\n=== Starting SAM2 + ICG Tracking (ROS) ===")
        
        # Wait for first frame
        print("Waiting for camera data...")
        rospy.sleep(1.0)
        
        with self.frame_lock:
            if self.latest_rgb is None or self.latest_depth is None:
                print("Error: No camera data received")
                return
            
            rgb_frame = self.latest_rgb.copy()
            depth_frame = self.latest_depth.copy()
        
        # Initialize
        if not self.initialize_tracking(rgb_frame, depth_frame):
            print("Initialization failed")
            return
        
        print("\n=== Tracking Active ===")
        print("Press 'q' to quit")
        
        fps_start_time = time.time()
        
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_rgb is None or self.latest_depth is None:
                    rospy.sleep(0.1)
                    continue
                
                rgb_frame = self.latest_rgb.copy()
                depth_frame = self.latest_depth.copy()
            
            # Visualize
            vis_frame = rgb_frame.copy()
            
            # Draw pose
            if self.current_pose is not None:
                vis_frame = draw_pose_axes(
                    vis_frame,
                    self.current_pose,
                    self.camera_intrinsic,
                    length=0.1
                )
            
            # Display info
            info_text = [f"Frame: {self.frame_count}"]
            if self.current_pose is not None:
                t = self.current_pose[:3, 3]
                info_text.append(f"Position: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            
            # Display progress if tracking
            if self.track_progress:
                if self.reference_pose is None:
                    frames_until_ref = self.reference_frame_index - self.frame_count + 1
                    if frames_until_ref > 0:
                        info_text.append(f"Progress: Waiting for reference frame ({frames_until_ref} frames)...")
                    else:
                        info_text.append("Progress: Waiting for reference pose...")
                else:
                    progress = self._calculate_progress()
                    if progress is not None:
                        progress_text = f"Progress: {progress:.1f}%"
                        if self.progress_mode == 'rotation':
                            axis_display = self.last_detected_axis.upper() if self.progress_axis.lower() == 'auto' and self.last_detected_axis else self.progress_axis.upper()
                            progress_text += f" ({self.goal_rotation} deg. around {axis_display})"
                        elif self.progress_mode == 'translation':
                            axis_display = self.last_detected_axis.upper() if self.progress_axis.lower() == 'auto' and self.last_detected_axis else self.progress_axis.upper()
                            progress_text += f" ({self.goal_translation:.2f}m along {axis_display})"
                        else:
                            progress_text += f" ({self.goal_distance:.2f}m distance)"
                        info_text.append(progress_text)
                        
                        # Progress bar
                        bar_width = 300
                        bar_height = 20
                        bar_x = 10
                        bar_y = len(info_text) * 25 + 25
                        
                        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                        fill_width = int(bar_width * progress / 100.0)
                        bar_color = (0, 255, 0)
                        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
                        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            
            y_offset = 20
            for i, text in enumerate(info_text):
                cv2.putText(
                    vis_frame,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # FPS
            fps = 1 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            cv2.putText(
                vis_frame,
                f"FPS: {fps:.1f}",
                (10, vis_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            if self.args.render_model and self.current_pose is not None:
                overlay_rgba = self._render_model_overlay(
                    self.cad_mesh, 
                    self.current_pose, 
                    self.camera_intrinsic, 
                    vis_frame.shape[1], 
                    vis_frame.shape[0]
                )
                vis_frame = overlay_rgba_on_bgr(vis_frame, overlay_rgba)
            
            cv2.imshow('SAM2 + ICG Tracking (ROS)', vis_frame)
            
            if self.save_frames:
                frame_filename = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            self.frame_count += 1
            rospy.sleep(0.01)
        
        cv2.destroyAllWindows()
        print("\nTracking stopped.")


def main():
    
    rospy.init_node("sam2_icg_tracker_ros", anonymous=False)

    # SAM2 parameters
    checkpoint = rospy.get_param("~checkpoint", "./sam2_video_predictor/checkpoints/sam2_hiera_small.pt")
    model_cfg = rospy.get_param("~model_cfg", "sam2_hiera_s.yaml")

    # Zed camera type
    zed_type = rospy.get_param("~zed_type", "zed")

    # Point cloud / registration
    voxel_size = rospy.get_param("~voxel_size", 0.005)

    # CAD model
    cad_model_path = rospy.get_param("~cad_model_path", "pose-estimation/ICG/examples/pearl_lab/handle.obj")
    cad_scale = rospy.get_param("~cad_scale", 0.001)

    # Other options
    verbose = rospy.get_param("~verbose", False)
    render_model = rospy.get_param("~render_model", False)
    save_frames = rospy.get_param("~save_frames", False)
    output_dir = rospy.get_param("~output_dir", "./data/output")

    # Progress tracking
    progress_mode = rospy.get_param("~progress_mode", None)
    goal_rotation = rospy.get_param("~goal_rotation", 90.0)
    goal_translation = rospy.get_param("~goal_translation", 0.5)
    goal_distance = rospy.get_param("~goal_distance", 0.5)
    progress_axis = rospy.get_param("~progress_axis", "auto")

    # Check checkpoint exists
    if not os.path.exists(checkpoint):
        rospy.logerr(f"SAM2 checkpoint not found: {checkpoint}")
        return

    # Build argument object (optional: can wrap in a simple class or dict)
    class Args:
        pass

    args = Args()
    args.checkpoint = checkpoint
    args.model_cfg = model_cfg
    args.zed_type = zed_type
    args.voxel_size = voxel_size
    args.cad_model_path = cad_model_path
    args.cad_scale = cad_scale
    args.verbose = verbose
    args.render_model = render_model
    args.save_frames = save_frames
    args.output_dir = output_dir
    args.progress_mode = progress_mode
    args.goal_rotation = goal_rotation
    args.goal_translation = goal_translation
    args.goal_distance = goal_distance
    args.progress_axis = progress_axis

    # Initialize tracker
    tracker = SAM2ICGTrackerROS(args)

    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

