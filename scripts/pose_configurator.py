#!/root/miniconda3/envs/articulation/bin/python3

import datetime
import math
import sys
import yaml
import os
import argparse
import cv2
import numpy as np
import time
import copy
import open3d as o3d
import rospy
import threading
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import message_filters
from scipy.spatial.transform import Rotation as R
from cam_utils import resize_K
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from point_cloud_utils import *
from utils import *
from ros_utils import *
from visualization_utils import draw_pose_axes, overlay_mask, overlay_rgba_on_bgr



class PoseConfigurator:
    """Configure an initial pose in a live camera-view via ROS."""
    
    def __init__(self, args):
        """Initialize tracker."""
        self.args = args
        
        # Image resize parameters
        self.target_width = self.args.target_width
        self.target_height = self.args.target_width
        
        self.detector_yaml_path = self.args.detector_yaml
        self.current_pose = None
        if self.detector_yaml_path:
            self.current_pose = load_pose_from_yaml(self.detector_yaml_path)
        
        if self.current_pose is not None:
            print(f"Loaded body2world_pose from {self.detector_yaml_path}")
        else:
            print(f"Failed to load pose from {self.detector_yaml_path}, will use normal initialization")
            self.current_pose = np.ones((4, 4))
        
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
        
        # Current frame data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.camera_intrinsic = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        
        # Load CAD model
        print(f"Loading CAD model from {args.cad_model_path}...")
        self.cad_mesh, self.cad_pcd = load_cad_model(args.cad_model_path, args.cad_scale)
        print(f"CAD model loaded: {len(self.cad_pcd.points)} points")
        
        # Preprocess CAD model
        cad_pcd_copy = copy.deepcopy(self.cad_pcd)
        cad_center = cad_pcd_copy.get_center()
        print(f"CAD model centered at origin (offset: {cad_center})")
        
        # Initialize model renderer
        if self.args.render_model:
            # Will be initialized when we know image size
            self.renderer = None
            self.scene = None
            self.material = None
    
    def image_callback(self, rgb_msg, depth_msg, camera_info_msg):
        """Callback for synchronized image messages."""
        try:
            # Convert ROS messages to OpenCV
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            
            # Extract camera intrinsic from CameraInfo
            K = np.array(camera_info_msg.K).reshape(3, 3)
            
            # Center crop images if target size is specified
            if self.target_width is not None and self.target_height is not None:
                original_height, original_width = rgb.shape[:2]
                crop_width = min(self.target_width, original_width)
                crop_height = min(self.target_height, original_height)
                
                # Calculate center crop coordinates
                x1 = (original_width - crop_width) // 2
                y1 = (original_height - crop_height) // 2
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                
                # Crop RGB image (numpy indexing: [rows, cols] = [y, x])
                rgb = rgb[y1:y2, x1:x2]
                
                # Crop depth image
                depth = depth[y1:y2, x1:x2]
                
                # Adjust camera intrinsics for cropping
                # Focal lengths stay the same, but principal point shifts
                K = K.copy()
                K[0, 2] = K[0, 2] - x1  # cx
                K[1, 2] = K[1, 2] - y1  # cy
            
            with self.frame_lock:
                self.latest_rgb = rgb
                self.latest_depth = depth
                self.latest_camera_info = camera_info_msg
                self.camera_intrinsic = K
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def _reset_pose(self):
        
        if self.detector_yaml_path is None:
            print("No YAML path available for reset.")
            return

        pose = load_pose_from_yaml(self.detector_yaml_path)

        if pose is not None:
            self.current_pose = pose
            print("Pose reset to original YAML.")
        else:
            print("Failed to reload pose.")
    
    
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
        print("\n=== Starting Pose Configurator ===")
        
        # Wait for first frame
        print("Waiting for camera data...")
        rospy.sleep(1.0)
        
        with self.frame_lock:
            if self.latest_rgb is None or self.latest_depth is None:
                print("Error: No camera data received")
                return
            
            rgb_frame = self.latest_rgb.copy()
            depth_frame = self.latest_depth.copy()
        
                   
        print("\n=== Pose Configurator Controls ===")
        print("W/S A/D R/F → translate")
        print("Q/E → yaw")
        print("T/G → pitch")
        print("Z/C → roll")
        
        print("Press 'Esc' to quit")
        
        fps_start_time = time.time()
        
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_rgb is None or self.latest_depth is None:
                    rospy.sleep(0.1)
                    continue
                
                rgb_frame = self.latest_rgb.copy()
                depth_frame = self.latest_depth.copy()
            
            key = cv2.waitKey(1) & 0xFF
            self.current_pose = configure_pose(key, self.current_pose)
            
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
                try:
                    overlay_rgba = self._render_model_overlay(
                        self.cad_mesh, 
                        self.current_pose, 
                        self.camera_intrinsic, 
                        vis_frame.shape[1], 
                        vis_frame.shape[0]
                    )
                    vis_frame = overlay_rgba_on_bgr(vis_frame, overlay_rgba)
                except Exception as e:
                    print(f"Error when drawing model overlay: {e}")
                    print("It is likely caused because the model is not visible in the camera frame.")
            
            cv2.imshow('Pose Configurator', vis_frame)
                   
            # Save     
            if key == ord('p'):
                # timestamp with date + time
                stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                base_path = self.detector_yaml_path
                if base_path is not None:
                    root, ext = os.path.splitext(base_path)
                    save_path = f"{root}_edit_{stamp}{ext}"
                else:
                    save_path = f"pose_{stamp}{ext}"
                print("No detector YAML path defined.")
                save_pose_yaml(save_path, self.current_pose)

            # Reset
            elif key == ord('x'):
                self._reset_pose()
            
            # Exit
            elif key == 27: # Esc
                break
            
            self.frame_count += 1
            #rospy.sleep(0.01)
        
        cv2.destroyAllWindows()
        print("\Pose Configuration stopped.")


def main():
    
    rospy.init_node("pose_configurator", anonymous=False)
    
    detector_yaml = rospy.get_param('~detector_yaml', '')

    # Zed camera type
    zed_type = rospy.get_param("~zed_type", "zed")

    # CAD model
    cad_model_path = rospy.get_param("~cad_model_path", "articulation-tracker/meta_files/handle.obj")
    cad_scale = rospy.get_param("~cad_scale", 0.001)

    # Other options
    verbose = rospy.get_param("~verbose", False)
    render_model = rospy.get_param("~render_model", False)
    
    # Image resize parameters (None means no resizing)
    image_width = rospy.get_param("~image_width", None)
    image_height = rospy.get_param("~image_height", None)
    
    target_width = rospy.get_param('~target_width', None)
    target_height = rospy.get_param('~target_height', None)

    # Build argument object (optional: can wrap in a simple class or dict)
    class Args:
        pass

    args = Args()
    args.detector_yaml = detector_yaml
    args.zed_type = zed_type
    args.cad_model_path = cad_model_path
    args.cad_scale = cad_scale
    args.verbose = verbose
    args.render_model = render_model
    args.image_width = image_width
    args.image_height = image_height
    args.target_width = target_width
    args.target_height = target_height

    # Initialize pose configurator
    configurator = PoseConfigurator(args)

    try:
        configurator.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

