#!/usr/bin/env python3
"""
SAM2 + ICG Tracker Integration

Pipeline:
1. Read frames (recorded or live)
2. Apply SAM2 segmentation
3. Create point cloud from segmented area
4. Globally match point cloud against CAD model to find initial pose
5. Pass initial pose to ICG tracker
6. Receive poses from ICG tracker at every step
"""

# run: cd pose-estimation/ICG/build && make -j && cd ../../..
# run: python articulation-sim/sam2_icg_tracker.py --camera_source recorded --camera_metafile pose-estimation/ICG/examples/pearl_lab/zed_color.yaml --body_metafile pose-estimation/ICG/examples/pearl_lab/handle.yaml --cad_model_path pose-estimation/ICG/examples/pearl_lab/handle.obj --render_model

import sys
import os
import argparse
import cv2
import numpy as np
import torch
import time
import copy
import open3d as o3d

# Add paths for sam2 and icg
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sam2_video_predictor'))
# Add icg python bindings
icg_build_path = os.path.join(os.path.dirname(__file__), '..', 'pose-estimation', 'ICG', 'build', 'python_bindings')
icg_src_path = os.path.join(os.path.dirname(__file__), '..', 'pose-estimation', 'ICG', 'python_bindings')
if os.path.exists(icg_build_path):
    sys.path.insert(0, icg_build_path)
elif os.path.exists(icg_src_path):
    sys.path.insert(0, icg_src_path)

from sam2.build_sam import build_sam2_camera_predictor
from cam_utils import ZEDStreamer
from point_cloud_utils import (
    depth_to_point_cloud, 
    preprocess_point_cloud, 
    load_cad_model,
    requires_ransac,
    downsample_and_extract_features,
    global_ransac_registration
)
from recorded_streamer import RecordedZEDStreamer
from visualization_utils import draw_pose_axes, overlay_mask, depth_to_colormap, overlay_rgba_on_bgr

import icg_tracker  # build with BUILD_PYTHON_BINDINGS=ON


class ICGTrackerWrapper:
    """Wrapper for ICG tracker with Python interface."""
    
    def __init__(self, camera_metafile, body_metafile, temp_directory, 
                 region_model_path=None):
        """
        Initialize ICG tracker.
        
        Args:
            camera_metafile: path to camera YAML file
            body_metafile: path to body YAML file
            temp_directory: directory for temporary files (region model)
            region_model_path: optional path to pre-computed region model
        """
        
        self.temp_directory = temp_directory
        os.makedirs(temp_directory, exist_ok=True)
        
        # Create tracker and renderer geometry
        self.tracker = icg_tracker.Tracker("tracker")
        self.renderer_geometry = icg_tracker.RendererGeometry("renderer_geometry")
        
        # Create camera
        self.camera = icg_tracker.LoaderColorCamera("color_camera", camera_metafile)
        
        # Create body
        self.body = icg_tracker.Body("body", body_metafile)
        self.renderer_geometry.add_body(self.body)
        
        # Create region model
        if region_model_path is None:
            region_model_path = os.path.join(temp_directory, "body_region_model.bin")
        self.region_model = icg_tracker.RegionModel(
            "body_region_model", 
            self.body, 
            region_model_path
        )
        
        # Create region modality
        self.region_modality = icg_tracker.RegionModality(
            "body_region_modality",
            self.body,
            self.camera,
            self.region_model
        )
        
        # Create optimizer
        self.optimizer = icg_tracker.Optimizer("body_optimizer")
        self.optimizer.add_modality(self.region_modality)
        self.tracker.add_optimizer(self.optimizer)
        
        # Create Python callback publisher (callback will be set later)
        self.pose_callback = None
        self.publisher = icg_tracker.PythonCallbackPublisher("python_publisher")
        self.publisher.set_body_ptrs([self.body])
        
        # set a dummy callback initially so SetUp() does not fail
        # -> the real callback will be set later via set_pose_callback()
        self.publisher.set_callback(lambda it, name, pose: None)
        self.tracker.add_publisher(self.publisher)
        
        # dn't set up tracker yet - wait for initial pose to create detector
        self.detector = None
        self.initialized = False
        self.last_loaded_frame = -1  # Track last loaded frame index
    
    def set_pose_callback(self, callback):
        """Set callback function for pose updates."""
        self.pose_callback = callback
        # Update the callback (publisher is already set up)
        self.publisher.set_callback(callback)
    
    def set_initial_pose_and_setup(self, pose_matrix):
        """
        Set initial pose using StaticDetector and set up tracker.
        
        Args:
            pose_matrix: numpy array (4, 4) - initial pose in camera frame
        """
        # Create StaticDetector with initial pose
        self.detector = icg_tracker.StaticDetector(
            "body_detector",
            self.body,
            pose_matrix.astype(np.float32)
        )
        self.tracker.add_detector(self.detector)
        
        # Now set up tracker (this will set up all objects including detector)
        if not self.tracker.set_up():
            raise RuntimeError("Failed to set up ICG tracker")
        
        # Execute detection to set the body pose
        if not self.tracker.execute_detection_cycle(0):
            raise RuntimeError("Failed to execute detection")
        
        # last loaded frame is initial_load_index
        initial_load_index = self.camera.load_index() - 1  # SetUp() incremented it
        self.last_loaded_frame = initial_load_index
        
        self.initialized = True
    
    def track_frame(self, frame_index):
        """
        Track object in current frame.
        
        Args:
            frame_index: int - frame index to load
        
        Returns:
            bool - True if tracking successful
        """
        if not self.initialized:
            print("Tracker not initialized")
            return False
        
        # Check whether to change load_index
        current_load_index = self.camera.load_index()
        
        if current_load_index != frame_index:
            self.camera.set_load_index(frame_index)
            if not self.camera.set_up():
                print(f"Failed to set up camera for frame {frame_index}")
                return False

            skip_update = True
        else:
            skip_update = False
        
        if not skip_update:
            if not self.tracker.update_cameras(False):
                print("Failed to update cameras")
                return False
        
        self.last_loaded_frame = frame_index
        
        # Execute tracking cycle
        if not self.tracker.execute_tracking_cycle(0):
            print("Failed to execute tracking cycle")
            return False
        
        return True
    
    def get_current_pose(self):
        """Get current pose from body."""
        return self.body.body2world_pose()


class SAM2ICGTracker:
    """Main tracker combining SAM2 detection with ICG tracking."""
    
    def __init__(self, args):
        """Initialize tracker."""
        self.args = args
        
        # Initialize camera
        self.camera_source = args.camera_source
        if self.camera_source == "recorded":
            print("Initializing recorded RGB-D streamer...")
            recorded_root = args.recorded_root
            color_dir = os.path.join(recorded_root, args.recorded_color_subdir)
            depth_dir = os.path.join(recorded_root, args.recorded_depth_subdir)
            intrinsic_path = (
                args.recorded_intrinsic
                if args.recorded_intrinsic is not None
                else os.path.join(recorded_root, "intrinsics", "intrinsic_color.txt")
            )
            self.camera = RecordedZEDStreamer(
                color_dir=color_dir,
                depth_dir=depth_dir,
                intrinsic_path=intrinsic_path,
                depth_scale=args.recorded_depth_scale,
            )
            self.camera.set_params(
                width=args.width,
                height=args.height,
                fps=args.fps,
                close=args.depth_min,
                far=args.depth_max,
            )
        else:
            print("Initializing ZED camera...")
            self.camera = ZEDStreamer()
            self.camera.set_params(
                width=args.width,
                height=args.height,
                fps=args.fps,
                depth_mode=args.depth_mode,
                resolution=args.resolution,
                close=args.depth_min,
                far=args.depth_max,
            )
        
        self.camera.start()
        self.camera_intrinsic = getattr(self.camera, "intrinsic", None)
        if self.camera_intrinsic is None:
            raise RuntimeError("Camera intrinsic matrix unavailable")
        
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
        # Center CAD model at origin in its own coordinate frame
        cad_pcd_copy = copy.deepcopy(self.cad_pcd)
        cad_center = cad_pcd_copy.get_center()
        # cad_pcd_copy.translate(-cad_center)  # Center at origin
        print(f"CAD model centered at origin (offset: {cad_center})")
        
        self.cad_pcd_processed = preprocess_point_cloud(
            cad_pcd_copy,
            voxel_size=args.voxel_size,
            estimate_normals=True
        )
        
        print("Initializing ICG tracker...")
        self.icg_tracker = ICGTrackerWrapper(
            camera_metafile=args.camera_metafile,
            body_metafile=args.body_metafile,
            temp_directory=args.temp_directory
        )
        
        # Set pose callback
        self.icg_tracker.set_pose_callback(self._on_pose_update)
        
        # Tracking state
        self.frame_count = 0
        self.tracking_active = False
        self.current_pose = None
        self.last_icg_pose = None
        
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
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.camera.width, self.camera.height)
            self.scene = self.renderer.scene
            self.scene.set_background([0, 0, 0, 0])
            self.material = o3d.visualization.rendering.MaterialRecord()
            self.material.shader = 'defaultLit' # 'defaultLitTransparency' is not producing rgba image
            self.material.base_color = [0, 0, 255, 1]

        self.save_frames = args.save_frames
        self.output_dir = args.output_dir
        
        # Setup output directory if saving frames
        if self.save_frames:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Frames will be saved to: {self.output_dir}")
    
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
    
    def _on_pose_update(self, iteration, body_name, pose_array):
        """Callback for ICG pose updates."""
        # Convert pose array to numpy matrix
        pose = np.array(pose_array).reshape(4, 4)
        self.last_icg_pose = pose
        if self.args.verbose:
            print(f"ICG pose update (iteration {iteration}, body {body_name}):")
            print(f"  Translation: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")
    
    def _estimate_initial_pose(self, object_pcd):
        """
        Estimate initial pose by matching object point cloud to CAD model.
        
        The CAD model is in its own coordinate frame (model frame). We search for 
        the transformation T_model_to_camera that places the CAD model in the camera 
        frame (world frame). This transformation is the body2world_pose for ICG.
        
        Registration: source (CAD model in model frame) -> target (observed object in camera frame)
        Result: T_model_to_camera transforms points from model frame to camera frame
        
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
        
        # Debug: Check point cloud centers
        object_center = np.mean(np.asarray(object_pcd_processed.points), axis=0)
        cad_center = np.mean(np.asarray(self.cad_pcd_processed.points), axis=0)
        print(f"Object center (camera frame): {object_center}")
        print(f"CAD center (model frame): {cad_center}")
        
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
            
            # Global registration: source (object in camera frame) -> target (CAD in model frame)
            # -> gives T_camera_to_model, we need T_model_to_camera
            result_ransac = global_ransac_registration(
                object_down,        # source: object in camera frame
                cad_down,           # target: CAD in model frame
                object_fpfh, 
                cad_fpfh, 
                self.args.voxel_size
            )

            # return np.array([
            #     [0.0, 0.86, 0.5, 0.011876],
            #     [1, 0, 0, -0.25546736],
            #     [0.0, 0.5, -0.86, 1.118],
            #     [0, 0, 0, 1 ]])
            
            if result_ransac.fitness < 0.1:
                print(f"RANSAC registration failed (fitness: {result_ransac.fitness:.3f})")
                return None
            
            print(f"RANSAC fitness: {result_ransac.fitness:.3f}")
            T_model_to_camera = np.linalg.inv(result_ransac.transformation)
            print(f"RANSAC transform translation: {T_model_to_camera[:3, 3]}")
            # Result is T_camera_to_model, invert to get T_model_to_camera
        else:
            print("Using ICP directly (good initial alignment)...")
            T_model_to_camera = np.eye(4)

        return T_model_to_camera
    
    
    def _render_model_overlay(self, mesh, T_model_to_camera, K, width, height):        

        # Transform mesh
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.transform(T_model_to_camera.copy())

        # Add to scene
        self.scene.clear_geometry()
        self.scene.add_geometry("model", mesh_copy, self.material)
        

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            K[0, 0],  # fx
            K[1, 1],  # fy
            K[0, 2],  # cx
            K[1, 2]   # cy
        )

        T_camera_to_world = np.eye(4, dtype=np.float32)

        # Set up camera
        self.renderer.setup_camera(
            intrinsic,
            T_camera_to_world
        )

        # Render RGBA image
        img = self.renderer.render_to_image()
        img = np.asarray(img)  # H × W × 4 (RGBA)

        if img.shape[2] == 3:  # no alpha channel
            # construct an alpha channel
            # and asign 0 for black pixels, 0.8 for others
            alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
            img = np.concatenate([img, alpha], axis=2)  # now HxWx4
            # print(f"Rendered image shape: {img.shape}")
            # cv2.imshow("Overlay", img)
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0  # shape (H,W,1)
            mask_non_black = np.any(rgb >= 10, axis=2, keepdims=True)
            alpha[:] = 0.8 * mask_non_black.astype(np.float32)
            img[:, :, 3] = (alpha[:, :, 0] * 255).astype(np.uint8)

        return img
    
    def initialize_tracking(self, rgb_frame, depth_frame):
        """Initialize tracking with first frame."""
        print("\n=== Initialization Phase ===")
        print("Draw a bounding box around the object and press 's' to start tracking")
        
        cv2.namedWindow('Initialization - Draw Bounding Box')
        cv2.setMouseCallback('Initialization - Draw Bounding Box', self.mouse_callback)
        
        while True:
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
        
        # Set initial pose in ICG tracker using StaticDetector
        if self.icg_tracker:
            print("Setting initial pose in ICG tracker using StaticDetector...")
            self.icg_tracker.set_initial_pose_and_setup(initial_pose)
            self.current_pose = initial_pose
            self.tracking_active = True
            # Start tracking
            self.icg_tracker.tracker.start_tracking()
        
        return True
    
    def track_frame(self, rgb_frame, depth_frame):
        """Track object in current frame using ICG only (no SAM2 after first frame)."""
        # Track with icg only, no sam2 segmentation after initialization
        if self.icg_tracker and self.tracking_active:
            if not self.icg_tracker.track_frame(self.frame_count):
                print(f"Warning: ICG tracking failed for frame {self.frame_count}")
            
            # Get pose from icg
            if self.last_icg_pose is not None:
                self.current_pose = self.last_icg_pose
        
        # empty mask
        mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=bool)
        return mask
    
    def run(self):
        """Main tracking loop."""
        print("\n=== Starting SAM2 + ICG Tracking ===")
        
        # Get first frame
        rgb_frame, depth_frame = self.camera.get_frame()
        if rgb_frame is None:
            print("Error: Could not get frame from camera")
            return
        
        # Initialize
        if not self.initialize_tracking(rgb_frame, depth_frame):
            print("Initialization failed")
            self.camera.stop()
            return
        
        print("\n=== Tracking Active ===")
        print("Press 'q' to quit")
        
        fps_start_time = time.time()
        
        while True:
            rgb_frame, depth_frame = self.camera.get_frame()
            if rgb_frame is None:
                break
            
            # Track
            mask = self.track_frame(rgb_frame, depth_frame)
            
            # Visualize
            vis_frame = rgb_frame.copy()
            # Only show mask if we have one (first frame)
            if mask.sum() > 0:
                vis_frame = overlay_mask(vis_frame, mask, color=(0, 255, 0), alpha=0.3)
            
            # Draw pose
            if self.current_pose is not None:
                vis_frame = draw_pose_axes(
                    vis_frame,
                    self.current_pose,
                    self.camera_intrinsic,
                    length=0.1
                )
            
            # Display info
            info_text = [
                f"Frame: {self.frame_count}",
            ]
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

            if self.args.render_model:
                overlay_rgba = self._render_model_overlay(self.cad_mesh, self.current_pose, self.camera_intrinsic, self.camera.width, self.camera.height)
                vis_frame = overlay_rgba_on_bgr(vis_frame, overlay_rgba)
            
            cv2.imshow('SAM2 + ICG Tracking', vis_frame)

            if self.save_frames:
                frame_filename = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            self.frame_count += 1
        
        self.camera.stop()
        cv2.destroyAllWindows()
        print("\nTracking stopped.")


def main():
    parser = argparse.ArgumentParser(description="SAM2 + ICG 6D Pose Tracking")
    
    # Camera source
    parser.add_argument("--camera_source", choices=["zed", "recorded"], default="recorded")
    parser.add_argument("--recorded_root", type=str, default=os.path.join("data", "scene0010_00"))
    parser.add_argument("--recorded_color_subdir", type=str, default="color")
    parser.add_argument("--recorded_depth_subdir", type=str, default="depth")
    parser.add_argument("--recorded_intrinsic", type=str, default=None)
    parser.add_argument("--recorded_depth_scale", type=float, default=1000.0)
    
    # Camera parameters
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth_mode", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--depth_min", type=float, default=0.2)
    parser.add_argument("--depth_max", type=float, default=10.0)
    
    # SAM2 parameters
    parser.add_argument("--checkpoint", type=str, default="./sam2_video_predictor/checkpoints/sam2_hiera_small.pt")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_s.yaml")
    
    # Point cloud and registration parameters
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for point cloud processing")
    
    # ICG parameters
    parser.add_argument("--camera_metafile", type=str, default="pose-estimation/ICG/examples/pearl_lab/zed_color.yaml")
    parser.add_argument("--body_metafile", type=str, default="pose-estimation/ICG/examples/pearl_lab/handle.yaml")
    parser.add_argument("--cad_model_path", type=str, default="pose-estimation/ICG/examples/pearl_lab/handle.obj")
    parser.add_argument("--cad_scale", type=float, default=0.001, help="Scale factor to convert units of cad model to meters")
    parser.add_argument("--temp_directory", type=str, default="./temp_icg")
    
    # Other
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--render_model", action="store_true", help="Render model overlay")
    parser.add_argument("--save_frames", action="store_true", help="Save visualization frames to output directory")
    parser.add_argument("--output_dir", type=str, default="./data/output", help="Directory to save frames (default: /data/output)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.checkpoint}")
        return
    
    tracker = SAM2ICGTracker(args)
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tracker.camera.started:
            tracker.camera.stop()


if __name__ == "__main__":
    main()

