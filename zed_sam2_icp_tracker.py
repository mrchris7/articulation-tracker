#!/usr/bin/env python3

import sys
import os
import argparse
import cv2
import numpy as np
import torch
import time
import open3d as o3d

# parent directory to path to import SAM2 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sam2_video_predictor'))

from sam2.build_sam import build_sam2_camera_predictor
from cam_utils import ZEDStreamer

from point_cloud_utils import depth_to_point_cloud, preprocess_point_cloud
from icp_pose_estimator import ICPPoseEstimator
from visualization_utils import draw_pose_axes, project_mesh_to_image, overlay_mask
from recorded_streamer import RecordedZEDStreamer


class PoseTracker:
    """Main pose tracking class."""
    
    def __init__(self, args):
        """
        Initialize pose tracker.
        
        Args:
            args: argparse.Namespace - command line arguments
        """
        self.args = args
        
        # Initialize camera (live ZED or recorded sequence)
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
            raise RuntimeError("Camera intrinsic matrix unavailable from the selected camera source.")
        
        # Initialize SAM2
        print("Initializing SAM2...")
        self.sam2_predictor = build_sam2_camera_predictor(
            config_file=args.model_cfg,
            ckpt_path=args.checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sam2_initialized = False
        
        # Initialize ICP pose estimator
        print(f"Loading CAD model from {args.cad_model}...")
        self.icp_estimator = ICPPoseEstimator(
            cad_model_path=args.cad_model,
            voxel_size=args.voxel_size,
            max_correspondence_distance=args.max_correspondence_distance
        )
        
        # Tracking state
        self.current_pose = None
        self.frame_count = 0
        self.bbox = None
        self.tracking_active = False
        
        # Mouse callback data for bounding box drawing
        self.mouse_data = {
            'drawing': False,
            'start_point': None,
            'end_point': None,
            'bbox': None,
            'new_bbox': False
        }
    
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
    
    def initialize_tracking(self, rgb_frame, depth_frame):
        """
        Initialize tracking on first frame with user-drawn bounding box.
        
        Args:
            rgb_frame: numpy array (H, W, 3) - RGB frame
            depth_frame: numpy array (H, W) - depth frame
        
        Returns:
            bool - True if initialization successful
        """
        print("\n=== Initialization Phase ===")
        print("Draw a bounding box around the object and press 's' to start tracking")
        print("Press 'q' to quit")
        
        # Create window and set mouse callback
        cv2.namedWindow('Initialization - Draw Bounding Box')
        cv2.setMouseCallback('Initialization - Draw Bounding Box', self.mouse_callback)
        
        while True:
            # Display frame with bounding box
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
        
        print(f"Bounding box: {bbox_array}")
        
        # Initialize SAM2 with first frame
        print("Running SAM2 on first frame...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam2_predictor.load_first_frame(rgb_frame)
            
            # Add bounding box prompt
            frame_idx = 0
            obj_id = 1
            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_prompt(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox=bbox_array
            )
        
        # Get mask from SAM2 output
        # out_mask_logits shape is (batch_size, 1, H, W) or (H, W)
        mask_logits = out_mask_logits[0]
        if mask_logits.dim() == 3:  # (1, H, W)
            mask_logits = mask_logits.squeeze(0)  # (H, W)
        mask = (mask_logits > 0.0).cpu().numpy()
        
        # Resize mask to match depth frame if needed
        if mask.shape != depth_frame.shape:
            mask = cv2.resize(mask.astype(np.uint8), 
                            (depth_frame.shape[1], depth_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        print(f"Mask area: {mask.sum()} pixels")
        
        if mask.sum() == 0:
            print("Error: No mask generated. Please try again with a different bounding box.")
            return False
        
        # Extract object point cloud from depth and mask
        print("Extracting object point cloud...")
        object_pcd = depth_to_point_cloud(
            depth_frame,
            self.camera_intrinsic,
            mask=mask
        )
        
        if len(object_pcd.points) == 0:
            print("Error: No valid points in point cloud. Check depth data.")
            return False
        
        print(f"Object point cloud: {len(object_pcd.points)} points")
        
        # Estimate initial pose using ICP
        print("Estimating initial pose...")
        initial_pose, fitness, rmse = self.icp_estimator.estimate_pose(object_pcd)
        
        print(f"Initial pose estimated:")
        print(f"  Fitness: {fitness:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Translation: {initial_pose[:3, 3]}")
        
        # Store initial pose
        self.current_pose = initial_pose
        self.tracking_active = True
        self.sam2_initialized = True
        
        return True

    def _get_next_frame(self):
        """Abstract camera read so both live and recorded streams are supported."""
        if hasattr(self.camera, "zed"):
            status = self.camera.zed.grab(self.camera.runtime_params)
            if status != self.camera.sl.ERROR_CODE.SUCCESS:
                return None, None
        rgb_frame, depth_frame = self.camera.get_frame()
        if rgb_frame is not None:
            intrinsic = getattr(self.camera, "intrinsic", None)
            if intrinsic is not None:
                self.camera_intrinsic = intrinsic
        return rgb_frame, depth_frame
    
    def track_frame(self, rgb_frame, depth_frame):
        """
        Track object in current frame.
        
        Args:
            rgb_frame: numpy array (H, W, 3) - RGB frame
            depth_frame: numpy array (H, W) - depth frame
        
        Returns:
            tuple: (mask, pose, fitness, rmse)
        """
        # SAM2 mask propagation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = self.sam2_predictor.track(rgb_frame)
        
        # Get mask (assuming single object)
        # out_mask_logits shape is (batch_size, 1, H, W) or (H, W)
        mask_logits = out_mask_logits[0]
        if mask_logits.dim() == 3:  # (1, H, W)
            mask_logits = mask_logits.squeeze(0)  # (H, W)
        mask = (mask_logits > 0.0).cpu().numpy()
        
        # Resize mask to match depth frame if needed
        if mask.shape != depth_frame.shape:
            mask = cv2.resize(mask.astype(np.uint8), 
                            (depth_frame.shape[1], depth_frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Extract object point cloud
        object_pcd = depth_to_point_cloud(
            depth_frame,
            self.camera_intrinsic,
            mask=mask
        )


        # Temp visualization
        object_pcd.paint_uniform_color([1, 0.706, 0])   # yellow-ish
        #self.icp_estimator.cad_pcd.paint_uniform_color([0, 0.651, 0.929]) # blue-ish
        o3d.visualization.draw_geometries([object_pcd, self.icp_estimator.cad_pcd])

        
        if len(object_pcd.points) == 0:
            # Return previous pose if no points
            return mask, self.current_pose, 0.0, float('inf')
        
        # Estimate pose using ICP with previous pose as initial guess
        pose, fitness, rmse = self.icp_estimator.estimate_pose(
            object_pcd,
            previous_pose=self.current_pose
        )
        
        # Update current pose
        self.current_pose = pose
        
        return mask, pose, fitness, rmse
    
    def run(self):
        """Main tracking loop."""
        print("\n=== Starting Pose Tracking ===")
        
        # Get first frame for initialization
        rgb_frame, depth_frame = self._get_next_frame()
        print("rgb_frame: ", rgb_frame)
        print("depth_frame: ", depth_frame)
        if rgb_frame is None:
            print("Error: Could not get frame from camera")
            return
        
        # Initialize tracking
        if not self.initialize_tracking(rgb_frame, depth_frame):
            print("Initialization failed or cancelled")
            self.camera.stop()
            return
        
        print("\n=== Tracking Active ===")
        print("Press 'q' to quit")
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            rgb_frame, depth_frame = self._get_next_frame()
            if rgb_frame is None:
                break
            
            # Track object
            mask, pose, fitness, rmse = self.track_frame(rgb_frame, depth_frame)
            
            # Visualize
            vis_frame = rgb_frame.copy()
            
            # Overlay mask
            vis_frame = overlay_mask(vis_frame, mask, color=(0, 255, 0), alpha=0.3)
            
            # Draw pose axes
            if pose is not None:
                print("pose found: ", pose)

                vis_frame = draw_pose_axes(
                    vis_frame,
                    pose,
                    self.camera_intrinsic,
                    length=0.1
                )
                
                # Optionally project mesh
                if self.args.show_mesh:
                    vis_frame = project_mesh_to_image(
                        vis_frame,
                        self.icp_estimator.cad_mesh,
                        pose,
                        self.camera_intrinsic
                    )
            else:
                print("No pose found")
            
            # Display info
            info_text = [
                f"Frame: {self.frame_count}",
                f"Fitness: {fitness:.3f}",
                f"RMSE: {rmse:.4f}",
                f"Mask pixels: {mask.sum()}",
            ]
            
            if pose is not None:
                t = pose[:3, 3]
                info_text.append(f"Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            
            y_offset = 20
            for i, text in enumerate(info_text):
                cv2.putText(
                    vis_frame,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = 30.0 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_counter = 0
                cv2.putText(
                    vis_frame,
                    f"FPS: {fps:.1f}",
                    (10, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            cv2.imshow('Pose Tracking', vis_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            self.frame_count += 1
        
        # Cleanup
        self.camera.stop()
        cv2.destroyAllWindows()
        print("\nTracking stopped.")


def main():
    parser = argparse.ArgumentParser(description="6D Pose Tracking with ZED, SAM2, and ICP")
    
    # source selection
    parser.add_argument("--camera_source", choices=["zed", "recorded"], default="zed", help="Select live ZED camera or recorded frames")
    parser.add_argument("--recorded_root", type=str, default=os.path.join("data", "scene0000_00"), help="Root folder for recorded RGB-D sequence")
    parser.add_argument("--recorded_color_subdir", type=str, default="color", help="Relative path under recorded_root that contains color frames")
    parser.add_argument("--recorded_depth_subdir", type=str, default="depth", help="Relative path under recorded_root that contains depth frames")
    parser.add_argument("--recorded_intrinsic", type=str, default=None, help="Optional explicit path to intrinsic matrix (defaults to recorded_root/intrinsics/intrinsic_color.txt)")
    parser.add_argument("--recorded_depth_scale", type=float, default=1000.0, help="Scale factor that converts recorded depth values to meters")
    
    # camera parameters
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--depth_mode", type=int, default=2, help="ZED depth mode (0=ULTRA, 1=QUALITY, 2=PERFORMANCE, 3=NEURAL)")
    parser.add_argument("--resolution", type=int, default=2, help="ZED resolution (0=HD2K, 1=HD1080, 2=HD720, 3=VGA)")
    parser.add_argument("--depth_min", type=float, default=0.2, help="Minimum depth (meters)")
    parser.add_argument("--depth_max", type=float, default=10.0, help="Maximum depth (meters)")
    
    # SAM2 parameters
    parser.add_argument("--checkpoint", type=str, default="./sam2_video_predictor/checkpoints/sam2_hiera_small.pt", help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_s.yaml", help="SAM2 model config file")
    
    # CAD model
    parser.add_argument("--cad_model", type=str, default="./data/models/handle.stl", required=True, help="Path to CAD model file (.obj, .ply, .stl)")
    
    # ICP parameters
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for point cloud downsampling (meters)")
    parser.add_argument("--max_correspondence_distance", type=float, default=0.05, help="Max correspondence distance for ICP (meters)")
    
    # visualization
    parser.add_argument("--show_mesh", action="store_true", help="Show projected mesh wireframe")

    args = parser.parse_args()
    
    if not os.path.exists(args.cad_model):
        print(f"Error: CAD model file not found: {args.cad_model}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: SAM2 checkpoint not found: {args.checkpoint}")
        return
    
    tracker = PoseTracker(args)
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

