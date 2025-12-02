"""
ICP-based pose tracking module.

Pipeline:
    1. First frame: Store point cloud as reference, pose = identity (translation [0, 0, 0])
    2. Subsequent frames: ICP between previous frame's point cloud and current frame's point cloud
    3. Compose transformations to get pose in first frame's coordinate system
"""
import copy
import numpy as np
import open3d as o3d
from point_cloud_utils import preprocess_point_cloud


class ICPPoseEstimator:
    """
    ICP-based 6D pose tracker.
    
    Tracks the handle's pose by comparing consecutive frames' point clouds.
    Pose is expressed in the first frame's coordinate system.
    """
    
    def __init__(self, voxel_size=0.01, max_correspondence_distance=0.05):
        """
        Initialize ICP pose tracker.
        
        Args:
            voxel_size: float - voxel size for point cloud preprocessing
            max_correspondence_distance: float - maximum correspondence distance for ICP
        """
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        
        # Reference point cloud from first frame
        self.reference_pcd = None
        self.previous_pcd = None
        
        # Pose in first frame's coordinate system
        self.pose_in_reference_frame = np.eye(4)  # Identity for first frame
    
    def initialize(self, first_frame_pcd):
        """
        Initialize tracking with first frame's point cloud.
        
        Args:
            first_frame_pcd: open3d.geometry.PointCloud - first frame's point cloud
        """
        # Preprocess and store reference point cloud
        self.reference_pcd = preprocess_point_cloud(
            copy.deepcopy(first_frame_pcd),
            voxel_size=self.voxel_size,
            estimate_normals=True
        )
        self.previous_pcd = copy.deepcopy(self.reference_pcd)
        
        # Store reference center (handle center in first frame)
        self.reference_center = self.reference_pcd.get_center()
        
        # First frame: pose translation is the actual handle position (reference_center)
        # This ensures visualization is at the correct location
        self.pose_in_reference_frame = np.eye(4)
        self.pose_in_reference_frame[:3, 3] = self.reference_center
        
        print(f"Initialized tracking: {len(self.reference_pcd.points)} reference points")
        print(f"Reference center: {self.reference_center}")
    
    def track_frame(self, current_pcd):
        """
        Track handle pose by comparing current frame with previous frame.
        
        Pipeline: ICP(previous_pcd, current_pcd) → T_prev_to_curr → compose with previous pose
        
        Args:
            current_pcd: open3d.geometry.PointCloud - current frame's point cloud
        
        Returns:
            tuple: (pose_in_reference_frame (4, 4), fitness score, inlier_rmse)
        """
        if self.previous_pcd is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        
        if len(current_pcd.points) == 0:
            return self.pose_in_reference_frame, 0.0, float('inf')
        
        # Preprocess current point cloud
        current_pcd_processed = preprocess_point_cloud(
            copy.deepcopy(current_pcd),
            voxel_size=self.voxel_size,
            estimate_normals=True
        )
        
        if len(current_pcd_processed.points) == 0:
            return self.pose_in_reference_frame, 0.0, float('inf')
        
        # Calculate adaptive correspondence distance
        prev_center = self.previous_pcd.get_center()
        curr_center = current_pcd_processed.get_center()
        center_distance = np.linalg.norm(curr_center - prev_center)
        
        adaptive_max_distance = max(
            self.max_correspondence_distance,
            center_distance * 1.5,
            self.max_correspondence_distance * 3.0
        )
        adaptive_max_distance = min(adaptive_max_distance, 0.5)
        
        # Run ICP: previous_pcd (source) -> current_pcd (target)
        # This gives T_prev_to_curr: transforms previous frame coords to current frame coords
        if center_distance > self.max_correspondence_distance * 2:
            # Two-stage ICP for poor initial alignment
            result_coarse = o3d.pipelines.registration.registration_icp(
                self.previous_pcd,  # source
                current_pcd_processed,  # target
                max_correspondence_distance=adaptive_max_distance,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=30
                )
            )
            
            if result_coarse.fitness > 0:
                # Refine with point-to-plane ICP
                scene_center_after_coarse = (result_coarse.transformation[:3, :3] @ prev_center) + result_coarse.transformation[:3, 3]
                center_dist_after_coarse = np.linalg.norm(scene_center_after_coarse - curr_center)
                refine_max_distance = max(
                    self.max_correspondence_distance,
                    center_dist_after_coarse * 1.5,
                    result_coarse.inlier_rmse * 3.0
                )
                refine_max_distance = min(refine_max_distance, 0.5)
                
                result = o3d.pipelines.registration.registration_icp(
                    self.previous_pcd,
                    current_pcd_processed,
                    max_correspondence_distance=refine_max_distance,
                    init=result_coarse.transformation,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=30
                    )
                )
                
                if result.fitness == 0.0:
                    result = result_coarse
            else:
                result = result_coarse
        else:
            # Good alignment, use point-to-plane ICP directly
            result = o3d.pipelines.registration.registration_icp(
                self.previous_pcd,  # source
                current_pcd_processed,  # target
                max_correspondence_distance=adaptive_max_distance,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=50
                )
            )
        
        if result.fitness == 0.0:
            # ICP failed, keep previous pose
            return self.pose_in_reference_frame, 0.0, float('inf')
        
        # T_prev_to_curr transforms previous frame coords to current frame coords
        T_prev_to_curr = result.transformation
        
        # Get handle centers
        current_center = current_pcd_processed.get_center()
        
        # Compute pose in reference frame:
        # 1. Translation: current_center (actual handle position in camera coordinates)
        #    This includes the initial reference_center position + displacement
        # 2. Rotation: compose rotations from ICP transformations
        
        # For rotation: compose with previous rotation: T_prev_to_curr rotation tells us how handle rotated from previous to current frame
        R_prev_to_curr = T_prev_to_curr[:3, :3]
        R_ref_to_prev = self.pose_in_reference_frame[:3, :3]
        R_ref_to_curr = R_ref_to_prev @ R_prev_to_curr
        
        # For translation: use actual handle center position in camera coordinates
        # This is reference_center + displacement, which equals current_center
        translation = current_center
        
        # Construct pose matrix
        self.pose_in_reference_frame = np.eye(4)
        self.pose_in_reference_frame[:3, :3] = R_ref_to_curr
        self.pose_in_reference_frame[:3, 3] = translation
        
        # Update previous point cloud for next frame
        self.previous_pcd = current_pcd_processed
        
        return self.pose_in_reference_frame, result.fitness, result.inlier_rmse
    
    def get_handle_center(self):
        """
        Get handle center position in reference frame.
        
        Returns:
            numpy array (3,) - handle center position [x, y, z] in reference frame
        """
        if self.reference_pcd is None:
            return np.array([0.0, 0.0, 0.0])
        
        # Handle center in reference frame is the translation part of the pose
        return self.pose_in_reference_frame[:3, 3]
