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
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.transform import Slerp


class ICPPoseEstimator:
    """
    ICP-based 6D pose tracker.
    
    Tracks the handle's pose by comparing consecutive frames' point clouds.
    Pose is expressed in the first frame's coordinate system.
    """
    
    def __init__(self, voxel_size=0.01, max_correspondence_distance=0.05, 
                 visualize_point_clouds=False, visualization_frame_interval=30,
                 enable_reference_reregistration=False, reference_reregistration_interval=30,
                 reference_reregistration_min_fitness=0.3,
                 enable_rotation_constraints=False, max_angular_velocity=2.0,
                 rotation_smoothing_alpha=0.7):
        """
        Initialize ICP pose tracker.
        
        Args:
            voxel_size: float - voxel size for point cloud preprocessing
            max_correspondence_distance: float - maximum correspondence distance for ICP
            visualize_point_clouds: bool - whether to show point cloud visualization
            visualization_frame_interval: int - show visualization every N frames
            enable_reference_reregistration: bool - enable periodic re-registration with reference frame
            reference_reregistration_interval: int - re-register with reference every N frames
            reference_reregistration_min_fitness: float - minimum fitness to trigger re-registration
            enable_rotation_constraints: bool - enable rotation constraint validation
            max_angular_velocity: float - maximum angular velocity in rad/s
            rotation_smoothing_alpha: float - rotation smoothing factor (0.0-1.0, higher = more smoothing)
        """
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        self.visualize_point_clouds = visualize_point_clouds
        self.visualization_frame_interval = visualization_frame_interval
        
        # Reference frame re-registration parameters
        self.enable_reference_reregistration = enable_reference_reregistration
        self.reference_reregistration_interval = reference_reregistration_interval
        self.reference_reregistration_min_fitness = reference_reregistration_min_fitness
        self.frame_count = 0
        
        # Rotation constraint parameters
        self.enable_rotation_constraints = enable_rotation_constraints
        self.max_angular_velocity = max_angular_velocity
        self.rotation_smoothing_alpha = rotation_smoothing_alpha
        self.previous_rotation = None
        self.previous_pose = None
        self.frame_time = 1.0 / 30.0  # Assume 30 fps, will be updated if available
        
        # Reference point cloud from first frame
        self.reference_pcd = None
        self.previous_pcd = None
        self.current_pcd_processed = None
        self.previous_pcd_for_viz = None
        
        # Pose in first frame's coordinate system
        self.pose_in_reference_frame = np.eye(4)  # Identity for first frame
        
        # Store last transformation for visualization
        self.last_transformation = None
        self.last_fitness = 0.0
        self.last_rmse = 0.0
    
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
        
        # Initialize rotation smoothing state
        if self.enable_rotation_constraints:
            self.previous_rotation = scipyR.from_matrix(np.eye(3))
            self.previous_pose = np.eye(4)
            self.previous_pose[:3, 3] = self.reference_center
    
    def _register_with_reference(self, current_pcd_processed):
        """
        Register current frame with reference frame to correct drift.
        
        Args:
            current_pcd_processed: open3d.geometry.PointCloud - preprocessed current point cloud
        
        Returns:
            tuple: (transformation (4, 4), fitness, rmse) or None if registration fails
        """
        if self.reference_pcd is None or len(current_pcd_processed.points) == 0:
            return None
        
        # Calculate adaptive correspondence distance based on expected displacement
        ref_center = self.reference_pcd.get_center()
        curr_center = current_pcd_processed.get_center()
        center_distance = np.linalg.norm(curr_center - ref_center)
        
        adaptive_max_distance = max(
            self.max_correspondence_distance,
            center_distance * 1.5,
            self.max_correspondence_distance * 3.0
        )
        adaptive_max_distance = min(adaptive_max_distance, 0.5)
        
        # Use current pose as initial guess for reference-to-current transformation
        # Current pose is T_ref_to_curr, so we can use it directly
        init_transform = self.pose_in_reference_frame.copy()
        
        # Run two-stage ICP for better alignment
        result_coarse = o3d.pipelines.registration.registration_icp(
            self.reference_pcd,  # source
            current_pcd_processed,  # target
            max_correspondence_distance=adaptive_max_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=30
            )
        )
        
        if result_coarse.fitness > 0:
            # Refine with point-to-plane ICP
            refine_max_distance = max(
                self.max_correspondence_distance,
                result_coarse.inlier_rmse * 3.0
            )
            refine_max_distance = min(refine_max_distance, 0.5)
            
            result = o3d.pipelines.registration.registration_icp(
                self.reference_pcd,
                current_pcd_processed,
                max_correspondence_distance=refine_max_distance,
                init=result_coarse.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=50
                )
            )
            
            if result.fitness == 0.0:
                result = result_coarse
        else:
            result = result_coarse
        
        if result.fitness > 0:
            return result.transformation, result.fitness, result.inlier_rmse
        else:
            return None
    
    def _validate_and_smooth_rotation(self, R_new, dt=None):
        """
        Validate rotation against angular velocity constraints and apply smoothing.
        
        Args:
            R_new: numpy array (3, 3) - new rotation matrix
            dt: float - time delta since last frame (seconds), defaults to self.frame_time
        
        Returns:
            numpy array (3, 3) - validated and smoothed rotation matrix
        """
        if dt is None:
            dt = self.frame_time
        
        current_rotation = scipyR.from_matrix(R_new)
        
        # Validate angular velocity
        if self.previous_rotation is not None:
            # Compute relative rotation
            R_rel = self.previous_rotation.inv() * current_rotation
            angle = R_rel.magnitude()
            angular_velocity = angle / dt if dt > 0 else 0.0
            
            if angular_velocity > self.max_angular_velocity:
                # Rotation too fast, reject and use previous rotation
                print(f"WARNING: Angular velocity {angular_velocity:.3f} rad/s exceeds max {self.max_angular_velocity:.3f} rad/s. Using previous rotation.")
                current_rotation = self.previous_rotation
        
        # Apply smoothing using SLERP
        if self.previous_rotation is not None and self.rotation_smoothing_alpha < 1.0:
            key_times = [0, 1]
            key_rots = scipyR.from_matrix([
                self.previous_rotation.as_matrix(),
                current_rotation.as_matrix()
            ])
            slerp = Slerp(key_times, key_rots)
            # Interpolate at (1 - alpha) to get smoothed rotation
            smoothed_rotation = slerp([1 - self.rotation_smoothing_alpha])[0]
            current_rotation = smoothed_rotation
        
        # Update previous rotation
        self.previous_rotation = current_rotation
        
        return current_rotation.as_matrix()
    
    def track_frame(self, current_pcd, dt=None):
        """
        Track handle pose by comparing current frame with previous frame.
        
        Pipeline: ICP(previous_pcd, current_pcd) → T_prev_to_curr → compose with previous pose
        
        Args:
            current_pcd: open3d.geometry.PointCloud - current frame's point cloud
            dt: float - time delta since last frame (seconds), for rotation validation
        
        Returns:
            tuple: (pose_in_reference_frame (4, 4), fitness score, inlier_rmse)
        """
        if dt is not None:
            self.frame_time = dt
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
        
        # Store transformation for visualization
        self.last_transformation = T_prev_to_curr
        self.last_fitness = result.fitness
        self.last_rmse = result.inlier_rmse
        
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
        
        # Apply rotation constraint validation and smoothing if enabled
        if self.enable_rotation_constraints:
            R_ref_to_curr = self._validate_and_smooth_rotation(R_ref_to_curr, dt)
        
        # For translation: use actual handle center position in camera coordinates
        # This is reference_center + displacement, which equals current_center
        translation = current_center
        
        # Construct pose matrix from frame-to-frame tracking
        pose_from_frame_to_frame = np.eye(4)
        pose_from_frame_to_frame[:3, :3] = R_ref_to_curr
        pose_from_frame_to_frame[:3, 3] = translation
        
        # Reference frame re-registration
        should_reregister = False
        if self.enable_reference_reregistration:
            # Check if we should re-register (periodic or low fitness)
            self.frame_count += 1
            if (self.frame_count % self.reference_reregistration_interval == 0 or
                result.fitness < self.reference_reregistration_min_fitness):
                should_reregister = True
        
        if should_reregister:
            ref_result = self._register_with_reference(current_pcd_processed)
            if ref_result is not None:
                T_ref_to_curr_ref, ref_fitness, ref_rmse = ref_result
                
                # Blend between frame-to-frame and reference-based poses based on confidence
                # Use reference result more when frame-to-frame fitness is low
                frame_to_frame_weight = result.fitness
                ref_weight = ref_fitness
                
                # Normalize weights
                total_weight = frame_to_frame_weight + ref_weight
                if total_weight > 0:
                    frame_to_frame_weight /= total_weight
                    ref_weight /= total_weight
                else:
                    frame_to_frame_weight = 0.5
                    ref_weight = 0.5
                
                # Blend rotations using SLERP
                R_ref_to_curr_ref = T_ref_to_curr_ref[:3, :3]
                R_ref_to_curr_frame = pose_from_frame_to_frame[:3, :3]
                
                rot_ref = scipyR.from_matrix(R_ref_to_curr_ref)
                rot_frame = scipyR.from_matrix(R_ref_to_curr_frame)
                
                key_times = [0, 1]
                key_rots = scipyR.from_matrix([rot_frame.as_matrix(), rot_ref.as_matrix()])
                slerp = Slerp(key_times, key_rots)
                blended_rotation = slerp([ref_weight])[0]
                
                # Blend translations
                translation_ref = T_ref_to_curr_ref[:3, 3]
                blended_translation = (frame_to_frame_weight * translation + 
                                     ref_weight * translation_ref)
                
                # Update pose
                self.pose_in_reference_frame = np.eye(4)
                self.pose_in_reference_frame[:3, :3] = blended_rotation.as_matrix()
                self.pose_in_reference_frame[:3, 3] = blended_translation
                
                print(f"Reference re-registration: fitness={ref_fitness:.3f}, "
                      f"blended weights: frame={frame_to_frame_weight:.2f}, ref={ref_weight:.2f}")
            else:
                # Re-registration failed, use frame-to-frame result
                self.pose_in_reference_frame = pose_from_frame_to_frame
        else:
            # No re-registration, use frame-to-frame result
            self.pose_in_reference_frame = pose_from_frame_to_frame
        
        # Store current and previous point clouds 
        self.current_pcd_processed = current_pcd_processed
        self.previous_pcd_for_viz = copy.deepcopy(self.previous_pcd)
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
    
    def get_visualization_data(self):
        """
        Get data for point cloud visualization.
        
        Returns:
            tuple: (previous_pcd, current_pcd, transformation, fitness, rmse) or None if not available
        """
        if not self.visualize_point_clouds or self.last_transformation is None:
            return None
        
        return (
            self.previous_pcd,
            None,  # current pcd needs to be provided by caller
            self.last_transformation,
            self.last_fitness,
            self.last_rmse
        )
