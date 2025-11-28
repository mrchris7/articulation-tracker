"""
ICP-based pose estimation module.
"""
import numpy as np
import open3d as o3d
from point_cloud_utils import debug_pointclouds, downsample_and_extract_features, global_ransac_registration, preprocess_point_cloud, requires_ransac
from visualization_utils import visualize_icp_alignment


class ICPPoseEstimator:
    """ICP-based 6D pose estimator."""
    
    def __init__(self, cad_model_path, voxel_size=0.01, max_correspondence_distance=0.05):
        """
        Initialize ICP pose estimator.
        
        Args:
            cad_model_path: str - path to CAD model file
            voxel_size: float - voxel size for point cloud preprocessing
            max_correspondence_distance: float - maximum correspondence distance for ICP
        """
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        
        # Load CAD model
        from point_cloud_utils import load_cad_model
        self.cad_mesh, self.cad_pcd = load_cad_model(cad_model_path)
        
        # Preprocess CAD model point cloud
        self.cad_pcd = preprocess_point_cloud(
            self.cad_pcd, 
            voxel_size=voxel_size, 
            estimate_normals=True
        )
        
        print(f"Loaded CAD model with {len(self.cad_pcd.points)} points")
    
    def estimate_initial_pose(self, scene_pcd):
        """
        Estimate initial pose using point-to-point ICP (coarse alignment).
        
        Args:
            scene_pcd: open3d.geometry.PointCloud - scene point cloud
        
        Returns:
            numpy array (4, 4) - initial transformation matrix
        """
        if len(scene_pcd.points) == 0:
            return np.eye(4)
        
        # Preprocess scene point cloud
        scene_pcd = preprocess_point_cloud(
            scene_pcd, 
            voxel_size=self.voxel_size, 
            estimate_normals=True
        )
        
        if len(scene_pcd.points) == 0:
            return np.eye(4)
        
        # Coarse alignment using point-to-point ICP
        result = o3d.pipelines.registration.registration_icp(
            scene_pcd,
            self.cad_pcd,
            max_correspondence_distance=self.max_correspondence_distance * 2,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=30
            )
        )
        
        return result.transformation


    def estimate_initial_pose_ransac(self, scene_pcd, voxel_size):
        scene_down, scene_fpfh = downsample_and_extract_features(scene_pcd,    voxel_size)
        cad_down,   cad_fpfh   = downsample_and_extract_features(self.cad_pcd, voxel_size)

        print("Running global RANSAC...")
        result_ransac = global_ransac_registration(
            scene_down, cad_down, scene_fpfh, cad_fpfh, voxel_size
        )
        print("RANSAC result:")
        print(result_ransac)

        return result_ransac.transformation

    
    def refine_pose(self, scene_pcd, initial_pose):
        """
        Refine pose using point-to-plane ICP.
        
        Args:
            scene_pcd: open3d.geometry.PointCloud - scene point cloud
            initial_pose: numpy array (4, 4) - initial transformation matrix
        
        Returns:
            tuple: (refined_pose (4, 4), fitness score, inlier_rmse)
        """
        if len(scene_pcd.points) == 0:
            return initial_pose, 0.0, float('inf')
        
        # Preprocess scene point cloud
        scene_pcd = preprocess_point_cloud(
            scene_pcd, 
            voxel_size=self.voxel_size, 
            estimate_normals=True
        )
        
        if len(scene_pcd.points) == 0:
            return initial_pose, 0.0, float('inf')

        translation = scene_pcd.get_center() - self.cad_pcd.get_center()
        self.cad_pcd.translate(translation)

        if requires_ransac(scene_pcd, self.cad_pcd):
            #visualize_icp_alignment(scene_pcd, self.cad_pcd, initial_pose)
            print("Running RANSAC first...")
            initial_pose = self.estimate_initial_pose_ransac(scene_pcd, voxel_size=0.01)
            #visualize_icp_alignment(scene_pcd, self.cad_pcd, initial_pose)
        else:
            print("Skipping RANSAC, ICP should work directly.")

        #debug_pointclouds(scene_pcd, self.cad_pcd)
        
        # Point-to-plane ICP for fine alignment
        result = o3d.pipelines.registration.registration_icp(
            scene_pcd,
            self.cad_pcd,
            max_correspondence_distance=self.max_correspondence_distance,
            init=initial_pose,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=50
            )
        )

        #debug_pointclouds(scene_pcd, self.cad_pcd, result=result)
        visualize_icp_alignment(scene_pcd, self.cad_pcd, result.transformation)

        
        return result.transformation, result.fitness, result.inlier_rmse
    
    def estimate_pose(self, scene_pcd, previous_pose=None):
        """
        Estimate pose from scene point cloud.
        
        Args:
            scene_pcd: open3d.geometry.PointCloud - scene point cloud
            previous_pose: numpy array (4, 4) - previous frame's pose (for tracking)
        
        Returns:
            tuple: (pose (4, 4), fitness score, inlier_rmse)
        """
        if previous_pose is not None:
            # Use previous pose as initial guess
            initial_pose = previous_pose

        else:
            # Estimate initial pose
            initial_pose = self.estimate_initial_pose(scene_pcd)
        
        # Refine pose
        refined_pose, fitness, rmse = self.refine_pose(scene_pcd, initial_pose)
        
        return refined_pose, fitness, rmse



