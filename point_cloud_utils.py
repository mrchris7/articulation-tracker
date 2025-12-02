"""
Point cloud utilities for converting depth maps to point clouds and processing them.
"""
import numpy as np
import open3d as o3d
import cv2


def depth_to_point_cloud(depth_map, intrinsic, mask=None):
    """
    Convert depth map to point cloud using camera intrinsics.
    
    Args:
        depth_map: numpy array (H, W) - depth values in meters
        intrinsic: numpy array (3, 3) - camera intrinsic matrix
        mask: numpy array (H, W) - optional boolean mask to filter points
    
    Returns:
        open3d.geometry.PointCloud - point cloud object
    """
    h, w = depth_map.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D points
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        valid_mask = (depth_map > 0) & mask
    else:
        valid_mask = depth_map > 0
    
    # Flatten and filter valid points
    points_flat = points[valid_mask]
    
    if len(points_flat) == 0:
        return o3d.geometry.PointCloud()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat)
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.01, estimate_normals=True):
    """
    Preprocess point cloud: downsample and estimate normals.
    
    Args:
        pcd: open3d.geometry.PointCloud
        voxel_size: float - voxel size for downsampling
        estimate_normals: bool - whether to estimate normals
    
    Returns:
        open3d.geometry.PointCloud - processed point cloud
    """
    if len(pcd.points) == 0:
        return pcd
    
    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals
    if estimate_normals and len(pcd.points) > 0:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        # Orient normals consistently (towards camera)
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    return pcd


def load_cad_model(file_path):
    """
    Load CAD model from file (supports .obj, .ply, .stl).
    
    Args:
        file_path: str - path to CAD model file
    
    Returns:
        open3d.geometry.TriangleMesh - loaded mesh
    """
    if file_path.endswith('.obj'):
        mesh = o3d.io.read_triangle_mesh(file_path)
    elif file_path.endswith('.ply'):
        mesh = o3d.io.read_triangle_mesh(file_path)
    elif file_path.endswith('.stl'):
        mesh = o3d.io.read_triangle_mesh(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {file_path}")
    
    # Compute normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # scale to meter unit
    mesh.scale(0.001, center=mesh.get_center())
    
    # Convert to point cloud for ICP (sample points from mesh surface)
    # We'll use the mesh vertices directly, but could also sample uniformly
    pcd = mesh.sample_points_uniformly(number_of_points=10000)    
    
    return mesh, pcd


def requires_ransac(scene_pcd, cad_pcd):

    # Centroids
    c_scene = np.mean(np.asarray(scene_pcd.points), axis=0)
    c_cad   = np.mean(np.asarray(cad_pcd.points), axis=0)

    centroid_distance = np.linalg.norm(c_scene - c_cad)

    # Extents
    scene_extent = np.max(np.asarray(scene_pcd.points), axis=0) - np.min(np.asarray(scene_pcd.points), axis=0)
    cad_extent   = np.max(np.asarray(cad_pcd.points),   axis=0) - np.min(np.asarray(cad_pcd.points),   axis=0)

    max_extent = max(scene_extent.max(), cad_extent.max())
    extent_ratio = scene_extent.max() / cad_extent.max()

    # --- Decision logic ---
    if centroid_distance > 3 * max_extent:
        return True     # too far apart -> ICP impossible

    if extent_ratio > 1.5 or extent_ratio < 0.66:
        return True     # size too different -> ICP unreliable

    return False        # looks close enough -> ICP OK



def downsample_and_extract_features(pcd, voxel_size):
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0,
            max_nn=30
        )
    )

    # Extract FPFH feature
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0,
            max_nn=100
        )
    )

    return pcd_down, fpfh


def global_ransac_registration(scene_down, cad_down, scene_fpfh, cad_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        scene_down,               # source
        cad_down,                 # target
        scene_fpfh,               # source_feature
        cad_fpfh,                 # target_feature
        True,                     # mutual_filter MUST be positional
        distance_threshold,       # max_correspondence_distance
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result


def debug_pointclouds(scene_pcd, cad_pcd, result=None):
    print("\n================= POINT CLOUD DEBUG INFO =================")

    # Bounding boxes
    scene_extent = scene_pcd.get_max_bound() - scene_pcd.get_min_bound()
    cad_extent   = cad_pcd.get_max_bound() - cad_pcd.get_min_bound()

    print("Scene extent (XYZ):", scene_extent)
    print("CAD extent   (XYZ):", cad_extent)
    print("Scene max extent:", np.max(scene_extent))
    print("CAD max extent:", np.max(cad_extent))

    # Centroids
    print("\nScene centroid:", scene_pcd.get_center())
    print("CAD centroid:  ", cad_pcd.get_center())

    # Relative scale estimate
    scale_ratio = np.max(scene_extent) / np.max(cad_extent)
    print("\nApprox scale ratio (scene / CAD):", scale_ratio)

    if scale_ratio > 10:
        print("CAD is probably in millimeters → needs scaling by 0.001")
    elif scale_ratio < 0.1:
        print("CAD may be in meters and scene in millimeters")

    # Centroid distance
    dist_centers = np.linalg.norm(scene_pcd.get_center() - cad_pcd.get_center())
    print("\nDistance between centroids:", dist_centers)
    if dist_centers > np.max(scene_extent) * 2:
        print("Centroids are far apart → ICP will fail")

    # ICP debug (if provided)
    if result is not None:
        print("\n---------------- ICP RESULT ----------------")
        print("Fitness:", result.fitness)
        print("RMSE:", result.inlier_rmse)
        print("Transformation:\n", result.transformation)

        if result.fitness < 0.1:
            print("Very low fitness → clouds did not align")
        if result.inlier_rmse > np.max(scene_extent) * 0.1:
            print("ICP RMSE is very large → alignment likely failed")

    print("==========================================================\n")



