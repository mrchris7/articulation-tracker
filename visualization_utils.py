"""
Visualization utilities for pose tracking.
"""
import copy
import numpy as np
import cv2
import open3d as o3d


def draw_pose_axes(image, pose, intrinsic, length=0.1):
    """
    Draw 3D coordinate axes on image using pose and camera intrinsics.
    
    Args:
        image: numpy array (H, W, 3) - RGB image
        pose: numpy array (4, 4) - transformation matrix
        intrinsic: numpy array (3, 3) - camera intrinsic matrix
        length: float - length of axes in meters
    
    Returns:
        numpy array - image with axes drawn
    """
    # Define axis endpoints in object frame
    axis_points = np.array([
        [0, 0, 0],           # origin
        [length, 0, 0],      # X axis (red)
        [0, length, 0],     # Y axis (green)
        [0, 0, length]      # Z axis (blue)
    ])
    
    # Transform to camera frame
    axis_points_cam = (pose[:3, :3] @ axis_points.T).T + pose[:3, 3]
    
    # Project to image
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    points_2d = []
    for point in axis_points_cam:
        if point[2] > 0:  # Only project points in front of camera
            x = int(fx * point[0] / point[2] + cx)
            y = int(fy * point[1] / point[2] + cy)
            points_2d.append((x, y))
        else:
            points_2d.append(None)
    
    # Draw axes
    if points_2d[0] is not None:
        origin = points_2d[0]
        
        # X axis (red)
        if points_2d[1] is not None:
            cv2.line(image, origin, points_2d[1], (0, 0, 255), 3)
            cv2.circle(image, points_2d[1], 5, (0, 0, 255), -1)
        
        # Y axis (green)
        if points_2d[2] is not None:
            cv2.line(image, origin, points_2d[2], (0, 255, 0), 3)
            cv2.circle(image, points_2d[2], 5, (0, 255, 0), -1)
        
        # Z axis (blue)
        if points_2d[3] is not None:
            cv2.line(image, origin, points_2d[3], (255, 0, 0), 3)
            cv2.circle(image, points_2d[3], 5, (255, 0, 0), -1)
    
    return image


def project_mesh_to_image(image, mesh, pose, intrinsic):
    """
    Project mesh vertices to image and draw wireframe.
    
    Args:
        image: numpy array (H, W, 3) - RGB image
        mesh: open3d.geometry.TriangleMesh - mesh to project
        pose: numpy array (4, 4) - transformation matrix
        intrinsic: numpy array (3, 3) - camera intrinsic matrix
    
    Returns:
        numpy array - image with projected mesh
    """
    # Get mesh vertices
    vertices = np.asarray(mesh.vertices)
    
    # Transform to camera frame
    vertices_cam = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    
    # Project to image
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    vertices_2d = []
    for point in vertices_cam:
        if point[2] > 0:  # Only project points in front of camera
            x = int(fx * point[0] / point[2] + cx)
            y = int(fy * point[1] / point[2] + cy)
            vertices_2d.append((x, y))
        else:
            vertices_2d.append(None)
    
    # Draw wireframe
    triangles = np.asarray(mesh.triangles)
    for triangle in triangles:
        pts = [vertices_2d[i] for i in triangle]
        if all(p is not None for p in pts):
            # Check if all points are within image bounds
            h, w = image.shape[:2]
            if all(0 <= p[0] < w and 0 <= p[1] < h for p in pts):
                cv2.line(image, pts[0], pts[1], (255, 255, 0), 1)
                cv2.line(image, pts[1], pts[2], (255, 255, 0), 1)
                cv2.line(image, pts[2], pts[0], (255, 255, 0), 1)
    
    return image


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.3):
    """
    Overlay mask on image.
    
    Args:
        image: numpy array (H, W, 3) - RGB image
        mask: numpy array (H, W) - boolean mask
        color: tuple - RGB color for mask
        alpha: float - transparency
    
    Returns:
        numpy array - image with mask overlay
    """
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask] = color
    cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0, overlay)
    return overlay


def depth_to_colormap(depth_frame, depth_min=None, depth_max=None, colormap=cv2.COLORMAP_JET):
    """
    Convert depth frame to colorized visualization using colormap.
    
    Args:
        depth_frame: numpy array (H, W) - depth values in meters
        depth_min: float - minimum depth for normalization (None = auto)
        depth_max: float - maximum depth for normalization (None = auto)
        colormap: int - OpenCV colormap (default: COLORMAP_JET)
    
    Returns:
        numpy array (H, W, 3) - colorized depth image
    """
    # Create a copy to avoid modifying original
    depth_vis = depth_frame.copy()
    
    # Mask invalid depths
    valid_mask = (depth_vis > 0) & np.isfinite(depth_vis)
    
    if valid_mask.sum() == 0:
        # No valid depth, return black image
        return np.zeros((depth_vis.shape[0], depth_vis.shape[1], 3), dtype=np.uint8)
    
    # Get valid depth range
    if depth_min is None:
        depth_min = depth_vis[valid_mask].min()
    if depth_max is None:
        depth_max = depth_vis[valid_mask].max()
    
    # Normalize depth to 0-255 range
    depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
    if depth_max > depth_min:
        depth_normalized[valid_mask] = np.clip(
            ((depth_vis[valid_mask] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8),
            0, 255
        )
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]
    
    return depth_colored




def visualize_frame_to_frame_icp(previous_pcd, current_pcd, reference_pcd=None, transformation=None, fitness=0.0, rmse=0.0):
    """
    Visualize frame-to-frame ICP alignment between previous and current point clouds.
    
    Args:
        previous_pcd: open3d.geometry.PointCloud - previous frame's point cloud (in camera coordinates)
        current_pcd: open3d.geometry.PointCloud - current frame's point cloud (in camera coordinates)
        reference_pcd: open3d.geometry.PointCloud - first frame's reference point cloud (in camera coordinates)
        transformation: numpy array (4, 4) - T_prev_to_curr transformation from ICP
        fitness: float - ICP fitness score
        rmse: float - ICP inlier RMSE
    """
    # Copy original clouds to avoid modifying them
    prev_vis = copy.deepcopy(previous_pcd)
    curr_vis = copy.deepcopy(current_pcd)
    ref_vis = copy.deepcopy(reference_pcd) if reference_pcd is not None else None
    
    print("\n--- Frame-to-Frame ICP Visualization ---")
    print(f"Previous PCD: center={prev_vis.get_center()}, {len(prev_vis.points)} points")
    print(f"Current PCD: center={curr_vis.get_center()}, {len(curr_vis.points)} points")
    if ref_vis is not None:
        print(f"Reference PCD (first frame): center={ref_vis.get_center()}, {len(ref_vis.points)} points")
    
    if transformation is not None:
        # Transform previous frame to current frame coordinates for visualization
        prev_vis.transform(transformation)
        print(f"Previous PCD (transformed to current frame): center={prev_vis.get_center()}")
        print(f"Distance between centers (after transform): {np.linalg.norm(prev_vis.get_center() - curr_vis.get_center()):.4f} m")
        print(f"ICP Fitness: {fitness:.4f}, RMSE: {rmse:.6f} m")
        print(f"Transformation T_prev_to_curr:\n{transformation}")
    
    # Color clouds for display
    prev_vis.paint_uniform_color([1.0, 0.706, 0.0])  # orange/yellow for previous frame
    curr_vis.paint_uniform_color([0.0, 0.651, 0.929])  # blue for current frame
    if ref_vis is not None:
        ref_vis.paint_uniform_color([0.0, 1.0, 0.0])  # green for reference (first frame)
    
    # Create coordinate frames at centers
    prev_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=prev_vis.get_center())
    curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=curr_vis.get_center())
    
    # Color the frames
    prev_frame.paint_uniform_color([1.0, 0.706, 0.0])  # Match previous color
    curr_frame.paint_uniform_color([0.0, 0.651, 0.929])  # Match current color
    
    # Create reference frame if reference PCD exists
    geometries = [prev_vis, curr_vis, prev_frame, curr_frame]
    if ref_vis is not None:
        ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=ref_vis.get_center())
        ref_frame.paint_uniform_color([0.0, 1.0, 0.0])  # Match reference color
        geometries.append(ref_vis)
        geometries.append(ref_frame)
    
    # Create camera coordinate frame at origin
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    camera_frame.paint_uniform_color([1.0, 0.0, 0.0])  # Red for camera frame
    geometries.append(camera_frame)
    
    print("Yellow/Orange = Previous frame point cloud (transformed to current frame)")
    print("Blue = Current frame point cloud")
    if ref_vis is not None:
        print("Green = Reference point cloud (first frame)")
    print("Red = Camera coordinate frame (origin)")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Frame-to-Frame ICP (Fitness: {fitness:.3f}, RMSE: {rmse:.4f}m)",
        width=1600,
        height=900,
        point_show_normal=False
    )


def visualize_icp_alignment(scene_pcd, cad_pcd, transformation=None, show_in_camera_frame=True):
    """
    Visualize how well the scene point cloud aligns with the CAD reference.
    
    Args:
        scene_pcd: open3d.geometry.PointCloud - scene point cloud (in camera coordinates)
        cad_pcd: open3d.geometry.PointCloud - CAD model point cloud (in CAD model coordinates)
        transformation: numpy array (4, 4) - transformation matrix
                       If show_in_camera_frame=True: T_cad_to_camera (transforms CAD to camera coords)
                       If show_in_camera_frame=False: T_camera_to_cad (transforms scene to CAD coords)
        show_in_camera_frame: bool - if True, show both in camera coordinates; if False, show both in CAD coordinates
    """
    # Copy original clouds to avoid modifying them
    scene_vis = copy.deepcopy(scene_pcd)
    cad_vis = copy.deepcopy(cad_pcd)

    if transformation is not None:
        print("\n--- Point Cloud Visualization ---")
        print(f"Scene PCD (camera coords): center={scene_vis.get_center()}, {len(scene_vis.points)} points")
        print(f"CAD PCD (CAD coords): center={cad_vis.get_center()}, {len(cad_vis.points)} points")
        
        if show_in_camera_frame:
            # Transform CAD to camera coordinates for visualization
            # transformation is T_cad_to_camera
            cad_vis.transform(transformation)
            print(f"CAD PCD (transformed to camera coords): center={cad_vis.get_center()}")
            print(f"Distance between scene and CAD centers (in camera frame): {np.linalg.norm(scene_vis.get_center() - cad_vis.get_center()):.4f} m")
            print(f"Transformation T_cad_to_camera:\n{transformation}")
        else:
            # Transform scene to CAD coordinates for visualization
            # transformation is T_camera_to_cad
            scene_vis.transform(transformation)
            print(f"Scene PCD (transformed to CAD coords): center={scene_vis.get_center()}")
            print(f"Distance between scene and CAD centers (in CAD frame): {np.linalg.norm(scene_vis.get_center() - cad_vis.get_center()):.4f} m")
            print(f"Transformation T_camera_to_cad:\n{transformation}")

    # Color clouds for display
    scene_vis.paint_uniform_color([1.0, 0.706, 0.0])  # orange/yellow for scene
    cad_vis.paint_uniform_color([0.0, 0.651, 0.929])  # blue for CAD

    # Create coordinate frames at origins for reference
    scene_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=scene_vis.get_center())
    cad_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=cad_vis.get_center())
    
    # Color the frames
    scene_frame.paint_uniform_color([1.0, 0.706, 0.0])  # Match scene color
    cad_frame.paint_uniform_color([0.0, 0.651, 0.929])  # Match CAD color

    # Create camera coordinate frame at origin
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    camera_frame.paint_uniform_color([1.0, 0.0, 0.0])  # Red for camera frame

    print("\nOpening visualization window...")
    if show_in_camera_frame:
        print("Yellow/Orange = Scene point cloud (camera coordinates)")
        print("Blue = CAD model point cloud (transformed to camera coordinates)")
        print("Red = Camera coordinate frame (origin)")
    else:
        print("Yellow/Orange = Scene point cloud (transformed to CAD coordinates)")
        print("Blue = CAD model point cloud (CAD coordinates)")
    
    o3d.visualization.draw_geometries(
        [scene_vis, cad_vis, scene_frame, cad_frame, camera_frame],
        window_name="ICP Alignment - Scene (yellow) vs CAD (blue)",
        width=1600,
        height=900,
        point_show_normal=False
    )


def overlay_rgba_on_bgr(frame_bgr, overlay_rgba):
    if overlay_rgba is None or overlay_rgba.size == 0:
        print("Overlay is empty -> returning original frame")
        return frame_bgr

    if overlay_rgba.shape[2] < 4:
        print("Overlay has no alpha -> returning original frame")
        return frame_bgr

    overlay_rgb = overlay_rgba[:, :, :3]
    alpha = overlay_rgba[:, :, 3:4] / 255.0

    # Resize if needed
    if overlay_rgb.shape[:2] != frame_bgr.shape[:2]:
        overlay_rgb = cv2.resize(overlay_rgb, (frame_bgr.shape[1], frame_bgr.shape[0]))
        alpha = cv2.resize(alpha, (frame_bgr.shape[1], frame_bgr.shape[0]))

    blended = (alpha * overlay_rgb + (1 - alpha) * frame_bgr).astype(np.uint8)
    return blended
