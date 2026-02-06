"""
Open3D Headless 3D OCC Visualization
Uses Open3D's offscreen rendering - works on servers without display.
"""
import os
import numpy as np
import argparse
import pickle
import open3d as o3d

# Enable headless rendering
os.environ['OPEN3D_HEADLESS'] = '1'

FREE_LABEL = 17
VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

# Color map for each class (RGB, 0-1 range)
colormap = np.array([
    [0, 0, 0],           # 0 others
    [255, 120, 50],      # 1 barrier
    [255, 192, 203],     # 2 bicycle
    [255, 255, 0],       # 3 bus
    [0, 150, 245],       # 4 car
    [0, 255, 255],       # 5 construction_vehicle
    [200, 180, 0],       # 6 motorcycle
    [255, 0, 0],         # 7 pedestrian
    [255, 240, 150],     # 8 traffic_cone
    [135, 60, 0],        # 9 trailer
    [160, 32, 240],      # 10 truck
    [255, 0, 255],       # 11 driveable_surface
    [139, 137, 137],     # 12 other_flat
    [75, 0, 75],         # 13 sidewalk
    [150, 240, 80],      # 14 terrain
    [230, 230, 250],     # 15 manmade
    [0, 175, 0],         # 16 vegetation
], dtype=np.float32) / 255.0


def create_voxel_grid(occ_pred):
    """Create Open3D point cloud from occupancy prediction."""
    # Get non-free voxels
    occupied_mask = occ_pred != FREE_LABEL
    x, y, z = np.where(occupied_mask)
    
    if len(x) == 0:
        return None
    
    # Convert to world coordinates
    points = np.stack([
        x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0],
        y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1],
        z * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2]
    ], axis=1)
    
    # Get colors
    labels = occ_pred[occupied_mask]
    colors = colormap[labels % len(colormap)]
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def render_and_save(pcd, save_path, width=1920, height=1080):
    """Render point cloud using Open3D offscreen renderer."""
    # Create voxel grid from point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)
    
    # Create offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # Setup material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    
    # Add geometry
    render.scene.add_geometry("voxels", voxel_grid, mat)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    render.scene.add_geometry("coord", coord_frame, mat)
    
    # Setup camera - front-ish view
    bounds = voxel_grid.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    
    # Camera position: looking from front-left, slightly elevated
    eye = np.array([center[0] - 60, center[1] - 40, center[2] + 30])
    up = np.array([0, 0, 1])
    
    render.setup_camera(60.0, center, eye, up)
    
    # Set background to white
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])
    
    # Render and save
    img = render.render_to_image()
    o3d.io.write_image(save_path, img)
    print(f"Saved: {save_path}")


def render_multiple_views(pcd, save_dir, base_name):
    """Render multiple views of the point cloud."""
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)
    
    render = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    
    render.scene.add_geometry("voxels", voxel_grid, mat)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    render.scene.add_geometry("coord", coord_frame, mat)
    
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])
    
    bounds = voxel_grid.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    
    views = [
        ("front", np.array([center[0], center[1] - 80, center[2] + 20])),
        ("side", np.array([center[0] - 80, center[1], center[2] + 20])),
        ("3d", np.array([center[0] - 60, center[1] - 60, center[2] + 40])),
        ("top", np.array([center[0], center[1], center[2] + 100])),
    ]
    
    up = np.array([0, 0, 1])
    
    for view_name, eye in views:
        render.setup_camera(60.0, center, eye, up)
        img = render.render_to_image()
        save_path = os.path.join(save_dir, f"{base_name}_{view_name}.png")
        o3d.io.write_image(save_path, img)
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Open3D Headless OCC Visualization')
    parser.add_argument('pred_dir', help='Directory containing pred.npz files')
    parser.add_argument('--save-path', type=str, default='./vis_3d_open3d')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--multi-view', action='store_true', 
                        help='Generate multiple view angles')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Find npz files
    npz_files = []
    for root, dirs, files in os.walk(args.pred_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))
    
    if not npz_files:
        print(f"No npz files found in {args.pred_dir}")
        return
    
    print(f"Found {len(npz_files)} npz files")
    npz_files = sorted(npz_files)[:args.num_samples]
    
    for i, npz_path in enumerate(npz_files):
        print(f"\nProcessing [{i+1}/{len(npz_files)}]: {npz_path}")
        
        try:
            data = np.load(npz_path)
            pred_occ = data['pred']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
        
        # Create point cloud
        pcd = create_voxel_grid(pred_occ)
        if pcd is None:
            print("No occupied voxels found")
            continue
        
        # Generate name
        base_name = f"{i:04d}"
        
        if args.multi_view:
            render_multiple_views(pcd, args.save_path, base_name)
        else:
            save_path = os.path.join(args.save_path, f"{base_name}_3d.png")
            render_and_save(pcd, save_path)
    
    print(f"\nDone! Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
