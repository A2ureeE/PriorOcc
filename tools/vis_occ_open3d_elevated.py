"""
Open3D Elevated Front View OCC Visualization
Uses Open3D OffscreenRenderer for headless rendering with GPU.
"""
import os
import numpy as np
import argparse

# Set headless mode BEFORE importing open3d
os.environ['OPEN3D_ENABLE_HEADLESS_RENDERING'] = '1'

import open3d as o3d
import open3d.visualization.rendering as rendering

FREE_LABEL = 17
SKIP_CLASSES = [11, 12, 13, 14]  # Ground classes

# Color map (RGB 0-1)
COLORMAP = np.array([
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


def get_voxel_coords(occ_pred):
    """Convert voxel indices to world coordinates."""
    shape = occ_pred.shape
    if shape[0] == 200:
        voxel_size = [0.4, 0.4, 0.4]
        pc_range = [-40, -40, -1, 40, 40, 5.4]
    else:
        voxel_size = [0.4, 0.4, 0.25]
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # Get occupied voxels (skip free space and ground)
    mask = (occ_pred != FREE_LABEL)
    for skip_cls in SKIP_CLASSES:
        mask = mask & (occ_pred != skip_cls)
    
    x_idx, y_idx, z_idx = np.where(mask)
    labels = occ_pred[x_idx, y_idx, z_idx]
    
    # Convert to world coordinates
    x = x_idx * voxel_size[0] + pc_range[0] + voxel_size[0]/2
    y = y_idx * voxel_size[1] + pc_range[1] + voxel_size[1]/2
    z = z_idx * voxel_size[2] + pc_range[2] + voxel_size[2]/2
    
    return np.stack([x, y, z], axis=1), labels, voxel_size


def create_voxel_mesh(points, labels, voxel_size):
    """Create mesh geometry from voxels (as cubes)."""
    if len(points) == 0:
        return None
    
    # Limit to 30000 voxels for performance
    if len(points) > 30000:
        idx = np.random.choice(len(points), 30000, replace=False)
        points = points[idx]
        labels = labels[idx]
    
    combined_mesh = o3d.geometry.TriangleMesh()
    
    # Full voxel size (no gap)
    hx, hy, hz = voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]/2
    
    print(f"Creating {len(points)} voxel cubes...")
    
    for i in range(len(points)):
        box = o3d.geometry.TriangleMesh.create_box(hx*2, hy*2, hz*2)
        box.translate(points[i] - np.array([hx, hy, hz]))
        
        color = COLORMAP[labels[i] if labels[i] < len(COLORMAP) else 0]
        box.paint_uniform_color(color)
        box.compute_vertex_normals()
        
        combined_mesh += box
    
    return combined_mesh


def render_with_open3d(mesh, save_path, view='front_elevated'):
    """Render mesh using Open3D OffscreenRenderer."""
    # Create renderer
    render = rendering.OffscreenRenderer(1920, 1080)
    
    # Material
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    
    # Add geometry
    render.scene.add_geometry("voxels", mesh, mat)
    
    # Lighting
    render.scene.scene.set_sun_light([0.5, -0.5, -1], [1, 1, 1], 75000)
    render.scene.scene.enable_sun_light(True)
    
    # Background
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])
    
    # Camera setup for different views
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    
    if view == 'front_elevated':
        # Looking from behind and above (like car roof looking forward)
        eye = np.array([-30, 0, 15])  # Behind and above
        center = np.array([20, 0, 0])  # Looking forward
        up = np.array([0, 0, 1])
    elif view == 'side':
        eye = np.array([0, -60, 10])
        center = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
    elif view == '3d':
        eye = np.array([-40, -40, 30])
        center = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
    
    render.setup_camera(60.0, center, eye, up)
    
    # Render
    img = render.render_to_image()
    o3d.io.write_image(save_path, img)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', help='Directory with npz files')
    parser.add_argument('--save-path', default='vis_open3d_elevated')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Find npz files
    npz_files = []
    for root, dirs, files in os.walk(args.pred_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))
    
    npz_files = sorted(npz_files)[:args.num_samples]
    print(f"Found {len(npz_files)} files")
    
    for i, npz_path in enumerate(npz_files):
        print(f"\n[{i+1}/{len(npz_files)}] Processing {npz_path}")
        
        try:
            pred = np.load(npz_path)['pred']
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        print(f"  Shape: {pred.shape}")
        
        # Get voxels
        points, labels, voxel_size = get_voxel_coords(pred)
        print(f"  Voxels: {len(points)}")
        
        if len(points) == 0:
            continue
        
        # Create mesh
        mesh = create_voxel_mesh(points, labels, voxel_size)
        
        base_name = f"{i:04d}"
        
        # Render views
        render_with_open3d(mesh, os.path.join(args.save_path, f"{base_name}_front_elevated.png"), 'front_elevated')
        render_with_open3d(mesh, os.path.join(args.save_path, f"{base_name}_3d.png"), '3d')
        render_with_open3d(mesh, os.path.join(args.save_path, f"{base_name}_side.png"), 'side')
    
    print(f"\nDone! Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
