"""
Matplotlib-based 3D OCC Visualization
No EGL/OpenGL dependency - works in any headless environment.
"""
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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


def render_3d_scatter(occ_pred, save_path, max_points=30000, dpi=150):
    """Render occupancy as 3D scatter plot with Matplotlib."""
    # Get non-free voxels
    occupied_mask = occ_pred != FREE_LABEL
    x, y, z = np.where(occupied_mask)
    
    if len(x) == 0:
        print("No occupied voxels found")
        return False
    
    # Downsample if too many points
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    
    # Convert to world coordinates
    wx = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
    wy = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
    wz = z * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2]
    
    # Get colors
    labels = occ_pred[x, y, z]
    colors = colormap[labels % len(colormap)]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(wx, wy, wz, c=colors, s=1, alpha=0.8)
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set viewing angle (front-left elevated view)
    ax.view_init(elev=25, azim=-135)
    
    # Set axis limits
    ax.set_xlim([POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3]])
    ax.set_ylim([POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4]])
    ax.set_zlim([POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5]])
    
    # Equal aspect ratio
    ax.set_box_aspect([
        POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0],
        POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1],
        POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]
    ])
    
    ax.set_title(f'3D Occupancy ({len(x)} voxels)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def render_bev(occ_pred, save_path, max_points=200000, dpi=150):
    """Render bird's eye view (top-down) as 2D scatter plot."""
    # Get non-free voxels
    occupied_mask = occ_pred != FREE_LABEL
    x, y, z = np.where(occupied_mask)
    
    if len(x) == 0:
        print("No occupied voxels found")
        return False
    
    # Downsample only if really necessary
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    
    # Convert to world coordinates (only X, Y for BEV)
    wx = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
    wy = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
    
    labels = occ_pred[x, y, z]
    colors = colormap[labels % len(colormap)]
    
    # Create 2D figure for BEV
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Large square markers to look like filled voxels
    ax.scatter(wx, wy, c=colors, s=15, alpha=0.9, marker='s', edgecolors='none')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim([POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3]])
    ax.set_ylim([POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4]])
    ax.set_aspect('equal')
    ax.set_title(f'Bird\'s Eye View ({len(x)} voxels)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def render_multi_view(occ_pred, save_dir, base_name, max_points=30000, dpi=150):
    """Render multiple views of the occupancy."""
    # Get non-free voxels
    occupied_mask = occ_pred != FREE_LABEL
    x, y, z = np.where(occupied_mask)
    
    if len(x) == 0:
        print("No occupied voxels found")
        return False
    
    # Downsample
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    
    # Convert to world coordinates
    wx = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
    wy = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
    wz = z * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2]
    
    labels = occ_pred[x, y, z]
    colors = colormap[labels % len(colormap)]
    
    views = [
        ("3d", 25, -135),       # 3D perspective
        ("front", 0, -90),     # Front view
        ("side", 0, 0),        # Side view
        ("top", 90, -90),      # Top-down view
    ]
    
    for view_name, elev, azim in views:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(wx, wy, wz, c=colors, s=1, alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlim([POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3]])
        ax.set_ylim([POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4]])
        ax.set_zlim([POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5]])
        
        ax.set_box_aspect([
            POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0],
            POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1],
            POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]
        ])
        
        ax.set_title(f'{view_name.upper()} View ({len(x)} voxels)')
        
        save_path = os.path.join(save_dir, f"{base_name}_{view_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Matplotlib 3D OCC Visualization (Headless Compatible)')
    parser.add_argument('pred_dir', help='Directory containing pred.npz files')
    parser.add_argument('--save-path', type=str, default='./vis_3d_matplotlib')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--max-points', type=int, default=30000, 
                        help='Max points to render (downsampling for performance)')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--multi-view', action='store_true',
                        help='Generate multiple view angles')
    parser.add_argument('--bev', action='store_true',
                        help='Generate bird\'s eye view (top-down) only')
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
    
    for i, npz_path in enumerate(tqdm(npz_files, desc="Rendering")):
        try:
            data = np.load(npz_path)
            pred_occ = data['pred']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
        
        base_name = f"{i:04d}"
        
        if args.multi_view:
            render_multi_view(pred_occ, args.save_path, base_name, 
                              max_points=args.max_points, dpi=args.dpi)
        elif args.bev:
            # BEV (bird's eye view) - top-down
            render_bev(pred_occ, os.path.join(args.save_path, f"{base_name}_bev.png"),
                       max_points=args.max_points, dpi=args.dpi)
        else:
            save_path = os.path.join(args.save_path, f"{base_name}_3d.png")
            render_3d_scatter(pred_occ, save_path, 
                              max_points=args.max_points, dpi=args.dpi)
    
    print(f"\nDone! Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
