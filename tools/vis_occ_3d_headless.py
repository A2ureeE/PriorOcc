"""
3D Voxel OCC Visualization Script
Generates publication-quality 3D occupancy visualizations with camera images.
Works on headless servers using EGL/OSMesa backend.
"""
import os
import numpy as np
import argparse
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle

FREE_LABEL = 17

# Color map for each class (RGB, 0-255 range) - matching nuScenes OCC3D
colormap = np.array([
    [0, 0, 0],           # 0 others/undefined - black
    [255, 120, 50],      # 1 barrier - orange
    [255, 192, 203],     # 2 bicycle - pink
    [255, 255, 0],       # 3 bus - yellow
    [0, 150, 245],       # 4 car - blue
    [0, 255, 255],       # 5 construction_vehicle - cyan
    [200, 180, 0],       # 6 motorcycle - dark orange
    [255, 0, 0],         # 7 pedestrian - red
    [255, 240, 150],     # 8 traffic_cone - light yellow
    [135, 60, 0],        # 9 trailer - brown
    [160, 32, 240],      # 10 truck - purple
    [255, 0, 255],       # 11 driveable_surface - magenta
    [139, 137, 137],     # 12 other_flat - grey
    [75, 0, 75],         # 13 sidewalk - dark purple
    [150, 240, 80],      # 14 terrain - light green
    [230, 230, 250],     # 15 manmade - lavender
    [0, 175, 0],         # 16 vegetation - green
    [255, 255, 255],     # 17 free - white (skip)
], dtype=np.uint8)

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

# Grid config
VOXEL_SIZE = [0.4, 0.4, 0.4]  # meters
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]


def draw_voxels_matplotlib(occ_pred, save_path, title='', 
                           elev=25, azim=-60, figsize=(16, 12)):
    """
    Draw 3D voxels using matplotlib - produces clean voxel visualization.
    
    Args:
        occ_pred: (Dx, Dy, Dz) numpy array with class labels
        save_path: path to save the image
        title: title for the plot
        elev: elevation angle
        azim: azimuth angle
    """
    from matplotlib.colors import LightSource
    
    Dx, Dy, Dz = occ_pred.shape
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Use ax.voxels for cleaner visualization but it's slow for large arrays
    # So we'll downsample and use scatter with cube markers
    
    # Get occupied voxels (not free space)
    occupied_mask = occ_pred != FREE_LABEL
    
    # For large arrays, use voxel-based rendering with downsampling
    # Create a boolean array for each class and render separately
    
    all_x, all_y, all_z, all_colors = [], [], [], []
    
    for class_id in range(17):  # 0-16, skip 17 (free)
        class_mask = occ_pred == class_id
        if not np.any(class_mask):
            continue
            
        # Get voxel coordinates
        x, y, z = np.where(class_mask)
        
        # Convert to world coordinates
        x_world = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
        y_world = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
        z_world = z * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2]
        
        # Get color
        color = colormap[class_id] / 255.0
        
        all_x.extend(x_world)
        all_y.extend(y_world)
        all_z.extend(z_world)
        all_colors.extend([color] * len(x))
    
    if len(all_x) == 0:
        print(f"No occupied voxels found")
        return
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    all_colors = np.array(all_colors)
    
    # Downsample if too many points
    max_points = 50000
    if len(all_x) > max_points:
        indices = np.random.choice(len(all_x), max_points, replace=False)
        all_x = all_x[indices]
        all_y = all_y[indices]
        all_z = all_z[indices]
        all_colors = all_colors[indices]
    
    # Plot as 3D scatter with square markers to simulate voxels
    ax.scatter(all_x, all_y, all_z, c=all_colors, s=8, marker='s', 
               alpha=0.9, edgecolors='none')
    
    # Set axis labels and limits
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    
    # Set limits based on point cloud range
    ax.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax.set_zlim(POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5])
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # White background
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    if title:
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


def create_combined_visualization(occ_pred, camera_images, save_path, 
                                  title='', elev=25, azim=-60):
    """
    Create a combined visualization with camera images on top and 3D OCC below.
    Similar to the reference image user provided.
    
    Args:
        occ_pred: (Dx, Dy, Dz) numpy array
        camera_images: list of camera image paths or numpy arrays
        save_path: path to save
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid: 2 rows - top for cameras, bottom for 3D
    # Top row: 6 cameras (or however many provided)
    n_cams = min(len(camera_images), 6) if camera_images else 0
    
    if n_cams > 0:
        # Camera images row
        for i, img in enumerate(camera_images[:6]):
            ax = fig.add_subplot(2, 6, i + 1)
            if isinstance(img, str):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis('off')
    
    # 3D OCC visualization - full width of bottom row
    ax_3d = fig.add_subplot(2, 1, 2, projection='3d')
    
    # Draw voxels
    occupied_mask = occ_pred != FREE_LABEL
    all_x, all_y, all_z, all_colors = [], [], [], []
    
    for class_id in range(17):
        class_mask = occ_pred == class_id
        if not np.any(class_mask):
            continue
        x, y, z = np.where(class_mask)
        x_world = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
        y_world = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
        z_world = z * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2]
        color = colormap[class_id] / 255.0
        all_x.extend(x_world)
        all_y.extend(y_world)
        all_z.extend(z_world)
        all_colors.extend([color] * len(x))
    
    if len(all_x) > 0:
        all_x, all_y, all_z = np.array(all_x), np.array(all_y), np.array(all_z)
        all_colors = np.array(all_colors)
        
        # Downsample
        max_points = 30000
        if len(all_x) > max_points:
            indices = np.random.choice(len(all_x), max_points, replace=False)
            all_x, all_y, all_z = all_x[indices], all_y[indices], all_z[indices]
            all_colors = all_colors[indices]
        
        ax_3d.scatter(all_x, all_y, all_z, c=all_colors, s=5, marker='s', 
                      alpha=0.9, edgecolors='none')
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax_3d.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax_3d.set_zlim(POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5])
    ax_3d.view_init(elev=elev, azim=azim)
    ax_3d.set_facecolor('white')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined visualization: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='3D Voxel OCC Visualization')
    parser.add_argument('pred_dir', help='Directory containing pred.npz files (or single npz file)')
    parser.add_argument('--save-path', type=str, default='./vis_3d_voxel',
                        help='Output directory')
    parser.add_argument('--data-root', type=str, default='data/nuscenes',
                        help='Path to nuScenes data (for camera images)')
    parser.add_argument('--info-file', type=str, default=None,
                        help='Path to info pkl file (for camera image paths)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--elev', type=float, default=25,
                        help='Camera elevation angle')
    parser.add_argument('--azim', type=float, default=-60,
                        help='Camera azimuth angle')
    parser.add_argument('--with-images', action='store_true',
                        help='Include camera images in visualization')
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Find npz files
    if os.path.isfile(args.pred_dir) and args.pred_dir.endswith('.npz'):
        npz_files = [args.pred_dir]
    else:
        npz_files = []
        for root, dirs, files in os.walk(args.pred_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(root, f))
    
    if not npz_files:
        print(f"No npz files found in {args.pred_dir}")
        print("\n请先运行以下命令生成 npz 文件:")
        print("python tools/vis_occ.py --config <配置文件> --weights <权重> --viz-dir <输出目录> --save-npz")
        return
    
    print(f"Found {len(npz_files)} npz files")
    npz_files = sorted(npz_files)[:args.num_samples]
    
    # Load info file if provided (for camera images)
    info_data = None
    if args.info_file and os.path.exists(args.info_file):
        with open(args.info_file, 'rb') as f:
            info_data = pickle.load(f)
        print(f"Loaded info file with {len(info_data.get('infos', []))} samples")
    
    for i, npz_path in enumerate(npz_files):
        print(f"\nProcessing [{i+1}/{len(npz_files)}]: {npz_path}")
        
        try:
            data = np.load(npz_path)
            pred_occ = data['pred']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
        
        print(f"  Shape: {pred_occ.shape}, Classes: {np.unique(pred_occ)}")
        
        # Generate output filename
        rel_path = os.path.basename(os.path.dirname(npz_path))
        save_name = f"{i:04d}_{rel_path}_3d.png"
        save_path_3d = os.path.join(args.save_path, save_name)
        
        # Generate 3D visualization
        draw_voxels_matplotlib(
            pred_occ, 
            save_path_3d,
            title=f'Sample {i}: {rel_path}',
            elev=args.elev,
            azim=args.azim
        )
        
        # Also generate front view
        save_path_front = os.path.join(args.save_path, f"{i:04d}_{rel_path}_front.png")
        draw_voxels_matplotlib(
            pred_occ,
            save_path_front,
            title=f'Front View - Sample {i}',
            elev=5,    # Low elevation = more side/front view
            azim=-90   # Looking from front
        )
    
    print(f"\n完成! 可视化结果保存在: {args.save_path}")
    print("\n生成的文件:")
    print("  - *_3d.png: 3D 透视图")
    print("  - *_front.png: 前向视图")


if __name__ == '__main__':
    main()
