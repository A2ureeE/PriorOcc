"""
3D OCC Visualization Script for Headless Servers
Uses Matplotlib instead of Open3D to avoid display issues.
"""
import os
import numpy as np
import torch
import argparse
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FREE_LABEL = 17

# Color map for each class (RGB, 0-1 range)
colormap_to_colors = np.array([
    [0, 0, 0],           # 0 undefined/others
    [112, 128, 144],     # 1 barrier
    [220, 20, 60],       # 2 bicycle
    [255, 127, 80],      # 3 bus
    [255, 158, 0],       # 4 car
    [233, 150, 70],      # 5 construction vehicle
    [255, 61, 99],       # 6 motorcycle
    [0, 0, 230],         # 7 pedestrian
    [47, 79, 79],        # 8 traffic cone
    [255, 140, 0],       # 9 trailer
    [255, 99, 71],       # 10 truck
    [0, 207, 191],       # 11 driveable surface
    [175, 0, 75],        # 12 other flat
    [75, 0, 75],         # 13 sidewalk
    [112, 180, 60],      # 14 terrain
    [222, 184, 135],     # 15 manmade
    [0, 175, 0],         # 16 vegetation
    [255, 255, 255],     # 17 free (white, but we skip this)
], dtype=np.float32) / 255.0

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


def visualize_occ_3d(occ_pred, save_path, title='OCC Prediction', 
                     elevation=30, azimuth=-60, downsample=2):
    """
    Visualize 3D occupancy prediction using matplotlib.
    
    Args:
        occ_pred: (Dx, Dy, Dz) numpy array with class labels
        save_path: path to save the image
        title: title for the plot
        elevation: camera elevation angle
        azimuth: camera azimuth angle
        downsample: downsample factor to reduce points for faster rendering
    """
    # Get occupied voxels (not free space)
    occupied_mask = occ_pred != FREE_LABEL
    
    # Get indices of occupied voxels
    x, y, z = np.where(occupied_mask)
    
    if len(x) == 0:
        print(f"No occupied voxels found, skipping {save_path}")
        return
    
    # Downsample for faster rendering
    if downsample > 1:
        indices = np.arange(0, len(x), downsample)
        x, y, z = x[indices], y[indices], z[indices]
    
    # Get colors for each point
    labels = occ_pred[occupied_mask]
    if downsample > 1:
        labels = labels[::downsample]
    colors = colormap_to_colors[labels % len(colormap_to_colors)]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(x, y, z, c=colors, s=1, alpha=0.8)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Set equal aspect ratio
    max_range = max(occ_pred.shape) / 2
    mid_x, mid_y, mid_z = [s / 2 for s in occ_pred.shape]
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, occ_pred.shape[2])
    
    # Add legend
    legend_elements = []
    unique_labels = np.unique(labels)
    for label in unique_labels[:10]:  # Limit legend items
        if label < len(occ_class_names):
            color = colormap_to_colors[label]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=8,
                                              label=occ_class_names[label]))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to {save_path}")


def visualize_occ_multiview(occ_pred, save_path, title='OCC Prediction'):
    """
    Create a multi-view visualization (front, side, top views).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    occupied_mask = occ_pred != FREE_LABEL
    x, y, z = np.where(occupied_mask)
    
    if len(x) == 0:
        print(f"No occupied voxels found, skipping {save_path}")
        return
    
    # Downsample
    downsample = 3
    indices = np.arange(0, len(x), downsample)
    x, y, z = x[indices], y[indices], z[indices]
    
    labels = occ_pred[occupied_mask][::downsample]
    colors = colormap_to_colors[labels % len(colormap_to_colors)]
    
    views = [
        ('Front View', 0, -90),
        ('Side View', 0, 0),
        ('Top View (BEV)', 90, -90),
        ('3D View', 30, -60),
    ]
    
    for ax, (view_name, elev, azim) in zip(axes.flat, views):
        ax = fig.add_subplot(2, 2, list(axes.flat).index(ax) + 1, projection='3d')
        ax.scatter(x, y, z, c=colors, s=0.5, alpha=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(view_name)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-view visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='3D OCC Visualization for Headless Servers')
    parser.add_argument('pred_dir', help='Directory containing prediction npz files')
    parser.add_argument('--save-path', type=str, default='./vis_3d', 
                        help='Directory to save visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--view', type=str, default='3d', choices=['3d', 'front', 'multi'],
                        help='View type: 3d, front, or multi (all views)')
    parser.add_argument('--elevation', type=float, default=30,
                        help='Camera elevation for 3D view')
    parser.add_argument('--azimuth', type=float, default=-60,
                        help='Camera azimuth for 3D view')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Find all npz files
    npz_files = []
    for root, dirs, files in os.walk(args.pred_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))
    
    if not npz_files:
        print(f"No npz files found in {args.pred_dir}")
        print("Please run vis_occ.py with --save-npz first!")
        return
    
    print(f"Found {len(npz_files)} npz files")
    npz_files = npz_files[:args.num_samples]
    
    for i, npz_path in enumerate(npz_files):
        print(f"Processing {i+1}/{len(npz_files)}: {npz_path}")
        
        try:
            data = np.load(npz_path)
            pred_occ = data['pred']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
        
        # Generate save name
        rel_path = os.path.relpath(npz_path, args.pred_dir)
        save_name = rel_path.replace('/', '_').replace('.npz', '.png')
        save_path = os.path.join(args.save_path, save_name)
        
        if args.view == 'front':
            visualize_occ_3d(pred_occ, save_path, 
                            title=f'Front View - {os.path.basename(npz_path)}',
                            elevation=0, azimuth=-90)
        elif args.view == 'multi':
            visualize_occ_multiview(pred_occ, save_path,
                                   title=f'Multi-View - {os.path.basename(npz_path)}')
        else:  # 3d
            visualize_occ_3d(pred_occ, save_path,
                            title=f'3D View - {os.path.basename(npz_path)}',
                            elevation=args.elevation, azimuth=args.azimuth)
    
    print(f"\nDone! Visualizations saved to {args.save_path}")


if __name__ == '__main__':
    main()
