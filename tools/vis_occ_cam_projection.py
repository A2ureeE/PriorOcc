"""
Camera-View Occupancy Visualization
Projects 3D Occupancy voxels onto 2D camera images.
"""
import os
import cv2
import numpy as np
import argparse
import pickle
import torch
from pyquaternion import Quaternion
from tqdm import tqdm

FREE_LABEL = 17

# Grid config matching the base config (e.g., flashocc-r50-M0.py)
# Grid shape is [200, 200, 16] for [-40, -40, -1, 40, 40, 5.4]
# Or [256, 256, 32] for [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# Need to detect from prediction shape

VOXEL_SIZE_SMALL = [0.4, 0.4, 0.4]  # For 200x200x16 grid
VOXEL_SIZE_LARGE = [0.4, 0.4, 0.25]  # For 256x256x32 grid
POINT_CLOUD_RANGE_SMALL = [-40, -40, -1, 40, 40, 5.4]
POINT_CLOUD_RANGE_LARGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

def get_grid_config(pred_shape):
    """Detect grid config from prediction shape."""
    if pred_shape[0] == 200:
        return VOXEL_SIZE_SMALL, POINT_CLOUD_RANGE_SMALL
    else:
        return VOXEL_SIZE_LARGE, POINT_CLOUD_RANGE_LARGE

# Color map (BGR for OpenCV)
colormap = np.array([
    [0, 0, 0],           # 0 others
    [50, 120, 255],      # 1 barrier (Orange)
    [203, 192, 255],     # 2 bicycle
    [0, 255, 255],       # 3 bus
    [245, 150, 0],       # 4 car (Blue)
    [255, 255, 0],       # 5 construction_vehicle
    [0, 180, 200],       # 6 motorcycle
    [0, 0, 255],         # 7 pedestrian (Red)
    [150, 240, 255],     # 8 traffic_cone
    [0, 60, 135],        # 9 trailer
    [240, 32, 160],      # 10 truck
    [255, 0, 255],       # 11 driveable_surface
    [137, 137, 139],     # 12 other_flat
    [75, 0, 75],         # 13 sidewalk
    [80, 240, 150],      # 14 terrain
    [250, 230, 230],     # 15 manmade
    [0, 175, 0],         # 16 vegetation
], dtype=np.uint8)

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid

def get_lidar2camera(cam_info):
    # Calculate transform from Lidar to Camera
    lidar2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2lidar_rotation']).rotation_matrix)
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    return lidar2cam_rt.T

def lidar2img(points_lidar, camera_info):
    """
    Project lidar points to image plane.
    Exactly matches the implementation in analysis_tools/vis.py
    """
    # Homogeneous coordinates
    points_lidar_homogeneous = np.concatenate(
        [points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)], 
        axis=1
    )
    
    # camera2lidar transform (sensor2lidar)
    camera2lidar = np.eye(4, dtype=np.float32)
    
    rotation = camera_info['sensor2lidar_rotation']
    if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
        camera2lidar[:3, :3] = rotation
    elif len(rotation) == 4:
        camera2lidar[:3, :3] = Quaternion(rotation).rotation_matrix
    else:
        camera2lidar[:3, :3] = np.array(rotation)
    
    camera2lidar[:3, 3] = np.array(camera_info['sensor2lidar_translation'])
    
    # Invert to get lidar2camera
    lidar2camera = np.linalg.inv(camera2lidar)
    
    # Transform points to camera frame
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    
    # Validity check: points must be in front of camera
    valid = points_camera[:, 2] > 0.5
    depth = points_camera[:, 2].copy()
    
    # Normalize by Z (this is the key step!)
    points_camera = points_camera / (points_camera[:, 2:3] + 1e-6)
    
    # Apply camera intrinsic
    camera2img = np.array(camera_info['cam_intrinsic'])
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    
    return points_img, valid, depth

def render_projection(occ_pred, info, data_root, save_path, voxel_size=0.4):
    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    
    # Detect grid config from prediction shape
    VOXEL_SIZE, POINT_CLOUD_RANGE = get_grid_config(occ_pred.shape)
    print(f"  Grid shape: {occ_pred.shape}, Range: {POINT_CLOUD_RANGE}")
    
    # 1. Get occupied voxel centers in Lidar Frame
    occupied_mask = occ_pred != FREE_LABEL
    x_idx, y_idx, z_idx = np.where(occupied_mask)
    if len(x_idx) == 0:
        return
        
    labels = occ_pred[x_idx, y_idx, z_idx]
    
    # Voxel half sizes
    hx, hy, hz = VOXEL_SIZE[0]/2, VOXEL_SIZE[1]/2, VOXEL_SIZE[2]/2
    
    # 8 corners of a unit cube centered at origin
    cube_corners = np.array([
        [-hx, -hy, -hz],
        [+hx, -hy, -hz],
        [+hx, +hy, -hz],
        [-hx, +hy, -hz],
        [-hx, -hy, +hz],
        [+hx, -hy, +hz],
        [+hx, +hy, +hz],
        [-hx, +hy, +hz],
    ])
    
    # Process each camera
    for view in views:
        cam_info = info['cams'][view]
        
        # Fix path
        raw_path = cam_info['data_path']
        if 'samples/' in raw_path:
            rel_path = raw_path[raw_path.find('samples/'):]
        elif 'sweeps/' in raw_path:
            rel_path = raw_path[raw_path.find('sweeps/'):]
        else:
            rel_path = os.path.basename(raw_path)
        img_path = os.path.join(data_root, rel_path)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        overlay = img.copy()
        
        # Get lidar2camera transform
        lidar2camera = np.eye(4, dtype=np.float32)
        rotation = cam_info['sensor2lidar_rotation']
        if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            lidar2camera[:3, :3] = rotation
        elif len(rotation) == 4:
            lidar2camera[:3, :3] = Quaternion(rotation).rotation_matrix
        else:
            lidar2camera[:3, :3] = np.array(rotation)
        lidar2camera[:3, 3] = np.array(cam_info['sensor2lidar_translation'])
        lidar2camera = np.linalg.inv(lidar2camera)
        
        intrinsic = np.array(cam_info['cam_intrinsic'])
        
        # Collect all voxels with their depth for sorting
        voxel_data = []
        
        for i in range(len(x_idx)):
            # Voxel center in lidar frame
            cx = x_idx[i] * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0] + VOXEL_SIZE[0]/2
            cy = y_idx[i] * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1] + VOXEL_SIZE[1]/2
            cz = z_idx[i] * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2] + VOXEL_SIZE[2]/2
            
            center = np.array([cx, cy, cz, 1.0])
            center_cam = lidar2camera @ center
            depth = center_cam[2]
            
            if depth < 0.5:
                continue
                
            voxel_data.append((depth, i, cx, cy, cz))
        
        # Sort by depth (far to near)
        voxel_data.sort(key=lambda x: -x[0])
        
        # Draw voxels
        for depth, i, cx, cy, cz in voxel_data:
            label = labels[i]
            
            # Skip ground classes to avoid occlusion
            # 11: driveable_surface, 12: other_flat, 13: sidewalk, 14: terrain
            if label in [11, 12, 13, 14]:
                continue
                
            color = colormap[label if label < len(colormap) else 0].tolist()
            
            # Get 8 corners in world
            corners_world = cube_corners + np.array([cx, cy, cz])
            corners_h = np.hstack([corners_world, np.ones((8, 1))])
            
            # Project to camera
            corners_cam = (lidar2camera @ corners_h.T).T[:, :3]
            
            # Skip if behind camera
            if np.any(corners_cam[:, 2] < 0.5):
                continue
            
            # Project to image: normalize by Z first, then apply intrinsic
            corners_cam_normalized = corners_cam / (corners_cam[:, 2:3] + 1e-6)
            corners_img = (intrinsic @ corners_cam_normalized.T).T
            corners_2d = corners_img[:, :2]
            
            # Check if any corner is in image
            in_img = (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < w) & \
                     (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < h)
            if not np.any(in_img):
                continue
            
            corners_2d = corners_2d.astype(np.int32)
            
            # Draw cube faces (front-facing ones)
            # Face definitions: indices of corners
            faces = [
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # front
                [2, 3, 7, 6],  # back
                [0, 3, 7, 4],  # left
                [1, 2, 6, 5],  # right
            ]
            
            for face_idx in faces:
                pts = corners_2d[face_idx]
                
                # Simple visibility check: face normal dot view direction
                # Skip for simplicity, just draw all faces
                
                # Draw filled polygon
                cv2.fillPoly(overlay, [pts], color)
                
                # Draw edges for block effect
                edge_color = [max(0, c - 40) for c in color]
                cv2.polylines(overlay, [pts], True, edge_color, 1)
        
        # Blend
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Save
        view_save_path = save_path.replace('.jpg', f'_{view}.jpg')
        cv2.imwrite(view_save_path, img)
        print(f"Saved {view_save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', help='Path to npz dir')
    parser.add_argument('--root_path', default='data/nuscenes', help='Data root')
    parser.add_argument('--info-file', default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl')
    parser.add_argument('--save-path', default='vis_cam')
    parser.add_argument('--use-mini', action='store_true')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()
    
    if args.use_mini and 'mini' not in args.info_file:
        args.info_file = 'data/nuscenes/bevdetv2-nuscenes-mini_infos_val.pkl'
        
    print(f"Loading info from {args.info_file}")
    with open(args.info_file, 'rb') as f:
        dataset = pickle.load(f)
    infos = dataset['infos']
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Map token to info
    token2info = {info['token']: info for info in infos}
    print(f"Loaded {len(infos)} samples from info file")
    
    # Find npz files with FULL paths
    npz_files = []
    for root, dirs, files in os.walk(args.pred_dir):
        for f in files:
            if f.endswith('.npz'):
                full_path = os.path.join(root, f)
                # Token is the parent directory name: .../scene/TOKEN/pred.npz
                token = os.path.basename(os.path.dirname(full_path))
                npz_files.append((full_path, token))
    
    print(f"Found {len(npz_files)} npz files")
    
    # Debug: Show first few tokens
    if len(npz_files) > 0:
        print(f"Sample npz tokens: {[t for _, t in npz_files[:3]]}")
        print(f"Sample info tokens: {list(token2info.keys())[:3]}")
    
    count = 0
    for npz_path, token in npz_files:
        if count >= args.num_samples: 
            break
            
        if token not in token2info:
            print(f"Token {token} not found in info file, skipping...")
            continue
        
        info = token2info[token]
        print(f"[{count+1}/{args.num_samples}] Processing {token}...")
        
        try:
            pred = np.load(npz_path)['pred']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
            
        save_name = os.path.join(args.save_path, f"{token}.jpg")
        render_projection(pred, info, args.root_path, save_name)
        count += 1
    
    print(f"\nDone! Processed {count} samples, saved to {args.save_path}")

if __name__ == '__main__':
    main()
