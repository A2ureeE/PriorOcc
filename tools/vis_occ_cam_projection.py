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
VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

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

def project_points(points, cam_info):
    # 1. Lidar to Camera
    lidar2camera = np.eye(4, dtype=np.float32)
    lidar2camera[:3, :3] = Quaternion(cam_info['sensor2lidar_rotation']).rotation_matrix
    lidar2camera[:3, 3] = cam_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(lidar2camera) # Camera to Lidar -> Lidar to Camera
    
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_cam_h = points_h @ lidar2camera.T
    points_cam = points_cam_h[:, :3]
    
    # 2. Camera to Image
    depth = points_cam[:, 2]
    valid_depth = depth > 0.1
    
    intrinsic = np.array(cam_info['cam_intrinsic'])
    points_img = points_cam @ intrinsic.T
    points_img = points_img[:, :2] / points_img[:, 2:3]
    
    return points_img, valid_depth, depth

def render_projection(occ_pred, info, data_root, save_path, voxel_size=0.4):
    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    
    # 1. Get occupied voxel centers in Lidar Frame
    occupied_mask = occ_pred != FREE_LABEL
    x_idx, y_idx, z_idx = np.where(occupied_mask)
    if len(x_idx) == 0:
        return
        
    labels = occ_pred[x_idx, y_idx, z_idx]
    
    # Convert index to physical coordinates (center of voxel)
    # BEVDet/FlashOCC coordinates: x is forward, y is left
    # Grid range: [-40, -40, -1]
    
    pts_x = x_idx * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0] + VOXEL_SIZE[0]/2
    pts_y = y_idx * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1] + VOXEL_SIZE[1]/2
    pts_z = z_idx * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2] + VOXEL_SIZE[2]/2
    
    points_lidar = np.stack([pts_x, pts_y, pts_z], axis=1)
    
    # Process each camera
    for view in views:
        cam_info = info['cams'][view]
        img_path = os.path.join(data_root, cam_info['data_path'])
        
        # Handle mini dataset path difference
        if not os.path.exists(img_path):
            # Try to handle relative path issue
            if img_path.startswith('/'):
                 # Try finding 'samples' or 'sweeps'
                 idx = img_path.find('samples')
                 if idx == -1: idx = img_path.find('sweeps')
                 if idx != -1:
                     img_path = os.path.join(data_root, img_path[idx:])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        
        # Project points
        pts_img, valid_depth, depth = project_points(points_lidar, cam_info)
        
        # Filter points inside image
        valid_uv = check_point_in_img(pts_img, h, w)
        valid = np.logical_and(valid_depth, valid_uv)
        
        # Get valid points
        pts_draw = pts_img[valid]
        depth_draw = depth[valid]
        labels_draw = labels[valid]
        
        # Sort by depth (far to near) so near points overwrite far ones
        sort_idx = np.argsort(depth_draw)[::-1]
        pts_draw = pts_draw[sort_idx]
        labels_draw = labels_draw[sort_idx]
        
        # Draw on image
        # Simple point rendering. For "block" effect, we would need 
        # to project corners of cubes, which is much slower.
        # Making points larger simulates blocks.
        overlay = img.copy()
        
        for i in range(len(pts_draw)):
            pt = pts_draw[i]
            label = labels_draw[i]
            color = colormap[label if label < len(colormap) else 0].tolist()
            
            # Simple circle/rectangle
            # Size depends on depth (closer = larger)
            # Simple approximation: size = f / depth
            pt_size = max(1, int(120 / depth_draw[sort_idx[i]]))
            
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), pt_size, color, -1)
        
        # Blend with original image
        alpha = 0.5
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
    
    # Find npz files
    npz_files = []
    for root, dirs, files in os.walk(args.pred_dir):
        for f in files:
            if f.endswith('.npz'):
                token = os.path.basename(os.path.dirname(f)) # .../token/pred.npz
                # Or sometimes .../scene/token/pred.npz
                # Try to map path to token
                npz_files.append((f, token))
    
    print(f"Found {len(npz_files)} predictions")
    
    count = 0
    for npz_path, token in npz_files:
        if token not in token2info:
            # Try parent dir name
            token = os.path.basename(os.path.dirname(npz_path))
            
        if token not in token2info:
            continue
            
        if count >= args.num_samples: break
        
        info = token2info[token]
        print(f"Processing {token}...")
        
        try:
            pred = np.load(npz_path)['pred']
        except:
            continue
            
        save_name = os.path.join(args.save_path, f"{token}.jpg")
        render_projection(pred, info, args.root_path, save_name)
        count += 1

if __name__ == '__main__':
    main()
