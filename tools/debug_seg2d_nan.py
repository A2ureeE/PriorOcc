#!/usr/bin/env python3
"""
调试 loss_2d_seg 返回 NaN 的问题
"""
import os
import cv2
import numpy as np
from glob import glob

def check_seg_labels(seg_dir):
    """检查分割标签的统计信息"""
    print(f"\n检查分割标签目录: {seg_dir}")
    
    cam_dirs = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    all_labels_255 = True
    total_files = 0
    valid_files = 0
    
    for cam in cam_dirs:
        cam_path = os.path.join(seg_dir, 'samples', cam)
        if not os.path.exists(cam_path):
            print(f"  ⚠ 相机目录不存在: {cam}")
            continue
        
        files = glob(os.path.join(cam_path, '*.png'))
        total_files += len(files)
        
        if len(files) == 0:
            print(f"  ⚠ {cam}: 无 PNG 文件")
            continue
        
        # 检查前几个文件
        for f in files[:3]:
            label = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if label is None:
                print(f"  ✗ 无法读取: {f}")
                continue
            
            unique_vals = np.unique(label)
            non_ignore = unique_vals[unique_vals != 255]
            
            if len(non_ignore) > 0:
                all_labels_255 = False
                valid_files += 1
                print(f"  ✓ {cam}/{os.path.basename(f)}:")
                print(f"    形状: {label.shape}")
                print(f"    唯一值: {unique_vals}")
                print(f"    有效类别 (非255): {non_ignore}")
                print(f"    255 占比: {(label == 255).sum() / label.size * 100:.1f}%")
            else:
                print(f"  ⚠ {cam}/{os.path.basename(f)}: 全为 255 (ignore)")
    
    print(f"\n总计: {total_files} 个文件")
    if all_labels_255:
        print("⚠ 警告: 所有检查的标签都只包含 255 (ignore)!")
        print("  这会导致 CrossEntropy loss 返回 NaN")
    else:
        print(f"✓ 发现 {valid_files} 个包含有效类别的标签文件")
    
    return not all_labels_255


def main():
    seg_dir = 'data/nuscenes/seg_2d_labels'
    
    if not os.path.exists(seg_dir):
        print(f"目录不存在: {seg_dir}")
        return
    
    check_seg_labels(seg_dir)


if __name__ == '__main__':
    main()
