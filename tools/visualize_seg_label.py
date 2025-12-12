#!/usr/bin/env python3
"""
可视化 2D 语义分割标签

用法:
    python tools/visualize_seg_label.py --label /tmp/debug_seg_xxx/samples/CAM_FRONT/debug.png
"""
import os
import argparse
import numpy as np
import cv2

# FlashOcc 17 类调色板 (RGB)
PALETTE = np.array([
    [0, 0, 0],        # 0: others (黑色)
    [255, 120, 50],   # 1: barrier (橙色)
    [255, 192, 203],  # 2: bicycle (粉色)
    [255, 255, 0],    # 3: bus (黄色)
    [0, 150, 245],    # 4: car (蓝色)
    [0, 255, 255],    # 5: construction_vehicle (青色)
    [255, 127, 0],    # 6: motorcycle (橙红色)
    [255, 0, 0],      # 7: pedestrian (红色)
    [255, 240, 150],  # 8: traffic_cone (浅黄色)
    [135, 60, 0],     # 9: trailer (棕色)
    [160, 32, 240],   # 10: truck (紫色)
    [255, 0, 255],    # 11: driveable_surface (品红)
    [139, 137, 137],  # 12: other_flat (灰色)
    [75, 0, 75],      # 13: sidewalk (深紫)
    [150, 240, 80],   # 14: terrain (浅绿)
    [230, 230, 250],  # 15: manmade (淡紫)
    [0, 175, 0],      # 16: vegetation (绿色)
], dtype=np.uint8)

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]


def colorize_label(label):
    """将标签图转为彩色可视化"""
    h, w = label.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(17):
        mask = (label == i)
        color_img[mask] = PALETTE[i]
    
    # ignore (255) 用灰色
    color_img[label == 255] = [128, 128, 128]
    
    return color_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, required=True, help='Path to label PNG')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: label_colored.png)')
    args = parser.parse_args()
    
    # 读取标签
    label = cv2.imread(args.label, cv2.IMREAD_GRAYSCALE)
    if label is None:
        print(f'Error: Cannot read {args.label}')
        return
    
    print(f'Label shape: {label.shape}')
    print(f'Unique values: {np.unique(label)}')
    
    # 统计各类别像素数
    print('\n类别统计:')
    for i in range(17):
        count = np.sum(label == i)
        if count > 0:
            pct = count / label.size * 100
            print(f'  {i:2d} {CLASS_NAMES[i]:20s}: {count:6d} px ({pct:.1f}%)')
    
    ignore_count = np.sum(label == 255)
    if ignore_count > 0:
        pct = ignore_count / label.size * 100
        print(f' 255 {"ignore":20s}: {ignore_count:6d} px ({pct:.1f}%)')
    
    # 生成彩色图
    color_img = colorize_label(label)
    
    # 保存
    output_path = args.output or 'label_colored.png'
    # OpenCV 使用 BGR，需要转换
    cv2.imwrite(output_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    print(f'\n✓ 彩色可视化已保存到: {output_path}')


if __name__ == '__main__':
    main()
