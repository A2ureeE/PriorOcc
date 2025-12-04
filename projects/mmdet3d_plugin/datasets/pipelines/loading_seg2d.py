# Copyright (c) 2024. All rights reserved.
"""
加载 2D 语义分割伪标签的 Pipeline 模块

功能：
- 从预生成的伪标签 PNG 文件中读取 2D 语义分割标签
- 支持多相机视角（nuScenes 6 个相机）
- 与图像预处理同步进行 resize/crop/flip 变换
- 输出 gt_semantic_2d 供 SemanticInjector 训练使用

中文注释已添加以便理解每一步。
"""

import os.path as osp
import numpy as np
import cv2

from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadSemanticSeg2D:
    """加载 2D 语义分割标签（伪标签）的 Pipeline。
    
    用于从预生成的 PNG 文件中读取每个相机视角的 2D 语义分割标签，
    并应用与图像相同的几何变换（resize, crop, flip）。
    
    Args:
        seg_prefix (str): 伪标签文件的根目录路径。
            默认为 'data/nuscenes/seg_2d_labels'
        num_classes (int): 语义类别数量。默认为 17（nuScenes）
        ignore_index (int): 忽略像素的索引值。默认为 255
        to_float32 (bool): 是否转换为 float32。默认为 False
    
    输入 results 需包含:
        - cams (dict): 相机信息，包含 data_path
        - 可选: img_aug_matrix 用于同步几何变换
    
    输出:
        - gt_semantic_2d (np.ndarray): shape (N_views, H, W)，dtype uint8
    """
    
    def __init__(self,
                 seg_prefix='data/nuscenes/seg_2d_labels',
                 num_classes=17,
                 ignore_index=255,
                 to_float32=False):
        self.seg_prefix = seg_prefix
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.to_float32 = to_float32
        
        # nuScenes 相机名称（与 PrepareImageInputs 中顺序一致）
        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
    
    def _get_seg_path(self, img_path):
        """根据图像路径构建对应的伪标签路径。
        
        Args:
            img_path (str): 原始图像路径，如 
                'data/nuscenes/samples/CAM_FRONT/xxx.jpg' 或绝对路径
        
        Returns:
            str: 伪标签路径，如 
                'data/nuscenes/seg_2d_labels/samples/CAM_FRONT/xxx.png'
        """
        # 提取 samples/CAM_XXX/filename 部分
        if 'samples/' in img_path:
            rel_path = 'samples/' + img_path.split('samples/')[-1]
        else:
            # 如果路径格式不符合预期，使用文件名
            rel_path = osp.basename(img_path)
        
        # 构建伪标签路径
        seg_path = osp.join(self.seg_prefix, rel_path)
        # 替换扩展名为 .png
        seg_path = seg_path.rsplit('.', 1)[0] + '.png'
        
        return seg_path
    
    def _load_seg_label(self, seg_path, target_size=None):
        """加载单个伪标签文件。
        
        Args:
            seg_path (str): 伪标签文件路径
            target_size (tuple): 目标尺寸 (H, W)，用于 resize
        
        Returns:
            np.ndarray: 语义标签，shape (H, W)，dtype uint8
        """
        if osp.exists(seg_path):
            # 读取单通道 PNG
            seg_label = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if seg_label is None:
                # 读取失败，返回全 ignore
                seg_label = np.full(
                    target_size if target_size else (900, 1600),
                    self.ignore_index, dtype=np.uint8
                )
        else:
            # 文件不存在，返回全 ignore
            seg_label = np.full(
                target_size if target_size else (900, 1600),
                self.ignore_index, dtype=np.uint8
            )
        
        return seg_label
    
    def _apply_transform(self, seg_label, resize_dims=None, crop=None, flip=False):
        """对标签应用与图像相同的几何变换。
        
        Args:
            seg_label (np.ndarray): 原始标签 (H, W)
            resize_dims (tuple): resize 后的尺寸 (W, H)
            crop (tuple): (x1, y1, x2, y2) 裁剪区域
            flip (bool): 是否水平翻转
        
        Returns:
            np.ndarray: 变换后的标签
        """
        # Resize
        if resize_dims is not None:
            seg_label = cv2.resize(
                seg_label, resize_dims, 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Crop
        if crop is not None:
            x1, y1, x2, y2 = [int(c) for c in crop]
            H, W = seg_label.shape
            # 处理边界情况（crop 可能超出图像范围）
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - W)
            pad_bottom = max(0, y2 - H)
            
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                # 需要 padding
                seg_label = np.pad(
                    seg_label,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=self.ignore_index
                )
                x1 += pad_left
                y1 += pad_top
                x2 += pad_left
                y2 += pad_top
            
            seg_label = seg_label[y1:y2, x1:x2]
        
        # Flip
        if flip:
            seg_label = np.flip(seg_label, axis=1).copy()
        
        return seg_label
    
    def __call__(self, results):
        """Pipeline 主入口。
        
        Args:
            results (dict): 包含图像和相机信息的字典
        
        Returns:
            dict: 添加了 'gt_semantic_2d' 键的结果字典
        """
        gt_semantic_2d = []
        
        # 获取相机列表（可能被 PrepareImageInputs 过滤过）
        if 'cam_names' in results:
            cam_names = results['cam_names']
        else:
            cam_names = self.cam_names
        
        # 获取变换参数（如果存在）
        # PrepareImageInputs 会存储这些参数
        img_augs = results.get('img_augs', None)
        
        for idx, cam_name in enumerate(cam_names):
            # 获取图像路径
            if 'cams' in results:
                img_path = results['cams'][cam_name]['data_path']
            elif 'img_filename' in results:
                img_paths = results['img_filename']
                img_path = img_paths[idx] if isinstance(img_paths, list) else img_paths
            else:
                # 无法获取路径，使用全 ignore
                gt_semantic_2d.append(
                    np.full((256, 704), self.ignore_index, dtype=np.uint8)
                )
                continue
            
            # 构建伪标签路径并加载
            seg_path = self._get_seg_path(img_path)
            seg_label = self._load_seg_label(seg_path)
            
            # 应用几何变换（与图像保持一致）
            if img_augs is not None and idx < len(img_augs):
                aug = img_augs[idx]
                seg_label = self._apply_transform(
                    seg_label,
                    resize_dims=aug.get('resize_dims'),
                    crop=aug.get('crop'),
                    flip=aug.get('flip', False)
                )
            
            gt_semantic_2d.append(seg_label)
        
        # Stack: (N_views, H, W)
        gt_semantic_2d = np.stack(gt_semantic_2d, axis=0)
        
        if self.to_float32:
            gt_semantic_2d = gt_semantic_2d.astype(np.float32)
        
        results['gt_semantic_2d'] = gt_semantic_2d
        
        return results
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'seg_prefix={self.seg_prefix}, '
                f'num_classes={self.num_classes}, '
                f'ignore_index={self.ignore_index})')
