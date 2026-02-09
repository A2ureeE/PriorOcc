#!/usr/bin/env python3
"""
调试数据流程 - 检查 gt_semantic_2d 是否正确传递到模型
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 plugin 以注册自定义模块
import projects.mmdet3d_plugin

import torch
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmcv.parallel import collate
from functools import partial

def test_data_pipeline():
    """测试数据 pipeline 是否正确加载 gt_semantic_2d"""
    print("=" * 60)
    print("测试 gt_semantic_2d 数据流程")
    print("=" * 60)
    
    # 加载配置
    cfg = Config.fromfile('projects/configs/flashocc/flashocc-r50-mini.py')
    
    # 构建数据集
    print("\n[1] 构建训练数据集...")
    dataset = build_dataset(cfg.data.train)
    print(f"    样本数量: {len(dataset)}")
    
    # 获取一个样本
    print("\n[2] 获取第一个样本...")
    sample = dataset[0]
    
    print(f"    样本 keys: {sample.keys()}")
    
    # 检查 gt_semantic_2d
    if 'gt_semantic_2d' in sample:
        gt_seg = sample['gt_semantic_2d']
        print(f"\n    ✓ gt_semantic_2d 存在!")
        print(f"      类型: {type(gt_seg)}")
        if hasattr(gt_seg, 'data'):
            gt_seg = gt_seg.data
        if isinstance(gt_seg, torch.Tensor):
            print(f"      形状: {gt_seg.shape}")
            print(f"      dtype: {gt_seg.dtype}")
            unique = torch.unique(gt_seg)
            print(f"      唯一值: {unique.numpy()}")
            non_255 = unique[unique != 255]
            print(f"      有效类别: {non_255.numpy()}")
            ignore_ratio = (gt_seg == 255).sum() / gt_seg.numel() * 100
            print(f"      255 占比: {ignore_ratio:.1f}%")
        elif isinstance(gt_seg, np.ndarray):
            print(f"      形状: {gt_seg.shape}")
            print(f"      dtype: {gt_seg.dtype}")
            unique = np.unique(gt_seg)
            print(f"      唯一值: {unique}")
            non_255 = unique[unique != 255]
            print(f"      有效类别: {non_255}")
            ignore_ratio = (gt_seg == 255).sum() / gt_seg.size * 100
            print(f"      255 占比: {ignore_ratio:.1f}%")
    else:
        print("\n    ✗ gt_semantic_2d 不存在!")
        print(f"      可用 keys: {list(sample.keys())}")
    
    # 使用 dataloader 测试
    print("\n[3] 测试 DataLoader collate...")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate, samples_per_gpu=1)
    )
    
    batch = next(iter(dataloader))
    print(f"    Batch keys: {batch.keys()}")
    
    if 'gt_semantic_2d' in batch:
        gt_seg = batch['gt_semantic_2d']
        if hasattr(gt_seg, 'data'):
            gt_seg = gt_seg.data[0] if isinstance(gt_seg.data, list) else gt_seg.data
        print(f"    ✓ gt_semantic_2d 在 batch 中存在")
        print(f"      形状: {gt_seg.shape if hasattr(gt_seg, 'shape') else 'N/A'}")
        print(f"      dtype: {gt_seg.dtype}")
        unique = torch.unique(gt_seg)
        print(f"      唯一值: {unique.numpy()}")
        non_255 = unique[unique != 255]
        print(f"      有效类别: {non_255.numpy()}")
        ignore_ratio = (gt_seg == 255).sum() / gt_seg.numel() * 100
        print(f"      255 占比: {ignore_ratio:.1f}%")
        
        # 检查是否所有像素都是 255
        if (gt_seg == 255).all():
            print("\n    ⚠ 警告：所有像素都是 255 (ignore)！这会导致 loss = NaN")
        else:
            print(f"\n    ✓ 有 {(gt_seg != 255).sum()} 个有效像素")
    else:
        print("    ✗ gt_semantic_2d 不在 batch 中")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_data_pipeline()
