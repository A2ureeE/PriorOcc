#!/usr/bin/env python3
"""
调试脚本：验证 LoadSemanticSeg2D 数据加载器的功能

功能：
- 测试路径映射逻辑
- 测试标签加载和几何变换
- 使用模拟数据验证完整流程

中文注释已添加以便理解每一步。
"""

import os
import sys
import tempfile
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_path_mapping():
    """测试路径映射逻辑"""
    print('=' * 60)
    print('测试 1: 路径映射逻辑')
    print('=' * 60)
    
    from projects.mmdet3d_plugin.datasets.pipelines.loading_seg2d import LoadSemanticSeg2D
    
    loader = LoadSemanticSeg2D(seg_prefix='data/nuscenes/seg_2d_labels')
    
    # 测试各种路径格式
    test_cases = [
        'data/nuscenes/samples/CAM_FRONT/n008-2018-xxx.jpg',
        '/abs/path/to/samples/CAM_BACK/yyy.jpg',
        'samples/CAM_FRONT_RIGHT/zzz.jpeg',
    ]
    
    for img_path in test_cases:
        seg_path = loader._get_seg_path(img_path)
        print(f'  输入: {img_path}')
        print(f'  输出: {seg_path}')
        print()
    
    print('✓ 路径映射测试完成')


def test_label_loading():
    """测试标签加载（使用临时文件）"""
    print('=' * 60)
    print('测试 2: 标签加载')
    print('=' * 60)
    
    import cv2
    from projects.mmdet3d_plugin.datasets.pipelines.loading_seg2d import LoadSemanticSeg2D
    
    # 创建临时目录和模拟伪标签
    tmpdir = tempfile.mkdtemp(prefix='debug_loading_')
    seg_dir = os.path.join(tmpdir, 'samples', 'CAM_FRONT')
    os.makedirs(seg_dir, exist_ok=True)
    
    # 创建模拟伪标签（64x128，随机类别）
    rng = np.random.RandomState(42)
    mock_label = rng.randint(0, 17, size=(64, 128), dtype=np.uint8)
    mock_label[0:10, :] = 255  # 添加一些 ignore 区域
    seg_path = os.path.join(seg_dir, 'test.png')
    cv2.imwrite(seg_path, mock_label)
    
    print(f'  创建临时伪标签: {seg_path}')
    print(f'  标签尺寸: {mock_label.shape}')
    print(f'  类别范围: {mock_label.min()} - {mock_label.max()}')
    
    # 测试加载
    loader = LoadSemanticSeg2D(seg_prefix=tmpdir)
    loaded = loader._load_seg_label(seg_path)
    
    print(f'  加载后尺寸: {loaded.shape}')
    print(f'  加载后类别: {np.unique(loaded)}')
    
    # 验证一致性
    assert np.array_equal(mock_label, loaded), '加载的标签与原始不一致！'
    print('✓ 标签加载测试完成')


def test_transform():
    """测试几何变换"""
    print('=' * 60)
    print('测试 3: 几何变换')
    print('=' * 60)
    
    from projects.mmdet3d_plugin.datasets.pipelines.loading_seg2d import LoadSemanticSeg2D
    
    loader = LoadSemanticSeg2D()
    
    # 创建测试标签
    label = np.zeros((100, 200), dtype=np.uint8)
    label[25:75, 50:150] = 1  # 中间区域为类别 1
    
    # 测试 resize
    resized = loader._apply_transform(label, resize_dims=(100, 50))
    print(f'  原始尺寸: {label.shape}')
    print(f'  Resize 后: {resized.shape}')
    assert resized.shape == (50, 100), f'Resize 失败: {resized.shape}'
    
    # 测试 crop
    cropped = loader._apply_transform(label, crop=(50, 25, 150, 75))
    print(f'  Crop 后: {cropped.shape}')
    assert cropped.shape == (50, 100), f'Crop 失败: {cropped.shape}'
    
    # 测试 flip
    flipped = loader._apply_transform(label, flip=True)
    print(f'  Flip 后: {flipped.shape}')
    assert flipped.shape == label.shape, f'Flip 失败: {flipped.shape}'
    
    print('✓ 几何变换测试完成')


def test_full_pipeline():
    """测试完整 pipeline 流程"""
    print('=' * 60)
    print('测试 4: 完整 Pipeline 流程')
    print('=' * 60)
    
    import cv2
    from projects.mmdet3d_plugin.datasets.pipelines.loading_seg2d import LoadSemanticSeg2D
    
    # 创建临时目录和 6 个相机的模拟伪标签
    tmpdir = tempfile.mkdtemp(prefix='debug_pipeline_')
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    rng = np.random.RandomState(0)
    
    for cam in cam_names:
        seg_dir = os.path.join(tmpdir, 'samples', cam)
        os.makedirs(seg_dir, exist_ok=True)
        
        # 创建模拟标签
        mock_label = rng.randint(0, 17, size=(900, 1600), dtype=np.uint8)
        cv2.imwrite(os.path.join(seg_dir, 'test_frame.png'), mock_label)
    
    # 构造模拟 results
    results = {
        'cams': {
            cam: {'data_path': f'{tmpdir}/samples/{cam}/test_frame.jpg'}
            for cam in cam_names
        }
    }
    
    # 运行 pipeline
    loader = LoadSemanticSeg2D(seg_prefix=tmpdir)
    results = loader(results)
    
    gt_semantic_2d = results['gt_semantic_2d']
    print(f'  输出 shape: {gt_semantic_2d.shape}')
    print(f'  输出 dtype: {gt_semantic_2d.dtype}')
    print(f'  类别范围: {gt_semantic_2d.min()} - {gt_semantic_2d.max()}')
    print(f'  唯一类别数: {len(np.unique(gt_semantic_2d))}')
    
    # 验证
    assert gt_semantic_2d.shape == (6, 900, 1600), f'Shape 不正确: {gt_semantic_2d.shape}'
    assert gt_semantic_2d.dtype == np.uint8, f'Dtype 不正确: {gt_semantic_2d.dtype}'
    
    print('✓ 完整 Pipeline 测试完成')


def test_missing_file():
    """测试文件不存在时的处理"""
    print('=' * 60)
    print('测试 5: 缺失文件处理')
    print('=' * 60)
    
    from projects.mmdet3d_plugin.datasets.pipelines.loading_seg2d import LoadSemanticSeg2D
    
    loader = LoadSemanticSeg2D(seg_prefix='/non/existent/path')
    
    # 构造模拟 results（指向不存在的文件）
    results = {
        'cams': {
            'CAM_FRONT': {'data_path': '/non/existent/samples/CAM_FRONT/xxx.jpg'},
            'CAM_FRONT_RIGHT': {'data_path': '/non/existent/samples/CAM_FRONT_RIGHT/xxx.jpg'},
            'CAM_FRONT_LEFT': {'data_path': '/non/existent/samples/CAM_FRONT_LEFT/xxx.jpg'},
            'CAM_BACK': {'data_path': '/non/existent/samples/CAM_BACK/xxx.jpg'},
            'CAM_BACK_LEFT': {'data_path': '/non/existent/samples/CAM_BACK_LEFT/xxx.jpg'},
            'CAM_BACK_RIGHT': {'data_path': '/non/existent/samples/CAM_BACK_RIGHT/xxx.jpg'},
        }
    }
    
    # 应该不报错，而是返回全 ignore
    results = loader(results)
    gt_semantic_2d = results['gt_semantic_2d']
    
    print(f'  输出 shape: {gt_semantic_2d.shape}')
    print(f'  唯一值: {np.unique(gt_semantic_2d)}')
    
    # 所有值应该是 255 (ignore)
    assert np.all(gt_semantic_2d == 255), '缺失文件应该返回全 ignore'
    
    print('✓ 缺失文件处理测试完成')


def main():
    print('LoadSemanticSeg2D 调试脚本')
    print('=' * 60)
    
    try:
        test_path_mapping()
        print()
        test_label_loading()
        print()
        test_transform()
        print()
        test_full_pipeline()
        print()
        test_missing_file()
        print()
        
        print('=' * 60)
        print('✅ 所有测试通过！')
        print('=' * 60)
        
    except Exception as e:
        print(f'\n❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
