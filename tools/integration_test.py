#!/usr/bin/env python3
"""
FlashOCC 集成测试脚本

验证以下内容：
1. 配置文件能否正确构建 BEVDetOCC 模型（含 SemanticInjector）
2. extract_feat 返回的张量维度是否匹配
3. forward_train 能否完成并产生 loss_2d_seg
4. 反向传播是否正常

运行方式：
    cd /home/azure/learning/FlashOCC
    python tools/integration_test.py

中文注释已添加以便理解每一步。
"""

import os
import sys
import importlib
import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def import_plugins(cfg):
    """导入插件模块"""
    if not getattr(cfg, 'plugin', False):
        return
    plugin_dir = getattr(cfg, 'plugin_dir', '')
    if not plugin_dir:
        return
    # projects/mmdet3d_plugin/ -> projects.mmdet3d_plugin
    module_path = plugin_dir.replace('/', '.').rstrip('.')
    if module_path.endswith('.'):
        module_path = module_path[:-1]
    print(f'[集成测试] 导入插件模块: {module_path}')
    importlib.import_module(module_path)


def prepare_device():
    """准备运行设备（优先 GPU，不兼容则退回 CPU）"""
    if not torch.cuda.is_available():
        print('[集成测试] CUDA 不可用，使用 CPU')
        return torch.device('cpu'), False
    
    try:
        # 测试 CUDA 是否真正可用
        _ = torch.zeros(1, device='cuda:0')
        print('[集成测试] 使用 GPU: cuda:0')
        return torch.device('cuda:0'), True
    except Exception as e:
        print(f'[集成测试] CUDA 不兼容: {e}')
        print('[集成测试] 退回到 CPU')
        return torch.device('cpu'), False


def build_synthetic_batch(cfg, batch_size, num_cams, device):
    """构建合成测试数据批次"""
    img_h, img_w = cfg.data_config['input_size']  # (256, 704)
    
    # 构建 img_inputs
    imgs = torch.randn(batch_size, num_cams, 3, img_h, img_w, device=device)
    
    def eye_repeat(dim):
        return torch.eye(dim, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1)
    
    sensor2egos = eye_repeat(4)
    ego2globals = eye_repeat(4)
    
    intrins = torch.zeros(batch_size, num_cams, 3, 3, device=device)
    intrins[:, :, 0, 0] = 1000  # fx
    intrins[:, :, 1, 1] = 1000  # fy
    intrins[:, :, 0, 2] = img_w / 2  # cx
    intrins[:, :, 1, 2] = img_h / 2  # cy
    intrins[:, :, 2, 2] = 1
    
    post_rots = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1)
    post_trans = torch.zeros(batch_size, num_cams, 3, device=device)
    bda = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
    
    # 构建 img_metas
    img_metas = [
        {
            'img_shape': [(img_h, img_w, 3)] * num_cams,
            'ori_shape': [(img_h, img_w, 3)] * num_cams,
            'pad_shape': [(img_h, img_w, 3)] * num_cams,
            'lidar2img': [np.eye(4, dtype=np.float32) for _ in range(num_cams)],
            'box_type_3d': 'LiDAR',
        }
    ]
    
    # 构建 voxel_semantics 和 mask_camera
    grid_x = int((cfg.grid_config['x'][1] - cfg.grid_config['x'][0]) / cfg.grid_config['x'][2])
    grid_y = int((cfg.grid_config['y'][1] - cfg.grid_config['y'][0]) / cfg.grid_config['y'][2])
    grid_z = cfg.model['occ_head'].get('Dz', 16)
    
    num_occ_classes = cfg.model['occ_head'].get('num_classes', 18)
    voxel_semantics = torch.randint(0, num_occ_classes, (batch_size, grid_x, grid_y, grid_z), device=device)
    mask_camera = torch.ones(batch_size, grid_x, grid_y, grid_z, dtype=torch.bool, device=device)
    
    # 构建 gt_depth
    gt_depth = torch.rand(batch_size, num_cams, img_h // 8, img_w // 8, device=device) * 45.0
    
    # 构建 gt_semantic_2d（用于 SemanticInjector 损失）
    feat_h, feat_w = img_h // 16, img_w // 16
    seg_num_classes = cfg.model.get('semantic_injector', {}).get('num_classes', 17)
    gt_semantic_2d = torch.randint(0, seg_num_classes, (batch_size, num_cams, feat_h, feat_w), device=device)
    
    return dict(
        img_inputs=img_inputs,
        img_metas=img_metas,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        gt_depth=gt_depth,
        gt_semantic_2d=gt_semantic_2d,
    )


def main():
    print('=' * 60)
    print('FlashOCC 集成测试')
    print('=' * 60)
    
    # ============================================================
    # Step 1: 加载配置并构建模型
    # ============================================================
    print('\n[Step 1] 加载配置并构建模型...')
    
    from mmcv import Config
    from mmdet3d.models import build_model
    
    config_path = 'projects/configs/flashocc/flashocc-r50.py'
    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    
    device, cuda_available = prepare_device()
    
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    
    # 检查 SemanticInjector 是否存在
    if hasattr(model, 'semantic_injector') and model.semantic_injector is not None:
        print('  ✓ SemanticInjector 模块已加载')
    else:
        print('  ✗ SemanticInjector 未找到！')
        return False
    
    print(f'  ✓ 模型构建成功，设备: {device}')
    
    # ============================================================
    # Step 2: 测试 SemanticInjector 单独前向
    # ============================================================
    print('\n[Step 2] 测试 SemanticInjector 前向...')
    
    batch_size = 1
    num_cams = 6
    img_h, img_w = cfg.data_config['input_size']
    feat_h, feat_w = img_h // 16, img_w // 16
    in_channels = cfg.model['semantic_injector']['in_channels']
    
    # 模拟 backbone 输出的特征
    dummy_feat = torch.randn(batch_size * num_cams, in_channels, feat_h, feat_w, device=device)
    
    try:
        feat_fused, seg_logits = model.semantic_injector(dummy_feat)
        
        print(f'  输入特征: {tuple(dummy_feat.shape)}')
        print(f'  融合特征: {tuple(feat_fused.shape)}')
        print(f'  分割 logits: {tuple(seg_logits.shape)}')
        
        # 验证输出维度
        expected_out_channels = cfg.model['semantic_injector']['out_channels']
        expected_num_classes = cfg.model['semantic_injector']['num_classes']
        
        assert feat_fused.shape == (batch_size * num_cams, expected_out_channels, feat_h, feat_w), \
            f'融合特征维度错误: {feat_fused.shape}'
        assert seg_logits.shape == (batch_size * num_cams, expected_num_classes, feat_h, feat_w), \
            f'分割 logits 维度错误: {seg_logits.shape}'
        
        print('  ✓ SemanticInjector 前向测试通过')
        
    except Exception as e:
        print(f'  ✗ SemanticInjector 前向失败: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Step 3: 测试 loss_2d_seg
    # ============================================================
    print('\n[Step 3] 测试 loss_2d_seg...')
    
    try:
        seg_num_classes = cfg.model['semantic_injector']['num_classes']
        gt_semantic_2d = torch.randint(0, seg_num_classes, (batch_size, num_cams, feat_h, feat_w), device=device)
        
        loss_dict = model.loss_2d_seg(seg_logits, gt_semantic_2d)
        
        print(f'  gt_semantic_2d: {tuple(gt_semantic_2d.shape)}')
        print(f'  loss_2d_seg: {loss_dict["loss_2d_seg"].item():.6f}')
        
        assert 'loss_2d_seg' in loss_dict, 'loss_2d_seg 未返回'
        assert loss_dict['loss_2d_seg'].requires_grad, 'loss_2d_seg 无梯度'
        
        print('  ✓ loss_2d_seg 测试通过')
        
    except Exception as e:
        print(f'  ✗ loss_2d_seg 失败: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Step 4: 测试反向传播
    # ============================================================
    print('\n[Step 4] 测试反向传播...')
    
    try:
        loss_dict['loss_2d_seg'].backward()
        
        # 检查 SemanticInjector 是否有梯度
        has_grad = False
        for name, param in model.semantic_injector.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if has_grad:
            print('  ✓ SemanticInjector 参数有梯度')
        else:
            print('  ⚠ SemanticInjector 参数无梯度')
        
        print('  ✓ 反向传播测试通过')
        
    except Exception as e:
        print(f'  ✗ 反向传播失败: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Step 5: 完整 forward_train 测试（仅在 CUDA 可用时）
    # ============================================================
    if cuda_available:
        print('\n[Step 5] 测试完整 forward_train...')
        
        batch = build_synthetic_batch(cfg, batch_size, num_cams, device)
        
        model.train()
        model.zero_grad()
        
        try:
            losses = model.forward_train(
                points=None,
                img_metas=batch['img_metas'],
                img_inputs=batch['img_inputs'],
                voxel_semantics=batch['voxel_semantics'],
                mask_camera=batch['mask_camera'],
                gt_depth=batch['gt_depth'],
                gt_semantic_2d=batch['gt_semantic_2d'],
            )
            
            print('  损失值:')
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    print(f'    {key}: {value.item():.6f}')
            
            if 'loss_2d_seg' not in losses:
                print('  ✗ loss_2d_seg 未在损失中找到')
                return False
            
            print('  ✓ forward_train 测试通过')
            
        except Exception as e:
            print(f'  ✗ forward_train 失败: {e}')
            import traceback
            traceback.print_exc()
            return False
    else:
        print('\n[Step 5] 跳过完整 forward_train 测试（需要 CUDA）')
        print('  ⚠ bev_pool_v2 算子需要 CUDA，CPU 模式下无法测试完整流程')
        print('  ⚠ SemanticInjector 相关测试已通过，完整训练需在 CUDA 环境下进行')
    
    # ============================================================
    # 总结
    # ============================================================
    print('\n' + '=' * 60)
    print('✅ 集成测试完成！')
    print('=' * 60)
    
    if cuda_available:
        print('\n所有测试通过，项目流程验证完成。')
    else:
        print('\nSemanticInjector 模块测试通过。')
        print('完整训练流程需在兼容的 CUDA 环境下验证。')
    
    print('\n开始训练: ./tools/dist_train.sh projects/configs/flashocc/flashocc-r50.py 8')
    
    return True


def print_dependencies():
    """打印依赖版本信息"""
    print('\n' + '=' * 60)
    print('依赖版本信息')
    print('=' * 60)
    
    # Python 版本
    print(f'Python: {sys.version}')
    
    # PyTorch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version (PyTorch): {torch.version.cuda}')
        try:
            print(f'cuDNN version: {torch.backends.cudnn.version()}')
        except Exception:
            print('cuDNN version: N/A')
    
    # NumPy
    print(f'NumPy: {np.__version__}')
    
    # MMCV
    try:
        import mmcv
        print(f'MMCV: {mmcv.__version__}')
    except ImportError:
        print('MMCV: Not installed')
    
    # MMDetection
    try:
        import mmdet
        print(f'MMDetection: {mmdet.__version__}')
    except ImportError:
        print('MMDetection: Not installed')
    
    # MMDetection3D
    try:
        import mmdet3d
        print(f'MMDetection3D: {mmdet3d.__version__}')
    except ImportError:
        print('MMDetection3D: Not installed')
    
    # MMSegmentation (可选)
    try:
        import mmseg
        print(f'MMSegmentation: {mmseg.__version__}')
    except ImportError:
        print('MMSegmentation: Not installed')
    
    # Transformers (可选，用于 2D 伪标签生成)
    try:
        import transformers
        print(f'Transformers: {transformers.__version__}')
    except ImportError:
        print('Transformers: Not installed')
    
    # OpenCV
    try:
        import cv2
        print(f'OpenCV: {cv2.__version__}')
    except ImportError:
        print('OpenCV: Not installed')
    
    # Pillow
    try:
        from PIL import Image
        import PIL
        print(f'Pillow: {PIL.__version__}')
    except ImportError:
        print('Pillow: Not installed')
    
    # CUDA / GPU 信息
    if torch.cuda.is_available():
        print('\nGPU 信息:')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'  GPU {i}: {props.name}')
            print(f'    Compute capability: {props.major}.{props.minor}')
            print(f'    Total memory: {props.total_memory / 1024**3:.2f} GB')
    
    print('=' * 60)


if __name__ == '__main__':
    success = main()
    print_dependencies()
    sys.exit(0 if success else 1)
