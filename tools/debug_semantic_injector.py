#!/usr/bin/env python
"""
Debug script for SemanticInjector integration.
This script validates:
1. Model builds correctly with SemanticInjector
2. Tensor dimensions match through the forward pass
3. Memory usage is reasonable
4. forward_train runs without errors

Usage:
    cd /home/azure/learning/FlashOCC
    python tools/debug_semantic_injector.py
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("SemanticInjector Debug Script")
    print("=" * 60)
    
    # ============================================================
    # Step 1: Load config and build model
    # ============================================================
    print("\n[Step 1] Loading config and building model...")
    
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmcv.runner import load_checkpoint
    
    config_path = 'projects/configs/flashocc/flashocc-r50.py'
    cfg = Config.fromfile(config_path)
    
    # Import plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"    Importing plugin from: {_module_path}")
            plg_lib = importlib.import_module(_module_path)
    
    # Build model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # Check if semantic_injector exists
    if hasattr(model, 'semantic_injector') and model.semantic_injector is not None:
        print("    ✓ SemanticInjector module found!")
        print(f"    SemanticInjector config: {model.semantic_injector}")
    else:
        print("    ✗ SemanticInjector NOT found in model!")
        print("    Please check your config file.")
        return
    
    # Move to GPU if available and compatible
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Check CUDA compatibility
        try:
            _ = torch.randn(1, device='cuda:0')
            device = torch.device('cuda:0')
        except RuntimeError as e:
            print(f"    ⚠ CUDA not compatible: {e}")
            print("    Falling back to CPU...")
            use_cuda = False
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.train()
    print(f"    Model moved to: {device}")
    
    # ============================================================
    # Step 2: Generate dummy input data
    # ============================================================
    print("\n[Step 2] Generating dummy input data...")
    
    batch_size = 1
    num_cams = 6
    img_h, img_w = cfg.data_config['input_size']  # (256, 704)
    
    # img_inputs: (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)
    imgs = torch.randn(batch_size, num_cams, 3, img_h, img_w, device=device)
    sensor2egos = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1)
    ego2globals = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1)
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
    
    # img_metas
    img_metas = [
        {
            'img_shape': [(img_h, img_w, 3)] * num_cams,
            'ori_shape': [(img_h, img_w, 3)] * num_cams,
            'pad_shape': [(img_h, img_w, 3)] * num_cams,
            'lidar2img': [np.eye(4) for _ in range(num_cams)],
            'box_type_3d': 'LiDAR',
        }
    ]
    
    # voxel_semantics and mask_camera (for OCC loss)
    # Grid size from config: x=200, y=200, z=16
    grid_x = int((cfg.grid_config['x'][1] - cfg.grid_config['x'][0]) / cfg.grid_config['x'][2])
    grid_y = int((cfg.grid_config['y'][1] - cfg.grid_config['y'][0]) / cfg.grid_config['y'][2])
    grid_z = 16  # typically 16
    
    voxel_semantics = torch.randint(0, 18, (batch_size, grid_x, grid_y, grid_z), device=device)
    mask_camera = torch.ones(batch_size, grid_x, grid_y, grid_z, dtype=torch.bool, device=device)
    
    # gt_depth (for depth supervision if needed)
    gt_depth = torch.rand(batch_size, num_cams, img_h, img_w, device=device) * 45.0  # depth range 0-45m
    
    # gt_semantic_2d (for SemanticInjector loss) - Optional
    # Feature map size after backbone (typically input_size / 16)
    feat_h, feat_w = img_h // 16, img_w // 16
    gt_semantic_2d = torch.randint(0, 17, (batch_size, num_cams, feat_h, feat_w), device=device)
    
    print(f"    imgs shape: {imgs.shape}")
    print(f"    voxel_semantics shape: {voxel_semantics.shape}")
    print(f"    gt_semantic_2d shape: {gt_semantic_2d.shape}")
    
    # ============================================================
    # Step 3: Run forward_train
    # ============================================================
    print("\n[Step 3] Running forward_train...")
    
    # Clear GPU cache
    if use_cuda:
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated(device) / 1024**2
        print(f"    GPU memory before forward: {mem_before:.2f} MB")
    
    try:
        losses = model.forward_train(
            points=None,
            img_metas=img_metas,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            img_inputs=img_inputs,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            gt_depth=gt_depth,
            gt_semantic_2d=gt_semantic_2d,  # Pass 2D semantic GT
        )
        
        print("    ✓ forward_train completed successfully!")
        print("\n    Losses:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"      {k}: {v.item():.4f}")
            else:
                print(f"      {k}: {v}")
        
        # Check if loss_2d_seg is present
        if 'loss_2d_seg' in losses:
            print("\n    ✓ loss_2d_seg is being computed!")
        else:
            print("\n    ⚠ loss_2d_seg NOT found in losses (check if gt_semantic_2d is passed correctly)")
            
    except Exception as e:
        print(f"    ✗ forward_train FAILED!")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # Step 4: Check memory usage
    # ============================================================
    if use_cuda:
        mem_after = torch.cuda.memory_allocated(device) / 1024**2
        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\n[Step 4] GPU Memory Usage:")
        print(f"    After forward: {mem_after:.2f} MB")
        print(f"    Peak usage: {mem_peak:.2f} MB")
    else:
        print("\n[Step 4] GPU Memory Usage: N/A (running on CPU)")
    
    # ============================================================
    # Step 5: Test backward pass
    # ============================================================
    print("\n[Step 5] Testing backward pass...")
    
    try:
        total_loss = sum([v for k, v in losses.items() if isinstance(v, torch.Tensor)])
        total_loss.backward()
        print("    ✓ Backward pass completed successfully!")
        
        # Check if SemanticInjector gradients exist
        if model.semantic_injector is not None:
            has_grad = False
            for name, param in model.semantic_injector.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    break
            if has_grad:
                print("    ✓ SemanticInjector has gradients!")
            else:
                print("    ⚠ SemanticInjector has NO gradients (check loss computation)")
                
    except Exception as e:
        print(f"    ✗ Backward pass FAILED!")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    print("✓ Model built successfully with SemanticInjector")
    print("✓ forward_train completed")
    print("✓ backward pass completed")
    print("✓ All tensor dimensions matched correctly")
    print("\nYou can now proceed to prepare gt_semantic_2d data and start training!")
    print("=" * 60)


if __name__ == '__main__':
    main()
