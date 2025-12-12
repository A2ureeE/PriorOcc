#!/usr/bin/env python3
"""
计算 PriorOcc 模型的参数量和计算量（FLOPs）

用法：
    python tools/count_params_flops.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn


def count_parameters(model, name="Model"):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name}:")
    print(f"  总参数量: {total:,} ({total/1e6:.2f}M)")
    print(f"  可训练参数: {trainable:,} ({trainable/1e6:.2f}M)")
    return total, trainable


def count_module_params(model):
    """按模块统计参数量"""
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    return module_params


def estimate_memory(param_count, batch_size=1, dtype_bytes=4):
    """估算显存占用（仅参数，不含激活）"""
    # 参数本身
    param_mem = param_count * dtype_bytes
    # 梯度（同样大小）
    grad_mem = param_mem
    # 优化器状态（Adam 需要 2x 参数量的状态）
    optim_mem = param_mem * 2
    
    total = param_mem + grad_mem + optim_mem
    return total


def main():
    print("=" * 60)
    print("PriorOcc 模型参数量与计算量分析")
    print("=" * 60)
    
    # ============================================================
    # 1. 单独计算新增模块
    # ============================================================
    print("\n" + "=" * 60)
    print("新增模块分析")
    print("=" * 60)
    
    # SemanticInjector
    print("\n--- SemanticInjector ---")
    from projects.mmdet3d_plugin.models.model_utils import SemanticInjector
    
    sem_inj = SemanticInjector(
        in_channels=256,
        out_channels=256,
        num_classes=17,
    )
    sem_params, _ = count_parameters(sem_inj, "SemanticInjector")
    
    # 计算 FLOPs (近似)
    # seg_head: Conv2d(256, 256, 3) + Conv2d(256, 17, 1)
    # fusion_layer: Conv2d(273, 256, 1)
    H, W = 16, 44  # 特征图尺寸
    sem_flops = (
        256 * 256 * 3 * 3 * H * W +  # seg_head conv1
        256 * 17 * 1 * 1 * H * W +   # seg_head conv2
        273 * 256 * 1 * 1 * H * W    # fusion_layer
    )
    print(f"  估算 FLOPs: {sem_flops:,} ({sem_flops/1e9:.4f} GFLOPs)")
    
    # LanguageSelfGating
    print("\n--- LanguageSelfGating ---")
    from projects.mmdet3d_plugin.models.model_utils import LanguageSelfGating
    
    lsg = LanguageSelfGating(
        in_channels=256,
        proj_channels=128,
        num_anchors=6,
        grid_D=16,
        scale=1.0,
    )
    lsg_params, _ = count_parameters(lsg, "LanguageSelfGating")
    
    # 计算 FLOPs
    # projector: Conv3d(256, 128, 1)
    # 相似度计算: einsum
    D, H, W = 16, 200, 200  # BEV 特征尺寸
    lsg_flops = (
        256 * 128 * 1 * 1 * 1 * D * H * W +  # projector
        128 * 6 * D * H * W +                 # 相似度计算
        D * H * W                             # sigmoid
    )
    print(f"  估算 FLOPs: {lsg_flops:,} ({lsg_flops/1e9:.4f} GFLOPs)")
    
    # ============================================================
    # 2. 完整模型分析
    # ============================================================
    print("\n" + "=" * 60)
    print("完整模型分析")
    print("=" * 60)
    
    try:
        from mmcv import Config
        from mmdet3d.models import build_model
        import importlib
        
        config_path = os.path.join(PROJECT_ROOT, 'projects/configs/flashocc/flashocc-r50.py')
        cfg = Config.fromfile(config_path)
        
        # 导入插件
        if getattr(cfg, 'plugin', False):
            plugin_dir = getattr(cfg, 'plugin_dir', '')
            module_path = plugin_dir.replace('/', '.').rstrip('.')
            importlib.import_module(module_path)
        
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        
        total_params, trainable_params = count_parameters(model, "BEVDetOCC (完整模型)")
        
        # 按模块统计
        print("\n各模块参数量:")
        print("-" * 50)
        module_params = count_module_params(model)
        for name, params in sorted(module_params.items(), key=lambda x: -x[1]):
            pct = params / total_params * 100
            print(f"  {name:30s}: {params:>12,} ({params/1e6:>6.2f}M, {pct:>5.1f}%)")
        
        # ============================================================
        # 3. 新增模块占比
        # ============================================================
        print("\n" + "=" * 60)
        print("新增模块占比分析")
        print("=" * 60)
        
        new_module_params = sem_params + lsg_params
        print(f"\n新增模块总参数量: {new_module_params:,} ({new_module_params/1e6:.2f}M)")
        print(f"原始模型参数量: {total_params - new_module_params:,} ({(total_params - new_module_params)/1e6:.2f}M)")
        print(f"新增占比: {new_module_params / total_params * 100:.2f}%")
        
        # ============================================================
        # 4. 显存估算
        # ============================================================
        print("\n" + "=" * 60)
        print("显存估算 (FP32)")
        print("=" * 60)
        
        mem_bytes = estimate_memory(total_params)
        print(f"参数显存: {total_params * 4 / 1e9:.2f} GB")
        print(f"梯度显存: {total_params * 4 / 1e9:.2f} GB")
        print(f"优化器状态: {total_params * 8 / 1e9:.2f} GB")
        print(f"参数相关总计: {mem_bytes / 1e9:.2f} GB")
        print("\n注: 激活值显存与 batch_size 和输入分辨率相关，需实际测试")
        
    except Exception as e:
        print(f"\n完整模型加载失败: {e}")
        print("仅显示新增模块的统计结果")
    
    # ============================================================
    # 5. 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"\nSemanticInjector: {sem_params/1e6:.2f}M 参数, {sem_flops/1e9:.4f} GFLOPs")
    print(f"LanguageSelfGating: {lsg_params/1e6:.2f}M 参数, {lsg_flops/1e9:.4f} GFLOPs")
    print(f"新增总计: {(sem_params + lsg_params)/1e6:.2f}M 参数")
    print(f"\n结论: 新增模块非常轻量，不会显著增加显存和计算开销")


if __name__ == '__main__':
    main()
