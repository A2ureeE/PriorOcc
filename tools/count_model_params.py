#!/usr/bin/env python3
"""
计算完整模型的参数量和计算量

用法：
    python tools/count_model_params.py projects/configs/flashocc/flashocc-r50.py
    python tools/count_model_params.py projects/configs/flashocc/flashocc-r50.py --checkpoint work_dirs/test3/latest.pth
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='统计模型参数量')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--checkpoint', help='权重文件路径（可选）', default=None)
    parser.add_argument('--detail', action='store_true', help='显示各模块详细参数')
    return parser.parse_args()


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()
    
    print("=" * 70)
    print("模型参数量统计")
    print("=" * 70)
    print(f"\n配置文件: {args.config}")
    if args.checkpoint:
        print(f"权重文件: {args.checkpoint}")
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 导入插件
    if getattr(cfg, 'plugin', False):
        import importlib
        plugin_dir = getattr(cfg, 'plugin_dir', '')
        module_path = plugin_dir.replace('/', '.').rstrip('.')
        print(f"加载插件: {module_path}")
        importlib.import_module(module_path)
    
    # 构建模型
    from mmdet3d.models import build_model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.eval()
    
    # 加载权重（可选）
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        print("✓ 权重加载成功")
    
    # 统计总参数量
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "=" * 70)
    print("总体统计")
    print("=" * 70)
    print(f"\n  总参数量:     {total_params:>15,} ({total_params/1e6:.2f} M)")
    print(f"  可训练参数:   {trainable_params:>15,} ({trainable_params/1e6:.2f} M)")
    print(f"  冻结参数:     {total_params - trainable_params:>15,} ({(total_params - trainable_params)/1e6:.2f} M)")
    
    # 显存估算
    print("\n" + "-" * 70)
    print("显存估算 (FP32)")
    print("-" * 70)
    param_mem = total_params * 4 / 1e9  # GB
    print(f"  参数显存:     {param_mem:.2f} GB")
    print(f"  梯度显存:     {param_mem:.2f} GB")
    print(f"  优化器(Adam): {param_mem * 2:.2f} GB")
    print(f"  总计(不含激活): {param_mem * 4:.2f} GB")
    
    # 详细模块统计
    if args.detail:
        print("\n" + "=" * 70)
        print("各模块参数量")
        print("=" * 70)
        
        module_params = {}
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            module_params[name] = params
        
        print(f"\n{'模块名称':<35} {'参数量':>15} {'占比':>10}")
        print("-" * 70)
        for name, params in sorted(module_params.items(), key=lambda x: -x[1]):
            pct = params / total_params * 100 if total_params > 0 else 0
            print(f"  {name:<33} {params:>12,} ({params/1e6:>6.2f}M)  {pct:>5.1f}%")
    
    print("\n" + "=" * 70)
    print(f"✓ 模型总参数量: {total_params/1e6:.2f} M")
    print("=" * 70)


if __name__ == '__main__':
    main()
