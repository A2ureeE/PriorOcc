#!/usr/bin/env python3
"""
计算完整模型的参数量和计算量（FLOPs）

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
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='统计模型参数量和FLOPs')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--checkpoint', help='权重文件路径（可选）', default=None)
    parser.add_argument('--detail', action='store_true', help='显示各模块详细参数')
    return parser.parse_args()


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, cfg):
    """
    计算模型 FLOPs
    使用 fvcore 库进行计算
    """
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        # 创建模拟输入
        # 6个相机视角，3通道，H x W 分辨率
        data_config = cfg.data_config if hasattr(cfg, 'data_config') else cfg.model.get('data_config', {})
        input_size = data_config.get('input_size', (256, 704))
        H, W = input_size
        
        # 构造输入数据
        imgs = torch.randn(1, 6, 3, H, W)  # (B, N_views, C, H, W)
        
        # 相机内参/外参 (简化)
        sensor2egos = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1)
        ego2globals = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1)
        intrins = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1)
        intrins[:, :, 0, 0] = 1000  # fx
        intrins[:, :, 1, 1] = 1000  # fy
        intrins[:, :, 0, 2] = W / 2  # cx
        intrins[:, :, 1, 2] = H / 2  # cy
        post_rots = torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1)
        post_trans = torch.zeros(1, 6, 2)
        bda = torch.eye(3).unsqueeze(0)
        
        img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
        
        # 使用 fvcore 计算 FLOPs
        model.eval()
        
        # 尝试使用 image_encoder 单独计算
        if hasattr(model, 'img_backbone'):
            # 计算 backbone FLOPs
            backbone_input = imgs.view(-1, 3, H, W)  # (B*N, C, H, W)
            flops_backbone = FlopCountAnalysis(model.img_backbone, (backbone_input,))
            backbone_flops = flops_backbone.total()
        else:
            backbone_flops = 0
        
        return backbone_flops, None
        
    except ImportError:
        print("  ⚠ fvcore 未安装，尝试使用 thop...")
        try:
            from thop import profile, clever_format
            
            data_config = cfg.data_config if hasattr(cfg, 'data_config') else cfg.model.get('data_config', {})
            input_size = data_config.get('input_size', (256, 704))
            H, W = input_size
            
            # 单独计算 backbone
            if hasattr(model, 'img_backbone'):
                backbone_input = torch.randn(6, 3, H, W)  # 6 views
                flops, _ = profile(model.img_backbone, inputs=(backbone_input,), verbose=False)
                return flops, None
            
            return 0, None
            
        except ImportError:
            print("  ⚠ thop 未安装，使用手动估算...")
            return None, None
    except Exception as e:
        print(f"  ⚠ FLOPs 计算失败: {e}")
        return None, None


def estimate_flops_manual(model, cfg):
    """手动估算 FLOPs（基于模型结构动态计算，只计算启用的模块）"""
    data_config = cfg.data_config if hasattr(cfg, 'data_config') else cfg.model.get('data_config', {})
    input_size = data_config.get('input_size', (256, 704))
    H, W = input_size
    N_views = 6
    
    total_flops = 0
    flops_breakdown = {}
    
    # 1. Backbone (ResNet50): ~4.1 GFLOPs per image at 224x224，按分辨率缩放
    resnet50_base = 4.1e9
    scale = (H * W) / (224 * 224)
    backbone_flops = resnet50_base * scale * N_views
    flops_breakdown['img_backbone'] = backbone_flops
    total_flops += backbone_flops
    
    # 2. Neck (FPN): ~0.5 GFLOPs per view
    neck_flops = 0.5e9 * N_views
    flops_breakdown['img_neck'] = neck_flops
    total_flops += neck_flops
    
    # 3. SemanticInjector (检查是否真正启用)
    semantic_injector_enabled = (
        hasattr(model, 'semantic_injector') and 
        model.semantic_injector is not None and
        sum(p.numel() for p in model.semantic_injector.parameters()) > 0
    )
    if semantic_injector_enabled:
        # seg_head: Conv2d(256, 256, 3) + Conv2d(256, 17, 1)
        # fusion_layer: Conv2d(273, 256, 1)
        fH, fW = H // 16, W // 16  # 特征图尺寸
        sem_flops = (
            256 * 256 * 3 * 3 * fH * fW * N_views +  # seg_head conv1
            256 * 17 * 1 * 1 * fH * fW * N_views +   # seg_head conv2
            273 * 256 * 1 * 1 * fH * fW * N_views    # fusion_layer
        ) * 2  # MAC to FLOPs
        flops_breakdown['semantic_injector ✓'] = sem_flops
        total_flops += sem_flops
    
    # 4. View Transformer (LSS): ~2-3 GFLOPs
    vt_flops = 2.5e9
    
    # 检查是否有 SGDM (Semantic Gating) - 通过检查模型属性
    sgdm_enabled = False
    if hasattr(model, 'img_view_transformer'):
        vt = model.img_view_transformer
        if hasattr(vt, 'depthnet') and hasattr(vt.depthnet, 'use_semantic_gating'):
            sgdm_enabled = vt.depthnet.use_semantic_gating
            if sgdm_enabled:
                fH, fW = H // 16, W // 16
                sgdm_flops = (
                    17 * 256 * 1 * 1 * fH * fW * N_views +  # sem_proj
                    256 * 2 * 64 * fH * fW * N_views +      # SE-block
                    256 * fH * fW * N_views                  # gating multiply
                ) * 2
                vt_flops += sgdm_flops
                flops_breakdown['SGDM ✓'] = sgdm_flops
    
    flops_breakdown['img_view_transformer'] = vt_flops
    total_flops += vt_flops
    
    # 5. BEV Encoder: ~5 GFLOPs
    bev_encoder_flops = 5e9
    flops_breakdown['img_bev_encoder'] = bev_encoder_flops
    total_flops += bev_encoder_flops
    
    # 6. Language Self-Gating (检查是否真正启用)
    # 通过配置和模型属性双重检查
    lsg_enabled = (
        hasattr(model, 'language_self_gating') and 
        model.language_self_gating is not None and
        cfg.model.get('use_language_self_gating', False) == True
    )
    if lsg_enabled:
        # projector: Conv3d(256, 128, 1)
        D, Hb, Wb = 16, 200, 200  # BEV 特征尺寸
        lsg_flops = (
            256 * 128 * 1 * 1 * 1 * D * Hb * Wb +  # projector
            128 * 6 * D * Hb * Wb +                 # 相似度计算
            D * Hb * Wb                             # sigmoid
        ) * 2
        flops_breakdown['language_self_gating ✓'] = lsg_flops
        total_flops += lsg_flops
    
    # 7. OCC Head: ~1 GFLOPs
    occ_head_flops = 1e9
    flops_breakdown['occ_head'] = occ_head_flops
    total_flops += occ_head_flops
    
    return total_flops, flops_breakdown


def main():
    args = parse_args()
    
    print("=" * 70)
    print("模型参数量与FLOPs统计")
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
    
    # FLOPs 计算
    print("\n" + "=" * 70)
    print("FLOPs 计算")
    print("=" * 70)
    
    flops, _ = count_flops(model, cfg)
    if flops is not None:
        print(f"\n  Backbone FLOPs: {flops/1e9:.2f} GFLOPs")
    
    # 手动估算总 FLOPs (传入 model)
    estimated_flops, flops_breakdown = estimate_flops_manual(model, cfg)
    
    print(f"\n  各模块 FLOPs 分解:")
    print("-" * 70)
    for name, f in sorted(flops_breakdown.items(), key=lambda x: -x[1]):
        pct = f / estimated_flops * 100 if estimated_flops > 0 else 0
        print(f"    {name:<30} {f/1e9:>8.2f} GFLOPs  ({pct:>5.1f}%)")
    
    print(f"\n  估算总 FLOPs:  {estimated_flops/1e9:.2f} GFLOPs")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n  ✓ 模型总参数量: {total_params/1e6:.2f} M")
    print(f"  ✓ 估算总 FLOPs: {estimated_flops/1e9:.2f} GFLOPs")
    print("=" * 70)


if __name__ == '__main__':
    main()
