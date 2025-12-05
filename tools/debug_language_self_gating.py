#!/usr/bin/env python3
"""
调试脚本：验证 LanguageSelfGating 在 CPU 上的前向和反向传播
运行：
    python tools/debug_language_self_gating.py
"""

import os
import sys

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from projects.mmdet3d_plugin.models.model_utils.language_self_gating import LanguageSelfGating


def main():
    device = torch.device('cpu')
    B, C, D, H, W = 2, 256, 8, 6, 6
    model = LanguageSelfGating(in_channels=C, proj_channels=128, num_anchors=6, grid_D=D)
    model.to(device)

    x = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    gated, gate_map = model(x)
    print('gated shape:', gated.shape)
    print('gate_map shape:', gate_map.shape)

    loss = gated.mean() + gate_map.mean() * 0.1
    loss.backward()

    # 检查梯度
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            print(f'param {name} grad sum: {p.grad.abs().sum().item():.6f}')

    if has_grad:
        print('✅ gradients exist')
    else:
        print('❌ no gradients')


if __name__ == '__main__':
    main()
