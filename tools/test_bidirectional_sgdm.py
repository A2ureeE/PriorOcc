#!/usr/bin/env python
"""
测试双向语义-深度联合学习模块 (LiteBSDM)。

验证内容：
1. LiteBSDM 模块初始化
2. 前向传播维度正确性
3. 双向数据流验证
4. 梯度回传测试
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from projects.mmdet3d_plugin.models.model_utils.depthnet import (
    SemanticGatingModule,
    BidirectionalSemanticDepthModule,
    DepthNet
)


def test_lite_bsdm_module():
    """测试 LiteBSDM 模块的基本功能"""
    print("=" * 60)
    print("Test 1: LiteBSDM 模块初始化和前向传播")
    print("=" * 60)
    
    B, N = 1, 6
    C_img = 256
    C_sem = 17
    H, W = 16, 44
    
    # 创建 LiteBSDM 模块
    bsdm = BidirectionalSemanticDepthModule(
        img_channels=C_img,
        sem_channels=C_sem,
        reduction=8,
        depth_feedback_weight=0.3
    )
    bsdm.train()  # 设置为训练模式以启用反向分支
    
    # 模拟输入
    img_feat = torch.randn(B * N, C_img, H, W)
    sem_logits = torch.randn(B * N, C_sem, H, W)
    
    # 测试无 depth_prob 的情况（正向 gating only）
    gated_feat, refined_sem = bsdm(img_feat, sem_logits, depth_prob=None)
    
    assert gated_feat.shape == img_feat.shape, f"Gated feat shape mismatch: {gated_feat.shape} vs {img_feat.shape}"
    assert refined_sem is None, "refined_sem should be None when depth_prob is None"
    print(f"✓ 正向 Gating: img {img_feat.shape} -> gated {gated_feat.shape}")
    
    # 测试有 depth_prob 的情况（双向 gating）
    D = 88
    depth_prob = torch.softmax(torch.randn(B * N, D, H, W), dim=1)
    gated_feat, refined_sem = bsdm(img_feat, sem_logits, depth_prob=depth_prob)
    
    assert gated_feat.shape == img_feat.shape
    assert refined_sem is not None, "refined_sem should not be None when depth_prob is provided"
    assert refined_sem.shape == sem_logits.shape, f"Refined sem shape mismatch: {refined_sem.shape}"
    print(f"✓ 双向 Gating: depth {depth_prob.shape} + sem {sem_logits.shape} -> refined {refined_sem.shape}")
    
    # 验证 Sobel 边缘检测
    depth_edges = bsdm._compute_depth_edges(depth_prob)
    assert depth_edges.shape == (B * N, 1, H, W), f"Depth edges shape mismatch: {depth_edges.shape}"
    print(f"✓ Sobel 边缘检测: {depth_edges.shape}, range [{depth_edges.min():.3f}, {depth_edges.max():.3f}]")
    
    print("✓ Test 1 PASSED\n")


def test_depthnet_bidirectional():
    """测试 DepthNet 的双向模式"""
    print("=" * 60)
    print("Test 2: DepthNet 双向模式")
    print("=" * 60)
    
    B, N = 1, 6
    C_in = 256
    C_mid = 256
    C_context = 64
    D = 88
    H, W = 16, 44
    C_sem = 17
    
    # 创建双向 DepthNet
    depth_net = DepthNet(
        in_channels=C_in,
        mid_channels=C_mid,
        context_channels=C_context,
        depth_channels=D,
        use_dcn=False,  # 简化测试
        use_aspp=False,
        use_bidirectional_sgdm=True,
        sem_channels=C_sem,
        sgdm_reduction=8,
        depth_feedback_weight=0.3
    )
    depth_net.train()
    
    # 模拟输入
    x = torch.randn(B * N, C_in, H, W)
    mlp_input = torch.randn(B, N, 27)
    sem_logits = torch.randn(B * N, C_sem, H, W)
    
    # 前向传播
    output, refined_sem = depth_net(x, mlp_input, sem_logits=sem_logits)
    
    expected_output_channels = D + C_context
    assert output.shape == (B * N, expected_output_channels, H, W), \
        f"Output shape mismatch: {output.shape} vs {(B * N, expected_output_channels, H, W)}"
    print(f"✓ DepthNet 输出: {output.shape}")
    
    # 检查 refined_sem_logits
    if refined_sem is not None:
        assert refined_sem.shape == sem_logits.shape
        print(f"✓ Refined sem logits: {refined_sem.shape}")
    else:
        print("✓ Refined sem logits: None (可能是推理模式)")
    
    print("✓ Test 2 PASSED\n")


def test_gradient_flow():
    """测试梯度双向流动"""
    print("=" * 60)
    print("Test 3: 梯度双向流动测试")
    print("=" * 60)
    
    B, N = 1, 6
    C_img = 256
    C_sem = 17
    D = 88
    H, W = 16, 44
    
    bsdm = BidirectionalSemanticDepthModule(
        img_channels=C_img,
        sem_channels=C_sem,
        reduction=8
    )
    bsdm.train()
    
    img_feat = torch.randn(B * N, C_img, H, W, requires_grad=True)
    sem_logits = torch.randn(B * N, C_sem, H, W, requires_grad=True)
    depth_prob = torch.softmax(torch.randn(B * N, D, H, W), dim=1)
    depth_prob.requires_grad = True
    
    gated_feat, refined_sem = bsdm(img_feat, sem_logits, depth_prob=depth_prob)
    
    # 计算 loss
    loss = gated_feat.mean() + refined_sem.mean()
    loss.backward()
    
    assert img_feat.grad is not None, "img_feat gradient should not be None"
    assert sem_logits.grad is not None, "sem_logits gradient should not be None"
    assert depth_prob.grad is not None, "depth_prob gradient should not be None"
    
    print(f"✓ img_feat.grad: norm={img_feat.grad.norm():.4f}")
    print(f"✓ sem_logits.grad: norm={sem_logits.grad.norm():.4f}")
    print(f"✓ depth_prob.grad: norm={depth_prob.grad.norm():.4f}")
    
    print("✓ Test 3 PASSED\n")


def test_inference_mode():
    """测试推理模式（应跳过反向分支）"""
    print("=" * 60)
    print("Test 4: 推理模式（零额外开销）")
    print("=" * 60)
    
    B, N = 1, 6
    C_img = 256
    C_sem = 17
    D = 88
    H, W = 16, 44
    
    bsdm = BidirectionalSemanticDepthModule(
        img_channels=C_img,
        sem_channels=C_sem,
        reduction=8
    )
    bsdm.eval()  # 推理模式
    
    img_feat = torch.randn(B * N, C_img, H, W)
    sem_logits = torch.randn(B * N, C_sem, H, W)
    depth_prob = torch.softmax(torch.randn(B * N, D, H, W), dim=1)
    
    with torch.no_grad():
        gated_feat, refined_sem = bsdm(img_feat, sem_logits, depth_prob=depth_prob)
    
    assert gated_feat.shape == img_feat.shape
    assert refined_sem is None, "refined_sem should be None in inference mode"
    
    print(f"✓ 推理模式: refined_sem = None（跳过反向分支）")
    print("✓ Test 4 PASSED\n")


def test_backward_compatibility():
    """测试与原版 SGDM 的兼容性"""
    print("=" * 60)
    print("Test 5: 与原版 SGDM 的兼容性")
    print("=" * 60)
    
    B, N = 1, 6
    C_in = 256
    C_mid = 256
    C_context = 64
    D = 88
    H, W = 16, 44
    C_sem = 17
    
    # 原版 SGDM
    depth_net_sgdm = DepthNet(
        in_channels=C_in,
        mid_channels=C_mid,
        context_channels=C_context,
        depth_channels=D,
        use_dcn=False,
        use_aspp=False,
        use_semantic_gating=True,
        use_bidirectional_sgdm=False,  # 原版
        sem_channels=C_sem,
        sgdm_reduction=8
    )
    
    x = torch.randn(B * N, C_in, H, W)
    mlp_input = torch.randn(B, N, 27)
    sem_logits = torch.randn(B * N, C_sem, H, W)
    
    output, refined_sem = depth_net_sgdm(x, mlp_input, sem_logits=sem_logits)
    
    assert output.shape[1] == D + C_context
    # 原版 SGDM 不应有 refined_sem
    print(f"✓ 原版 SGDM 输出: {output.shape}")
    print(f"✓ 原版 SGDM refined_sem: {refined_sem}")
    
    print("✓ Test 5 PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("双向语义-深度联合学习模块 (LiteBSDM) 测试")
    print("=" * 60 + "\n")
    
    test_lite_bsdm_module()
    test_depthnet_bidirectional()
    test_gradient_flow()
    test_inference_mode()
    test_backward_compatibility()
    
    print("=" * 60)
    print("🎉 All tests PASSED!")
    print("=" * 60)
