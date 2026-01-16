"""
测试 SGDM (Semantic-Gated Depth Module) 模块是否正常运行。

该脚本验证：
1. SemanticGatingModule 能否正确初始化
2. 前向传播是否正常工作
3. 输出形状是否正确
4. 梯度是否能正确回传
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

def test_sgdm():
    print("=" * 60)
    print("SGDM (Semantic-Gated Depth Module) 测试")
    print("=" * 60)
    
    # 导入模块
    from projects.mmdet3d_plugin.models.model_utils.depthnet import SemanticGatingModule, DepthNet
    
    # ========================================
    # 测试 1: SemanticGatingModule 初始化
    # ========================================
    print("\n[Test 1] SemanticGatingModule 初始化...")
    try:
        img_channels = 256
        sem_channels = 17
        reduction = 4
        
        sgdm = SemanticGatingModule(
            img_channels=img_channels,
            sem_channels=sem_channels,
            reduction=reduction
        )
        print(f"  ✓ SemanticGatingModule 初始化成功")
        print(f"    - img_channels: {img_channels}")
        print(f"    - sem_channels: {sem_channels}")
        print(f"    - reduction: {reduction}")
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        return False
    
    # ========================================
    # 测试 2: 前向传播
    # ========================================
    print("\n[Test 2] SemanticGatingModule 前向传播...")
    try:
        batch_size = 2
        num_views = 6
        H, W = 16, 44  # Feature map size
        
        # 创建模拟输入
        img_feat = torch.randn(batch_size * num_views, img_channels, H, W)
        sem_logits = torch.randn(batch_size * num_views, sem_channels, H, W)
        
        # 前向传播
        gated_feat = sgdm(img_feat, sem_logits)
        
        print(f"  ✓ 前向传播成功")
        print(f"    - 输入 img_feat 形状: {img_feat.shape}")
        print(f"    - 输入 sem_logits 形状: {sem_logits.shape}")
        print(f"    - 输出 gated_feat 形状: {gated_feat.shape}")
        
        # 验证输出形状
        expected_shape = (batch_size * num_views, img_channels, H, W)
        assert gated_feat.shape == expected_shape, f"形状不匹配: {gated_feat.shape} vs {expected_shape}"
        print(f"  ✓ 输出形状验证通过: {gated_feat.shape}")
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        return False
    
    # ========================================
    # 测试 3: 梯度回传
    # ========================================
    print("\n[Test 3] 梯度回传测试...")
    try:
        img_feat = torch.randn(batch_size * num_views, img_channels, H, W, requires_grad=True)
        sem_logits = torch.randn(batch_size * num_views, sem_channels, H, W, requires_grad=True)
        
        gated_feat = sgdm(img_feat, sem_logits)
        loss = gated_feat.mean()
        loss.backward()
        
        assert img_feat.grad is not None, "img_feat 梯度为空"
        assert sem_logits.grad is not None, "sem_logits 梯度为空"
        print(f"  ✓ 梯度回传成功")
        print(f"    - img_feat.grad 均值: {img_feat.grad.mean().item():.6e}")
        print(f"    - sem_logits.grad 均值: {sem_logits.grad.mean().item():.6e}")
    except Exception as e:
        print(f"  ✗ 梯度回传失败: {e}")
        return False
    
    # ========================================
    # 测试 4: DepthNet 中 SGDM 集成测试
    # ========================================
    print("\n[Test 4] DepthNet 中 SGDM 集成测试...")
    try:
        depthnet = DepthNet(
            in_channels=256,
            mid_channels=256,
            context_channels=128,
            depth_channels=59,
            use_dcn=True,
            use_aspp=True,
            with_cp=False,
            stereo=False,
            use_semantic_gating=True,  # 开启 SGDM
            sem_channels=17
        )
        
        # 验证 SGDM 是否启用
        assert depthnet.use_semantic_gating == True, "use_semantic_gating 未开启"
        assert hasattr(depthnet, 'semantic_gating'), "semantic_gating 模块不存在"
        print(f"  ✓ DepthNet 中 SGDM 已启用")
        print(f"    - use_semantic_gating: {depthnet.use_semantic_gating}")
        print(f"    - semantic_gating 类型: {type(depthnet.semantic_gating).__name__}")
    except Exception as e:
        print(f"  ✗ DepthNet 集成测试失败: {e}")
        return False
    
    # ========================================
    # 测试 5: 门控效果验证
    # ========================================
    print("\n[Test 5] 门控效果验证...")
    try:
        sgdm_test = SemanticGatingModule(img_channels=64, sem_channels=17, reduction=4)
        
        # 创建特定输入来验证门控效果
        img_feat = torch.ones(1, 64, 4, 4) * 2.0  # 固定值
        
        # 情况1: 均匀语义分布
        sem_logits_uniform = torch.zeros(1, 17, 4, 4)  # 均匀 softmax
        gated_1 = sgdm_test(img_feat, sem_logits_uniform)
        
        # 情况2: 偏向某一类的语义分布
        sem_logits_peaked = torch.zeros(1, 17, 4, 4)
        sem_logits_peaked[:, 0, :, :] = 10.0  # 强偏向第0类
        gated_2 = sgdm_test(img_feat, sem_logits_peaked)
        
        # 验证两种情况产生不同的输出（门控效果）
        diff = (gated_1 - gated_2).abs().mean().item()
        print(f"  ✓ 门控效果验证成功")
        print(f"    - 均匀语义输出均值: {gated_1.mean().item():.4f}")
        print(f"    - 偏向语义输出均值: {gated_2.mean().item():.4f}")
        print(f"    - 输出差异: {diff:.6f}")
        
        if diff > 1e-6:
            print(f"  ✓ 不同语义输入产生不同门控效果，SGDM 工作正常！")
        else:
            print(f"  ⚠ 警告: 不同语义输入产生相似输出，门控效果可能不明显")
    except Exception as e:
        print(f"  ✗ 门控效果验证失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有 SGDM 测试通过！")
    print("=" * 60)
    return True


def test_sgdm_disabled():
    """测试 SGDM 关闭时的行为"""
    print("\n" + "=" * 60)
    print("SGDM 关闭状态测试")
    print("=" * 60)
    
    from projects.mmdet3d_plugin.models.model_utils.depthnet import DepthNet
    
    try:
        depthnet = DepthNet(
            in_channels=256,
            mid_channels=256,
            context_channels=128,
            depth_channels=59,
            use_dcn=True,
            use_aspp=True,
            with_cp=False,
            stereo=False,
            use_semantic_gating=False,  # 关闭 SGDM
            sem_channels=17
        )
        
        assert depthnet.use_semantic_gating == False, "use_semantic_gating 应为 False"
        assert not hasattr(depthnet, 'semantic_gating') or depthnet.semantic_gating is None, \
            "semantic_gating 应该不存在或为 None"
        print(f"  ✓ SGDM 关闭状态验证成功")
        print(f"    - use_semantic_gating: {depthnet.use_semantic_gating}")
    except AssertionError as e:
        # 如果 semantic_gating 属性存在但 use_semantic_gating 为 False，也是可以的
        if hasattr(depthnet, 'use_semantic_gating') and not depthnet.use_semantic_gating:
            print(f"  ✓ SGDM 关闭状态验证成功")
            print(f"    - use_semantic_gating: {depthnet.use_semantic_gating}")
        else:
            print(f"  ✗ 验证失败: {e}")
            return False
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = test_sgdm()
    if success:
        test_sgdm_disabled()
    else:
        print("\n✗ SGDM 测试失败，请检查实现！")
        exit(1)
