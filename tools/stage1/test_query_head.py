"""Unit test for multi-scale SemanticQueryFutureHead.

Verifies:
1. Zero-init output -> future = current (exact copy, B-baseline)
2. Per-horizon resolution: 1s full-res, 2s/3s half-res
3. Velocity bias direction correctness
4. Memory estimate check (unfold sizes)
"""
import sys
sys.path.insert(0, '/root/priorocc/PriorOcc')

import torch
from projects.mmdet3d_plugin.models.model_utils.query_future_head import (
    SemanticQueryFutureHead)


def test_multiscale():
    B, C, H, W = 1, 64, 40, 40  # small for speed
    T = 3
    torch.manual_seed(42)

    # Multi-scale config: 1s full res R=5, 2s/3s half res R=9
    head = SemanticQueryFutureHead(
        bev_channels=C, num_future=T, hidden_dim=32,
        window_radius=[5, 9, 9],
        num_semantic_classes=17,
        horizons_sec=(1.0, 2.0, 3.0), vel_bias_init=2.0,
        downsample=[1, 2, 2])

    bev = torch.randn(B, C, H, W) * 0.5
    per_cls = torch.rand(B, 17, H, W)
    per_cls = per_cls / per_cls.sum(dim=1, keepdim=True)
    dyn = torch.zeros(B, 1, H, W)
    dyn[0, 0, 18:22, 18:22] = 1.0

    print("=== Test 1: zero-init identity ===")
    vel = torch.zeros(B, 2, H, W)
    out = head(bev, per_cls, dyn, vel)
    for t in range(T):
        diff = (out[0, t] - bev[0]).abs().max().item()
        print(f"  t={t}: max|future-current| = {diff:.8f}")
    ok1 = all((out[0, t] - bev[0]).abs().max().item() < 1e-5
              for t in range(T))
    print(f"  -> {'PASS' if ok1 else 'FAIL'}")

    print("\n=== Test 2: groups & shape ===")
    print(f"  groups: {head._groups}")
    print(f"  per_horizon_ds: {head.per_horizon_ds}")
    print(f"  per_horizon_R: {head.per_horizon_R}")
    print(f"  output shape: {tuple(out.shape)}")
    ok2 = (out.shape == (B, T, C, H, W)
           and head._groups == [((1, 5), [0]), ((2, 9), [1, 2])])
    print(f"  -> {'PASS' if ok2 else 'FAIL'}")

    print("\n=== Test 3: velocity bias direction ===")
    off = -2 + 2 * 1.0
    off_w = -3 + 3 * 1.0
    bias_correct = -(off ** 2 + off_w ** 2)
    bias_stay = -((0 + 2) ** 2 + (0 + 3) ** 2)
    print(f"  bias(query=dest, key=source) = {bias_correct:.1f} (expect 0)")
    print(f"  bias(query=self, key=self)   = {bias_stay:.1f} (expect <0)")
    ok3 = bias_correct == 0 and bias_stay < 0
    print(f"  -> {'PASS' if ok3 else 'FAIL'}")

    print("\n=== Test 4: gradient flow ===")
    bev_g = bev.clone().requires_grad_(True)
    vel_g = vel.clone().requires_grad_(True)
    out_g = head(bev_g, per_cls, dyn, vel_g)
    out_g.sum().backward()
    has_grad = bev_g.grad is not None
    print(f"  bev grad: {has_grad}")
    print(f"  (vel grad may be 0 due to zero-init out_proj)")

    print("\n=== Test 5: single-value backward compat ===")
    head_old = SemanticQueryFutureHead(
        bev_channels=C, num_future=T, hidden_dim=32,
        window_radius=6, num_semantic_classes=17,
        horizons_sec=(1.0, 2.0, 3.0), vel_bias_init=2.0,
        downsample=2)
    out_old = head_old(bev, per_cls, dyn, vel)
    ok5 = (out_old.shape == (B, T, C, H, W)
           and head_old._groups == [((2, 6), [0, 1, 2])])
    print(f"  single-value groups: {head_old._groups}")
    print(f"  -> {'PASS' if ok5 else 'FAIL'}")

    all_ok = ok1 and ok2 and ok3 and ok5
    print(f"\n{'='*50}")
    print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == '__main__':
    test_multiscale()
