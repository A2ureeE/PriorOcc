"""Synthetic unit test for PriorCorrespondenceFlow (PCF).

Verifies:
1. A blob moved by a known displacement → PCF measures the correct displacement
2. Static regions (dyn_mask=0) → zero flow addition
3. Zero-init residual → flow is pure measurement at initialization
4. Correct flow convention (dx=W, dy=H, positive=moves content forward)
"""
import sys
sys.path.insert(0, '/root/priorocc/PriorOcc')

import torch
from projects.mmdet3d_plugin.models.model_utils.prior_correspondence import (
    PriorCorrespondenceFlow)
from projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder import (
    warp_feature)


def test_pcf_displacement():
    B, C, H, W = 1, 64, 50, 50
    T = 3
    torch.manual_seed(42)

    pcf = PriorCorrespondenceFlow(
        gate_mode='none',
        bev_channels=C, num_future=T,
        search_radius=10, temperature=0.05,
        horizons_sec=(1.0, 2.0, 3.0),
        history_lag=1.0, history_index=2)

    # Different noise for each frame (so background doesn't create spurious
    # zero-displacement matches)
    noise = torch.randn(B, C, H, W) * 0.3  # SAME noise both frames (realistic static)
    noise_cur = noise.clone()

    # Moving blob: at (25, 25) in history → (30, 28) in current
    # Displacement: +5 H, +3 W over 1.0s → velocity (5, 3) cells/s
    bev_hist = noise.clone()
    bev_cur = noise_cur.clone()

    blob = torch.randn(C) * 3  # strong signal
    for dh in range(-2, 3):
        for dw in range(-2, 3):
            w = 1.0 / (1 + dh * dh + dw * dw)  # Gaussian-ish weight
            bev_hist[0, :, 25 + dh, 25 + dw] += blob * w
            bev_cur[0, :, 30 + dh, 28 + dw] += blob * w

    # Dynamic mask: 1 at blob's current position
    dyn_mask = torch.zeros(B, 1, H, W)
    dyn_mask[0, 0, 27:34, 25:32] = 1.0

    flow_add = pcf(bev_cur, bev_hist, dyn_mask)

    print("=== Test 1: blob displacement at center (30, 28) ===")
    exp = [(3, 5), (6, 10), (9, 15)]  # (dx, dy) per horizon
    ok = True
    for k in range(T):
        dx = flow_add[0, k, 0, 30, 28].item()
        dy = flow_add[0, k, 1, 30, 28].item()
        exp_dx, exp_dy = exp[k]
        print(f"  {k+1}s: dx={dx:+.2f} (expect ~+{exp_dx}), "
              f"dy={dy:+.2f} (expect ~+{exp_dy})")
        if abs(dx - exp_dx) > 1.5 or abs(dy - exp_dy) > 1.5:
            ok = False
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n=== Test 2: static region (dyn_mask=0) at (5, 5) ===")
    ok2 = True
    for k in range(T):
        dx = flow_add[0, k, 0, 5, 5].item()
        dy = flow_add[0, k, 1, 5, 5].item()
        print(f"  {k+1}s: dx={dx:.4f}, dy={dy:.4f} (expect 0)")
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            ok2 = False
    print(f"  -> {'PASS' if ok2 else 'FAIL'}")

    print(f"\n=== Test 3: zero-init residual ===")
    wn = pcf.residual.weight.norm().item()
    bn = pcf.residual.bias.norm().item()
    print(f"  weight norm = {wn:.6f} (expect 0)")
    print(f"  bias norm = {bn:.6f} (expect 0)")
    ok3 = wn == 0.0 and bn == 0.0
    print(f"  -> {'PASS' if ok3 else 'FAIL'}")

    print(f"\n=== Test 4: flow shape ===")
    print(f"  flow_add shape = {tuple(flow_add.shape)} "
          f"(expect ({B}, {T}, 2, {H}, {W}))")
    ok4 = flow_add.shape == (B, T, 2, H, W)
    print(f"  -> {'PASS' if ok4 else 'FAIL'}")

    print("\n=== Test 5: warp direction (blob moves +H,+W) ===")
    # Warp current BEV with 1s flow; blob should move from (30,28) toward (35,33)
    flow_1s = flow_add[0, 0]  # (2, H, W)
    warped = warp_feature(bev_cur, flow_1s.unsqueeze(0))
    # Find the blob center in the warped BEV (max activation position)
    blob_act = (warped[0] * blob.view(-1, 1, 1)).sum(dim=0)
    max_pos = blob_act.argmax()
    max_h, max_w = max_pos // W, max_pos % W
    print(f"  blob after warp: ({max_h.item()}, {max_w.item()}) "
          f"(expect ~(35, 31))")
    ok5 = abs(max_h.item() - 35) <= 2 and abs(max_w.item() - 31) <= 2
    print(f"  -> {'PASS' if ok5 else 'FAIL'}")

    all_ok = ok and ok2 and ok3 and ok4 and ok5
    print(f"\n{'='*50}")
    print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == '__main__':
    test_pcf_displacement()
