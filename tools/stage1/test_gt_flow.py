#!/usr/bin/env python
"""GT flow 生成核心函数 + 合成测试（锁死 单位/符号/轴/cap）。

真实数据帧系(velocity global vs ego)由 --real 探针另行验证。
flow 约定与 warp_feature 对齐：
  flow[...,0]=dx (世界 x, 沿 Dx/W 轴)，flow[...,1]=dy (世界 y, 沿 Dy/H 轴)
  backward warp: output(Q)=input(Q-flow) → flow = 物体位移(正速度=正flow)
"""
import numpy as np
import torch
import torch.nn.functional as F

PC = [-40.0, 40.0]      # 世界 x/y 物理范围 (grid_config)
CELL = 0.4             # m/cell
D = 200               # Dy=Dx=200


def world_to_cell(v):
    return (v - PC[0]) / CELL


def build_gt_flow(boxes9, dt, cap_cells):
    """boxes9: (N,9) [x,y,z,dx,dy,dz,yaw,vx,vy]  已含 m/s 速度。
    返回 flow (2,H,W) cell 单位, mask (1,H,W) 动态覆盖=1。"""
    flow = np.zeros((2, D, D), dtype=np.float32)
    mask = np.zeros((1, D, D), dtype=np.float32)
    for b in boxes9:
        x, y, z, lx, ly, lz, yaw, vx, vy = b
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            continue
        disp_x = np.clip(vx * dt / CELL, -cap_cells, cap_cells)
        disp_y = np.clip(vy * dt / CELL, -cap_cells, cap_cells)
        # 目标位置约定: backward warp output(Q)=input(Q-flow(Q)) →
        # flow 要放在物体未来的格 (源心 + 位移)，值=位移。
        cx = world_to_cell(x) + disp_x
        cy = world_to_cell(y) + disp_y
        hx, hy = (lx / CELL) / 2, (ly / CELL) / 2
        # 旋转矩形采样填充（粗采样）
        cos_s, sin_s = np.cos(yaw), np.sin(yaw)
        for yy in range(D):
            for xx in range(D):
                wx, wy = xx - cx, yy - cy
                # 反旋转到框局部系
                lxr = cos_s * wx + sin_s * wy
                lyr = -sin_s * wx + cos_s * wy
                if abs(lxr) <= hx and abs(lyr) <= hy:
                    flow[0, yy, xx] = disp_x
                    flow[1, yy, xx] = disp_y
                    mask[0, yy, xx] = 1.0
    return flow, mask


def test_synthetic():
    ok = True
    exp = 5.0 / CELL  # 12.5 cells
    # 1) 正 x 速度 → flow ch0 值=位移, 峰值格在中心右侧(未来)
    box = np.array([[0, 0, 0, 2, 4, 1.5, 0, 5.0, 0.0]])  # vx=5 m/s
    f, m = build_gt_flow(box, dt=1.0, cap_cells=36)
    cc = int(round(world_to_cell(0)))
    peak_col = np.unravel_index(f[0].argmax(), f[0].shape)[1]
    ok &= abs(f[0].max() - exp) < 1e-3 and abs(f[1].max()) < 1e-3 and peak_col > cc
    print(f"[1] +x: max dx={f[0].max():.2f} (exp {exp:.2f}), peak col={peak_col} (> {cc})")
    # 2) 正 y 速度 → flow ch1 值=位移, 峰值格在中心下方(未来)
    box = np.array([[0, 0, 0, 2, 4, 1.5, 0, 0.0, 5.0]])
    f, _ = build_gt_flow(box, dt=1.0, cap_cells=36)
    peak_row = np.unravel_index(f[1].argmax(), f[1].shape)[0]
    ok &= abs(f[1].max() - exp) < 1e-3 and abs(f[0].max()) < 1e-3 and peak_row > cc
    print(f"[2] +y: max dy={f[1].max():.2f} (exp {exp:.2f}), peak row={peak_row} (> {cc})")
    # 3) cap 裁剪：v=20,dt=3 → 60m=150cell 应裁到 36
    box = np.array([[0, 0, 0, 2, 4, 1.5, 0, 20.0, 0.0]])
    f, _ = build_gt_flow(box, dt=3.0, cap_cells=36)
    ok &= abs(f[0].max() - 36.0) < 1e-3
    print(f"[3] clip: max dx={f[0].max():.2f} (exp cap 36)")
    # 4) 静态(v=0) 不写 mask
    box = np.array([[0, 0, 0, 2, 4, 1.5, 0, 0.0, 0.0]])
    _, m = build_gt_flow(box, dt=1.0, cap_cells=36)
    ok &= m.sum() == 0
    print(f"[4] static mask sum={m.sum():.1f} (exp 0)")
    # 5) 端到端: 用 flow warp 当前 blob → 应落到 +x 位移处（backward warp）
    bev = np.zeros((1, 1, D, D), np.float32)
    bev[0, 0, cc, cc-2:cc+3] = 1.0          # x 方向一小条
    box = np.array([[0, 0, 0, 6, 4, 1.5, 0, 5.0, 0.0]])  # vx=5 → +12.5cell
    f, _ = build_gt_flow(box, dt=1.0, cap_cells=36)
    flow_t = torch.from_numpy(f).unsqueeze(0)  # (1,2,H,W)
    # backward warp: sample_grid = base - flow_norm
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, D), torch.linspace(-1, 1, D), indexing='ij')
    base = torch.stack([xs, ys], -1).unsqueeze(0)
    fn = torch.empty_like(flow_t)
    fn[:, 0] = flow_t[:, 0] * (2.0 / (D - 1))
    fn[:, 1] = flow_t[:, 1] * (2.0 / (D - 1))
    grid = base - fn.permute(0, 2, 3, 1)
    warped = F.grid_sample(torch.from_numpy(bev), grid, align_corners=True)
    # flow 只盖未来区, 源区恒等留残影; 正确判据=未来区(cc+6..cc+30)出现质量
    future_mass = float(warped[0, 0, cc, cc + 6:cc + 30].sum())
    ok &= future_mass > 0
    print(f"[5] warp future-region mass={future_mass:.1f} (>0 exp)")
    print("SYNTHETIC:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_synthetic() else 1)
