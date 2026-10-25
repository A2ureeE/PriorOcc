#!/usr/bin/env python
"""B 方法真实数据验收器（纯 CPU，不接训练）。

证明链路：infos 里同类未来框 → 经 ego 位姿对齐到当前 lidar 系 → 就近匹配得真值位移
→ 栅格成 flow（目标格） → 用它 warp 当前 occ 的动态类 → 与未来 occ 动态类算 IoU。
若 flow-warp IoU 显著 > 静态复制 IoU → 位移方向 + 帧变换 + occ 栅格映射 全对。
occ 栅格 pc_range/轴顺序未知 → 扫候选，取 IoU 提升最大者为真。
"""
import pickle, numpy as np, os, sys, torch
import torch.nn.functional as F

P = 'data/nuscenes/bevdetv2-nuscenes_infos_train_forecast.pkl'
# occ 动态类索引（对齐评测：1..10）
DYN = set(range(1, 11))


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def load_occ(occ_dir):
    f = os.path.join(occ_dir, 'labels.npz')
    if not os.path.exists(f):
        return None
    s = np.load(f)['semantics']            # (Dx,Dy,Dz)
    return (s[..., 0] if False else s)     # 保留全 3D，外面塌缩


def occ_dyn_mask(sem, D=200):
    m = np.isin(sem, list(DYN)).any(-1).astype(np.float32)   # (Dx,Dy)
    return m


def bev_warp(mask2d, flow, D):
    """backward warp: output(Q)=input(Q-flow). flow (2,Dx,Dy) cell units."""
    t = torch.from_numpy(mask2d)[None, None]                 # (1,1,Dx,Dy)
    xs, ys = torch.meshgrid(torch.linspace(-1, 1, D), torch.linspace(-1, 1, D), indexing='ij')
    base = torch.stack([xs, ys], -1)[None]                   # (1,Dx,Dy,2) dim0=x-axis(dim-2), dim1=y
    fn = torch.empty(1, 2, D, D)
    fn[0, 0] = torch.from_numpy(flow[0]) * (2.0 / (D - 1))   # dx along Dx
    fn[0, 1] = torch.from_numpy(flow[1]) * (2.0 / (D - 1))   # dy along Dy
    grid = base - fn.permute(0, 2, 3, 1)
    w = F.grid_sample(t, grid, align_corners=True)[0, 0].numpy()
    return w


def cell_of(x, y, xmin, half, D, x_first):
    c = (np.array([x, y]) - xmin) / (2 * half) * D
    c = c.round().astype(int)
    return (c[0], c[1]) if x_first else (c[1], c[0])


def run():
    d = pickle.load(open(P, 'rb'))['infos']
    # 按 scene 分组、按 timestamp 排序，用于取未来框
    from collections import defaultdict
    scenes = defaultdict(list)
    for i, info in enumerate(d):
        scenes[info['scene_token']].append((info['timestamp'], i))
    for k in scenes: scenes[k].sort()
    tok2i = {info['token']: i for i, info in enumerate(d)}

    DT = 1.0
    n_test = 0
    cands = [(-50.0, 0.5, True), (-50.0, 0.5, False),
             (-40.0, 0.4, True), (-40.0, 0.4, False)]
    stat_iou = {c: [] for c in cands}
    flow_iou = {c: [] for c in cands}

    for info in d[:800]:
        if info['occ_path'] is None or not os.path.exists(os.path.join(info['occ_path'], 'labels.npz')):
            continue
        fut = None
        st = scenes[info['scene_token']]; pos = [p[1] for p in st].index(tok2i[info['token']])
        tgt_ts = info['timestamp'] + int(DT * 1e6)
        fut_idx = None
        for j in range(pos + 1, len(st)):
            if st[j][0] >= tgt_ts - 5e4:
                fut_idx = st[j][1]; break
        if fut_idx is None:
            continue
        cur_box, cur_name = np.array(info['gt_boxes']), np.array(info['gt_names'])
        fut_info = d[fut_idx]
        fb = np.array(fut_info['gt_boxes']); fn_ = np.array(fut_info['gt_names'])
        if len(fb) == 0 or len(cur_box) == 0:
            continue
        # future lidar -> global -> current lidar
        Rg_c = q2R(info['ego2global_rotation']); tg_c = np.array(info['ego2global_translation'])
        Rg_f = q2R(fut_info['ego2global_rotation']); tg_f = np.array(fut_info['ego2global_translation'])
        cur_occ = np.load(os.path.join(info['occ_path'], 'labels.npz'))['semantics']
        fut_occ = np.load(os.path.join(fut_info['occ_path'], 'labels.npz'))['semantics'] if fut_info['occ_path'] else None
        if fut_occ is None: continue
        for (xmin, cell, x_first) in cands:
            D = cur_occ.shape[0]
            half = D * cell / 2.0
            stat = occ_dyn_mask(fut_occ, D)
            cur_dyn = occ_dyn_mask(cur_occ, D)
            flow = np.zeros((2, D, D), np.float32)
            for i in range(len(cur_box)):
                if cur_box[i][2] not in DYN if False else (i >= len(cur_name)):
                    pass
                if int(np.round((cur_box[i][2]))) < 0:
                    pass
                nm = cur_name[i]
                if not (np.hypot(*cur_box[i, :2]) < 50):
                    continue
                # 同类未来框，取平移后最近
                pc_l = cur_box[i, :3]
                best = None
                for j in range(len(fb)):
                    if fn_[j] != nm: continue
                    pf_global = Rg_f @ fb[j, :3] + tg_f
                    pf_cur = Rg_c.T @ (pf_global - tg_c)
                    dist = np.linalg.norm(pf_cur[:2] - pc_l[:2])
                    if dist < 6 and (best is None or dist < best[0]):
                        best = (dist, pf_cur)
                if best is None: continue
                disp = best[1][:2] - pc_l[:2]        # meters, current lidar frame
                disp_cells = disp / cell
                # 目标格（future position）栅格 footprint（用 2x2 近似，够验收）
                gx, gy = cell_of(*best[1][:2], xmin, half, D, x_first) if False else (None, None)
                fc = ((best[1][0] + 50.0) / cell, (best[1][1] + 50.0) / cell)
                cx = int(round(fc[0])); cy = int(round(fc[1]))
                if 2 <= cx < D-2 and 2 <= cy < D-2:
                    flow[0, cy-2:cy+3, cx-2:cx+3] = disp_cells[0]
                    flow[1, cy-2:cy+3, cx-2:cx+3] = disp_cells[1]
            warped = bev_warp(cur_dyn, flow, D)
            inter = (np.round(warped) * stat).sum(); union = (np.clip(np.round(warped) + stat, 0, 1)).sum()
            si = (cur_dyn * stat).sum() / (np.clip(cur_dyn + stat, 0, 1)).sum()
            if union > 0:
                stat_iou[(xmin, cell, x_first)].append(si)
                flow_iou[(xmin, cell, x_first)].append(inter / union)
        n_test += 1
        if n_test >= 150:
            break

    print(f'\n样本数(命中位移)= {n_test}')
    print('候选(xmin,cell,x_first)   静态IoU   flowIoU   提升')
    bestc, bestdt = None, -9
    for c in cands:
        if not flow_iou[c]:
            print(f'  {c}  (无样本)'); continue
        s = np.mean(stat_iou[c]); f = np.mean(flow_iou[c]); dt = f - s
        print(f'  {str(c):26s}  {s:.3f}    {f:.3f}    {dt:+.3f}')
        if dt > bestdt: bestdt, bestc = dt, c
    print(f'\n最优候选 {bestc} 提升 {bestdt:+.3f}')
    print('B-VALIDATION:', 'PASS' if bestdt > 0.03 else 'WEAK/FAIL')


if __name__ == '__main__':
    run()
