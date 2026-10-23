"""Cam4DOcc 官方 inflated-GMO 未来 GT 生成（用 OpenOccupancy infos 的 gt_boxes，无需 devkit 标注）。

变换链（与 Cam4DOcc 官方语义一致：所有未来帧 GT 落在 present 帧的 lidar 系）：
    box(lidar_k) --corners--> ego_k = R_l2e @ p + t_l2e
                --> global = R_e2g_k @ p + t_e2g_k
                --> ego_present = R_e2g_p^T @ (p - t_e2g_p)
                --> lidar_present = R_l2e_p^T @ (p - t_l2e_p)
  8 角点 -> 体素索引（0.2m, pc_range [-51.2,-51.2,-5, 51.2,51.2,3]）-> AABB 填充（inflated）

GMO 类别（Cam4DOcc V1.2 的 8 类）：bicycle / bus / car / construction / motorcycle / trailer / truck / pedestrian
（traffic_cone、barrier 不算）

用法：
    python gen_gmo_gt.py --validate 3
    python gen_gmo_gt.py --limit 100 --start 0
"""
import argparse
import os
import pickle

import numpy as np
from pyquaternion import Quaternion

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
DIM = [512, 512, 40]
RES = np.array([0.2, 0.2, 0.2], dtype=np.float64)
START = np.array(PC_RANGE[:3], dtype=np.float64)

GMO_NAMES = {'bicycle', 'bus', 'car', 'construction', 'motorcycle',
             'trailer', 'truck', 'pedestrian'}

# 我们的 occ 类 ID（CLASS_NAMES）：1..16 语义，17=free
# GMO 中我们有的 8 类：bicycle(2) bus(3) car(4) construction(5) motorcycle(6)
#                       pedestrian(7) trailer(9) truck(10)
OUR_GMO_IDS = [2, 3, 4, 5, 6, 7, 9, 10]


def _quat_from(rec, prefix):
    return Quaternion([rec[prefix + '_rotation'][3], rec[prefix + '_rotation'][0],
                       rec[prefix + '_rotation'][1], rec[prefix + '_rotation'][2]])


def box_corners_in_present_lidar(info_k, info_p):
    """把 info_k 的所有 GMO box 角点变换到 info_p 的 lidar 坐标系。"""
    # frame k: lidar2ego, ego2global
    t_l2e_k = np.array(info_k['lidar2ego_translation'], dtype=np.float64)
    R_l2e_k = _quat_from(info_k, 'lidar2ego').rotation_matrix
    t_e2g_k = np.array(info_k['ego2global_translation'], dtype=np.float64)
    R_e2g_k = _quat_from(info_k, 'ego2global').rotation_matrix
    # present
    t_l2e_p = np.array(info_p['lidar2ego_translation'], dtype=np.float64)
    R_l2e_p = _quat_from(info_p, 'lidar2ego').rotation_matrix
    t_e2g_p = np.array(info_p['ego2global_translation'], dtype=np.float64)
    R_e2g_p = _quat_from(info_p, 'ego2global').rotation_matrix

    boxes = np.asarray(info_k['gt_boxes'], dtype=np.float64)
    names = np.asarray(info_k['gt_names'])
    out = []
    for box, name in zip(boxes, names):
        if name not in GMO_NAMES:
            continue
        x, y, z, dx, dy, dz, yaw = box
        # 8 corners in lidar_k
        c = np.array([[dx / 2, dy / 2, dz / 2], [dx / 2, dy / 2, -dz / 2],
                      [dx / 2, -dy / 2, dz / 2], [dx / 2, -dy / 2, -dz / 2],
                      [-dx / 2, dy / 2, dz / 2], [-dx / 2, dy / 2, -dz / 2],
                      [-dx / 2, -dy / 2, dz / 2], [-dx / 2, -dy / 2, -dz / 2]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        corners_local = (Rz @ c.T).T + np.array([x, y, z])
        # lidar_k -> ego_k -> global
        p = (R_l2e_k @ corners_local.T).T + t_l2e_k
        p = (R_e2g_k @ p.T).T + t_e2g_k
        # global -> ego_present -> lidar_present
        p = (R_e2g_p.T @ (p - t_e2g_p).T).T
        p = (R_l2e_p.T @ (p - t_l2e_p).T).T
        out.append(p)
    return out


def voxelize(corner_list):
    """AABB 填充 -> 稀疏占用体素坐标 [N,3] int16。"""
    vox = set()
    for c in corner_list:
        if not (PC_RANGE[0] <= c[:, 0].min() and c[:, 0].max() <= PC_RANGE[3]
                and PC_RANGE[1] <= c[:, 1].min() and c[:, 1].max() <= PC_RANGE[4]
                and PC_RANGE[2] <= c[:, 2].min() and c[:, 2].max() <= PC_RANGE[5]):
            continue
        idx = np.round((c - START + RES / 2.0) / RES).astype(np.int64)
        lo, hi = idx.min(0), idx.max(0)
        lo = np.clip(lo, 0, np.array(DIM) - 1)
        hi = np.clip(hi, 0, np.array(DIM) - 1)
        xs = np.arange(lo[0], hi[0] + 1)
        ys = np.arange(lo[1], hi[1] + 1)
        zs = np.arange(lo[2], hi[2] + 1)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)
        vox.update(map(tuple, pts.tolist()))
    return np.array(sorted(vox), dtype=np.int16) if vox else np.zeros((0, 3), np.int16)


def build_sequence(infos, idx):
    """连续 7 帧（present = idx+2），返回 (present_token, [7 x sparse voxels])。"""
    seq = infos[idx:idx + 7]
    if len(seq) < 7 or len({x['scene_token'] for x in seq}) != 1:
        return None
    info_p = seq[2]
    steps = [voxelize(box_corners_in_present_lidar(x, info_p)) for x in seq]
    return info_p['token'], steps


def validate(infos, gts_root, n=3):
    print('[validate] Cam4DOcc GT vs 我们的 labels.npz（present 帧，GMO 二值）')
    done = 0
    for idx in range(len(infos) - 7):
        res = build_sequence(infos, idx)
        if res is None:
            continue
        tok, steps = res
        scene = infos[idx]['scene_token']
        path = os.path.join(gts_root, scene, tok, 'labels.npz')
        if not os.path.exists(path):
            continue
        our = np.load(path)['semantics']
        our_gmo = np.isin(our, OUR_GMO_IDS)
        gmo = np.zeros(DIM, dtype=bool)
        v = steps[2]
        if len(v):
            gmo[v[:, 0], v[:, 1], v[:, 2]] = True
        inter = (gmo & our_gmo).sum()
        union = (gmo | our_gmo).sum()
        iou = inter / max(union, 1)
        print(f'  [{tok[:10]}] 直接比对 IoU={iou:.3f} '
              f'(cam4docc={gmo.sum()} ours={our_gmo.sum()})')
        # z 与 x 平移扫描，判断是否有系统性偏移
        best = (iou, 0, 0)
        for dz in (-12, -10, -9, -8, 0, 8, 9, 10, 12):
            for dx in (-6, -5, -4, 0, 4, 5, 6):
                g2 = np.roll(np.roll(gmo, dz, axis=2), dx, axis=0)
                i2 = (g2 & our_gmo).sum()
                u2 = (g2 | our_gmo).sum()
                s = i2 / max(u2, 1)
                if s > best[0]:
                    best = (s, dz, dx)
        print(f'      最佳平移 dz={best[1]} dx={best[2]} -> IoU={best[0]:.3f}')
        done += 1
        if done >= n:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--out', type=str,
                    default='/root/autodl-tmp/dataforpriorocc/cam4docc_data/gmo_gt')
    ap.add_argument('--infos', type=str,
                    default='/root/autodl-tmp/dataforpriorocc/cam4docc_data/nuscenes_occ_infos_val.pkl')
    ap.add_argument('--gts-root', type=str,
                    default='/root/autodl-tmp/dataforpriorocc/nuscenes/gts')
    ap.add_argument('--validate', type=int, default=0)
    args = ap.parse_args()

    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)['infos']
    print(f'[gen] infos={len(infos)}')

    if args.validate:
        validate(infos, args.gts_root, args.validate)
        return

    os.makedirs(args.out, exist_ok=True)
    done = 0
    for idx in range(args.start, min(len(infos) - 7, args.start + args.limit)):
        res = build_sequence(infos, idx)
        if res is None:
            continue
        tok, steps = res
        out_path = os.path.join(args.out, tok + '.npz')
        if os.path.exists(out_path):
            done += 1
            continue
        np.savez_compressed(out_path, **{f'step{k}': steps[k] for k in range(7)})
        done += 1
        if done % 50 == 0:
            print(f'[gen] {done} sequences', flush=True)
    print(f'[gen] DONE {done} -> {args.out}')


if __name__ == '__main__':
    main()
