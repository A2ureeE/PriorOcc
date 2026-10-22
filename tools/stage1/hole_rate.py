"""Hole rate of continuous (background) classes in occupancy predictions.

A "hole" is a camera-visible voxel whose GT is a continuous class C, whose
prediction is NOT C, while at least ``--min-nb`` of its 6 voxel neighbours are
predicted as C. That is exactly the artifact a hole-filling (continuity)
regularizer is supposed to remove, so the hole rate answers "did the
regularizer fill holes?" independently of global mIoU drift.

Reported per class: hole_rate = holes / GT_C_voxels (percent).

Usage:
    python tools/stage1/hole_rate.py <ckpt> --cfg <config> --limit 600
"""
import argparse
import os
import sys
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import projects.mmdet3d_plugin  # noqa: F401,E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel, collate  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

CONT = {11: 'driveable_surface', 12: 'other_flat', 13: 'sidewalk',
        14: 'terrain', 15: 'manmade', 16: 'vegetation'}


def unwrap(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return np.asarray(x)


def neighbor_count(mask):
    """Count 6-neighbour agreement of a boolean 3D mask (no wraparound)."""
    nb = np.zeros(mask.shape, dtype=np.uint8)
    for axis in (0, 1, 2):
        nb[1:] += mask[:-1].astype(np.uint8)
        nb[:-1] += mask[1:].astype(np.uint8)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--limit', type=int, default=600)
    ap.add_argument('--min-nb', type=int, default=3,
                    help='neighbours (of 6) that must already be class C')
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    n_all = len(ds)
    limit = args.limit if args.limit > 0 else n_all
    stride = max(1, n_all // limit) if limit < n_all else 1
    idxs = list(range(0, n_all, stride))[:limit]
    dl = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                    num_workers=4,
                    collate_fn=partial(collate, samples_per_gpu=1))

    holes = {c: 0 for c in CONT}
    gts = {c: 0 for c in CONT}
    with torch.no_grad():
        for j, data in enumerate(dl):
            res = model(return_loss=False, rescale=True, **data)
            pred = unwrap(res['occ_current'] if isinstance(res, dict)
                          else res).astype(np.int64)
            info = ds.data_infos[idxs[j]]
            gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
            sem = gt['semantics'].astype(np.int64)
            vis = gt['mask_camera'].astype(bool)
            for c in CONT:
                gt_c = (sem == c) & vis
                if not gt_c.any():
                    continue
                pred_c = pred == c
                nb = neighbor_count(pred_c)
                hole = gt_c & (~pred_c) & (nb >= args.min_nb)
                holes[c] += int(hole.sum())
                gts[c] += int(gt_c.sum())
            if (j + 1) % 100 == 0:
                print(f'  {j + 1}/{len(idxs)}', flush=True)

    print(f'\n==== HOLE-RATE on {len(idxs)} samples '
          f'(min_nb={args.min_nb}/6) ckpt={os.path.basename(args.ckpt)} ====')
    vals = []
    for c, name in CONT.items():
        if gts[c] == 0:
            continue
        r = 100.0 * holes[c] / gts[c]
        vals.append(r)
        print(f'===> {name:18s} hole_rate = {r:6.3f}%  '
              f'(holes {holes[c]} / gt {gts[c]})')
    if vals:
        print(f'===> MEAN hole_rate = {np.mean(vals):6.3f}%')


if __name__ == '__main__':
    main()
