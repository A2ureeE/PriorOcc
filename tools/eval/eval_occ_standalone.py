import os, sys, argparse, numpy as np, torch
from functools import partial
sys.path.insert(0, '/root/priorocc/PriorOcc'); os.chdir('/root/priorocc/PriorOcc')
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader, Subset
import projects.mmdet3d_plugin
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.core.evaluation.occ_metrics import Metric_mIoU

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--cfg', default='projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-nodepth.py')
ap.add_argument('--no-points', action='store_true', dest='no_points',
                help='strip LiDAR-point loading from test pipeline (camera-only eval)')
a = ap.parse_args()

cfg = Config.fromfile(a.cfg); cfg.data.test.test_mode = True
if a.no_points:
    def _strip(pl):
        out = []
        for s in pl:
            t = s.get('type', '')
            if t in ('LoadPointsFromFile', 'LoadPointsFromMultiSweeps', 'PointToMultiViewDepth'):
                continue
            s = dict(s)
            if t == 'MultiScaleFlipAug3D':
                s['transforms'] = [dict(tt, keys=[k for k in tt.get('keys', []) if k != 'points'])
                                   if tt.get('type') == 'Collect3D' else tt for tt in s.get('transforms', [])]
            if t == 'Collect3D':
                s['keys'] = [k for k in s.get('keys', []) if k != 'points']
            out.append(s)
        return out
    if 'test' in cfg.data:
        cfg.data.test.pipeline = _strip(cfg.data.test.pipeline)
    if cfg.get('evaluation', None) is not None and 'pipeline' in cfg.evaluation:
        cfg.evaluation.pipeline = _strip(cfg.evaluation.pipeline)
ds = build_dataset(cfg.data.test)
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, a.ckpt, map_location='cpu')
model = MMDataParallel(model.cuda(), device_ids=[0]); model.eval()

# 跨场景分层子集：等距取样覆盖全部场景
N = len(ds)
limit = a.limit if a.limit > 0 else N
step = max(1, N // limit) if limit < N else 1
idxs = list(range(0, N, step))[:limit]
print(f'SELECTED {len(idxs)}/{N} samples, stride={step} (scene-spread)')
sub = Subset(ds, idxs)
dl = DataLoader(sub, batch_size=1, shuffle=False, num_workers=4,
                collate_fn=partial(collate, samples_per_gpu=1))

def unwrap(x):
    if isinstance(x, (list, tuple)): x = x[0]
    if isinstance(x, torch.Tensor): x = x.cpu().numpy()
    return np.asarray(x)

cur_m = Metric_mIoU(num_classes=18, use_lidar_mask=False, use_image_mask=True)
fut_m = [Metric_mIoU(num_classes=18, use_lidar_mask=False, use_image_mask=True) for _ in range(3)]
printed = False; ncur = 0; nfut = 0
with torch.no_grad():
    for j, data in enumerate(dl):
        res = model(return_loss=False, rescale=True, **data)
        if not printed:
            print('RESULT_TYPE', type(res).__name__, list(res.keys()) if isinstance(res, dict) else '')
            printed = True
        info = ds.data_infos[idxs[j]]
        if isinstance(res, dict) and 'occ_current' in res: cur = res['occ_current']
        elif isinstance(res, dict) and 'pred_occ' in res: cur = res['pred_occ']
        else: cur = res
        cur = unwrap(cur).astype(np.uint8)
        gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
        cur_m.add_batch(cur, gt['semantics'], gt['mask_lidar'].astype(bool), gt['mask_camera'].astype(bool)); ncur += 1
        fc = info.get('forecast', None)
        if isinstance(res, dict) and 'occ_future' in res and fc is not None:
            for k in range(3):
                fp = unwrap(res['occ_future'][k]).astype(np.uint8)
                fgt = np.load(os.path.join(fc['future_occ_paths'][k], 'labels.npz'))
                fut_m[k].add_batch(fp, fgt['semantics'], fgt['mask_lidar'].astype(bool), fgt['mask_camera'].astype(bool))
            nfut += 1
print(f'\n==== EVAL on {ncur} samples (future {nfut}) stride={step} ckpt={os.path.basename(a.ckpt)} ====')
print('=== CURRENT-frame mIoU ==='); cur_m.count_miou()
for k in range(3):
    print(f'=== FUTURE {k+1}s mIoU ==='); fut_m[k].count_miou()
