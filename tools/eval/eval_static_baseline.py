import os, sys, numpy as np
sys.path.insert(0,'/root/priorocc/PriorOcc'); os.chdir('/root/priorocc/PriorOcc')
from mmcv import Config
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.core.evaluation.occ_metrics import Metric_mIoU

cfg = Config.fromfile('projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-nodepth.py')
cfg.data.test.test_mode = True
ds = build_dataset(cfg.data.test)
# static-copy(GT版): 未来k的预测 = 直接拷贝"当前帧GT语义", 用未来GT自身的mask_camera评测
fut = [Metric_mIoU(num_classes=18, use_lidar_mask=False, use_image_mask=True) for _ in range(3)]
n=0; nskip=0
for info in ds.data_infos:
    fc = info.get('forecast', None)
    if fc is None or 'occ_path' not in info: nskip+=1; continue
    try:
        cur = np.load(os.path.join(info['occ_path'],'labels.npz'))['semantics'].astype(np.uint8)
    except Exception:
        nskip+=1; continue
    ok=False
    for k in range(3):
        try:
            f = np.load(os.path.join(fc['future_occ_paths'][k],'labels.npz'))
        except Exception:
            continue
        fs = f['semantics'].astype(np.uint8)
        if fs.shape != cur.shape:
            continue
        fut[k].add_batch(cur, fs, f['mask_lidar'].astype(bool), f['mask_camera'].astype(bool))
        ok=True
    if ok: n+=1
    if n % 500 == 0: print(f'  processed {n} samples', flush=True)
print(f'\n==== STATIC-COPY(GT) baseline on {n} samples (skipped {nskip}) ====')
for k in range(3):
    print(f'=== FUTURE {k+1}s mIoU (static-copy) ==='); fut[k].count_miou()
