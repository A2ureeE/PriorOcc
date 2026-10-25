import os, sys, argparse, numpy as np, torch
from functools import partial
sys.path.insert(0,'/root/priorocc/PriorOcc'); os.chdir('/root/priorocc/PriorOcc')
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader, Subset
import projects.mmdet3d_plugin
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.core.evaluation.occ_metrics import Metric_mIoU
ap=argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--limit',type=int,default=0)
ap.add_argument('--cfg',default='projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-nodepth.py')
a=ap.parse_args()
cfg=Config.fromfile(a.cfg); cfg.data.test.test_mode=True
ds=build_dataset(cfg.data.test)
model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
load_checkpoint(model,a.ckpt,map_location='cpu')
model=MMDataParallel(model.cuda(),device_ids=[0]); model.eval()
N=len(ds); limit=a.limit if a.limit>0 else N
step=max(1,N//limit) if limit<N else 1
idxs=list(range(0,N,step))[:limit]
print(f'SELECTED {len(idxs)}/{N} stride={step}')
dl=DataLoader(Subset(ds,idxs),batch_size=1,shuffle=False,num_workers=4,
              collate_fn=partial(collate,samples_per_gpu=1))
def unwrap(x):
    if isinstance(x,(list,tuple)): x=x[0]
    if isinstance(x,torch.Tensor): x=x.cpu().numpy()
    return np.asarray(x)
M=lambda: Metric_mIoU(num_classes=18,use_lidar_mask=False,use_image_mask=True)
cur_m=M(); fut_m=[M() for _ in range(3)]; b_m=[M() for _ in range(3)]
n=0
with torch.no_grad():
    for j,data in enumerate(dl):
        res=model(return_loss=False,rescale=True,**data)
        info=ds.data_infos[idxs[j]]
        cur=unwrap(res['occ_current']).astype(np.uint8)
        gt=np.load(os.path.join(info['occ_path'],'labels.npz'))
        cur_m.add_batch(cur,gt['semantics'],gt['mask_lidar'].astype(bool),gt['mask_camera'].astype(bool))
        fc=info.get('forecast',None)
        if isinstance(res,dict) and 'occ_future' in res and fc is not None:
            for k in range(3):
                fp=unwrap(res['occ_future'][k]).astype(np.uint8)
                fg=np.load(os.path.join(fc['future_occ_paths'][k],'labels.npz'))
                fs=fg['semantics']; ml=fg['mask_lidar'].astype(bool); mc=fg['mask_camera'].astype(bool)
                fut_m[k].add_batch(fp,fs,ml,mc)
                b_m[k].add_batch(cur,fs,ml,mc)
            n+=1
print(f'\n==== B-baseline eval on {n} samples ckpt={os.path.basename(a.ckpt)} ====')
print('=== CURRENT-frame (model) ==='); cur_m.count_miou()
for k in range(3):
    print(f'=== FUTURE {k+1}s (MODEL) ==='); fut_m[k].count_miou()
for k in range(3):
    print(f'=== FUTURE {k+1}s (B-freeze-model-current) ==='); b_m[k].count_miou()
