"""实验A：把单帧 ep21 放到与 4D 完全相同的 4818 帧上评测，消除样本池差异。"""
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
ap=argparse.ArgumentParser()
ap.add_argument('--ckpt', default='/root/priorocc/PriorOcc/ckpts/priorocc-singleframe-ep21.pth')
ap.add_argument('--cfg', default='projects/configs/priorocc/priorocc-r50.py')
ap.add_argument('--limit', type=int, default=0)
a=ap.parse_args()
# 1) 4D 评测所用的 4818 帧 token
c4=Config.fromfile('projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-nodepth.py'); c4.data.test.test_mode=True
d4=build_dataset(c4.data.test); tok=set(i['token'] for i in d4.data_infos)
print('TOKENS_4D', len(tok), flush=True)
# 2) 单帧数据集按 token 取交集（保持原顺序）
cfg=Config.fromfile(a.cfg); cfg.data.test.test_mode=True
# 纯视觉环境无 LiDAR 点云：摘掉 LoadPointsFromFile，否则取数报 FileNotFoundError
cfg.data.test.pipeline=[st for st in cfg.data.test.pipeline if st.get('type')!='LoadPointsFromFile']

def _strip_points(pipe):
    """纯视觉环境: 递归移除依赖 LiDAR 点云的 Collect 键(points/gt_depth)。"""
    for st in pipe:
        if not isinstance(st, dict):
            continue
        if isinstance(st.get('transforms'), (list, tuple)):
            _strip_points(st['transforms'])
        keys = st.get('keys')
        if isinstance(keys, (list, tuple)) and keys:
            kept = [k for k in keys if k not in ('points', 'gt_depth')]
            if len(kept) != len(keys):
                st['keys'] = kept
_strip_points(cfg.data.test.pipeline)
ds=build_dataset(cfg.data.test)
idxs=[k for k,i in enumerate(ds.data_infos) if i['token'] in tok]
print('SUBSET', len(idxs), '/', len(ds.data_infos), flush=True)
if a.limit>0: idxs=idxs[:a.limit]
# 3) 单帧模型 + ep21
model=build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, a.ckpt, map_location='cpu')
model=MMDataParallel(model.cuda(), device_ids=[0]); model.eval()
dl=DataLoader(Subset(ds,idxs), batch_size=1, shuffle=False, num_workers=4,
              collate_fn=partial(collate, samples_per_gpu=1))
def unwrap(x):
    if isinstance(x,(list,tuple)): x=x[0]
    if isinstance(x,torch.Tensor): x=x.cpu().numpy()
    return np.asarray(x)
m=Metric_mIoU(num_classes=18, use_lidar_mask=False, use_image_mask=True)
n=0
with torch.no_grad():
    for j,data in enumerate(dl):
        res=model(return_loss=False, rescale=True, **data)
        cur=res['occ_current'] if isinstance(res,dict) and 'occ_current' in res else \
            (res['pred_occ'] if isinstance(res,dict) and 'pred_occ' in res else res)
        cur=unwrap(cur).astype(np.uint8)
        info=ds.data_infos[idxs[j]]
        gt=np.load(os.path.join(info['occ_path'],'labels.npz'))
        m.add_batch(cur, gt['semantics'], gt['mask_lidar'].astype(bool), gt['mask_camera'].astype(bool))
        n+=1
print(f'\n==== ep21 single-frame on SAME 4818-frame subset: {n} samples ====')
m.count_miou()
