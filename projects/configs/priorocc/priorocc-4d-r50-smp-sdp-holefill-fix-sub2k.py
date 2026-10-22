# 假设 2 快速验证：ep15_ema + 两个 bug 修复（逐 horizon cap + 非自回归）
# 2000 帧 forecast 子集、2 轮（1000 iter/轮）短训 ≈ 30 min，然后
# eval_b_baseline.py 同一次推理给出 MODEL 与 B 档（冻结自身当前帧），
# 直接看 2s/3s 动态 IoU 的 Δ(MODEL − B档) 是否由负转正。
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-fix.py']

data_root = 'data/nuscenes/'

data = dict(
    train=dict(
        ann_file=data_root +
        'bevdetv2-nuscenes_infos_train_forecast_sub2000.pkl'),
)

# 2000 帧 / batch2 = 1000 iter/轮，2 轮 = 2000 iter ≈ 30 min
runner = dict(type='EpochBasedRunner', max_epochs=2)
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
# policy='fixed' 不接受 step，必须用 _delete_ 整块替换继承来的 lr_config
lr_config = dict(
    _delete_=True,
    policy='fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1)

load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'smp-warmstart/epoch_15_ema.pth')
