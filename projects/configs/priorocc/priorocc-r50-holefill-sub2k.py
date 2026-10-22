# 阶段 1 快速假设验证：2000 帧子集 + 4 轮短训（NOT the full stage-1 run）
#
# 目的：在 40 min 内回答"补洞开启后，单帧当前帧的连续类
# （driveable_surface / sidewalk / terrain / manmade / vegetation / other_flat）
# IoU 是否真的涨"。A/B 两臂除补洞 loss 外完全一致（同子集、同 seed、同 lr、
# 同数据顺序），所以差值可直接归因到补洞。
#
#   A 臂(off): --cfg-options model.enable_sem_continuity=False
#   B 臂(on) : 本配置原样（三层权重按实测原始量级对齐，见下）
#
# 补洞权重为什么不是 0.05/0.05/0.05：
#   实测三项原始量级 cont_2d≈0.52 / cont_bev≈0.24 / cont_occ≈0.0104，
#   统一 0.05 时 occ 项只占总 loss 的 0.14%（几乎无约束）。这里把 occ 提到
#   2.5 使其与 cont_2d 等量级（各 ≈7%），其余保持方案值。
_base_ = ['./priorocc-r50-holefill-full.py']

data_root = 'data/nuscenes/'

model = dict(
    sem_continuity_loss_weights=dict(
        cont_2d=0.05, cont_bev=0.05, cont_occ=2.5),
    continuity_apply_occ=True,
)

data = dict(
    train=dict(
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train_sub2000.pkl'),
)

# 2000 帧 / batch4 = 500 iter/轮，4 轮 = 2000 iter ≈ 15 min
runner = dict(type='EpochBasedRunner', max_epochs=4)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
# policy='fixed' 不接受 step，必须用 _delete_ 整块替换继承来的 lr_config
lr_config = dict(
    _delete_=True,
    policy='fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1)
