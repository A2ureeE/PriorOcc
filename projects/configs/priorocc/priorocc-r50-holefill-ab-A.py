# 补洞 A/B —— A 臂（补洞 ON，cont_occ=0）
#
# 与 B 臂（priorocc-r50-holefill-ab-B.py）唯一差异：enable_sem_continuity。
# 共同约定（来自任务书）：
#   - load_from = 单帧 ep21（4818 同池 33.83）
#   - lr = 1e-5（ep21 末段实际学习率量级，避免复现 1e-4 发散）
#   - grad clip max_norm=5，linear warmup 200 iters
#   - 全量 28130 帧、单帧路径（num_adj=0 / BEV 入口 64 通道）
#   - enable_future_prediction=False
#   - cont_occ 必须为 0（occupancy 层补洞关闭），故 continuity_apply_occ=False
#     （不计算 occ TV，等价于该支路不存在，省算力）
#
# max_epochs 写 8：A/B 阶段用 --cfg-options runner.max_epochs=1 只跑 1 轮；
# 若判定“补洞有用”，直接对同一 work_dir 用 --auto-resume 续到 8 轮产出 SF8，
# 无需重跑已完成的第 1 轮。
_base_ = ['./priorocc-r50-holefill-full.py']

model = dict(
    enable_sem_continuity=True,
    continuity_apply_occ=False,
    sem_continuity_loss_weights=dict(
        cont_2d=0.05, cont_bev=0.05, cont_occ=0.0),
    enable_future_prediction=False,
)

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='fixed',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=8)

load_from = 'ckpts/priorocc-singleframe-ep21.pth'
