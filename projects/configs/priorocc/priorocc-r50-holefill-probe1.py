# 阶段 1「先跑一轮看看」探针配置（2026-10-22 用户裁定）
#
# 在 ep21 基础上用**全量 28130 帧**跑 1 轮（7033 iter），用于快速判断
# 补洞在全量帧下是否把连续类（背景）IoU 推上去。
#
# 与 canonical 阶段 1 配置（priorocc-r50-holefill-full.py）的差异只有两处：
#   1. cont_occ = 2.5（不是方案里的 0.05）。实测量级：三项原始值
#      cont_2d≈0.52 / cont_bev≈0.24 / cont_occ≈0.0104，统一 0.05 时 occ 项只占
#      总 loss 0.14%（形同关闭）；2.5 使其与 cont_2d 等量级（≈5%）。
#   2. max_epochs=1：只跑一轮，看结果再决定是否续到 12 轮。
#
# 评测口径（脚本 tools/stage1/probe1.sh 自动执行）：
#   - 全量 4818 当前帧 mIoU + 逐类，与 ep21（33.83 / 连续类 48.84）同池同类型对比
#   - 600 帧洞率，与 ep21（2.456%）对比
_base_ = ['./priorocc-r50-holefill-full.py']

model = dict(
    sem_continuity_loss_weights=dict(
        cont_2d=0.05, cont_bev=0.05, cont_occ=2.5),
    continuity_apply_occ=True,
)

runner = dict(type='EpochBasedRunner', max_epochs=1)
