# 动态类梯度加权续训 — SGDM 理念在 loss 层的落地
#
# 依据（实测数字）:
#   静态:动态体素比 75:1; class_balance 的 1/log 加权太温和
#   (路面梯度贡献 930万 vs 自行车 1.3万, 差 700 倍)
#   静态已饱和(ego 补偿增益到顶), 动态欠学 → 3x 加权把动态
#   梯度份额从 ~12% 提到 ~30%
#
# 基线（已留存）: ep14_ema + fix 架构零训练
#   全量4818: 当前帧 31.88 | 21.33/17.64/15.09 (Avg 18.02)
#   动态Δ: +0.80/+0.60/+0.26 全正
#   600帧: 21.55/17.65/15.23, 当前帧 31.96 ← ep2 判决用这个
#
# 训练: fix 架构(cap=[12,24,36] 使大位移可学) + lr=1e-5(原 run 末段值,
# 最保守) + 4 轮短续训; 效果不好回退零训练基线
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-fix.py']

model = dict(
    occ_head=dict(
        # 动态类(索引1..10: barrier..truck)在 1/log 频率权重基础上 ×3
        # 自归一化: loss 的 num_total_samples 分母同用 cls_weights,
        # 尺度稳定, 只是梯度再分配
        dynamic_class_boost=3.0,
    ),
)

lr_config = dict(
    _delete_=True,
    policy='fixed',
)

optimizer = dict(lr=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=4)

load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'smp-warmstart/epoch_14_ema.pth')
