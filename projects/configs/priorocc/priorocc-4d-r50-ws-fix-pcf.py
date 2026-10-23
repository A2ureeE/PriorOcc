# 联训续训 C 变体 — 修复版 + PCF 测速 + 安全对齐
#
# 起点与依据:
#   热启 ep14_ema (fix 架构 600帧: 21.55/17.65/15.23, 动态Δ全正 +0.81/+0.52/+0.32)
#   + 零训练修复 (非AR+cap) 已验证: 动态 Δ 全转正
#   + PCF 相关测速在两阶段已验证: 1s 动态 +0.80
#
# 关键设计 pcf_align_only=True:
#   PCF 的 correlation 需要 ego 对齐后的历史特征才能只测物体运动;
#   但热启编码器是按"未对齐"输入训练的, 直接全局对齐 = 分布漂移。
#   安全方案: 只给 PCF 一份对齐副本, 编码器继续吃未对齐输入 → 零漂移。
#
# 续训参数: lr=2e-5 (热启末段 1e-5 的 2 倍, 温和), 6 轮
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-fix.py']

model = dict(
    # PCF 相关测速 (零初始化残差, 起步恒等)
    enable_prior_correspondence=True,
    prior_correspondence=dict(
        type='PriorCorrespondenceFlow',
        bev_channels=64,
        num_future=3,
        search_radius=10,
        horizons_sec=(1.0, 2.0, 3.0),
        history_lag=1.0,
        history_index=2,
        gate_mode='none',
    ),
    # 安全对齐: PCF 拿对齐副本, 编码器输入不变
    align_after_view_transfromation=True,
    pcf_align_only=True,
)

# 温和续训: lr 2e-5 fixed (热启末段实际 lr=1e-5, 2x 属安全范围)
lr_config = dict(
    _delete_=True,
    policy='fixed',
    lr=2e-5,
)

runner = dict(type='EpochBasedRunner', max_epochs=6)

# 热启自热启 run 的 ep14_ema (fix 架构下未来最强)
load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'smp-warmstart/epoch_14_ema.pth')
