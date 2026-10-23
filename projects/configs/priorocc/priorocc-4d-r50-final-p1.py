# 最终模型 Phase 1 — 干净配置从头练 (ep21 热启)
#
# 与旧热启run的关键区别:
#   1. 无 SMP (位移收缩器移除, SCMF 直接校准)
#   2. + PCF 测速 (新鲜训练 = SCMF 学"让位", 无双重计费)
#   3. + 全局 ego 对齐 (新鲜训练 = 无分布漂移)
#   4. 非AR + 逐段cap (已验证的修复, 从第1轮就生效)
#
# 课程学习 Phase 1: 标准 loss, lr=1e-4, ep8 衰减
#   (ep8 = 静态/全局饱和点, 之后切换 Phase 2 动态加权)
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-nodepth.py']

model = dict(
    # 禁用 SMP (位移收缩器, rigid_smooth 奖励全局平移退化解)
    enable_motion_prior=False,
    motion_prior=None,

    # PCF 相关测速 (SCMF 从零学 → 让位给 PCF 测物体位移)
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

    # 全局 ego 对齐 (新鲜训练, 编码器从第1轮就学对齐特征)
    align_after_view_transfromation=True,

    # 修复版 SCMF: 非自回归 + 逐段上限
    scmf=dict(
        max_flow_cells=[12.0, 24.0, 36.0],
    ),
    future_predictor=dict(
        non_autoregressive=True,
        gate_init_bias=2.0,
    ),
)

# Phase 1: 标准 loss, lr=1e-4, ep8 衰减 (×0.1 → 1e-5)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8],
)

runner = dict(type='EpochBasedRunner', max_epochs=8)

# 热启自单帧 ep21 (4D 兼容版, 192ch 首层形状失配已过滤)
load_from = '/root/priorocc/PriorOcc/ckpts/priorocc-singleframe-ep21_4dcompat.pth'
