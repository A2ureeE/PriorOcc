# PriorOcc-4D 阶段 2：冻结单帧感知 + 轻量未来头（sep-frozen）
#
# 前提：阶段 1（priorocc-r50-holefill-full.py）已产出当前帧 > 33.83 的
# 补洞版单帧骨干。本配置把它整体冻结，只训未来预测分支，并修掉导致
# 2s/3s 崩溃的两个实现缺陷：
#
#   1. 逐 horizon flow 上限 [12, 24, 36] cells（替代全局 5 cells）。
#      GT 动态位移 p90 实测 ~8.6 / 14.1 / 17.9 cells（1 cell = 0.512 m）。
#   2. 非自回归：每个 horizon 都对同一个当前 BEV 做 warp，
#      不再 future_k -> future_{k+1} 递推（误差累积是 2s/3s 转负的直接原因：
#      ep19 动态 Δ(MODEL - 冻结自身) = +1.11 / -0.22 / -0.79）。
#   3. gate 初始化偏 warp（sigmoid(2.0)≈0.88），起步不低于冻结拷贝。
#
# 冻结范围（freeze_perception=True）：img_backbone / img_neck /
# semantic_injector / img_view_transformer / img_bev_encoder_* / occ_head。
# sep_perception=True 让 BEV 编码器只吃当前帧（64 通道），否则 192 通道时序
# 拼接的首层卷积无法从单帧权重继承，冻结住随机卷积会直接搞塌当前帧 mIoU。
# 运动链（bev_projector / dyn_sta_decoder / motion_encoder /
# semantic_attention / delta_combiner / scmf / future_predictor /
# motion_prior）全部可训——它们是本阶段新初始化的，冻结就没有梯度来源。
#
# 验收判据：2s/3s 动态 IoU 由负转正（> B 档 6.88 / 5.58）；整体不低于
# ep15（21.34 / 17.14 / 14.77）；当前帧 = 阶段 1 结果（冻结不动）。
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-nodepth.py']

numC_Trans = 64
# 阶段 2 恢复时序输入（历史 2 帧）供运动分支使用；但感知主干走
# sep_perception=True，BEV 编码器只吃当前帧的 64 通道，因此 stage-1
# 感知权重可 100% 1:1 继承（若用 192 通道时序拼接，首层卷积必然失配）。
multi_adj_frame_id_cfg = (1, 3, 1)

model = dict(
    num_adj=2,
    with_prev=True,
    # ---- 只训未来头 ----
    enable_future_prediction=True,
    enable_multimodal=False,
    enable_motion_prior=True,
    # 感知冻结后补洞 loss 的梯度没有下游可作用对象，关掉以免白算
    enable_sem_continuity=False,
    freeze_perception=True,
    sep_perception=True,
    # ---- 两个 bug 的修复 ----
    max_flow_cells=[12.0, 24.0, 36.0],
    scmf=dict(max_flow_cells=[12.0, 24.0, 36.0]),
    future_predictor=dict(
        non_autoregressive=True,
        gate_init_bias=2.0,
    ),
    img_bev_encoder_backbone=dict(numC_input=numC_Trans),
    # 与阶段 1 / ep21 对齐关闭 SDP，否则冻结的 prior table 是随机常量
    img_view_transformer=dict(
        depthnet_cfg=dict(use_semantic_depth_prior=False)),
)

# 阶段 1 产物（守护脚本会把当前帧最优 EMA 复制成 best_ema.pth）
load_from = '/root/autodl-tmp/dataforpriorocc/work_dirs/stage1-holefill/best_ema.pth'

# 未来头 6 轮足够：只训新初始化的轻量分支（6.8M），感知 52.4M 已固定。
runner = dict(type='EpochBasedRunner', max_epochs=6)
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[4])

evaluation = dict(interval=2, start=2)
