# PriorOcc-4D FIX v1 —— 短训验证用派生配置
#
# 继承单流 smp-sdp-holefill-nodepth，叠加三处针对性修复（均来自 ep15/ep19
# 三组对照诊断的结论）：
#
#  1. 逐 horizon flow 上限 [12, 24, 36] cells，替代原先全局 5 cells：
#     GT 动态目标位移 p90 实测约 8.6 / 14.1 / 17.9 cells（1 cell = 0.512 m），
#     5 cells (=2.56 m) 在物理上无法把移动目标放到正确位置。
#  2. 非自回归未来推演：每个 horizon 都对同一个当前 BEV 做 warp。
#     原先 future_k 作为第 k+1 步的输入，flow 误差沿链累积 —— 这是动态 IoU
#     在 2s/3s 反而劣于"冻结"基线的直接原因
#     （ep19: 动态 Δ(MODEL-B档) = +1.11 / -0.22 / -0.79）。
#  3. gate 初始化偏向 warp 分支（sigmoid(2.0)≈0.88），使模型起步不低于冻结拷贝。
#
# 短训（12 轮）重设 lr 调度：step=[8] 衰减，避免恒 1e-4 跑完无衰减。
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-nodepth.py']

model = dict(
    scmf=dict(
        max_flow_cells=[12.0, 24.0, 36.0],
    ),
    future_predictor=dict(
        non_autoregressive=True,
        gate_init_bias=2.0,
    ),
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8])
