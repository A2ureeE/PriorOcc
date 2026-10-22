# H2 归因拆分 B：只放大逐 horizon flow 上限（保持自回归递推）
# 用于回答"2s/3s 转正到底靠非自回归还是靠放大 flow 上限"
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-fix.py']

model = dict(
    max_flow_cells=[12.0, 24.0, 36.0],
    scmf=dict(max_flow_cells=[12.0, 24.0, 36.0]),
    future_predictor=dict(
        non_autoregressive=False,
        gate_init_bias=2.0,
    ),
)

load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'smp-warmstart/epoch_15_ema.pth')
