# H2 归因拆分 A：只开非自回归（flow 上限仍是旧的全局 5 cells）
# 用于回答"2s/3s 转正到底靠非自回归还是靠放大 flow 上限"
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-fix.py']

model = dict(
    max_flow_cells=5.0,
    scmf=dict(max_flow_cells=5.0),
    future_predictor=dict(
        non_autoregressive=True,
        gate_init_bias=2.0,
    ),
)

load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'smp-warmstart/epoch_15_ema.pth')
