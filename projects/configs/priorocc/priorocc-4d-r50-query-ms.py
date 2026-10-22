# Multi-scale Query Future Head — per-horizon resolution & window
#
# 修复 query-v1 的 1s 精度缺陷（下采样 2x 导致 ±1m 量化误差 + softmax 涂抹）:
#   1s: 全分辨率 200x200, R=5 (K=121) → 亚米级精度 + 更锐利的 softmax
#   2s: 半分辨率 100x100, R=9 (K=361) → ±9m 范围（共享 unfold）
#   3s: 半分辨率 100x100, R=9 (K=361) → ±9m 范围（共享 unfold）
#
# 显存: 1s unfold 4.9GB + 2s/3s 共享 3.7GB = 8.6GB (vs 单尺度 4.5GB)
# 预期: 1s 从 7.58 修复到 ≥8.95 (全分辨率), 3s 保持 6.21+ (范围还在)
#
# 热启: PCF ep6 (motion branch 训满 6 轮, 测速更准 → 速度偏置指路更准)
_base_ = ['./priorocc-4d-r50-pcf.py']

model = dict(
    enable_future_prediction=True,
    future_predictor=None,

    enable_query_future=True,
    query_future=dict(
        type='SemanticQueryFutureHead',
        bev_channels=256,
        num_future=3,
        hidden_dim=64,
        # 多尺度: 逐 horizon 分辨率与窗口
        window_radius=[5, 9, 9],
        downsample=[1, 2, 2],
        num_semantic_classes=17,
        horizons_sec=(1.0, 2.0, 3.0),
        vel_bias_init=2.0,
    ),
)

# 热启自 PCF ep6 (motion branch 满训; query head 零初始化)
load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'pcf-v1/epoch_6.pth')
