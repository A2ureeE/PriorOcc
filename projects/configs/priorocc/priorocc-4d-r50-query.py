# Query Future Head — cross-attention with velocity-matching bias
#
# 核心创新（方案 B）：用 attention 聚合替代 backward warp。
#   query 在未来位置 B 问"这里该有什么" → 速度偏置告诉它"去看 A"
#   bias(B,A) = -||B - A - v(A)·dt||²  （A 的速度恰好把内容送到 B 时最大）
#
# 组合: PCF 测速度（测量） + query head 用速度指路（聚合）
#   不再依赖 backward warp 的流场 → 解决"流在目标位置取不到物体"的结构性限制
_base_ = ['./priorocc-4d-r50-pcf.py']

model = dict(
    # 禁用旧的未来头（backward warp + GRU + gate）
    enable_future_prediction=True,   # 保留 flag 以进入 future loss 分支
    future_predictor=None,

    # 启用 query future head
    enable_query_future=True,
    query_future=dict(
        type='SemanticQueryFutureHead',
        bev_channels=256,
        num_future=3,
        hidden_dim=64,
        window_radius=10,
        num_semantic_classes=17,
        horizons_sec=(1.0, 2.0, 3.0),
        vel_bias_init=2.0,   # sigmoid(2)=0.88, 初始强信任速度
    ),
)

# 热启: 从 PCF ep2 (已验证 1s 动态 +0.80) 开始
load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'pcf-v1/epoch_2.pth')
