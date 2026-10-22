# PriorOcc-4D PCF (Prior Correspondence Flow) — 语义先验指路 + 对应测量位移
#
# 核心创新（用户授权）：不让模型"假设"位移（回归），而是从历史帧的
# 特征对应（RAFT 式 cost-volume）中"测量"位移。SI 先验（dyn_mask）
# 告诉模型"哪里是动态物体"，PCF 测量"它们怎么动"。
#
# 流分解:
#   total_flow = SCMF_flow(全局/自车分量, 学习) + PCF_flow(物体位移, 测量)
#   SCMF 保留：它已验证能学好全局自车平移（静态 IoU +1~+1.3）
#   SMP 关闭：实测 rigid_vehicle basis 把车辆位移砍到 0.34x，帮倒忙
#   PCF 新增：零参数测量 + 零初始化残差纠偏（<0.4K 参数）
#
# 预期: PCF 从第 1 个 iteration 就给出合理的动态位移（correlation 是
#   确定性计算，无需学习），动态 IoU 应从 ep1 就开始收窄与 B 档的差距，
#   不同于 SCMF-only 需要多轮才能追上（ep2 时动态 Δ 仍为 −2.33）。
#
# 前提: align_after_view_transfromation=True（历史帧自车对齐），
#   使 correlation 只测物体运动（自车运动由 SCMF 全局流负责）。
_base_ = ['./priorocc-4d-r50-sep-frozen.py']

model = dict(
    # 历史帧自车对齐（PCF 的前提）
    align_after_view_transfromation=True,

    # 关闭 SMP（实测在砍车辆位移，与 PCF 的测量位移冲突）
    enable_motion_prior=False,
    motion_prior=None,

    # 开启 PCF
    enable_prior_correspondence=True,
    prior_correspondence=dict(
        type='PriorCorrespondenceFlow',
        bev_channels=64,          # raw BEV (pre-encoder) channels
        num_future=3,
        search_radius=10,         # ±10 cells; covers p90 at 1.0s lag
        temperature=0.1,          # softmax 温度; 低=更尖锐的匹配
        horizons_sec=(1.0, 2.0, 3.0),
        history_lag=1.0,          # bev_feat_list[2] 距当前 1.0s
        history_index=2,          # 最远历史帧（1.0s back），速度估计更稳
    ),
)

# 用补洞 ep4_ema 骨干（4818 全量 33.94，连续类 +0.19）
load_from = ('/root/autodl-tmp/dataforpriorocc/work_dirs/'
             'ab-on/epoch_4_ema.pth')
