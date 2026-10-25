# Phase 3 — 逐类动态加权（从 Phase2 best ckpt 续训）
#
# 动机: ep10 全量逐类诊断显示两类"难学"病根不同——
#   bus/trailer/constr: 细长, bus 1s 动态 Δ=-0.47 (比静态复制还差, 位移精度问题)
#   bicycle/motorcycle/pedestrian: 极小欠学, 绝对 IoU 近零 (梯度预算不足)
# 策略: 给这两组更高的乘子, 已学好的 car/truck/barrier/cone 维持 ×3
#
# 继承 Phase2 配置 (loss_depth_weight=3.0 / PCF / 非AR / future-only boost 机制)
# 仅把 dynamic_class_boost 从标量 3.0 换成逐类字典
# 启动: 从 Phase2 best epoch_ema 的对应 epoch_N.pth --resume-from, 新 work_dir
_base_ = ['./priorocc-4d-r50-final-p1-depth.py']

# occ 类序: 0 others | 1 barrier 2 bicycle 3 bus 4 car 5 construction_vehicle
#           6 motorcycle 7 pedestrian 8 traffic_cone 9 trailer 10 truck
#           11..16 静态面 | 17 free
model = dict(
    occ_head=dict(
        # '_delete_': True 让 mmcv 用本 dict 整体替换基类的标量 3.0
        # （dict 不能覆盖 float）；int 键与字符串键在同一字面量里合法
        dynamic_class_boost={
            '_delete_': True,
            3: 5.0,    # bus        — 细长, 拉回非负
            9: 5.0,    # trailer    — 细长
            5: 5.0,    # construction_vehicle — 大且慢
            2: 6.0,    # bicycle    — 极小欠学
            6: 6.0,    # motorcycle — 极小欠学
            7: 6.0,    # pedestrian — 小目标
            4: 3.0,    # car        — 维持 (已学好)
            10: 3.0,   # truck      — 维持
            1: 3.0,    # barrier    — 维持 (名义动态, 物理少动)
            8: 3.0,    # traffic_cone — 维持
        },
    ),
)
