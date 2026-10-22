# PCF + Dynamic Attention Boost
#
# 继承 PCF 配置，仅追加 enable_dyn_boost=True：
#   refined_bev *= (1 + tanh(dyn_boost) * dyn_mask)
#   dyn_boost 零初始化 → tanh(0)=0 → 起步恒等（不改变已训好的 PCF 行为）
#   上界 2×（tanh 有界），允许学到抑制（tanh<0 → 因子<1）
_base_ = ['./priorocc-4d-r50-pcf.py']

model = dict(
    enable_dyn_boost=True,
)
