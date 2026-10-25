# 消融实验设计 + 数据对比（基于在手指权）

> 只用**当前已有的权重**能跑的消融，全部走 `tools/eval/eval_b_baseline.py` 三口径
> （CURRENT / MODEL / B档静态复制），全量 4818·EMA。协议见 `EVAL_METHOD.md`。
> “✓已测”= 数字已在本仓库；“⟳待跑”= 权重在手、需再评一次取数。

## 1. 在手权重清单（`released_ckpts/`，gitignore 不上传）

| 权重 | 代表配置 | 消融角色 |
|---|---|---|
| `final-p1-depth_phase2_ep10_ema.pth` | ★ 主模型：非SMP+非自回归+cap[12,24,36]+PCF+ego对齐+depth3.0+**uniform动态加权×3** | 完整方法 |
| `final-p1-depth_phase1_ep8_ema.pth` | 同上但**无动态加权**、lr1e-4 | “−动态加权” |
| `ablation_phase3_perepoch11_ema.pth` | 主模型基上换**逐类加权**(bus×5/小目标×6) | 加权策略对照(负结果) |
| `ablation_smp-warmstart_ep15_ema.pth` | **SMP + 自回归**旧架构(无PCF/无cap/无加权) | “−架构修复/含SMP” |
| `final-p1-depth_phase2_ep10.pth` | 主模型原始权重(含优化器) | 续训用 |
| `ckpts/priorocc-singleframe-ep21.pth` | 单帧(无未来分支) | “−预测”上限/当前帧参考 |

> B 档静态复制**不需权重**：把模型自身当前帧预测冻结当未来，评测时同帧同 mask 现算。

## 2. 消融矩阵（用手上权重即可覆盖）

| # | 消融问题 | 对照（同口径全量EMA） | 判据/结论 |
|---|---|---|---|
| A | 动态类加权是否有用 | ep8(−加权) vs ep10(+加权) | 动态Δ +1.26/+1.00/+0.74 → **+2.93/+2.67/+2.32**，未来mIoU +0.8~0.95，当前帧不掉 → ✓有效 |
| B | 逐类加权能否再涨 | ep10(uniform) vs phase3(逐类) | 全在±0.5噪声内、无提升 → ✓饱和/负结果 |
| C | 架构(SMP/AR → 非SMP/非AR+PCF+cap) | smp-ep15 vs ep10 | 旧臂2s/3s动态Δ曾 −0.22/−0.79(转负)，新配方全正 → ✓架构修复关键 |
| D | 预测相对静态复制的净增益 | 每个模型 MODEL vs B档 | 主表用 Δ(MODEL−B)；见 §3 |
| E | 单帧 vs 4D(当前帧代价) | ep21 33.83 vs ep10 30.58 | “预测代价” −3.25，换来未来能力 → ✓可解释 |
| F | PCF / SI / cap / 非自回归 单组件 | **需另训消融臂**（旧两阶段平台已定性：PCF 1s +0.80、attention 3s +0.24、warp 天花板≈−1.4） | 手上权重不含"关某组件"变体 → ⚠️如需精确单组件Δ要补训 |

## 3. 数据对比总表（全量 4818 · EMA）

**未来 mIoU（MODEL）：**

| 模型 | 1s | 2s | 3s | 未来均值 |
|---|---|---|---|---|
| ep8（无加权） | 21.17 | 17.85 | 15.86 | 18.29 |
| **ep10（主）** | **22.12** | **18.78** | **16.69** | **19.20** |
| ep11 | 22.09 | 18.73 | 16.65 | 19.16 |
| ep12 | 22.05 | 18.73 | 16.65 | 19.14 |
| phase3（逐类） | 22.06 | 18.76 | 16.65 | 19.16 |
| B档静态复制(ep10) | 18.70 | 14.95 | 13.07 | 15.57 |

**动态子集 Δ(MODEL−B)（核心指标）：**

| 模型 | 1s | 2s | 3s |
|---|---|---|---|
| ep8（无加权） | +1.26 | +1.00 | +0.74 |
| **ep10（主）** | **+2.93** | **+2.67** | **+2.32** |
| phase3（逐类） | +2.80 | +2.65 | +2.28 |

**当前帧 mIoU：** ep21单帧 33.83 ｜ ep8 30.36 ｜ ep10 30.58 ｜ phase3 30.75（加权不伤当前帧）。
**逐类明细（bus 负Δ、小目标低绝对、大车/静态正）** 见 `RESULTS_perclass_ep10.md`。

## 4. 复现每条消融的评测命令

```bash
CFG=projects/configs/priorocc/priorocc-4d-r50-final-p1-depth.py   # ep8/ep10/phase3 同结构可加载
for C in final-p1-depth_phase1_ep8_ema final-p1-depth_phase2_ep10_ema ablation_phase3_perepoch11_ema; do
  python tools/eval/eval_b_baseline.py released_ckpts/${C}.pth --cfg $CFG   # 三口径, 内含 B 档
done
# with-SMP 旧臂用其自身配置加载(smp-warmstart)：cfg=priorocc-4d-r50-smp-sdp-holefill-nodepth.py
```

## 5. 结论与缺口
- **可写进论文的消融（有数）**：动态加权(A)、逐类饱和(B)、架构修复(C)、vs 静态复制(D)、单帧代价(E) —— 均由手上权重支撑，数字已固化。
- **要更硬的单组件因果(F)**：PCF/SI/cap/非自回归 各自Δ，需在**统一基座上各训一版**（旧两阶段只给定性证据）。是否补看审稿要求；主卖点已由 A/C/D 支撑。
