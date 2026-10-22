# PriorOcc-4D 未来预测头重设计：诊断、创新与实施留档

> 2026-10-21 ~ 10-22。本文是"感知/预测分离"到"未来头重设计"的完整决策与创新留档，
> 所有结论均有实测数字支撑；原始日志在 `/root/autodl-tmp/dataforpriorocc/{auto,quick_hyp,probe1}/`。

## 0. 决策链时间线

| 时间 | 事件 | 依据 |
|---|---|---|
| 10-21 | 停 fix 短训，拍板两阶段分离（阶段1 补洞单帧 / 阶段2 冻结感知+未来头） | B档裁决：感知瓶颈 3.66 > 运动增益 1.11 |
| 10-21 | 小子集双假设验证：H2 成立（非自回归单独转正 2s/3s），H1 机制成立未转 IoU | quick_hyp/ |
| 10-22 | probe1 全量单轮暴露 lr=1e-4 续训发散（33.83→31.87），改 1e-5 | 训练 CE 自身抬升 = lr 病 |
| 10-22 | 补洞 A/B（lr 1e-5）：Δ(A−B)=+0.02 噪声级；8 轮曲线 ep4 峰值 | auto/ |
| 10-22 | **ep4_ema 骨干全量 4818 = 33.94**（补洞 +0.11，连续类 +0.19，动态 +0.04） | eval_best_ep4_full.log |
| 10-22 | 阶段2（冻结骨干+修好未来头）ep2 评测：动态 Δ(M−B档) = −2.33/−1.98/−1.50 | evb_s2_ep2.log |
| 10-22 | SMP 内部体检：车辆位移被基矩阵砍 2~3 倍 → 判定头设计问题，停训重设计 | 本文 §2 |
| 10-22 | git 存档 `a3f2b47..1533864`，实施 PCF 新头 | 用户授权算法创新 |

## 1. 已落地的代码改动与创新（全部实测验证）

| # | 改动 | 位置 | 机制与证据 |
|---|---|---|---|
| 1 | `freeze_perception` | `priorocc_4d.py`（PERCEPTION_PREFIXES + train() 覆写保持 BN eval） | 只训 6.76M/冻结 52.426M；阶段2 ep2 当前帧 34.05 与骨干逐位一致 → 冻结严格生效 |
| 2 | `sep_perception` | `priorocc_4d.py extract_img_feat` | BEV 编码器只吃当前帧 64 通道，历史帧仅供运动分支 → 感知权重 1:1 继承（dry-load 559 键全命中，0 失配） |
| 3 | 逐 horizon flow 上限 | `scmf.py`（`max_flow_cells` 支持列表） | GT 位移 p90 = 8.6/14.1/17.9 cells，全局 5 cells 物理不够 |
| 4 | 非自回归预测 | `scmf.py` `non_autoregressive` | 零训练消融：2s/3s 动态 Δ 由 −0.39/−0.83 翻成 **+0.50/+0.48**；当前帧三臂逐位一致证明消融干净 |
| 5 | gate 初始化偏 warp | `scmf.py` `gate_init_bias=2.0` | 起步 α≈0.88 不低于冻结拷贝 |
| 6 | occ 层补洞省显存重写 | `semantic_continuity.py`（z 分 slab + 稳定子集 softmax） | 与原公式数值等价（absdiff=0），显存 ~1/18 |
| 7 | 洞率指标 | `tools/stage1/hole_rate.py` | 机制级判据：GT 为 c、pred 非 c、六邻域≥3 为 c；补洞全量 1 轮洞率 −18.3% |
| 8 | 权重兼容审计 | `tools/dryload_check.py` | CPU 侧 build_model 逐 key 比形状，热启前 P0 关卡 |
| 9 | 逐类/分组报告与判定器 | `tools/stage1/perclass_report.py`、`ab_judge.py` | 动态(10)/静态(7)/连续类(6)分组 + 两日志 diff |
| 10 | 全自动分支流水线与守护 | `tools/stage1/auto_pipeline.sh` 等 | A/B 判定→分支训练→评测→结论落 `auto_run.log`；偶数轮评测走"暂停→独立脚本→续训"（runner 内建评测与 dict 输出不兼容，`KeyError: 0`） |

## 2. 诊断证据：为什么整个未来头要换（2026-10-22）

### 2.1 阶段2 ep2（600 帧，MODEL vs B档=冻结自身当前帧）
| 口径 | mIoU / 动态 | Δ(M−B) |
|---|---|---|
| 当前帧（冻结骨干） | 34.05 / 27.32 | = 骨干原值 |
| 未来 1s | 17.94 / 8.15 vs 19.05 / 10.48 | **−2.33（动态）** |
| 未来 2s | 14.37 / 5.35 vs 15.02 / 7.33 | **−1.98** |
| 未来 3s | 12.76 / 4.47 vs 13.11 / 5.97 | **−1.50** |

逐类：不动的 barrier **+0.92/+1.51/+1.44**、静态均值 +0.62/+1.24/+1.29（随 horizon 增大 = 自车补偿）；
bus −5.58 / trailer −5.73 / car −2.89（尺寸越大亏越多 = 位移不准的指纹）。

**结论：flow 头学到的是"全局平移（自车补偿）"，不是逐物体位移。**

### 2.2 SMP 内部体检（epoch_1/2_ema 实测，CPU 零成本）
| 语义组 | ep1 对角 | ep2 对角 | 解读 |
|---|---|---|---|
| rigid_vehicle | (0.80, 0.77) | **(0.62, 0.34)** | 车辆位移被砍 38%/66%，还在恶化 |
| rigid_small | (0.95, 0.90) | (0.90, 0.82) | barrier 所在组，基本保留 |
| static | (0.71, 0.60) | (0.49, 0.02) | 压死（设计意图） |

- `alpha` 0.44→0.32；三个正则项原始值≈0：static_flow 已满足（无空间）；**rigid_smooth 奖励"全场一致"，恰是全局平移退化解**；nonrigid_bound 未触发。
- gate 保持 0.86 warp-dominant → 不是"不敢动"，是"动不准 + 被先验缩放"。
- **SMP 提供了"少动少错"的逃逸通道（CE 驱动），这是设计层面的反效果。**

## 3. 文献调研：未来头四家族

| 家族 | 代表 | 位移来源 | 契合"先验指路、不假设位移" |
|---|---|---|---|
| 特征域查询/残差精修 | BeFormer (Cam4DOCC, CVPR24)、FPJ-LSTM、Implicit Residual World Model (2025) | 无显式位移 | 高 |
| 场景流 warp | 本仓库 SCMF；voxel-flow | 学 flow（文献共识：需 flow 监督） | 低（当前病灶） |
| 物体中心分解 | DOME (arXiv 2412.13772) | 动态实例轨迹 + 静态 warp | 高但需检测器 |
| token 自回归 | OccWorld | 离散 token | 中（重） |

仓库积木（关键：**动态先验不需要新感知**）：`dyn_sta_decoder` 的 `dyn_mask/per_cls_masks`、
`SemanticMotionAttention`、`DirectFutureOccupancyHead`、`bev_projector`、`shift_feature`（历史对齐）。

## 4. 新创新：PriorCorrespondenceFlow（PCF，先验门控对应测量流）

**核心思想（对"不假设位移"的工程实现）**：动态目标的位移不从回归"猜"，
而是**从历史帧的特征对应里测出来**（RAFT 式 cost-volume），由 SI 先验 `dyn_mask` 门控，
按 horizon 物理外推；与全局（自车）流相加后走既有 warp/GRU/gate 预测器。

```
v(x)     = soft-argmax_corr(bev_cur, bev_hist_aligned) / lag      # 每像素速度（测量）
flow_k   = flow_scmf(全局/自车分量, 已学好) + g(x)·(v(x)·Δt_k + δ_k)   # g=dyn_mask 门控, δ 零初始化
future_k = Predictor(bev_current, flow_k)                          # 既有非自回归 warp+GRU+gate
```

- 新文件：`projects/mmdet3d_plugin/models/model_utils/prior_correspondence.py`
- 接线：`PriorOcc4D._compute_motion_flow` 在 `flow = self.scmf(...)` 后 `flow = flow + self.prior_corr(...)`
- 配置：`priorocc-4d-r50-pcf.py` = sep-frozen 基座 + `align_after_view_transfromation=True`（历史 BEV 对齐到当前自车系，复用 shift_feature）+ `enable_prior_correspondence=True`
- 参数量：仅温度/残差 δ（每 horizon 2 通道 1×1 conv，零初始化）+ 可选置信投影，<0.1M
- 验证链：合成 blob 单测（已知位移→测出同号同幅）→ 50-iter smoke → ep1 快评动态 Δ 是否立刻收窄（对照旧头 ep2 的 −2.33）
- 风险与备选：相关匹配不稳（特征跨帧不一致）→ 降温/双帧平均/缩小窗口；仍不稳 → 退回 `SemanticDynamicRefiner`（语义门控注意力+残差）

## 5. 验证协议（口径不变量）

- 骨干固定 `work_dirs/ab-on/epoch_4_ema.pth`（4818 全量 33.94），冻结感知不变
- 评测：`eval_b_baseline.py`（600 帧逐偶数轮 + 全量末轮）+ `perclass_report.py`
- 预登记判据：
  1. 动态 Δ(MODEL−B档) 1s/2s/3s **全部转正**（主判据）
  2. 静态/barrier 不得低于 B 档超过 0.2（不丢自车补偿收益）
  3. 当前帧 = 33.94 逐位不动（冻结校验）
- 对照臂：旧 flow 头 ep2 数据（§2.1）已有；D1（DirectFutureOccupancyHead，零位移假设）作下界对照

## 6. 预算（2.18 元/h）

| 项 | 时长 | 费用 |
|---|---|---|
| 实现+单测+smoke | ~2 h | ~4.4 元 |
| PCF 主臂 6 轮+评测 | ~8.5 h | ~23 元 |
| D1 对照臂 | ~8 h | ~41 元（累计） |
| 一轮迭代缓冲 | ~8 h | ~58 元（累计） |
| 论文级（统一协议重跑+阶段3） | +20~30 h | 90~125 元 |

最小闭环（今晚出 PCF 判定）≈ 11 h ≈ 24 元；建议充 60 元。
