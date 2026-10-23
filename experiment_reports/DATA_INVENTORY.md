# PriorOcc-4D 实验数据清点（供写作 agent 引用）

> 用途：写作 agent 查阅精确数字、实验配置、结论归属。**本文不写论文段落**，只列事实。
> 所有数字均来自 `experiment_reports/auto_run.log` 与 `/root/autodl-tmp/dataforpriorocc/auto/evb_*.log` 原始评测输出。
> 最后更新：2026-10-23 17:40

---

## 0. 口径纪律（引用任何数字前必读）

| 维度 | 取值 | 说明 |
|---|---|---|
| 评测集 | **4818**（全量 val） / **600**（stride=8 子集） | **两种口径不可互比**。实测同权重 B 档：600 帧 = 19.05/15.02/13.11，4818 = 18.86/14.88/12.98（子集偏高 0.1~0.2） |
| 权重 | **EMA** / 非 EMA | EMA 稳定高 0.5~2.2。**对外一律报 EMA** |
| 指标 | mIoU（17 类均值） | Cam4DOcc OCFNet 报的是 IoU（≠mIoU），不可直接比 |
| 当前帧 | 4D 模型自身当前帧预测 | 与冻结骨干逐位一致时，证明冻结/未污染 |
| B 档 | 把**模型自己的当前帧预测**当未来预测 | 公平拷贝基线；每臂的 B 档不同（取决于该臂当前帧质量） |
| Δ(M−B) | MODEL − B 档 | 同一帧集内计算，跨臂可比性好于绝对值 |

**评测命令**：`eval_b_baseline.py <ckpt> --cfg <config> [--limit 600]`
**结果解析**：`perclass_report.py <log> --emit --section {0..6}`（0=CURRENT, 1-3=MODEL 1s/2s/3s, 4-6=B档 1s/2s/3s）

---

## 1. 模型与方法组成

基线架构：FlashOcc（LSS/BEVDepth）+ 时序 BEV 编码器（3 帧拼接，首层 192ch）。

### 先验使用链路（SI → dyn_mask/per_cls_masks）

| # | 消费者 | 位置 | 状态 |
|---|---|---|---|
| 1 | **SI（SemanticInjector）** 2D 语义嵌入图像特征 | `semantic_injector` | 全臂启用 |
| 2 | 2D 分割辅助损失 ×3 帧 | `loss_2d_seg_history_{0,1,2}` | 全臂启用 |
| 3 | **SCMF** 语义条件流场（per_cls_masks 条件化） | `scmf.motion_head` | 全臂启用 |
| 4 | **SemanticMotionAttention**（逐类掩码池化 query） | `semantic_attention` | 全臂启用 |
| 5 | **dyn_sta_decoder**（产出 per_cls_masks / dyn_mask） | `dyn_sta_decoder` | 全臂启用 |
| 6 | **PerClassDeltaCombiner**（逐类 delta 组合） | `delta_combiner` | 全臂启用 |
| 7 | **SMP（SemanticMotionPrior）** | `motion_prior` | ⚠️ 热启/PCF+boost 臂启用；**pcf/query/final-p1 臂关闭**（已证有害，见 §5.1） |
| 8 | **PCF（PriorCorrespondenceFlow）** | `prior_corr` | pcf / query / final-p1 启用 |
| 9 | **动态类加权**（loss 层） | `occ_head.dynamic_class_boost` | 仅 ws-fix-dynw（已弃用）/ final Phase2 计划 |

### 关键机制开关（不新增参数、不改 state_dict 形状，可零训练热切）

| 开关 | 取值 | 作用 |
|---|---|---|
| `future_predictor.non_autoregressive` | True/False | True=每个 horizon 都 warp 同一个当前 BEV（消除跨段漂移） |
| `scmf.max_flow_cells` | 5（旧）/ [12,24,36]（修） | 逐 horizon 位移上限（1 cell=0.512m） |
| `future_predictor.gate_init_bias` | 2.0 | gate sigmoid 偏 warp 分支（α≈0.88 起步） |
| `align_after_view_transfromation` | True/False | 历史帧 BEV 先 ego 对齐（PCF 必需） |
| `pcf_align_only` | True/False | 只给 PCF 对齐副本，编码器输入保持未对齐（联训续训防漂移） |
| `sep_perception` | True/False | BEV 编码器只看关键帧（两阶段冻结用） |
| `freeze_perception` | True/False | 冻结感知 52.426M，只训 6.763M |
| `enable_motion_prior` | True/False | SMP 开关 |
| `occ_head.dynamic_class_boost` | 1.0 / 3.0 | 动态类（索引 1..10）CE 权重 ×3 |

---

## 2. 实验臂总表

> 命名规则：work_dir 名即臂名；所有路径在 `/root/autodl-tmp/dataforpriorocc/work_dirs/`

| 臂 | work_dir | 配置 | 起点 | 轮数 | 状态 | 关键结论 |
|---|---|---|---|---|---|---|
| A 从头训 | `smp-sdp-holefill-nodepth` | `priorocc-4d-r50-smp-sdp-holefill-nodepth.py` | 无 | 11（停） | 完成对照 | 无热启=当前帧天花板 24.6，不可用 |
| B 热启联训 | `smp-warmstart` | 同上 | ep21_4dcompat | 20 | **完成（旧SOTA基线）** | 未来 mIoU 最强；SMP+旧SCMF |
| C 两阶段冻结 | `stage2-frozen` | `priorocc-4d-r50-sep-frozen.py` | ab-on/ep4_ema（冻结） | 6 | 完成 ep2 后崩→续 | 当前帧 34.05 完美；动态 Δ −2.33/−1.98/−1.50 |
| D 两阶段+PCF | `pcf-v1` | `priorocc-4d-r50-pcf.py` | ab-on/ep4_ema（冻结） | 6 | **完成（PCF 定量依据）** | 1s 动态 +0.80 vs 两阶段；仍全负 |
| E PCF+boost | `pcf-boost-v1` | `priorocc-4d-r50-pcf-boost.py` | pcf-v1/ep2 | 2（停） | 弃用 | dyn_boost 学到 **负值**（选择抑制） |
| F 两阶段+attention | `query-v1` | `priorocc-4d-r50-query.py` | pcf-v1/ep2 | 2（fallback 停） | 弃用 | **3s 动态首次转正 +0.24**；1s 精度回退 |
| G 多尺度 attention | `query-ms-v1` | `priorocc-4d-r50-query-ms.py` | pcf-v1/ep6 | <1（停） | 未验证 | 设计完成，无 ep2 数据 |
| H 联训+PCF(C变体) | `ws-fix-pcf-v1` | `priorocc-4d-r50-ws-fix-pcf.py` | smp-warmstart/ep14_ema | 2（停） | 回退 | **双重计费**→全线回退 |
| I 动态加权 | `ws-fix-dynw-v1` | `priorocc-4d-r50-ws-fix-dynw.py` | smp-warmstart/ep14_ema | <1（停） | 未验证 | 用户改走新车 |
| J **零训练修复版** | —（复用 B 的权重） | `priorocc-4d-r50-smp-sdp-holefill-fix.py` | smp-warmstart/ep14_ema | **0** | **当前最佳** | 动态 Δ **全正** |
| K 最终模型 Phase1 | `final-p1-v1` | `priorocc-4d-r50-final-p1.py` | ep21_4dcompat | 8（运行中） | **进行中** | 无SMP+PCF+对齐+非AR+cap 干净重训 |

---

## 3. 逐臂详细数据

### A. 从头训（`smp-sdp-holefill-nodepth`）

- 300 子集·非 EMA：ep3 19.47 / ep4 20.11 / ep6 21.85 / ep8 22.77 / ep10 22.92（当前帧）
- 全量 4818·EMA：
  - ep8：24.07 / 18.39 / 15.57 / 13.52
  - ep10：24.64 / 18.92 / 15.90 / 13.77
- 用途：证明「不热启、从头训」的当前帧只能到 ~24.6，比单帧 33.8 差 9 点。

### B. 热启联训（`smp-warmstart`）★旧基线

配置：`priorocc-4d-r50-smp-sdp-holefill-nodepth.py`（**SMP 开、旧SCMF：自回归+cap=5**），lr 1e-4，step=[14]，max_epochs=20。

**300 子集·非 EMA 曲线**（current/f1s/f2s/f3s）：
| ep | current | 1s | 2s | 3s |
|---|---|---|---|---|
| 2 | 28.53 | 18.09 | 14.41 | 12.38 |
| 4 | 29.15 | 19.12 | 15.52 | 13.43 |
| 6 | 29.56 | 19.08 | 15.38 | 13.48 |
| 8 | 30.22 | 20.04 | 16.18 | 14.19 |
| 10 | 29.65 | 19.49 | 15.45 | 13.55 |
| 12 | 30.25 | 20.62 | 16.16 | 14.05 |
| 14 | 29.93 | 20.56 | 16.42 | 14.39 |
| 16 | 31.84 | 21.51 | 16.93 | 14.56 |
| 18 | 31.91 | 21.75 | 17.35 | 15.02 |

**全量 4818·EMA**：
| ckpt | current | 1s | 2s | 3s |
|---|---|---|---|---|
| ep1 | 30.42 | 18.84 | 14.92 | 12.75 |
| ep15 | **32.10** | **21.34** | 17.14 | 14.77 |
| ep16 | 32.05 | 21.31 | 17.07 | 14.68 |
| ep17 | 31.95 | 21.26 | 17.01 | 14.62 |

- 用途：热启有效性证明（ep1 当前帧 30.42 vs 从头 ep10 的 24.64）；
- ep14 后 lr 衰减（1e-4→1e-5）带来 +1.91 跳升（ep16），为课程学习切换点提供依据。

### C. 两阶段冻结（`stage2-frozen`）

- 冻结 ab-on/ep4_ema 感知（52.426M frozen / 6.763M trainable），sep_perception=True（BEV 编码器只看关键帧）
- ep2·600帧·EMA：

| | current | 1s | 2s | 3s |
|---|---|---|---|---|
| MODEL | 34.05 | 17.94 | 14.37 | 12.76 |
| B 档 | — | 19.05 | 15.02 | 13.11 |
| **Δ(M−B)** | — | **−2.33** | **−1.98** | **−1.50** |
| MODEL 动态 | 27.32 | 8.15 | 5.35 | 4.47 |
| B 档动态 | — | 10.48 | 7.33 | 5.97 |

- 用途：证明「冻结强骨干」救不了未来预测（sep_perception 的 1 帧编码器看不到时间）。

### D. 两阶段+PCF（`pcf-v1`）★PCF 定量依据

配置：`priorocc-4d-r50-pcf.py` = 两阶段 + `align_after_view_transfromation=True` + **SMP 关** + PCF 开 + 非AR + cap[12,24,36]。

**ep2·600帧·EMA**：

| | current | 1s | 2s | 3s |
|---|---|---|---|---|
| MODEL | 34.05 | 18.77 | 14.77 | 12.90 |
| MODEL 动态 | 27.32 | **8.95** | 5.67 | 4.40 |
| Δ 动态 | — | −1.53 | −1.66 | −1.57 |

**ep6·全量4818·EMA**（PCF 最终值）：

| | current | 1s | 2s | 3s |
|---|---|---|---|---|
| MODEL | 33.94 | **18.69** | **14.94** | **13.10** |
| B 档 | — | 18.86 | 14.88 | 12.98 |
| **Δ(M−B) 总** | — | −0.17 | **+0.06** | **+0.12** |
| MODEL 动态 | 27.50 | 8.81 | 5.65 | 4.43 |
| B 档动态 | — | 10.26 | 7.05 | 5.76 |
| **Δ 动态** | — | **−1.45** | **−1.40** | **−1.33** |

分组（ep6 全量 Δ）：静态 **+1.65/+2.16/+2.19**；连续类 **+1.93/+2.54/+2.56**；动态 −1.45/−1.40/−1.33。

- 用途①：PCF 对 1s 动态的贡献 **+0.80**（两阶段 −2.33 → PCF −1.53，ep2 同口径）；
- 用途②：warp 路线 6 轮+lr 衰减后动态仍全负 → **结构性天花板 ≈ −1.4**；
- 用途③：静态/连续类大幅正增益，证明 ego 补偿正确。

### E. PCF+boost（`pcf-boost-v1`）✗已废弃

- 在 PCF 上叠加 `enable_dyn_boost=True`（`bev × (1+tanh(boost)×dyn_mask)`），从 pcf-v1/ep2 热启 2 轮。
- ep2·600帧·EMA：MODEL 18.81/14.86/13.02；动态 8.89/5.67/4.45；Δ 动态 −1.59/−1.66/−1.52
- **dyn_boost 实测值：ep1 = −0.106，ep2 = −0.136**（tanh 因子 0.865×）
- 结论：模型主动选择**抑制**动态区特征，与「放大动态特征」意图相反 → 特征幅值放大 ≠ 注意力偏置。

### F. 两阶段+Attention（`query-v1`）★attention 定量依据

配置：`priorocc-4d-r50-query.py`（PCF + `SemanticQueryFutureHead`，`future_predictor=None`）。

- ep2·600帧·EMA：

| | current | 1s | 2s | 3s |
|---|---|---|---|---|
| MODEL | 34.05 | 17.29 | 14.25 | 13.73 |
| MODEL 动态 | 27.32 | 7.58 | 5.49 | **6.21** |
| B 档动态 | — | 10.48 | 7.33 | 5.97 |
| **Δ 动态** | — | **−2.90** | −1.84 | **+0.24** |

- **3s 动态 +0.24 = 全部实验臂中首次 Δ(M−B)>0**
- 1s 回退 −2.90 的归因：注意力在 100×100 下采样上做（防 OOM），三重精度损失（±1m 量化 + 441 点 softmax 涂抹 + 双线性上采样模糊）
- 单测：零初始化恒等/形状/速度偏置方向全 PASS
- 工程数据：向量化 `F.unfold` 后 0.42 s/it、18.1 GB（Python 循环版 2.96 s/it 被否）

### G. 多尺度 attention（`query-ms-v1`）— 未验证

- 配置：`per_horizon_ds=[1,2,2]`、`per_horizon_R=[5,9,9]`（1s 全分辨率保精度、2s/3s 半分辨率保范围、按 (ds,R) 分组共享 unfold）
- 单测全 PASS；dry-load 629/668 命中、0 形状失配
- 实测 0.87 s/it（200×200 unfold 内存带宽翻倍）
- **在 ep2 前被 C 变体取代，无 ep2 数据** → 论文中只能作为设计提案，不能作为结论

### H. 联训+PCF（C 变体，`ws-fix-pcf-v1`）✗回退

- 配置：fix 架构 + PCF + `pcf_align_only=True`，lr 2e-5，从 smp-warmstart/ep14_ema 热启
- ep2·600帧：

| | 零训练基线(ep14+fix) | C变体 EMA | C变体 非EMA |
|---|---|---|---|
| current | **31.96** | 30.00 | 29.45 |
| 1s | **21.55** | 20.70 | 20.64 |
| 2s | **17.65** | 17.11 | 17.15 |
| 3s | **15.23** | 15.00 | 15.08 |

- EMA≈非EMA（差<0.1）→ 非预热问题，真实回退
- 归因：**双重计费**——SCMF 新 cap 下流已放大到正确幅度（H2 消融验证），PCF 又叠加一份测量位移 → 超调
- 教训：PCF 只在 SCMF **从零学**（两阶段）时有效；在已训熟的 SCMF 上叠加会双重计费

### I. 动态加权（`ws-fix-dynw-v1`）— 未验证

- 配置：fix + `occ_head.dynamic_class_boost=3.0`，lr 1e-5，4 轮，从 ep14_ema 热启
- dry-load 653/653 全命中；car:road 梯度比 0.110 → 0.330
- **ep2 前被最终方案（K）取代，无评测数据**

### J. 零训练修复版 ★当前最佳（论文可用数字）

- 权重：`smp-warmstart/epoch_14_ema.pth`（旧架构训出）
- 架构：`priorocc-4d-r50-smp-sdp-holefill-fix.py`（**只切推理开关**：非AR + cap[12,24,36] + gate_bias 2.0）
- **不训练一分钟**

**600帧·EMA**（对比用）：
| | current | 1s | 2s | 3s |
|---|---|---|---|---|
| MODEL | 31.96 | 21.55 | 17.65 | 15.23 |
| B 档 | — | 19.31 | 15.41 | 13.53 |
| Δ 动态(MODEL dyn − B dyn) | — | +0.81 | +0.52 | +0.32 |

**全量4818·EMA**（论文主数字）：
| | current | 1s | 2s | 3s |
|---|---|---|---|---|
| MODEL | 31.88 | **21.33** | **17.64** | **15.09** |
| B 档 | — | 19.15 | 15.34 | 13.41 |
| **Δ(M−B) 总** | — | **+2.18** | **+2.30** | **+1.68** |
| MODEL 动态 | 24.89 | 11.01 | 7.77 | 6.11 |
| B 档动态 | — | 10.21 | 7.17 | 5.85 |
| **Δ 动态** | — | **+0.80** | **+0.60** | **+0.26** |

- **Avg 未来 mIoU = (21.33+17.64+15.09)/3 = 18.02**
- 对比旧架构 ep15_ema（同全量口径）：1s 21.34→21.33（持平）、2s 17.14→**17.64（+0.50）**、3s 14.77→**15.09（+0.32）**、Avg 17.75→**18.02（+0.27）**
- **三个 horizon 动态 Δ 首次全部转正**

### K. 最终模型 Phase 1（`final-p1-v1`）— 运行中

- 配置：`priorocc-4d-r50-final-p1.py`
  - **无 SMP**（`enable_motion_prior=False`）
  - **PCF 开** + `align_after_view_transfromation=True`（新鲜训练，编码器从第 1 轮学对齐特征，无漂移）
  - 非AR + cap[12,24,36] + gate_bias 2.0
  - 标准 loss（无动态加权）
  - lr 1e-4，step=[8]（ep8 衰减到 1e-5），max_epochs=8
- 起点：`ckpts/priorocc-singleframe-ep21_4dcompat.pth`
- dry-load：557/649 感知键命中；92 missing 全为运动分支（随机初始化）；0 形状失配；0 unexpected
- 实测：1.09 s/it，19.9 GB；iter50→150 loss 5.34→2.17
- 计划 Phase 2：ep8-14，动态加权 3×，lr 1e-5
- **暂无评测数据**

---

## 4. 消融实验（可直接进论文消融表）

### 4.1 H2 零训练架构消融（ep15_ema，600 帧，只切开关）

| 配置 | 动态 1s | 动态 2s | 动态 3s | Δ(M−B) 2s/3s |
|---|---|---|---|---|
| 旧（cap5 + 自回归） | 10.98 | 6.69 | 4.99 | −0.39 / −0.83 |
| 只开非自回归（cap5） | 10.98 | 7.58 | 6.30 | +0.58 / +0.47 |
| 只放大 cap（仍自回归） | 11.03 | 6.60 | 4.90 | −0.41 / −0.89 |
| **两者齐开** | 11.03 | **7.66** | **6.29** | **+0.50 / +0.48** |

- 三臂 CURRENT 逐位一致（32.07）→ 消融干净（改动只作用于未来分支）
- 结论：**2s/3s 动态转正完全来自非自回归**；单独放大 cap 在未重训时无效甚至微负

### 4.2 零训练修复 vs 旧架构（全量 4818·EMA）

| 版本 | current | 1s | 2s | 3s | Avg |
|---|---|---|---|---|---|
| 旧架构 ep15 | 32.10 | 21.34 | 17.14 | 14.77 | 17.75 |
| **修复版 ep14** | 31.88 | 21.33 | **17.64** | **15.09** | **18.02** |
| 增量 | −0.22* | −0.01 | +0.50 | +0.32 | +0.27 |

\* current 差来自 ckpt 不同（ep14 vs ep15），非架构改动所致。

### 4.3 PCF 增益（同为两阶段冻结，600 帧 ep2）

| 版本 | 1s 动态 | 2s 动态 | 3s 动态 |
|---|---|---|---|
| 两阶段（无 PCF） | 8.15 | 5.35 | 4.47 |
| **两阶段+PCF** | **8.95** | 5.67 | 4.40 |
| 增量 | **+0.80** | +0.32 | −0.07 |

### 4.4 attention vs warp（600 帧 ep2，同一冻结骨干）

| 头 | 1s 动态 | 2s 动态 | 3s 动态 |
|---|---|---|---|
| warp + PCF | **8.95** | 5.67 | 4.40 |
| attention（100×100, R=10） | 7.58 | 5.49 | **6.21** |
| 差 | **−2.90** | −0.18 | **+0.24** |

- 结论：attention 的优势随 horizon 单调增长（−2.90 → −0.18 → +0.24），与「位移越大 warp 越失效」一致

### 4.5 动态特征放大（PCF vs PCF+boost，600 帧 ep2）

| 版本 | 1s 动态 | dyn_boost 值 |
|---|---|---|
| PCF | **8.95** | — |
| PCF+boost | 8.89 | **−0.136**（学到抑制） |

逐类 Δ(M−B) 1s 变化：barrier +1.99→+2.40；trailer −5.22→**−6.18**；bus −4.37→−4.79

---

## 5. 诊断证据（用于解释机制失效原因）

### 5.1 SMP 有害的三项证据

1. **前向基矩阵收缩位移**（纯 CPU 拆 ckpt，ep1→ep2）：
   - rigid_vehicle 对角：**(0.80, 0.77) → (0.62, 0.34)**（车辆位移被压到 34~62%）
   - rigid_small：(0.95,0.90)→(0.90,0.82)；static：(0.71,0.60)→(0.49,0.02)
   - 混合系数 α=sigmoid(alpha_logit)：0.44→0.32
2. **rigid_smooth 奖励退化解**：三项正则原始值 static_flow≈0.004 cell、rigid_smooth≈0、nonrigid_bound≈0（hinge 未触发）。rigid_smooth 对「纯全局平移」恰取最小值 → 模型住进退化解
3. **逐类 IoU 指纹**（stage2 ep2 动态 Δ）：barrier +0.92（物理不动却涨）、静态 7 类均值 +0.62、而 bus −5.58 / trailer −5.73 / car −2.89 → 「不动的普涨、大快的重亏」= 位移不准的签名

### 5.2 类别不平衡量化（动态加权依据）

- 体素比：静态合计 ≈24 亿 : 动态 10 类合计 ≈3200 万 = **75:1**
- `class_balance=True` 的 1/log 频率权重太温和：
  - 路面：权重 0.0527 × 1.76 亿 = **930 万**梯度贡献
  - 自行车：权重 0.0838 × 15 万 = **1.3 万** → 差 **~700 倍**
  - car（动态最大类）：0.0601 × 1700 万 = 102 万 → car:road = **0.110**
- 3× 加权后：car:road = **0.330**（设计目标达成）
- 小类欠学证据：**bicycle 1s IoU = 1.57、motorcycle = 0.61**（几乎为零，从未获得有效学习信号）

### 5.3 backward warp 结构限制

- 数学形式：`output(B) = input(B − flow(B))`，需要**目标位置 B** 的流指向源 A
- 但流头在 B 处没有物体信号（车在 A）→ 流不可靠 → 大位移目标取到背景
- 实测指纹：位移越大亏损越大（1s −1.45 → 3s −1.33 但绝对动态 IoU 递减 8.81→4.43），trailer/bus 最严重
- 该限制与训练量无关：PCF 6 轮 + lr 衰减后仍 −1.33/+ 动态全负

---

## 6. 与公开 SOTA 对照

> ⚠️ 我们的数字（4818 val，本仓库协议）与下表的 Cam4DOcc 协议**未严格同协议重跑**，仅作定位参考。

| 方法 | 会议 | 输入 | 1s | 2s | 3s | Avg |
|---|---|---|---|---|---|---|
| T3Former-O | 2025 | GT Occ | 46.32 | 33.23 | 28.73 | 36.09 |
| OccWorld | ECCV24 | GT Occ | 25.78 | 15.14 | 10.51 | 17.14 |
| FSF-Net | 2024 | GT Occ | 42.38 | — | 17.03 | — |
| T3Former-F | 2025 | Camera | 19.60 | — | — | — |
| Cam4DOcc OCFNet | CVPR24 | Camera | 29.36* | 28.30* | 27.44* | 26.82* |
| DOME | 2024 | Camera | 35.11 | 25.89 | 20.29 | 27.10 |
| Drive-OccWorld | AAAI25 | Camera | 36.3** | — | — | — |
| **PriorOcc-4D（零训练修复版）** | — | Camera | **21.33** | **17.64** | **15.09** | **18.02** |

\* IoU 口径，与 mIoU 不可混比　\** mIoU_f
- 我们超过：T3Former-F（1s +1.73）、OccWorld Avg（+0.88）
- 我们落后：DOME 约 9 点、Drive-OccWorld 更多
- 时间衰减保持力：3s/1s = 15.09/21.33 = **0.71**（DOME 0.58、OccWorld 0.41）→ 相对亮点

---

## 7. 废弃机制登记（论文「设计取舍」素材）

| 机制 | 实现位置 | 废弃证据 | 可写进论文的表述 |
|---|---|---|---|
| 自回归未来推演 | `future_predictor`（默认 AR） | H2 消融：非AR 单独把 2s/3s 动态 Δ 从 −0.39/−0.83 翻成 +0.58/+0.47 | 「自回归导致跨段误差累积」 |
| 全局 flow cap=5 | `scmf.max_flow_cells` | GT 位移 p90 = 8.6/14.1/17.9 cells，5 cells=2.56m 物理不足 | 「上限小于真实位移分布」 |
| SMP 语义运动先验 | `motion_prior` | 基矩阵收缩车辆位移至 34%；rigid_smooth 奖励全局平移退化解；逐类 IoU 指纹 | 「平滑先验与逐物体位移互斥」 |
| 动态特征放大 dyn_boost | `enable_dyn_boost` | 学到 −0.136（选择抑制）；动态零增益 | 「特征幅值≠注意力，GRU 按内容处理」 |
| 两阶段冻结感知 | `freeze_perception`+`sep_perception` | 当前帧 34.05 完美，但未来 mIoU 落后联训 2.65/2.20/1.67 | 「单帧编码器看不到时间；时序 BEV 是未来预测的关键」（同时说明两阶段作为**组件验证实验台**的价值） |
| 硬门控流场（dyn_mask × PCF 输出） | PCF `gate_mode` | 合成单测：目标位置流被清零 → 取到背景而非车 | 「backward warp 的流需在目标位置非零，源位置掩码不适用」 |
| PCF 直接叠加到已训 SCMF | C 变体 | ep2 全线回退（current −1.96）；双重计费 | 「measured displacement 与已放大流叠加会超调；PCF 需 SCMF 从零学」 |
| 下采样单尺度 attention | query-v1 | 1s 动态 −2.90（三重精度损失） | 「注意力的精度/范围权衡」 |

---

## 8. 进行中与后续

| 项 | 状态 | 判决线 |
|---|---|---|
| final-p1-v1（Phase 1，无SMP+PCF 干净重训，ep1-8） | **运行中**（Epoch[1]，1.09 s/it，ETA ~29.6h） | ep8 全量 vs 零训练基线 **Avg 18.02** |
| Phase 2（ep8-14，动态加权 3×） | 计划 | 动态 Δ 再抬 → 课程学习成立 |
| 多尺度 attention（query-ms） | 未验证 | 仅在论文作为设计提案 |

**最终模型配置（K）包含**：ep21 热启 + 无SMP + PCF + ego对齐 + 非AR + cap[12,24,36] +（Phase2）动态加权3×

---

## 9. 文件与日志索引

| 内容 | 路径 |
|---|---|
| 全部决策与评测落盘 | `/root/priorocc/PriorOcc/experiment_reports/auto_run.log` |
| 训练日志 | `/root/autodl-tmp/dataforpriorocc/auto/train_*.log` |
| 评测原始输出 | `/root/autodl-tmp/dataforpriorocc/auto/evb_*.log` |
| 配置目录 | `/root/priorocc/PriorOcc/projects/configs/priorocc/` |
| 权重目录 | `/root/autodl-tmp/dataforpriorocc/work_dirs/<臂名>/` |
| 单帧热启权重 | `/root/priorocc/PriorOcc/ckpts/priorocc-singleframe-ep21_4dcompat.pth` |
| 评测脚本 | `/root/autodl-tmp/dataforpriorocc/eval_b_baseline.py` |
| 逐类解析 | `tools/stage1/perclass_report.py` |
| 洞率指标 | `tools/stage1/hole_rate.py` |
| 设计文档（早期） | `experiment_reports/future_head_redesign.md` |
| SOTA 参照表 | `nextstep.md` L17-50 |
