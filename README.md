# PriorOcc-4D：语义先验时空化驱动的 4D 占用预测

**PriorOcc-4D** 在单帧 [PriorOcc](#项目脉络) / [FlashOCC](https://github.com/Yzichen/FlashOCC) 基础上，
把 2D 语义先验从"单帧空间注入"升级为"全链路时空驱动"，实现 **4D 占用预测（4D Occupancy
Forecasting）**：给定历史帧观测，预测未来 1s / 2s / 3s 的 3D 语义占用网格。

> **一句话原理**：语义类别先验不仅回答"*那里有什么*"，还能回答"*它将怎么动*"（约束运动学习空间）、
> "*它有多远*"（约束深度学习空间），并用"*背景是连续的*"这一先验补齐占用空洞。

**配套文档**
- 研究方案（SOTA 对比、故事线、路线图）：[`nextstep.md`](nextstep.md)
- 三项创新的方法与公式（技术细节）：[`PRIOROCC_4D_INNOVATIONS_README.md`](PRIOROCC_4D_INNOVATIONS_README.md)
- 启动/数据/训练（操作手册）：[`PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md`](PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md)、[`PRIOROCC_4D_LAUNCH_GUIDE.md`](PRIOROCC_4D_LAUNCH_GUIDE.md)
- 开发计划：[`PRIOROCC_4D_DEVELOPMENT_PLAN.md`](PRIOROCC_4D_DEVELOPMENT_PLAN.md)

---

## 目录
- [项目脉络](#项目脉络)
- [1. 核心原理](#1-核心原理)
- [2. 整体架构与数据流](#2-整体架构与数据流)
- [3. 关键模块详解（位置 / 输入输出 / 方法 / 公式 / 动机）](#3-关键模块详解)
- [4. 损失函数总表](#4-损失函数总表)
- [5. 训练配方](#5-训练配方)
- [6. 安装](#6-安装)
- [7. 数据准备](#7-数据准备)
- [8. 训练与消融](#8-训练与消融)
- [9. 训练前最小链路验证](#9-训练前最小链路验证)
- [10. 评测指标与协议](#10-评测指标与协议)
- [11. 数值稳定性与向后兼容](#11-数值稳定性与向后兼容)
- [12. 代码结构与文件索引](#12-代码结构与文件索引)
- [13. 致谢](#13-致谢)

---

## 项目脉络

| 阶段 | 任务 | 核心贡献 | 指标 |
|---|---|---|---|
| **FlashOCC** | 单帧 3D 占用 | Channel-to-Height (C2H) 高效解码，BEV→3D 无需 3D 卷积 | — |
| **PriorOcc** | 单帧 3D 占用 | SemanticInjector：backbone 后注入 2D 语义先验 + 2D 辅助监督 | 单帧 mIoU **32.08** |
| **PriorOcc-4D**（本仓库） | **4D 占用预测** | 语义先验时空化：语义引导动静解耦 + 语义条件化运动场(SCMF) + **语义运动先验(SMP)** + **语义深度先验(SDP)** + **语义连续性补洞** | 当前帧继承 PriorOcc；另报告未来 1/2/3s mIoU 与 Avg |

**赛道**：相机端到端 4D 占用预测，对标 Cam4DOcc(OCFNet)、Drive-OccWorld、DOME、T3Former-F 等
（详见 `nextstep.md` 的 SOTA 对比表）。

---

## 1. 核心原理

PriorOcc-4D 的全部设计围绕一个中心命题：**2D 语义先验是 4D 预测的核心驱动力**，而非辅助模块。
语义先验在五个层面被显式利用：

**① 语义 → 运动学习空间约束（创新 SMP）**
不同类别的运动模式天然不同：车辆是刚体、沿朝向相干平移；行人是非刚性、速度有界；建筑/地面静止。
若让网络在**无约束的 flow 空间**里自由学习，静态区易产生虚假运动、刚体易碎裂。SMP 用**每类可学习
运动基**把 flow 约束到各类别合理的运动子空间，并用三个正则损失（静态≈0 / 刚体相干 / 非刚性有界）
显式编码这一先验。

**② 语义 → 深度学习空间约束（创新 SDP，SGDM 升级）**
单目深度本质病态，是 2D→3D 投影噪声与占用空洞的根源。语义类别蕴含强深度先验（地面近、建筑/天空远）。
SDP 学习一张"每类深度分布表"，按语义概率混合后调制深度 logits，用类别先验正则化病态深度，
从源头减少投影空洞。它与原 SGDM（软前景门控）互补：SGDM 重塑特征，SDP 直接约束深度分布。

**③ 运动场是语义特征的函数（SCMF）**
运动场不是独立的运动建模模块，而是 `f(BEV特征, 运动差分特征, 语义概率, 注意力特征)` 的函数输出。
语义概率作为**必要条件输入**驱动运动场生成——运动是语义理解的直接推论。

**④ 背景连续性 → 补洞（创新 连续性 + warp 有效性门控）**
天空、地面等背景类空间连续、内部无洞。两处利用：
(a) **warp 有效性门控**——运动 warp 的 disocclusion 区域会被 `grid_sample` 零填充成空洞，用有效性掩码
喂入门控，让空洞自动回退到时序残差（对静态背景，上一帧即正确答案）；
(b) **语义连续性 TV 正则**——对背景类概率场做置信度加权全变分平滑，抑制孤立空洞。

**⑤ SemanticInjector 全链路驱动**
同一套 2D 语义先验（`seg_logits`）贯穿：SGDM/SDP 深度门控 → BEV 语义投影 → 动静分离 → 运动编码 →
注意力 query → 运动场条件 → SMP 运动基 → 连续性/时序一致性监督。语义先验在每个环节都有可量化贡献
（对应 `nextstep.md` 的 D1–D6 消融）。

---

## 2. 整体架构与数据流

### 2.1 结构总览

```
历史 3 帧 (t, t-1, t-2)  ── 每帧独立处理 ──────────────────────────────┐
  imgs (B,6,3,256,704)                                                  │
    └─ Backbone(R50)+FPN ─► (B,6,256,16,44)                             │
         └─ SemanticInjector ─► 融合特征 (B·6,256,16,44) + seg_logits (B·6,17,16,44)
              └─ ViewTransformer(LSS):                                   │
                   DepthNet[ SGDM 门控 + SDP 深度先验(创新③) ] ─► depth (B·6,88,16,44)
                   深度加权 voxel pooling ─► 每帧 BEV (B,64,200,200)     │
                                                                         ▼
  3 帧 BEV concat ─► (B,192,200,200) ─► BEVEncoder ─► Fused BEV (B,256,200,200)
                                                                         │
  seg_logits(key)+depth ─► SemanticBEVProjector ─► semantic_bev (B,17,200,200) + visibility (B,1,200,200)
                                          │
             SemanticDynStaSeparator ─► dyn_mask / sta_mask (B,1,200,200), per_cls_masks (B,17,200,200)
             SemanticMotionFeatureEncoder(增强1) ─► motion_feat (B,256,200,200)
             SemanticMotionAttention(增强2)      ─► attn_feat   (B,256,200,200)
             PerClassDeltaCombiner(增强3)        ─► Fused BEV += delta
                                          │
             SCMF 语义条件化运动场 ─► flow (B,3,2,200,200)
             SemanticMotionPrior(SMP, 创新①) ─► 约束后的 flow (B,3,2,200,200)
                                          │
             SCMFEnhancedPredictor: warp(有效性门控, 创新②) + ConvGRU 残差 + 门控融合
                                          │
             未来 BEV (B,3,256,200,200) ─► BEVOCCHead2D(C2H) ─► 未来占用 (B,200,200,16,18) ×3

  并行监督: loss_2d_seg(逐帧) · loss_depth(含SDP) · loss_occ(当前) · loss_occ_future(1/2/3s)
            · loss_sem_consistency(时序一致) · loss_static/rigid/nonrigid(SMP, 创新①)
            · loss_sem_continuity_2d/_bev(创新②补洞)
```

### 2.2 一次训练前向的张量流（关键形状）

| 阶段 | 张量 | 形状 |
|---|---|---|
| 多视图图像（每帧） | `imgs` | `(B, 6, 3, 256, 704)` |
| Backbone+FPN 特征 | `F_img` | `(B, 6, 256, 16, 44)` |
| 2D 语义 logits | `seg_logits` | `(B·6, 17, 16, 44)` |
| 深度分布（含 SDP） | `depth` | `(B·6, 88, 16, 44)` |
| 每帧 BEV | `bev_t` | `(B, 64, 200, 200)` |
| 融合 BEV | `fused_bev` | `(B, 256, 200, 200)` |
| BEV 语义概率 | `semantic_bev` | `(B, 17, 200, 200)` |
| 运动场（SMP 后） | `flow` | `(B, 3, 2, 200, 200)` |
| 未来 BEV | `future_bevs` | `(B, 3, 256, 200, 200)` |
| 未来占用 logits | `occ` | `(B, 200, 200, 16, 18)` ×3 |

（BEV 网格 200×200×16 @ 0.4m，x/y∈[-40,40]，z∈[-1,5.4]；`Ncams=6`，历史 3 帧，未来 3 步。）

---

## 3. 关键模块详解

记号：`φ`=运动场 `(B,T,2,H,W)`；`m`=每类语义概率 `(B,C,H,W)`；`C=17`；`T=3`；`D=88`；`H=W=200`；`ε=1e-6`。

### 3.1 SemanticInjector（2D 语义先验注入，继承 PriorOcc）
- **位置**：backbone/FPN 之后、view transformer 之前（逐帧）。
- **IO**：`F_img (B·N,256,16,44)` → `(F_enh (B·N,256,16,44), seg_logits (B·N,17,16,44))`。
- **方法/公式**：
  ```
  S_logits = SegHead(F_img)                       # Conv3x3→BN→ReLU→Conv1x1(→17)
  L_2d     = CrossEntropy(S_logits, Y_gt)         # ignore_index=255, weight 0.3
  F_enh    = Fusion([F_img, S_logits])            # Conv1x1(256+17→256)→BN→ReLU
  ```
- **动机**：用密集 2D 语义显式监督 backbone，并把语义回注特征，是全部下游语义先验的源头。

### 3.2 SGDM + SDP（语义门控深度 + 语义深度先验，创新③）
- **位置**：`DepthNet.forward` 内；SGDM 在 `reduce_conv` 后，SDP 在 `depth_conv` 后、`cat` 前。
- **SGDM（软前景门控，继承）**：
  ```
  fg_prob = Σ_{c<11} softmax(S_logits)_c                 # 前景概率
  gated   = F_img · (1 + fg_prob · Attn_SE([F_img, sem_proj(softmax S)]))
  ```
- **SDP（新增，表 `Θ∈R^{17×88}` 零初始化，~1.5K 参数）**：
  ```
  P_c = softmax(Θ_c / T_sdp)                             # 每类深度分布
  σ   = softmax(S_logits)                                # (BN,17,fH,fW)
  π   = einsum('bchw,cd->bdhw', σ, P)                    # (BN,88,fH,fW), Σ_d π=1
  depth_logits ← depth_logits + λ_sdp · clamp(log(π+ε), min=-10)
  ```
- **恒等且可学（关键性质）**：`Θ=0 ⇒ π` 均匀 ⇒ `log π` 对 d 为常数 ⇒ 对 `softmax_d(depth)` **值不变**
  （初始不扰动 baseline）；但对 `Θ` 的**梯度非零**：`∂L/∂Θ_{c,d} = (λ σ_c / T)·∂L/∂z_d ≠ 0`，故能学习。
- **动机**：用类别深度先验正则化病态深度，降低投影噪声/空洞；与 SGDM 特征门控互补。

### 3.3 时序 BEV 融合
- **位置**：3 帧各自过 view transformer 得 BEV，`align_after_view_transfromation` 下用 ego-motion 对齐。
- **方法**：`Concat[bev_t, bev_{t-1}', bev_{t-2}'] (B,192,200,200)` → `BEVEncoder(CustomResNet)` → `(B,256,200,200)`。
- **动机**：复用 FlashOCC/BEVDet4D 原生时序能力，零成本获得多帧运动线索。

### 3.4 SemanticBEVProjector + 动静分离
- **SemanticBEVProjector**：复用 view transformer 的 `get_ego_coor`+`voxel_pooling_v2`，把
  `softmax(seg_logits)×depth_prob` 抬升到 BEV，得 `semantic_bev (B,17,200,200)`，并以深度质量归一化得
  `visibility (B,1,200,200)`（后续作置信度/掩码）。
- **SemanticDynStaSeparator**：`dyn_mask=Σ_{c∈dyn} m_c`、`sta_mask=Σ_{c∈sta} m_c`、`per_cls_masks=m`
  （dyn=类 0–10，sta=类 11–16）。为 SMP / 注意力 / 连续性提供逐类掩码。

### 3.5 语义增强 1–3（SCMF 的前置支撑）
- **增强1 SemanticMotionFeatureEncoder**：`[Δ1, Δ2, accel, sem_proj(semantic_bev)]` → `motion_feat (B,256,·,·)`，
  其中 `Δ1=bev_{t-1}-bev_{t-2}`、`Δ2=bev_t-bev_{t-1}`、`accel=Δ2-Δ1`。让运动编码知道"什么类别在动"。
- **增强2 SemanticMotionAttention**：以逐类掩码从 BEV 加权池化出 per-class query，BEV 作 K/V，
  motion_feat 提供加性 bias，输出 `attn_feat (B,256,·,·)`。让语义类别驱动注意力推理。
- **增强3 PerClassDeltaCombiner**：`cls_to_motion(motion_to_cls(motion_feat)·masks)` 得逐类残差，
  `fused_bev += delta`。细粒度语义参与特征组合（含 `binary_mask` 开关支持消融 D3）。

### 3.6 SCMF 语义条件化运动场（核心）
- **IO**：`[fused_bev, motion_feat, attn_feat, semantic_bev]` → `flow φ (B,3,2,200,200)`。
- **方法/公式**：
  ```
  x     = Conv→BN→ReLU→Conv→BN→ReLU ( Concat[4 条件] )
  φ_raw = tanh( MotionHead_1x1(x) ) · max_flow_cells        # max_flow_cells=5.0
  ```
- **恒等初始化**：`MotionHead` 权重/偏置零初始化 ⇒ 初始 `φ=0`（恒等 warp），训练从"未来≈当前"稳定起步。
- **动机**：语义作为必要条件输入，运动场是语义理解的函数输出（原理③）。

### 3.7 SMP 语义运动先验（创新①：刚体/非刚性运动学习空间约束）
- **位置**：`priorocc_4d.py::_compute_motion_flow`，SCMF 产出 flow 之后。
- **类别分组**：

  | 组 | 类别 id | 运动先验 |
  |---|---|---|
  | `rigid_vehicle` | 0,1,2,3,4（car/truck/construction/bus/trailer） | 刚体、相干平移 |
  | `rigid_small` | 5,6,7,9（barrier/motorcycle/bicycle/traffic_cone） | 近刚体、小幅度 |
  | `nonrigid_ped` | 8（pedestrian） | 非刚性、幅度有界 |
  | `static` | 10–16（others + 背景 6 类） | flow ≈ 0 |

- **前向公式**：
  ```
  p_g   = Σ_{c∈g} m_c                             # einsum('bchw,gc->bghw', m, M)
  p̃_g  = p_g / (Σ_g p_g + ε)                      # 归一化 → 保证 init 恒等
  A_mix = einsum('bghw,gij->bijhw', p̃, A)         # A_g 每类运动基(2×2, init I), (B,2,2,H,W)
  trans[b,t,i] = Σ_j A_mix[b,i,j]·φ[b,t,j]
  φ_out = (1-α)·φ + α·trans,  α = sigmoid(alpha_logit)
  ```
  `basis_mode='per_group'`（每组独立 2×2）或 `'shared_modes'`（K 个共享基按组权重组合）。参数量 ≤ ~33。
- **正则损失**（补齐原方案缺失的 `L_motion_reg`）：
  ```
  L_static   = Σ p_sta·(|φx|+|φy|) / (T·Σ p_sta + ε)                              # 静态区不动
  L_rigid    = (Σ wx·|Δx φ| + Σ wy·|Δy φ|) / (2T·(Σwx+Σwy)+ε), wx=min(p_rig 邻域)  # 刚体相干(加权TV)
  L_nonrigid = Σ p_ped·relu(√(φx²+φy²+ε) − τ)² / (T·Σ p_ped + ε), τ=2.0            # 行人有界(hinge)
  ```
- **恒等性**：`A_g=I` ⇒ `φ_out=φ`（初始不扰动 SCMF）。训练中 `A_static→0`（杀静态 flow）、
  `A_vehicle→`相干平移、`A_ped→`有界——**每类被限制在其合理运动子空间**。

### 3.8 warp 有效性门控 + ConvGRU 融合（创新②：补洞）
- **位置**：`SCMFEnhancedPredictor`（自回归预测未来 BEV）。
- **空洞来源**：`warp_feature` 用 `grid_sample(padding_mode='zeros')`，disocclusion 越界处被填 0。
- **方法/公式**：
  ```
  validity = 1{ sample_grid_x∈[-1,1] 且 sample_grid_y∈[-1,1] }      # (B,1,H,W), 空洞=0
  coarse_k = warp(prev, φ_k)                                        # 粗预测(可能含空洞)
  delta_k  = ConvGRU(prev, h_{k-1})                                 # 数据驱动残差
  α_k      = gate([prev, coarse_k, delta_k, validity])              # 空洞处学 α→0
  future_k = α_k·coarse_k + (1-α_k)·(prev + delta_k)               # 空洞→时序残差
  prev ← future_k                                                   # 自回归链
  ```
- **动机**：空洞处自动回退到 `prev+delta`；对静态连续背景，上一帧即正确答案 ⇒ 空洞被自然填补。
- **恒等/兼容**：`use_warp_validity=False`（默认）时门控输入维度不变，baseline 逐位一致。

### 3.9 语义连续性补洞（创新②）
- **位置**：`forward_train` 损失项，作用于 `seg_logits`(2D)、`semantic_bev`(BEV)、可选 `occ_logits`(3D)。
- **连续类**：`Cont=[11..16]`（driveable_surface/other_flat/sidewalk/terrain/manmade/vegetation）。
- **方法/公式**（置信度加权 TV，权重 detach 防"把置信度压 0"的退化解；无可学习参数）：
  ```
  通用: L_TV = (Σ wx·|Δx q| + Σ wy·|Δy q|) / (K·(Σwx+Σwy)+ε),  wx=min(w 邻域).detach()
  2D : q=softmax(seg_logits),     w=Σ_{c∈Cont} q_c                → loss_sem_continuity_2d
  BEV: s=semantic_bev,            w=visibility·Σ_{c∈Cont} s_c      → loss_sem_continuity_bev
  OCC: po=softmax(occ_logits,-1), w=mask_camera·Σ_{c∈Cont} po_c    → loss_sem_continuity_occ (对 Dx,Dy; 默认关)
  ```
- **动机**：背景连续、内部无洞 ⇒ 平滑其概率场可抑制稀疏深度/warp 造成的孤立空洞。梯度回流到
  seg_head/depth_net/occ_head。

### 3.10 语义时序一致性（自监督）
- **方法**：每帧 `seg_logits` 在"高置信 ∩ 静态 ∩ 双帧可见"区域应一致，用对称 KL 约束：
  ```
  L_consist = SymKL( softmax(S_t) ‖ softmax(S_{t-1}) )  在有效掩码区域（历史帧 stop-gradient）
  ```
- **动机**：为时序融合提供额外自监督，无需额外标注。

### 3.11 未来占用头（C2H）
- 复用 `BEVOCCHead2D`（Channel-to-Height）把未来 BEV `(B,256,200,200)` 解码为 `(B,200,200,16,18)`，
  保持 FlashOCC 效率（无 3D 反卷积）。当前帧与 3 个未来步共享同一 occ head。

---

## 4. 损失函数总表

```
L_total = L_depth + L_occ(current) + Σ_{k=1..3} w_k·L_occ_future(k) + Σ_{i=0..2} L_2d_seg(frame_i)
        + L_sem_consistency + L_static + L_rigid + L_nonrigid
        + L_sem_continuity_2d + L_sem_continuity_bev (+ L_sem_continuity_occ)
```

| 损失键 | 含义 | 默认权重 | 来源模块 |
|---|---|---|---|
| `loss_depth` | 深度监督（含 SDP 调制） | 3.0 | ViewTransformer |
| `loss_occ` | 当前帧 3D 占用 CE + class balance | 1.0 | BEVOCCHead2D |
| `loss_occ_future_{1,2,3}s` | 未来各步占用 CE（时间衰减） | 1.0 / 0.7 / 0.5 | BEVOCCHead2D |
| `loss_2d_seg_history_{0,1,2}` | 逐帧 2D 语义 CE | 0.3 each | SemanticInjector |
| `loss_sem_consistency` | 语义时序一致性 KL | 0.05 | SemConsistencyLoss |
| `loss_static_flow` | 静态区 flow≈0（SMP，创新①） | 0.05 | SemanticMotionPrior |
| `loss_rigid_smooth` | 刚体 flow 相干 TV（SMP，创新①） | 0.02 | SemanticMotionPrior |
| `loss_nonrigid_bound` | 行人 flow 有界 hinge（SMP，创新①） | 0.02 | SemanticMotionPrior |
| `loss_sem_continuity_2d` | 2D 语义连续性 TV（创新②） | 0.05 | SemanticContinuityLoss |
| `loss_sem_continuity_bev` | BEV 语义连续性 TV（创新②） | 0.05 | SemanticContinuityLoss |
| `loss_sem_continuity_occ` | occ 连续性 TV（默认关） | 0.0 | SemanticContinuityLoss |
| `loss_future_semantic_*` | 未来语义预测（dormant，默认关） | 0.1 | FutureSemanticPredictor |

---

## 5. 训练配方

| 项 | 值 |
|---|---|
| 优化器 | AdamW, `lr=1e-4`, `weight_decay=1e-2` |
| 梯度裁剪 | `max_norm=5, norm_type=2` |
| 调度 | `EpochBasedRunner`, `max_epochs=30`（`lr_config` 见 config） |
| batch | `samples_per_gpu=4`, `workers_per_gpu=4` |
| 输入 | 6 相机 × 3 历史帧，`256×704` |
| 评测 | `interval=2, start=2` |
| 预训练 | `ckpts/bevdet-r50-cbgs.pth` |
| 参数量 | ~59.5M（新配置，含 SMP/SDP） |

---

## 6. 安装

参考 [FlashOCC 安装指南](doc/install.md)。验证环境：
```
conda activate flash     # Python 3.9, PyTorch 1.10+cu111, mmcv 1.5.3, mmdet3d 1.0.0rc4
pip install transformers # 2D 伪标签生成(SegFormer)
```
> **GPU 注意**：PyTorch 1.10+cu111 仅支持到 sm_86。若在 sm_120（如 RTX 5060）等新卡上，
> `torch.cuda` 不可用，需换兼容 GPU（A100/3090/V100 等）或升级 PyTorch；验证工具会自动回退 CPU。

---

## 7. 数据准备

| 数据 | 路径 | 来源 |
|---|---|---|
| nuScenes 图像 | `data/nuscenes/samples/`, `sweeps/` | nuScenes v1.0-trainval |
| 3D 占用 GT | `data/nuscenes/gts/{scene}/{token}/labels.npz`（含 `semantics/mask_lidar/mask_camera`） | 外部 CVPR2023-3D-Occupancy-Prediction（Occ3D 风格） |
| 2D 语义伪标签 | `data/nuscenes/seg_2d_labels/samples/CAM_*/*.png` | `tools/generate_2d_seg_labels.py`（SegFormer-B2 cityscapes） |
| BEVDet info | `bevdetv2-nuscenes_infos_{train,val}.pkl` | `tools/create_data_bevdet.py` |
| **4D 预测 info** | `bevdetv2-nuscenes_infos_{train,val}_forecast.pkl` | `tools/create_4d_forecast_infos.py`（**训练前必须生成**） |

```bash
# (1) BEVDet info
python tools/create_data_bevdet.py --root-path data/nuscenes --version v1.0-trainval
# (2) 2D 语义伪标签（一次性，需 GPU）
python tools/generate_2d_seg_labels.py --data-root data/nuscenes \
    --output-dir data/nuscenes/seg_2d_labels --split trainval --device cuda:0
# (3) 4D 预测 info：串联历史 3 帧 + 未来 1/2/3s（nuScenes 2Hz）
python tools/create_4d_forecast_infos.py --root-path data/nuscenes
#     HISTORY_OFFSETS=[2,1], FUTURE_OFFSETS=[2,4,6], horizons=[1,2,3]s
#     调试: --version v1.0-mini --max-samples 8 ; 仅校验: --verify-only
```
> 未来 GT 就是未来 token 的 `labels.npz`，**不依赖 Cam4DOcc**。`_forecast.pkl` 会保留无效锚点
> （`forecast=None`）以维持索引对齐。

---

## 8. 训练与消融

主配置：`projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py`
（继承 baseline `priorocc-4d-r50-stgdm-scmf.py`，开启三项创新）。

```bash
CFG=projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py
python tools/train.py $CFG --work-dir work_dirs/priorocc-4d-innov          # 单卡
bash tools/dist_train.sh $CFG 8 --work-dir work_dirs/priorocc-4d-innov     # 多卡
python tools/train.py $CFG --work-dir work_dirs/priorocc-4d-innov --auto-resume  # 续训
```

**消融开关**（`--cfg-options`，各创新独立可关）：
```bash
model.enable_motion_prior=False                                        # 去 SMP（创新①）
model.enable_sem_continuity=False                                      # 去连续性（创新②）
model.future_predictor.use_warp_validity=False                         # 去 warp 有效性门控
model.img_view_transformer.depthnet_cfg.use_semantic_depth_prior=False # 去 SDP（创新③）
model.continuity_apply_occ=True                                        # 开 occ 连续性
```

**消融实验设计**（详见 `nextstep.md`）：
- **主线 E 系列**：E0 单帧 baseline → E1 时序融合 → E2 直接多步 → E3 动静解耦 → E4/E5/E6 增强1/2/3
  → **E6+SCMF** → E7 未来语义 → E8 时序一致性 → E9 full。
- **语义先验深度消融 D 系列**：D1 去 sem_feat_bev / D2 随机 query / D3 二值 mask / **D4 去 SCMF 语义条件**
  / D5 去未来语义 / D6 去时序一致性——回答"语义先验在每个环节贡献多少"。
- **本次创新专项**：去 SMP / 去连续性 / 去有效性门控 / 去 SDP 的逐项对比。

---

## 9. 训练前最小链路验证

一次跑通"数据→前向→反向→optimizer.step→收敛"，提前暴露 loss 不收敛 / 维度不匹配 / 梯度断链：
```bash
python -m compileall -q projects/mmdet3d_plugin tools
python tools/audit_priorocc_4d_static.py --fail-on HIGH
python tools/verify_priorocc_4d.py --stage build           --config $CFG
python tools/verify_priorocc_4d.py --stage motion-pipeline --config $CFG
python tools/verify_priorocc_4d.py --stage minimal-chain   --config $CFG \
    --iters 5 --warmup 2 --overfit-lr 1e-3 --stats-out work_dirs/mc_stats.json
python tools/diagnose_priorocc_4d_training.py --stats-json work_dirs/mc_stats.json
```

**`minimal-chain` 检查项**：
- 真实 dataloader 取一个 batch（缺数据自动回退合成）；`warmup` 步预热 + `iters` 步记录，每步
  `forward_train → Σloss → backward → clip → optimizer.step`。
- 断言：所有期望 loss 键存在且**有限**；**收敛** `loss_total[末] < loss_total[首]`；**维度**
  （`flow(B,T,2,H,W)`、`semantic_bev(B,17,H,W)`、`future_bevs(B,T,C,H,W)`、`validity(B,1,H,W)`、SMP 输出）；
  **梯度到达新模块**（`SemanticMotionPrior`、`SemanticDepthPrior.table` 的确定性微测试）；step 后参数无 NaN。
- 输出 **stats JSON**（15 键，精确匹配 `diagnose_priorocc_4d_training.py`：`step, loss_total,
  loss_occ_future_total, grad_norm, param_changed_ratio, logits_entropy, dominant_class_ratio,
  future_pairwise_diff, flow_mean_abs, flow_saturation_ratio, flow_grad_norm, gate_mean, gate_std,
  semantic_bev_std, mask_overlap`）——**补上 diagnose 缺失的生产者**。
- 合成数据下 `semantic_bev` 近均匀，diagnose 需放宽 `--max-mask-overlap 0.3 --min-future-loss-drop 0.0`。

**当前状态**：`build / motion-pipeline / minimal-chain / diagnose / baseline(--stage all)` 全部 **PASS**；
`minimal-chain` 收敛 `16.49 → 11.81`，梯度到达 SMP 与 SDP.table；baseline 旧配置 `--stage all` 损失键
与改动前完全一致（向后兼容）。

---

## 10. 评测指标与协议

遵循 Cam4DOcc benchmark 协议：历史 3 帧 @2Hz → 预测未来 1s/2s/3s，报告每步 mIoU 与平均。
```bash
python tools/test.py $CFG work_dirs/priorocc-4d-innov/latest.pth --eval mIoU
```
指标：`mIoU_1.0s / mIoU_2.0s / mIoU_3.0s / mIoU_avg`（`NuScenes4DOccForecastDataset.evaluate`，
`Metric_mIoU(num_classes=18, use_image_mask=True)`）。训练中每 2 epoch 自动评测。

---

## 11. 数值稳定性与向后兼容

- **恒等初始化**（保证不扰动 baseline、训练稳定起步）：SCMF `MotionHead=0`（flow=0）；SMP `A_g=I`
  且组概率归一（`φ_out=φ`）；SDP `Θ=0`（π 均匀，softmax 不变）；`use_warp_validity=False` 默认（门控维度不变）。
- **默认 flag 关闭**：三项创新均由 config flag 控制，默认关闭时与已审计 baseline **逐位一致**；单帧
  PriorOcc 完全不受影响（SDP flag 在 `priorocc-r50.py` 显式为 `False`）。
- **数值细节**：`log(π+ε)` 加 `clamp_min=-10`；加权 TV 分母 `+ε`、权重用 `min(邻域)` 不跨真实边界惩罚；
  `√(·+ε)` 防 0 梯度爆炸；行人 hinge `relu(mag-τ)²` 不罚合理小幅运动；SMP 作用于 `tanh·max_flow` 之后，
  用 `flow_saturation_ratio` 监控饱和。
- **相机顺序修复**：新增文件 `loading_temporal_seg2d.py` 此前用硬编码相机顺序（与 `data_config['cams']`
  6 中 5 错位），已改读 `results['cam_names']`（与已验证正确的单帧 `LoadSemanticSeg2D` 一致），使
  `gt_semantic_2d_history` 与 `seg_logits_list` 相机对齐；形状不变、键缺失自动回退、不影响单帧 PriorOcc。

---

## 12. 代码结构与文件索引

```
projects/mmdet3d_plugin/models/
  detectors/priorocc_4d.py               # PriorOcc4D 主检测器（时序 + 运动 + 损失装配）
  model_utils/
    semantic_injector.py                 # SemanticInjector（2D 语义先验注入）
    depthnet.py                          # DepthNet + SGDM + SemanticDepthPrior(SDP, 创新③)
    dyn_sta_decoder.py                   # warp_feature(+validity) / 动静分离 / 运动编码 / 注意力 / BEV 投影
    scmf.py                              # SCMF / ConvGRU / warp 有效性门控预测器(创新②)
    semantic_motion_prior.py             # SemanticMotionPrior(SMP, 创新①)
    semantic_continuity.py               # SemanticContinuityLoss(创新②补洞)
    sem_consistency.py                   # 语义时序一致性
    future_semantic.py                   # 未来语义预测器(dormant)
    language_self_gating.py              # 体素级高度先验门控(备用, 默认禁用)
  datasets/
    nuscenes_4d_forecast_dataset.py      # 4D 预测数据集 + 评测
    pipelines/loading_future_occ.py      # 未来占用 GT 加载
    pipelines/loading_temporal_seg2d.py  # 时序 2D 语义加载（相机顺序已对齐）
projects/configs/priorocc/
    priorocc-4d-r50-smp-sdp-holefill.py  # 主配置（开启三项创新）
    priorocc-4d-r50-stgdm-scmf.py        # baseline 配置
    priorocc-r50.py                      # 单帧 PriorOcc 基配置
tools/
    create_data_bevdet.py                # 生成 BEVDet info
    generate_2d_seg_labels.py            # 生成 2D 语义伪标签(SegFormer)
    create_4d_forecast_infos.py          # 生成 _forecast.pkl
    verify_priorocc_4d.py                # 验证(含 minimal-chain + stats 生产者)
    diagnose_priorocc_4d_training.py     # 训练塌缩诊断
    audit_priorocc_4d_static.py          # 静态审计
    train.py / dist_train.sh / test.py   # 训练/评测入口
```

---

## 13. 致谢

- **FlashOCC**（Yzichen）：Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin.
- **PriorOcc**：显式 2D 语义先验注入（SemanticInjector），本项目的单帧基础。
- **Cam4DOcc / Occ3D**：4D 占用预测 benchmark 协议与占用标签参考。
- **nuScenes / SegFormer**：数据集与 2D 语义伪标签来源。
