# PriorOcc-4D 语义先验增强：新增内容、方法与公式

本篇是**技术总结**（做了什么 + 怎么做 + 数学）。运行/数据/训练步骤见
[`PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md`](PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md)。
整体研究方案见 [`nextstep.md`](nextstep.md)。

---

## 0. 一句话动机

> 语义类别先验不仅回答"那里有什么"，还能约束"**它将怎么动**"（运动学习空间）与
> "**它有多远**"（深度学习空间），并用"**背景是连续的**"这一先验补齐占用空洞。

本次在已实现的 PriorOcc-4D（SCMF 语义条件化运动场）基础上，新增三项创新 + 一个补洞门控
+ 一个训练前最小链路验证工具，**全部为后处理模块，不改骨干**，且**默认 flag 关闭时与
已审计 baseline 逐位一致**。

---

## 1. 审计对照（nextstep.md 实现状态）

| nextstep.md 组件 | 状态 | 说明 |
|---|---|---|
| Module 1 时序 BEV 融合 | 已实现 | FlashOCC 原生 3 帧 |
| Module 1 语义时序一致性 | 已实现（有偏差） | `SemConsistencyLoss` 对称 KL |
| Module 2 增强 1-3 | 已实现 | 运动编码 / 注意力 / 逐类 delta |
| Module 2 SCMF + GRU 门控 | 已实现（升级） | `ConvGRU2D` 替代逐像素 GRUCell |
| Module 3 未来占用头 | 已实现 | 直接多步 + 自回归 |
| Module 4 未来语义预测器 | dormant | config 关闭，本次不激活 |
| **`L_motion_reg`（静态区 flow≈0）** | **原缺失 → 本次由 SMP 补齐** | `loss_static_flow` |
| **刚体/非刚性运动学习空间约束** | **本次新增** | `SemanticMotionPrior` |
| **语义连续性补洞（天空/地面）** | **本次新增** | warp 有效性门控 + `SemanticContinuityLoss` |
| **SGDM 提升** | **本次新增** | `SemanticDepthPrior` |
| **训练前最小链路验证** | **本次新增** | `verify_priorocc_4d.py --stage minimal-chain` |

---

## 2. 创新一：语义运动先验 SMP（`SemanticMotionPrior`）

**文件**：`projects/mmdet3d_plugin/models/model_utils/semantic_motion_prior.py`
**接入**：`priorocc_4d.py::_compute_motion_flow`，在 SCMF 产出 flow 之后。

### 2.1 思想
不同语义类别的**运动模式学习空间**不同：车辆是刚体、沿朝向平移；行人是非刚性、幅度有界；
背景静止。SMP 用**每类可学习运动基**（2×2 仿射）按类别概率逐像素混合后作用于 SCMF 的原始
flow，把 flow 约束到各类别合理的运动子空间；再配合三个正则损失强化。

### 2.2 类别分组（默认）
| 组 | 类别 id | 运动先验 |
|---|---|---|
| `rigid_vehicle` | 0,1,2,3,4（car/truck/construction/bus/trailer） | 刚体、相干平移 |
| `rigid_small` | 5,6,7,9（barrier/motorcycle/bicycle/traffic_cone） | 近刚体、小幅度 |
| `nonrigid_ped` | 8（pedestrian） | 非刚性、幅度有界 |
| `static` | 10,11,12,13,14,15,16（others + 背景） | flow ≈ 0 |

### 2.3 前向公式
输入：SCMF flow `φ ∈ R^{B×T×2×H×W}`，每类概率 `m ∈ R^{B×C×H×W}`（=`per_cls_masks`）。
组成员矩阵 `M ∈ {0,1}^{G×C}`。

```
组概率     p_g = Σ_{c∈g} m_c                       # einsum('bchw,gc->bghw', m, M)
归一化     p̃_g = p_g / (Σ_g p_g + ε)               # 保证 init 恒等
每类运动基 A_g ∈ R^{2×2}                            # per_group: 直接学习; shared_modes: A_g=Σ_k softmax(logits)_gk·B_k
逐像素混合 A_mix[b,i,j,h,w] = Σ_g p̃[b,g,h,w]·A_g[i,j]   # einsum('bghw,gij->bijhw', p̃, A)
变换       transform[b,t,i] = Σ_j A_mix[b,i,j]·φ[b,t,j]
输出       φ_out = (1-α)·φ + α·transform,  α = sigmoid(alpha_logit)
```
**恒等初始化**：`A_g = I`、`p̃` 归一 ⇒ `A_mix = I` ⇒ `transform = φ` ⇒ `φ_out = φ`。
即初始不扰动 SCMF（其 `motion_head` 零初始化，flow 本就为 0），训练中 `A_static→0`
（杀静态 flow）、`A_vehicle→` 相干平移、`A_ped→` 有界。**参数量 ≤ ~33**。

### 2.4 正则损失（`regularization_losses`，未加权）
记 `φ=φ_out`，`p_sta/p_rig/p_ped` 为对应组概率，`ε=1e-6`，`τ=ped_max_flow_cells=2.0`：

```
(1) 静态区不动（补齐缺失的 L_motion_reg）:
    L_static = Σ p_sta·(|φx|+|φy|) / (T·Σ p_sta + ε)

(2) 刚体相干（置信度加权 TV，边界不罚）:
    Δxφ = φ[...,w+1]-φ[...,w];  Δyφ = φ[...,h+1,:]-φ[...,h,:]
    wx = min(p_rig[w], p_rig[w+1]);  wy = min(p_rig[h], p_rig[h+1])
    L_rigid = (Σ wx·|Δxφ| + Σ wy·|Δyφ|) / (2T·(Σwx+Σwy) + ε)

(3) 非刚性有界（hinge）:
    mag = sqrt(φx² + φy² + ε)
    L_nonrigid = Σ p_ped·relu(mag - τ)² / (T·Σ p_ped + ε)
```
**默认权重**：`static_flow=0.05, rigid_smooth=0.02, nonrigid_bound=0.02`
→ 损失键 `loss_static_flow / loss_rigid_smooth / loss_nonrigid_bound`。

---

## 3. 创新二：语义连续性补洞

空洞来源：`warp_feature` 用 `grid_sample(padding_mode='zeros')`，运动 warp 的 disocclusion
区域直接变 0（空洞）；稀疏深度投影也会在 BEV/占用产生空洞。用两招解决：

### 3.1 Warp 有效性门控（`dyn_sta_decoder.py` + `scmf.py`）
`warp_feature(bev, flow, return_validity=True)` 额外返回

```
validity[b,0,h,w] = 1{ sample_grid_x∈[-1,1] 且 sample_grid_y∈[-1,1] }   # (B,1,H,W)
```
`SCMFEnhancedPredictor(use_warp_validity=True)` 把 `validity` 拼进门控输入：

```
α = gate([prev, coarse, delta, validity])        # 原来只有前 3 项
future_k = α·coarse_k + (1-α)·(prev + delta_k)   # coarse_k = warp(prev, flow_k)
```
**效果**：空洞处 `validity=0` ⇒ 门控学 `α→0` ⇒ `future_k → prev+delta`（时序残差）而非 0。
对静态连续背景（地面），`prev` 正是正确内容 ⇒ 空洞被自然填补。**默认 flag 关，baseline 不变。**

### 3.2 语义连续性损失（`SemanticContinuityLoss`）
**文件**：`projects/mmdet3d_plugin/models/model_utils/semantic_continuity.py`
背景类（地面/人行道/terrain/manmade/vegetation，`continuous_class_ids=[11..16]`）空间连续，
用**置信度加权 TV** 平滑其概率场，抑制孤立空洞。**无可学习参数**（梯度回流到 seg_head/
depth_net/occ_head）；权重 detach，避免"把置信度压到 0"的退化解。

通用加权 TV（field `(...,K,d0,d1)`，weight `(...,1,d0,d1)`，K=连续类数）：
```
Δx = field[...,1:]-field[...,:-1];  Δy = field[...,1:,:]-field[...,:-1,:]
wx = min(weight[..., :-1], weight[..., 1:]).detach()
wy = min(weight[..., :-1,:], weight[..., 1:,:]).detach()
L_TV = (Σ wx·|Δx| + Σ wy·|Δy|) / (K·(Σwx+Σwy) + ε)
```
三个层级：
```
2D :  q = softmax(seg_logits_key);      conf = Σ_{c∈Cont} q_c            → loss_sem_continuity_2d
BEV:  s = semantic_bev;                 conf = visibility·Σ_{c∈Cont} s_c  → loss_sem_continuity_bev
OCC:  po = softmax(occ_logits, -1);     conf = mask_camera·Σ_{c∈Cont} po_c → loss_sem_continuity_occ
      (对 (B,Dx,Dy,Dz,C) 的 Dx,Dy 做 TV；默认关)
```
**默认权重**：`cont_2d=0.05, cont_bev=0.05, cont_occ=0.0`（`continuity_apply_occ=False`）。

---

## 4. 创新三：SGDM 升级 —— 语义深度先验 SDP（`SemanticDepthPrior`）

**文件**：`projects/mmdet3d_plugin/models/model_utils/depthnet.py`
**接入**：`DepthNet.forward`，在 `depth = self.depth_conv(...)` 之后、`cat([depth,context])` 之前。

### 4.1 思想
SGDM 现状是"软前景门控"（`img_feat·(1+fg_prob·attn)`），只重塑特征、不注入深度先验。
SDP 直接用**语义类别先验**约束病态深度：每类学一个深度分布，按语义概率混合后调制深度 logits。
这与 SMP 同主题——**类别先验约束学习空间**（一个约束运动、一个约束深度），并降低 2D→3D
投影噪声/空洞。

### 4.2 公式
每类深度表 `Θ ∈ R^{C×D}`（C=17, D=88, **零初始化**, ~1.5K 参数）：
```
P_c   = softmax(Θ_c / T_sdp)                              # (C,D) 每类深度分布
σ     = softmax(sem_logits, 1)                            # (BN,C,fH,fW)
π     = einsum('bchw,cd->bdhw', σ, P)                     # (BN,D,fH,fW), Σ_d π=1
depth_logits ← depth_logits + λ_sdp · clamp(log(π+ε), min=-10)
```
**恒等性 + 可学习**：`Θ=0 ⇒ π=1/D` 均匀 ⇒ `log π` 对 d 为常数 ⇒ 对 `softmax_d(depth)`
**值不变**（初始恒等、不扰动 baseline）；但对 `Θ` 的**梯度非零**
（`∂L/∂Θ_{c,d} = (λ σ_c/T)·∂L/∂z_d`，而 `∂L/∂z_d` 由深度损失给出，非零），故表能学习。
已由 `minimal-chain` 的 SDP 梯度微测试实测通过。

**flag**：`DepthNet(use_semantic_depth_prior=False, sdp_weight=1.0, sdp_temperature=1.0)`；
`priorocc-r50.py` 显式置 `False`（baseline 不变），新 config 置 `True`。
> 注：`LanguageSelfGating`（天空/地面高度先验，体素级门控）保持禁用——插入点不同，
> SDP（深度级）+ 连续性（语义/占用级）已覆盖空洞/伪响应；LSG 列为未来互补工作。

---

## 5. 训练前最小链路验证（`verify_priorocc_4d.py --stage minimal-chain`）

**目的**：训练前一次跑通，避免 loss 不收敛、维度不匹配、梯度不到达新模块等问题；
并**补上 `diagnose_priorocc_4d_training.py` 缺失的 stats 生产者**。

**流程**：build 模型 → 取一个 batch（真实 dataloader，缺数据自动回退合成）→
`warmup` 步预热 + `iters` 步记录，每步 `forward_train → Σloss → backward → clip → optimizer.step`：
- 断言所有期望 loss 键存在且**有限**（含新 `loss_static_flow/rigid_smooth/nonrigid_bound/sem_continuity_2d/_bev`）；
- **收敛**：`loss_total[末] < loss_total[首]`（overfit-a-batch）；
- **维度**：`flow(B,T,2,H,W)`、`semantic_bev(B,17,H,W)`、`future_bevs(B,T,C,H,W)`、`validity(B,1,H,W)`、SMP 输出；
- **梯度到达新模块**：`SemanticMotionPrior` 与 `SemanticDepthPrior.table` 的确定性微测试；
- step 后参数无 NaN/Inf；
- 输出 **stats JSON**（15 键，精确匹配 diagnose）到 `--stats-out`。

**参数**：`--iters 5 --warmup 2 --overfit-lr 1e-3 --stats-out PATH --use-real-data/--no-real-data`。
合成数据下 `semantic_bev` 近均匀，diagnose 需放宽：`--max-mask-overlap 0.3 --min-future-loss-drop 0.0`。

### 本次实测结果（CPU，合成 batch）
```
build           PASS   (PriorOcc4D, 59,485,223 params; motion_prior/SDP 已建)
motion-pipeline PASS   (新损失全部出现且有限, backward/simple_test/fallback OK)
minimal-chain   PASS   (convergence 16.4853 -> 11.8062; dim OK; grad->SMP OK; grad->SDP.table OK)
diagnose        PASS   (读取 mc_stats.json)
baseline --stage all  PASS (旧 config 损失键与改动前完全一致 → 向后兼容)
```

---

## 6. 公式与符号汇总

| 损失键 | 公式核心 | 默认权重 | 来源模块 |
|---|---|---|---|
| `loss_static_flow` | `Σ p_sta·(|φx|+|φy|)/(T Σp_sta)` | 0.05 | SMP |
| `loss_rigid_smooth` | rigid 区 flow 的置信度加权 TV | 0.02 | SMP |
| `loss_nonrigid_bound` | `Σ p_ped·relu(mag-τ)²/(T Σp_ped)` | 0.02 | SMP |
| `loss_sem_continuity_2d` | 2D sem 连续类加权 TV | 0.05 | Continuity |
| `loss_sem_continuity_bev` | BEV sem 连续类加权 TV | 0.05 | Continuity |
| `loss_sem_continuity_occ` | occ 连续类加权 TV（默认关） | 0.0 | Continuity |

**张量形状**：`flow(B,T=3,2,H=200,W=200)`；`semantic_bev(B,17,Dy=200,Dx=200)`；
`occ_logits(B,Dx=200,Dy=200,Dz=16,C=18)`；`seg_logits(B·N,17,fH=16,fW=44)`；depth bins `D=88`。

**符号**：`φ`=flow；`m`=每类概率；`p_g`=组概率；`A_g`=每类运动基；`α`=门控/残差系数；
`π`=混合深度先验；`Θ`=每类深度表；`ε=1e-6`；`τ=2.0`；`T`=未来步数=3。

---

## 7. 相机顺序修复（新增文件 BUG）

`loading_temporal_seg2d.py`（**新增文件**，非第一版 PriorOcc 用的 `loading_seg2d.py`）此前从
`results['curr'].get('cam_names', 硬编码顺序)` 取相机顺序，而 `results['curr']` 无该键 →
回退硬编码 `[FRONT,FRONT_RIGHT,FRONT_LEFT,BACK,BACK_LEFT,BACK_RIGHT]`，与
`PrepareImageInputs` 实际堆叠顺序 `data_config['cams']=[FRONT_LEFT,FRONT,FRONT_RIGHT,
BACK_LEFT,BACK,BACK_RIGHT]` **6 个里 5 个错位** → `loss_2d_seg_history_*` 静默监督到错误相机
（不报错，只降质量）。**修复**：改读顶层 `results['cam_names']`（与已验证正确的单帧
`LoadSemanticSeg2D` 同一套逻辑；`PrepareImageInputs` 第 1 步已设，第 7 步读取）。形状不变、
键缺失自动回退、**不影响单帧 PriorOcc**。

---

## 8. 消融开关与文件索引

**逐项消融**（`--cfg-options`）：
```
model.enable_motion_prior=False                                  # 关 SMP
model.enable_sem_continuity=False                                # 关连续性
model.future_predictor.use_warp_validity=False                   # 关补洞门控
model.img_view_transformer.depthnet_cfg.use_semantic_depth_prior=False  # 关 SDP
model.continuity_apply_occ=True                                  # 开 occ 连续性
```
**新增文件**：`semantic_motion_prior.py`、`semantic_continuity.py`、
`configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py`、本 README、启动指南。
**修改文件**：`depthnet.py`（SDP）、`dyn_sta_decoder.py`（`warp_feature` validity）、
`scmf.py`（warper/predictor 门控）、`priorocc_4d.py`（构建/接线/损失）、
`model_utils/__init__.py`（导出）、`loading_temporal_seg2d.py`（相机顺序）、
`tools/verify_priorocc_4d.py`（minimal-chain + stats 生产者）、
`configs/priorocc/priorocc-r50.py`（显式 `use_semantic_depth_prior=False`）。
