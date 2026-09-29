# PriorOcc-4D：语义先验时空化驱动的 4D 占用预测

**PriorOcc-4D** 在单帧 [PriorOcc](#项目脉络) / [FlashOCC](https://github.com/Yzichen/FlashOCC) 基础上，
把 2D 语义先验从"单帧空间注入"升级为"全链路时空驱动"，实现 **4D 占用预测（4D Occupancy
Forecasting）**：给定历史帧观测，预测未来 1s / 2s / 3s 的 3D 语义占用网格。

> **一句话原理**：语义类别先验不仅回答"*那里有什么*"，还能回答"*它将怎么动*"（约束运动学习空间）、
> "*它有多远*"（约束深度学习空间），并用"*背景是连续的*"这一先验补齐占用空洞。

- 完整研究方案：[`nextstep.md`](nextstep.md)
- 本次三项创新的方法与公式：[`PRIOROCC_4D_INNOVATIONS_README.md`](PRIOROCC_4D_INNOVATIONS_README.md)
- 启动/数据/训练：[`PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md`](PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md)、[`PRIOROCC_4D_LAUNCH_GUIDE.md`](PRIOROCC_4D_LAUNCH_GUIDE.md)

---

## 目录
- [项目脉络](#项目脉络)
- [1. 核心原理](#1-核心原理)
- [2. 整体架构](#2-整体架构)
- [3. 关键模块与公式](#3-关键模块与公式)
- [4. 损失函数总表](#4-损失函数总表)
- [5. 安装](#5-安装)
- [6. 数据准备](#6-数据准备)
- [7. 训练](#7-训练)
- [8. 训练前最小链路验证](#8-训练前最小链路验证)
- [9. 评测指标](#9-评测指标)
- [10. 代码结构与文件索引](#10-代码结构与文件索引)
- [11. 致谢](#11-致谢)

---

## 项目脉络

| 阶段 | 任务 | 核心 | 单帧 mIoU |
|---|---|---|---|
| **FlashOCC** | 单帧 3D 占用 | Channel-to-Height (C2H) 高效解码 | — |
| **PriorOcc** | 单帧 3D 占用 | SemanticInjector 注入 2D 语义先验 | 32.08 |
| **PriorOcc-4D**（本仓库） | **4D 占用预测** | 语义先验时空化：语义引导动静解耦 + 语义条件化运动场 + 语义运动/深度先验 + 语义连续性补洞 | 当前帧继承 PriorOcc，另报告未来 1/2/3s mIoU |

赛道：相机端到端 4D 占用预测，对标 Cam4DOcc(OCFNet)、Drive-OccWorld、DOME、T3Former 等。

---

## 1. 核心原理

1. **语义类别天然蕴含运动先验**：车辆会移动且运动与类别/朝向相关，行人横穿、幅度有界，
   建筑/地面静止。→ 用语义类别先验**约束运动模式的学习空间**（`SemanticMotionPrior`）。
2. **语义类别天然蕴含深度先验**：地面近、建筑/天空远。深度估计本质病态，是 2D→3D 投影噪声与
   空洞的根源。→ 用语义类别先验**约束深度分布**（`SemanticDepthPrior`，SGDM 升级）。
3. **运动场是语义特征的函数**：语义概率直接条件化运动场生成（`SCMF`），运动是语义理解的推论，
   而非独立运动模块。
4. **背景是连续的**：天空、地面等背景类空间连续、无洞。→ 用**语义连续性先验补洞**
   （warp 有效性门控 + 连续性 TV 正则）。
5. **SemanticInjector 全链路驱动**：同一套 2D 语义先验贯穿 运动编码 → 注意力 → 运动场 →
   残差组合 → 深度门控 → 时序一致性，语义先验处于绝对核心地位。

---

## 2. 整体架构

```
历史 3 帧 (t-2, t-1, t)
   │  每帧独立:
   ├─ Backbone+FPN ─► SemanticInjector ─► (融合特征, seg_logits)
   │                                   seg_logits ─► SGDM + SDP (语义门控/深度先验)
   ├─ ViewTransformer(LSS, 深度加权 voxel pooling) ─► 每帧 BEV (B,64,200,200)
   │
   └─► 时序 BEV 融合 (ego-motion 对齐 + Concat + Conv) ─► Fused BEV (B,256,200,200)
                                   │
             seg_logits(key) ─► SemanticBEVProjector ─► semantic_bev (B,17,200,200) + visibility
                                   │
             SemanticDynStaSeparator ─► dyn/sta/per-class masks
             SemanticMotionFeatureEncoder (增强1: 差分+语义) ─► motion_feat
             SemanticMotionAttention   (增强2: 语义驱动 query) ─► attn_feat
             PerClassDeltaCombiner     (增强3: 逐类残差) ─► refined BEV
                                   │
             SCMF 语义条件化运动场 ─► flow (B,3,2,200,200)
             SemanticMotionPrior (SMP) ─► 约束后的 flow   ← 创新①
                                   │
             SCMFEnhancedPredictor: warp(有效性门控) + ConvGRU 残差 + 门控融合  ← 创新②(补洞)
                                   │
             未来 BEV (t+1,t+2,t+3) ─► BEVOCCHead2D (C2H) ─► 未来 3D 占用
                                   
   并行监督: loss_2d_seg(逐帧) · loss_sem_consistency(时序一致) · loss_sem_continuity(连续性补洞, 创新②)
             · loss_static/rigid/nonrigid(SMP 正则, 创新①) · loss_depth(含 SDP, 创新③)
```

---

## 3. 关键模块与公式

记号：`φ`=运动场 flow `(B,T,2,H,W)`；`m`=每类语义概率 `(B,C,H,W)`；`C=17`；`T=3`；`D=88`(深度 bin)；
`H=W=200`(BEV)；`ε=1e-6`。

### 3.1 SemanticInjector（2D 语义先验注入，继承 PriorOcc）
置于 backbone/neck 之后、view transformer 之前：
```
S_logits = SegHead(F_img)                     # 2D 语义 logits
L_2d     = CrossEntropy(S_logits, Y_gt)       # 辅助监督 (ignore_index=255)
F_enh    = Fusion(F_img, S_logits)            # 语义回注特征
```
`S_logits` 同时驱动 SGDM/SDP、BEV 投影、时序一致性与连续性。

### 3.2 SGDM + SDP（语义门控深度 + 语义深度先验，创新③）
**SGDM**（软前景门控）：`gated = F_img · (1 + fg_prob · Attn_SE)`，`fg_prob=Σ_{c<11} softmax(S)_c`。
**SDP**（新增，每类深度分布先验，表 `Θ∈R^{C×D}` 零初始化 ~1.5K 参数）：
```
P_c = softmax(Θ_c / T_sdp)                         # 每类深度分布
σ   = softmax(S_logits)                            # 语义概率
π   = einsum('bchw,cd->bdhw', σ, P)                # 混合深度先验, Σ_d π = 1
depth_logits ← depth_logits + λ_sdp · clamp(log(π+ε), min=-10)
```
**恒等且可学**：`Θ=0 ⇒ π` 均匀 ⇒ `log π` 对 d 为常数 ⇒ 对 `softmax_d(depth)` **值不变**（不扰动
baseline），但对 `Θ` 的**梯度非零**（`∂L/∂Θ_{c,d} = (λ σ_c/T)·∂L/∂z_d`），故能学习。降低投影噪声/空洞。

### 3.3 SCMF 语义条件化运动场（核心）
```
φ_raw = tanh( MotionHead( Decoder([fused_bev, motion_feat, attn_feat, semantic_bev]) ) ) · max_flow
```
`MotionHead` 零初始化 ⇒ 初始 `φ=0`（恒等 warp，训练稳定）。

### 3.4 warp 有效性门控 + ConvGRU 融合（创新②补洞）
`warp_feature` 用 `grid_sample(padding_mode='zeros')`，disocclusion 越界处会变 0（空洞）。
返回有效性掩码并喂入门控，让空洞回退到时序残差：
```
validity = 1{ sample_grid ∈ [-1,1] }                         # (B,1,H,W), 空洞=0
coarse_k = warp(prev, φ_k)
delta_k  = ConvGRU(prev, h_{k-1})
α_k      = gate([prev, coarse_k, delta_k, validity])          # 空洞处 α→0
future_k = α_k · coarse_k + (1-α_k) · (prev + delta_k)        # 空洞→prev+delta(静态背景正确)
prev ← future_k                                               # 自回归
```

### 3.5 SMP 语义运动先验（创新①：刚体/非刚性运动学习空间约束）
按语义组学习 2×2 运动基，逐像素按类别概率混合后作用于 flow：

| 组 | 类别 id | 运动先验 |
|---|---|---|
| rigid_vehicle | 0,1,2,3,4 | 刚体、相干平移 |
| rigid_small | 5,6,7,9 | 近刚体、小幅度 |
| nonrigid_ped | 8 | 非刚性、幅度有界 |
| static | 10–16 | flow ≈ 0 |

```
p_g   = Σ_{c∈g} m_c ;  p̃_g = p_g /(Σ_g p_g + ε)
A_mix = einsum('bghw,gij->bijhw', p̃, A)          # A_g 每类运动基(init I)
φ_out = (1-α)·φ + α·(A_mix · φ),  α=sigmoid(alpha_logit)
```
`A_g=I` 初始化 ⇒ `φ_out=φ`（恒等）。训练中 `A_static→0`、`A_vehicle→`相干平移、`A_ped→`有界。
**正则损失**（补齐原缺失的 `L_motion_reg`）：
```
L_static   = Σ p_sta·(|φx|+|φy|) / (T·Σ p_sta + ε)                       # 静态不动
L_rigid    = (Σ wx·|Δx φ| + Σ wy·|Δy φ|) / (2T·(Σwx+Σwy)+ε), w=min(p_rig 邻域)  # 刚体相干(TV)
L_nonrigid = Σ p_ped·relu(√(φx²+φy²+ε) − τ)² / (T·Σ p_ped + ε), τ=2.0     # 行人有界(hinge)
```

### 3.6 语义连续性补洞（创新②）
背景类 `Cont=[11..16]`（地面/人行道/terrain/manmade/vegetation）空间连续，用**置信度加权 TV**
平滑其概率场，抑制孤立空洞（无可学习参数，权重 detach 防退化）：
```
通用: L_TV = (Σ wx·|Δx q| + Σ wy·|Δy q|) / (K·(Σwx+Σwy)+ε),  wx=min(w 邻域), w=置信度(detach)
2D : q=softmax(seg_logits),      w=Σ_{c∈Cont} q_c
BEV: s=semantic_bev,             w=visibility·Σ_{c∈Cont} s_c
OCC: po=softmax(occ_logits,-1),  w=mask_camera·Σ_{c∈Cont} po_c   (对 Dx,Dy; 默认关)
```

### 3.7 语义时序一致性（自监督）
每帧 `seg_logits` 在静止区域应一致，用对称 KL（置信度+静态掩码加权）约束时序融合质量：
```
L_consist = SymKL( softmax(S_t) , softmax(S_{t-1}) )  在 高置信∩静态∩可见 区域
```

### 3.8 未来占用头（C2H）
复用 `BEVOCCHead2D`（Channel-to-Height）把未来 BEV 解码为 `(B,Dx,Dy,Dz,18)` 占用，保持 FlashOCC 效率。

---

## 4. 损失函数总表

```
L_total = L_depth + L_occ(current) + Σ_k L_occ_future(k) + Σ_i L_2d_seg(frame_i)
        + L_sem_consistency + L_static + L_rigid + L_nonrigid
        + L_sem_continuity_2d + L_sem_continuity_bev (+ L_sem_continuity_occ)
```

| 损失键 | 含义 | 默认权重 | 来源 |
|---|---|---|---|
| `loss_depth` | 深度监督（含 SDP 调制） | 3.0 | ViewTransformer |
| `loss_occ` | 当前帧 3D 占用 CE + class balance | 1.0 | OccHead |
| `loss_occ_future_{1,2,3}s` | 未来各步占用 CE | 1.0 / 0.7 / 0.5 | OccHead |
| `loss_2d_seg_history_{0,1,2}` | 逐帧 2D 语义 CE | 0.3 each | SemanticInjector |
| `loss_sem_consistency` | 语义时序一致性 KL | 0.05 | SemConsistencyLoss |
| `loss_static_flow` | 静态区 flow≈0（SMP） | 0.05 | SemanticMotionPrior |
| `loss_rigid_smooth` | 刚体 flow 相干 TV（SMP） | 0.02 | SemanticMotionPrior |
| `loss_nonrigid_bound` | 行人 flow 有界（SMP） | 0.02 | SemanticMotionPrior |
| `loss_sem_continuity_2d` | 2D 语义连续性 TV | 0.05 | SemanticContinuityLoss |
| `loss_sem_continuity_bev` | BEV 语义连续性 TV | 0.05 | SemanticContinuityLoss |
| `loss_sem_continuity_occ` | occ 连续性 TV（默认关） | 0.0 | SemanticContinuityLoss |
| `loss_future_semantic_*` | 未来语义预测（dormant，默认关） | 0.1 | FutureSemanticPredictor |

> **向后兼容**：三项创新均由 flag 控制且默认关闭；关闭时与已审计 baseline 逐位一致
> （新模块 init 恒等：SCMF `MotionHead=0`、SMP `A=I`、SDP `Θ=0`、validity 默认关）。

---

## 5. 安装

参考 [FlashOCC 安装指南](doc/install.md)。本仓库验证环境：
```
conda activate flash     # Python 3.9, PyTorch 1.10+cu111, mmcv 1.5.3, mmdet3d 1.0.0rc4
pip install transformers # 2D 伪标签生成(SegFormer)
```
> **GPU 注意**：PyTorch 1.10+cu111 仅支持到 sm_86。若在 sm_120（如 RTX 5060）等新卡上，
> 需换兼容 GPU 或升级 PyTorch 构建，否则 `torch.cuda` 不可用（验证工具会自动回退 CPU）。

---

## 6. 数据准备

| 数据 | 路径 | 来源 |
|---|---|---|
| nuScenes 图像 | `data/nuscenes/samples/`, `sweeps/` | nuScenes v1.0-trainval |
| 3D 占用 GT | `data/nuscenes/gts/{scene}/{token}/labels.npz` | 外部 CVPR2023-3D-Occupancy-Prediction（Occ3D 风格） |
| 2D 语义伪标签 | `data/nuscenes/seg_2d_labels/samples/CAM_*/*.png` | `tools/generate_2d_seg_labels.py`（SegFormer-B2） |
| BEVDet info | `bevdetv2-nuscenes_infos_{train,val}.pkl` | `tools/create_data_bevdet.py` |
| **4D 预测 info** | `bevdetv2-nuscenes_infos_{train,val}_forecast.pkl` | `tools/create_4d_forecast_infos.py`（**训练前必须生成**） |

```bash
# (1) BEVDet info
python tools/create_data_bevdet.py --root-path data/nuscenes --version v1.0-trainval
# (2) 2D 语义伪标签
python tools/generate_2d_seg_labels.py --data-root data/nuscenes \
    --output-dir data/nuscenes/seg_2d_labels --split trainval --device cuda:0
# (3) 4D 预测 info（串联历史 3 帧 + 未来 1/2/3s；HISTORY=[2,1], FUTURE=[2,4,6] @2Hz）
python tools/create_4d_forecast_infos.py --root-path data/nuscenes
```
> 未来 GT 即未来 token 的 `labels.npz`，**不依赖 Cam4DOcc**。

---

## 7. 训练

主配置：`projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py`
（继承 baseline `priorocc-4d-r50-stgdm-scmf.py`，开启三项创新）。

```bash
CFG=projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py
python tools/train.py $CFG --work-dir work_dirs/priorocc-4d-innov          # 单卡
bash tools/dist_train.sh $CFG 8 --work-dir work_dirs/priorocc-4d-innov     # 多卡
```

**消融开关**（`--cfg-options`，各创新可独立开关）：
```bash
model.enable_motion_prior=False                                        # 去 SMP（创新①）
model.enable_sem_continuity=False                                      # 去连续性（创新②）
model.future_predictor.use_warp_validity=False                         # 去 warp 有效性门控
model.img_view_transformer.depthnet_cfg.use_semantic_depth_prior=False # 去 SDP（创新③）
model.continuity_apply_occ=True                                        # 开 occ 连续性
```

---

## 8. 训练前最小链路验证

一次跑通"数据→前向→反向→optimizer.step→收敛"，提前暴露 loss 不收敛 / 维度不匹配 / 梯度断链：
```bash
python tools/verify_priorocc_4d.py --stage build           --config $CFG
python tools/verify_priorocc_4d.py --stage motion-pipeline --config $CFG
python tools/verify_priorocc_4d.py --stage minimal-chain   --config $CFG \
    --iters 5 --warmup 2 --overfit-lr 1e-3 --stats-out work_dirs/mc_stats.json
python tools/diagnose_priorocc_4d_training.py --stats-json work_dirs/mc_stats.json
```
`minimal-chain` 断言：所有 loss 有限、维度正确、`loss_total` 下降（收敛）、梯度到达 SMP/SDP、
参数无 NaN，并输出与 `diagnose` 兼容的 stats（补上其缺失的生产者）。

**当前状态**：`build / motion-pipeline / minimal-chain / diagnose / baseline(--stage all)` 全部 **PASS**；
`minimal-chain` 收敛 `16.49 → 11.81`，梯度到达 SMP 与 SDP.table。

---

## 9. 评测指标

训练中自动评测（`evaluation=dict(interval=2, start=2)`），或：
```bash
python tools/test.py $CFG work_dirs/priorocc-4d-innov/latest.pth --eval mIoU
```
指标：`mIoU_1.0s / mIoU_2.0s / mIoU_3.0s / mIoU_avg`（`NuScenes4DOccForecastDataset.evaluate`，
遵循 Cam4DOcc benchmark 协议：历史 3 帧 @2Hz → 未来 1/2/3s）。

---

## 10. 代码结构与文件索引

```
projects/mmdet3d_plugin/models/
  detectors/priorocc_4d.py            # PriorOcc4D 主检测器（时序 + 运动 + 损失装配）
  model_utils/
    semantic_injector.py              # SemanticInjector（2D 语义先验注入）
    depthnet.py                       # DepthNet + SGDM + SemanticDepthPrior(SDP, 创新③)
    dyn_sta_decoder.py                # warp_feature(+validity) / 动静分离 / 运动编码 / 注意力 / BEV 投影
    scmf.py                           # SCMF 运动场 / ConvGRU / 有效性门控预测器(创新②)
    semantic_motion_prior.py          # SemanticMotionPrior(SMP, 创新①)
    semantic_continuity.py            # SemanticContinuityLoss(创新②补洞)
    sem_consistency.py                # 语义时序一致性
    future_semantic.py                # 未来语义预测器(dormant)
  datasets/
    nuscenes_4d_forecast_dataset.py   # 4D 预测数据集 + 评测
    pipelines/loading_future_occ.py   # 未来占用 GT 加载
    pipelines/loading_temporal_seg2d.py # 时序 2D 语义加载（相机顺序已对齐 seg_logits）
projects/configs/priorocc/
    priorocc-4d-r50-smp-sdp-holefill.py  # 主配置（开启三项创新）
    priorocc-4d-r50-stgdm-scmf.py        # baseline 配置
tools/
    create_4d_forecast_infos.py       # 生成 _forecast.pkl
    verify_priorocc_4d.py             # 验证(含 minimal-chain)  diagnose_priorocc_4d_training.py
    audit_priorocc_4d_static.py       # 静态审计
```

**文档**：`nextstep.md`（研究方案）· `PRIOROCC_4D_INNOVATIONS_README.md`（创新方法+公式）·
`PRIOROCC_4D_INNOVATIONS_LAUNCH_GUIDE.md` / `PRIOROCC_4D_LAUNCH_GUIDE.md`（启动）·
`PRIOROCC_4D_DEVELOPMENT_PLAN.md`（开发计划）。

---

## 11. 致谢

- **FlashOCC**（Yzichen）：Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin.
- **PriorOcc**：显式 2D 语义先验注入（SemanticInjector），本项目的单帧基础。
- **Cam4DOcc / Occ3D**：4D 占用预测 benchmark 与占用标签参考。
- **nuScenes / SegFormer**：数据集与 2D 语义伪标签来源。
