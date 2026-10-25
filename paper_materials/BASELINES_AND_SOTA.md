# 4D 占用预测：Baseline 与 SOTA 数据汇编（供写作 agent 引用）

> 用途：提供论文「相关工作 / 对比实验」章节所需的外部数字与协议说明。
> **核心警告：不同来源的数字口径不同（mIoU vs IoU、GT 输入 vs 相机、预测步长/频率/体素分辨率不同），不可混表。**
> 最后更新：2026-10-23

---

## 0. 三种评测协议对照（引用前必须确认）

| 协议 | 数据集 | 指标 | 输入 | 预测步长 | 体素 | 网格尺寸 | 类别 |
|---|---|---|---|---|---|---|---|
| **Cam4DOcc**（CVPR24） | nuScenes / Lyft-L5 | **IoU**（IoU_c / IoU_f / IoU~_f） | 相机（3 帧） | Nf=4 步，nuScenes 2Hz → 0.5/1.0/1.5/2.0s | 0.2m | 512×512×40 | GMO / GSO / free |
| **Occ3D-nuScenes**（OccWorld/DOME 等） | nuScenes | **mIoU**（17 类均值） | 相机或 GT Occ | 1s/2s/3s | 0.4m | 200×200×16 | 17 类 |
| **本仓库**（PriorOcc-4D） | nuScenes val_forecast | **mIoU**（17 类均值）+ 逐类 | 相机（3 帧） | 1s/2s/3s | 0.512m* | 200×200×16 | 17 类 |

\* 本仓库 `point_cloud_range=[-51.2,51.2]`、200 体素 → 1 cell = 0.512m；与 Occ3D 的 0.4m 略有差异，正式对比前需确认是否同网格。

**不可混用的三组数字**：
- Cam4DOcc 的 IoU（29.36、26.82 等）≠ mIoU
- OccWorld/T3Former-O 的 GT 占用输入 ≠ 相机输入
- 不同预测频率（2Hz vs 5Hz）的 2s 不是同一物理时长

---

## 1. GT 占用输入方法（非端到端，仅作参考上限）

| 方法 | 会议 | 架构 | 1s mIoU | 2s | 3s | Avg |
|---|---|---|---|---|---|---|
| **T3Former-O** | 2025 | Triplane delta 自回归 Transformer | 46.32 | 33.23 | 28.73 | **36.09** |
| **FSF-Net** | 2024 | VQ-Mamba + BEV flow warp | 42.38 | — | 17.03 | — |
| **OccWorld-O** | ECCV 2024 | VQ tokenizer + GPT 式自回归 | 25.78 | 15.14 | 10.51 | 17.14 |
| OccWorld-O（IoU） | 同上 | 同上 | 34.63 | 25.07 | 20.18 | 26.63 |
| **OccSTeP-WM** | 2025 | Tokenizer-free voxel + persistent state | — | — | — | ≈18.6（OccWorld +1.48） |

- 用途：说明「GT 输入 + 自回归」的上界；**不与我们做公平对比**（输入模态不同）
- 引用注意：OccWorld 的 3s/1s = 0.41，衰减陡峭；我们 0.71，说明时间保持力是相对亮点

## 2. 相机端到端方法（与 PriorOcc-4D 同赛道，mIoU 口径）

| 方法 | 会议 | 架构 | 1s | 2s | 3s | Avg | 备注 |
|---|---|---|---|---|---|---|---|
| **T3Former-F** | 2025 | Triplane delta（相机版） | 19.60 | — | — | — | 只报 1s |
| **DOME** | 2024 | Latent diffusion Transformer | 35.11 | 25.89 | 20.29 | **27.10** | 当前 mIoU 最强参考之一 |
| **Drive-OccWorld** | AAAI 2025 | C2H + 自回归 | 36.3（mIoU_f） | — | — | — | 只报 avg |
| **UniScene / COME** | 2025 | Scene-centric forecasting control | — | — | — | — | COME 比 DOME 高 26.3% mIoU（需查原文数字） |
| **STM4D** | 2025 | 2D+3D spatio-temporal | 待查（OpenReview 阻爬） | — | — | — | Occ3D-nuScenes 协议 |
| **PriorOcc-4D（本仓库，零训练修复版）** | — | LSS + 时序BEV + 语义先验 + PCF | **21.33** | **17.64** | **15.09** | **18.02** | 全量 4818·EMA |

**定位结论**：
- 超过：T3Former-F（1s +1.73）、OccWorld Avg（+0.88，但对方是 GT 输入，仅作参考）
- 落后：DOME 约 9 点、Drive-OccWorld 更多
- 我们的相对优势：**3s/1s = 0.71 的时间保持力**（DOME 0.58、OccWorld 0.41）
- 论文表述建议：「相机端到端方法中时间衰减最平缓之一；绝对精度落后 diffusion/LLM 类方法，但模型规模与推理成本低一个量级」

## 3. Cam4DOcc 官方 Baseline（**IoU 口径**，CVPR24）

协议：nuScenes，inflated GMO，序列长 7（Np=2 观测 + Nf=4 预测），0.2m 体素，512×512×40。

### 3.1 主表（inflated GMO 预测）

| 方法 | IoU_c | IoU_f（2s） | IoU~_f | 类型 |
|---|---|---|---|---|
| **OpenOccupancy-C** | 12.17 | 11.45 | 11.74 | 占用预测扩展 |
| **SPC**（SurroundDepth+PCPNet+Cylinder3D） | 1.27 | failed | failed | 点云预测+体素化 |
| **PowerBEV-3D** | 23.08 | 21.25 | 21.86 | 2D 实例预测+z 轴复制 |
| **OCFNet** | 27.86 | 23.89 | 24.77 | 端到端时空网络 |
| **OCFNet†**（更强配置） | **31.30** | **26.82** | **27.98** | 同上 |

### 3.2 细粒度 GMO（nuScenes-Occupancy）

| 方法 | IoU_c | IoU_f(2s) | IoU~_f |
|---|---|---|---|
| OpenOccupancy-C | 10.82 | 8.02 | 8.53 |
| SPC | 5.85 | 1.08 | 1.12 |
| PowerBEV-3D | 5.91 | 5.25 | 5.49 |
| OCFNet | 10.15 | 8.35 | 8.69 |
| OCFNet† | 11.45 | 9.68 | 10.10 |

### 3.3 ★关键：OCFNet 的 flow head 消融（对我们 PCF 的直接外部支持）

| 配置 | IoU_c | 0.5s | 1.0s | 1.5s | 2.0s | IoU_f | IoU~_f |
|---|---|---|---|---|---|---|---|
| OCFNet **w/o flow** | 26.84 | 25.01 | 24.04 | 23.38 | 22.99 | 23.86 | — |
| OCFNet（含 3D backward centripetal flow） | 27.86 | 25.95 | 24.92 | 24.33 | 23.89 | 24.77 | — |
| **提升** | +3.8% | +3.8% | +3.7% | +4.1% | +3.9% | +3.8% | — |

**用途**：
1. 外部证据表明「显式运动流（flow）」是 4D 预测的关键增益点（+3.8~4.1%），
   与我们 PCF 的「测量位移而非回归」路线同源，可作为相关工作/动机引用；
2. OCFNet 用的是 **GT 3D backward centripetal flow 监督**训练，
   我们 PCF 是**无监督相关测量**（无需 flow 标注）——这是可强调的差异点；
3. 说明社区主流做法与我们一致：**flow / 位移建模是 4D 预测的核心**。

---

## 4. 与本仓库实验的对应关系（写作时可引用的对比轴）

| 对比轴 | 外部参照 | 本仓库对应实验 | 结论 |
|---|---|---|---|
| flow 的作用 | OCFNet w/o flow vs w/ flow（+3.8%） | PCF vs 无PCF（1s 动态 +0.80） | 一致：flow 有效 |
| 静态拷贝基线 | Cam4DOcc 的「static-world hypothesis」baseline（论文定性提及，无具体数字） | B 档（拷贝当前帧） | 我们给出了定量 B 档，是更严格的基线 |
| 端到端 vs 分阶段 | OCFNet（端到端） | 两阶段冻结 vs 联训 | 联训未来 mIoU 高 2.2~2.7（编码器需看时序） |
| 时序输入 | Cam4DOcc 用 3 帧，OccWorld 用 2s 历史 | 我们用 3 帧（num_adj=2） | 同量级，可比 |
| 长时程保持力 | OccWorld 3s/1s=0.41；DOME 0.58 | 我们 0.71 | 相对优势 |

---

## 5. 引用清单（URL）

| 方法 | 出处 |
|---|---|
| Cam4DOcc（benchmark + OCFNet + 4 baselines） | CVPR 2024；arXiv:2311.17663v3；github.com/haomo-ai/Cam4DOcc |
| OccWorld | ECCV 2024 |
| OccSTeP（OccSTeP-WM） | arXiv:2512.15621（2025-12） |
| T3Former（O/F 两版） | 2025 |
| DOME | arXiv:2410.10429（2024-10） |
| Drive-OccWorld | AAAI 2025 |
| FSF-Net | 2024 |
| UniOcc | ICCV 2025 |
| STM4D | OpenReview vfDJiI0dbu（2025） |
| UniScene / COME | OpenReview EYxLmZRSK1（2025，COME 比 DOME +26.3% mIoU，需查原文） |
| OccProphet | 2025（Observer-Forecaster-Refiner 效率路线） |
| Geometry-Aware Spatio-Temporal | ECCV 2026 poster 5843（待查数字） |

---

## 6. 待补数据（写作 agent 若要写完整对比表需补）

| 缺项 | 补法 |
|---|---|
| STM4D 的 1s/2s/3s 具体数字 | OpenReview PDF（需绕验证）或作者主页 |
| COME / UniScene 的绝对数字 | 查 OpenReview EYxLmZRSK1 原文 |
| Geometry-Aware（ECCV26）数字 | 查 ECCV 2026 poster 5843 |
| Drive-OccWorld 的分 horizon 数字 | 查 AAAI25 原文（目前只有 mIoU_f=36.3） |
| T3Former-F 的 2s/3s | 原文可能只报 1s |
| **我们自己的 Cam4DOcc 协议重跑** | ⚠️ 论文正式对比前必须做：当前 21.33/17.64/15.09 是本仓库 4818 划分，非 Cam4DOcc 官方 split |
