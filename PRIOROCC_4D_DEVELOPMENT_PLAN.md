# PriorOcc-4D：以语义先验时空化为核心的开发方案

一句话总结：本项目通过将 PriorOcc 的 2D 语义先验扩展为多帧 SGDM、语义动静解耦、语义条件运动场与递推未来预测，实现更稳定的动态场景时空建模，最终得到一个面向未来 1/2/3 秒 occupancy 与语义占据预测的 4D 占据网络。

一句话创新点：创新性在于把 2D 语义先验从单帧深度辅助提升为贯穿时序融合、运动场生成、逐类动态建模和未来占据预测的显式条件，使占据网络能够按语义类别学习不同的时空演化规律。

本方案以 [`nextstep.md`](nextstep.md) 为唯一研究主线：不是把 PriorOcc 简化为“语义辅助的残余流”，而是将其已有的 2D 语义先验从单帧特征注入，系统扩展到**时序融合、动静解耦、运动推理、语义条件运动场、未来语义和跨帧一致性**。此前的最小 flow 版本仅作为工程排错基线，不代表最终研究方法。

## 1. 立项结论：可行且具有较高科研价值

| 维度 | 评分 | 判断 |
|---|---:|---|
| 工程可行性 | **7.0 / 10** | FlashOCC 已有 4D 历史帧读取、BEV ego-motion 对齐和 C2H occupancy 解码；PriorOcc 已有 SemanticInjector 与 SGDM。主要工作是补齐未来标签/评测，以及将这些能力贯通到新的 4D detector。 |
| 研究创新价值 | **8.0 / 10** | 核心不是“加一个语义 mask”，而是把显式 2D 语义作为运动编码、类别查询、运动场生成、未来语义预测和时序约束的共同条件，构成完整的“语义先验时空化”叙事。 |
| 可发表潜力 | **7.5 / 10** | 若严格证明 SCMF 在动态区域和长时预测上的独立增益，并完成全链路语义消融、效率和可解释性分析，具备较好的论文价值。 |
| 综合 | **7.7 / 10** | 建议按本方案推进；核心贡献应锁定为 SCMF 与全链路语义先验验证，而非仅宣称时序模型更强。 |

研究假设是：**语义类别不仅回答“那里有什么”，也提供“它可能如何运动”的可学习先验；因此运动场应由语义特征条件化生成，而不是只由 BEV 差分独立预测。**

需如实限定创新边界：动态/静态掩码、BEV warp、GRU 和未来预测均已有相关工作；本项目的创新价值来自 PriorOcc 语义先验在完整 4D 链路中的一致、可检验使用，以及以 SCMF 为中心的因果消融，而非这些基础算子的单独新颖性。

## 2. 目标架构：完整继承 `nextstep.md`

```text
历史图像 t-2, t-1, t
  └─ 每帧 Backbone + FPN
      └─ SemanticInjector ──> seg_logits
          └─ LSSViewTransformerBEVDepth + SGDM ──> raw BEV / depth
              └─ ego-motion 对齐与时序融合 ──> fused BEV
                  ├─ 语义 BEV 投影、动静分离、逐类 mask
                  ├─ 语义增强 MotionFeatureEncoder
                  ├─ 语义类别驱动 SemanticMotionAttention
                  ├─ SCMF：语义条件化运动场 ──> BEV warp（粗预测）
                  ├─ GRU 残差精炼 + 门控融合（细预测）
                  ├─ Per-Class Delta Combiner
                  ├─ Future Occupancy Head ──> future occupancy
                  └─ Future Semantic Predictor ──> future semantic / BEV 条件

对齐后的历史语义 ──> mask-aware Semantic Temporal Consistency
```

最终方法必须保留以下六个研究模块；开发时可分阶段打开，避免把“尚未验证”误写成“已不需要”。

| 模块 | `nextstep` 中的作用 | 最终实现定位 |
|---|---|---|
| 多帧 PriorOcc + SGDM | 每帧注入语义，并以语义门控深度估计 | 所有后续语义模块的共同源头 |
| 语义动静分离与逐类 mask | 区分可运动物体与静态区域 | 运动正则、attention 和逐类组合的空间依据 |
| 三项运动增强 | 语义进入运动编码、类别 query、逐类残差组合 | 证明语义不只是附加输入 |
| SCMF | 从 BEV、运动特征、语义 BEV 直接生成连续运动场 | 首要方法贡献 |
| GRU 残差与 future occupancy | warp 给粗预测，递推残差补细节 | 处理遮挡、形变和非刚体变化 |
| 未来语义与时序一致性 | 语义预测—占用预测链和跨帧约束 | 让语义先验覆盖未来及时间维度 |

## 3. 现有代码依据与改造边界

### 已有能力

- [`projects/configs/priorocc/priorocc-r50.py`](projects/configs/priorocc/priorocc-r50.py) 已启用 `SemanticInjector`，并配置 `LSSViewTransformerBEVDepth.depthnet_cfg.use_semantic_gating=True`、`sem_channels=17`、`sgdm_reduction=4`。这就是可复用的 SGDM 配置来源。
- [`projects/mmdet3d_plugin/models/detectors/bevdet.py`](projects/mmdet3d_plugin/models/detectors/bevdet.py) 的单帧 `extract_img_feat()` 已实现 `SemanticInjector → seg_logits → view_transformer(..., sem_logits=seg_logits)`，并兼容 SGDM 返回的 `(bev_feat, depth, refined_sem_logits)`。
- [`projects/mmdet3d_plugin/models/detectors/bevdet4d.py`](projects/mmdet3d_plugin/models/detectors/bevdet4d.py) 已有 `prepare_inputs()`、`shift_feature()` 和历史 BEV 对齐/拼接融合，但其 `prepare_bev_feat()` 仅调用 `image_encoder()` 与 view transformer，未调用 `SemanticInjector`，也未保留逐帧 `seg_logits`。这是 4D 语义主线的首个明确缺口。
- [`projects/mmdet3d_plugin/models/detectors/bevdet_occ.py`](projects/mmdet3d_plugin/models/detectors/bevdet_occ.py) 中的 `BEVDepth4DOCC` 已具备深度监督、当前占用损失和 `BEVOCCHead2D`，可作为新 detector 的父类或直接参考。
- [`projects/mmdet3d_plugin/datasets/pipelines/loading.py`](projects/mmdet3d_plugin/datasets/pipelines/loading.py) 只读取当前 `labels.npz`；[`nuscenes_dataset_occ.py`](projects/mmdet3d_plugin/datasets/nuscenes_dataset_occ.py) 也只评估当前帧。因此 forecasting 的 future index、future GT 和 evaluator 必须新增。

### 不应触碰的范围

- 不修改现有 `BEVDepth4DOCC`/`BEVDet4D` 的行为，以免影响 FlashOCC 既有配置；新增 `PriorOcc4D` detector。
- 不删除合法的 `projects/configs/priorocc/` 配置目录。
- 不另写一个与 LSS 几何不一致的 `project_to_bev()`；语义概率必须沿用 view transformer 的 frustum、geometry 与 BEV pooling。

## 4. 数据协议：先锁定，避免方向被错误标签带偏

`nextstep.md` 的研究目标保持不变，但其时间示例需要以实际 keyframe 协议校正。nuScenes 通常为 2 Hz：若 anchor 为 `t`，则索引 `t+1/t+2/t+3` 对应约 `0.5/1.0/1.5 s`；若论文报告 `1/2/3 s`，默认应取 `t+2/t+4/t+6`。最终必须以选定 Cam4DOcc 数据版本的官方定义和 evaluator 为真源。

### 4.1 新增 forecasting 索引

新增 `tools/create_4d_forecast_infos.py`，在现有 info 上建立：

```python
forecast = dict(
    history_indices=[t-2, t-1, t],
    history_tokens=[...],
    future_indices=[t+2, t+4, t+6],  # 对应 1/2/3 s；若官方协议不同则替换
    future_tokens=[...],
    future_occ_paths=[...],
    horizons_sec=[1.0, 2.0, 3.0],
)
```

约束：同一 scene、未来标签存在、时间差满足容差、语义类别和 `(200, 200, 16)` 体素形状合法。跨 scene、缺 future、标签异常的 anchor 必须丢弃并输出统计。

### 4.2 Dataset/Pipeline

新增 `NuScenes4DOccForecastDataset`（或在 `NuScenesDatasetOccpancy` 可选扩展）和以下 pipeline：

1. `LoadFutureOccGTFromFiles`：读取 `T=3` 个 `labels.npz`，输出 `future_voxel_semantics`、`future_mask_camera`、`future_mask_lidar`、`future_tokens`；必须同步已有 BDA flip。
2. `LoadTemporalSemanticSeg2D`：按 `PrepareImageInputs(sequential=True)` 的真实帧、相机顺序读取历史 3 帧伪标签，输出 `(3, 6, 256, 704)`；不能沿用现有 `LoadSemanticSeg2D` 缺标签时全 `255` 的静默退化行为。
3. 可选 `LoadFutureSemanticSeg2D`：读取真实未来 SegFormer 伪标签；这是未来语义预测的首选监督。只有 future 图像伪标签无法获得时，才增加 RAFT/ego-motion 生成的 teacher target，并保存置信度与遮挡 mask。

`Collect3D` 至少收集 `img_inputs`、`gt_depth`、当前 occupancy/mask、future occupancy/mask、`gt_semantic_2d_history`、`gt_semantic_2d_future`（可选）和 forecast meta。

### 4.3 Eval

新增 `evaluate_forecast()`：输出每个 horizon 的 mIoU、平均 mIoU、动态类 mIoU/GMO、静态类 mIoU/GSO、各 mask 下有效体素比例。当前 Occ3D mIoU、Cam4DOcc IoU 和其他论文 mIoU_f 必须分开列示，不能合并排名。

## 5. 详细模型开发设计

### 5.1 Detector 与 4D SGDM 迁移

新增 `projects/mmdet3d_plugin/models/detectors/priorocc_4d.py`：

```python
@DETECTORS.register_module()
class PriorOcc4D(BEVDepth4DOCC):
    ...
```

重写其 `prepare_bev_feat()`，对每一个历史帧执行以下等价于单帧 PriorOcc 的路径：

```text
image_encoder → reshape(B*Ncam,C,H,W) → SemanticInjector
→ (semantic-enhanced feature, seg_logits)
→ LSSViewTransformerBEVDepth(..., sem_logits=seg_logits)
→ (raw_bev, depth, refined_seg_logits)
```

返回 `raw_bev_list`、`seg_logits_list`、`depth_list` 与几何输入。历史 raw BEV 用父类 `shift_feature()` 对齐；对齐后再输入原有 `bev_encoder()` 得到 `fused_bev`。必须让每一历史帧可选反传；若显存受限，可先仅 key frame 反传，但在配置和论文中明确这一训练策略。

建议内部特征契约：

```text
aligned_bev:       (B, 3, 64, 200, 200)     # t-2, t-1, t，均在 t 坐标
fused_bev:         (B, 256, 200, 200)
seg_logits_history:(B, 3, 6, 17, fH, fW)
depth_history:     (B, 3, 6, D, fH, fW)
semantic_bev:      (B, 17, 200, 200)
```

### 5.2 语义 BEV、动静分离与逐类 mask

新增 `projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py`：

- `SemanticBEVProjector`：对 `softmax(seg_logits_t)` 与 depth probability 做类别—深度外积，并复用 LSS 的 geometry/pool helper，产生 `semantic_bev` 和 visibility；不得通过独立相机投影公式获得 BEV。
- `SemanticDynStaSeparator`：从 `semantic_bev` 取动态类别概率的可学习加权和，输出 `dyn_mask`、`sta_mask` 与每个动态类别的 `per_cls_masks`。初始类别集合按实际 17 类映射确定，不能直接硬编码 `nextstep.md` 的示例 id。
- 高置信静态区为 `sta_mask * visibility > threshold`，它是 motion regularization 与 semantic consistency 的有效区域。

### 5.3 三项语义增强

仍放入 `dyn_sta_decoder.py`，并由 config 逐个开关：

1. `SemanticMotionFeatureEncoder`：输入对齐后的 `delta_1 = BEV_t-BEV_t-1`、`delta_2 = BEV_t-1-BEV_t-2`、`accel = delta_1-delta_2` 和 `semantic_bev`，输出 `motion_feat`。这是语义特征参与运动编码。
2. `SemanticMotionAttention`：使用每类 `per_cls_mask` 加权池化 `fused_bev`，得到类别条件 query；以 BEV feature 为 K/V，`motion_feat` 提供 attention bias，且只在 dynamic mask 支持区域注意。实现应避免把每类全局 token 直接 `repeat` 到 `200×200` 造成无意义显存开销；可用 cross-attention 输出的类别 token 与局部 BEV 的投影融合。
3. `PerClassDeltaCombiner`：每类动态 mask 对应独立/分组的 delta gate，将静态 warp、SCMF/GRU 预测在类别级进行空间组合。首版可采用 group head，而不是为每个类别复制完整 decoder；但必须保留“二值 dynamic mask 替换 per-class mask”的消融。

### 5.4 SCMF、warp、GRU 残差

新增 `projects/mmdet3d_plugin/models/model_utils/scmf.py`：

- `SemanticConditionedMotionField`：输入 `[fused_bev, motion_feat, semantic_bev, semantic_motion_attention]`，经 2–3 层 Conv-BN-ReLU，输出 `(B, T, 2, 200, 200)` 的 residual motion field（单位为 BEV cell）。初始 motion head 置零，令训练起始为 identity warp；`tanh * max_flow_cells` 限制位移。
- `MotionFieldWarper`：以 `grid_sample(align_corners=True)` 将 cell 位移换算为 normalized grid offset；零 flow 必须严格为恒等映射。注意 grid 的采样方向需要用合成平移单测固定，避免“预测的是 forward flow，却按 backward sampling 使用”的符号错误。
- `SCMFEnhancedPredictor`：每个 horizon 先对当前/上一步 BEV warp，再以 GRUCell 递推 hidden state，输出残差 `delta_k` 和 gate `alpha_k`：

```text
coarse_k = warp(bev_{k-1}, flow_k)
refined_k = bev_{k-1} + GRU_delta_k
future_bev_k = alpha_k * coarse_k + (1-alpha_k) * refined_k
```

这里建议采用自回归 `bev_{k-1}`（`k=1` 时是 `fused_bev`），让 1/2/3 s 的运动演化有明确递推含义；并保留“direct independent horizon”作为对照。静态物体不依赖对象运动预测，而以 ego-motion 已对齐后的 current BEV 为稳定支路。

### 5.5 Future occupancy 与未来语义

- `FutureOccupancyHead`：复用一个 `BEVOCCHead2D` 对每个 `future_bev_k` 解码，确保 C2H 效率；共享 decoder是默认方案，独立 horizon decoder是参数量对照。
- `FutureSemanticPredictor`：不能直接把 BEV feature 误称为“未来 2D 语义图”。正确实现为：先为每个未来时刻预测/得到 future BEV semantic state，再通过对应 future camera calibration 和 differentiable projection/rendering 输出每相机 2D logits，或把它明确定义为 BEV semantic auxiliary head。若宣称“future 2D semantic”，必须采用前者并以 future image pseudo-label 监督。
- future semantic BEV embedding 通过 `1×1` projection 反馈进每步 occupancy refinement，形成 `future semantic → future occupancy` 条件链；训练时可先 teacher forcing/stop-gradient，稳定后再端到端联合。

### 5.6 语义时序一致性

新增 `sem_consistency.py`。不是在全部区域直接 KL：将历史语义以各帧相机几何/BEV 对齐到 current 坐标，仅在 `high-confidence ∩ static ∩ visible-in-both` 区域计算对称 KL 或 CE，动态目标、遮挡、新出现区域必须排除。这样保留 `nextstep` 的跨帧语义约束思想，同时避免把真实运动当成错误。

## 6. 损失与训练策略

完整训练目标：

```text
L = L_depth + L_occ,current + Σ_k w_k L_occ,future,k
  + λ2d L_sem,history
  + λmotion L_static-motion
  + λcons L_sem-temporal
  + λfsem L_future-semantic
```

建议起始权重为 `w=[1.0, 0.7, 0.5]`、`λ2d=0.3`、`λmotion=0.05`、`λcons=0.05`、`λfsem=0.1`，用小规模验证集搜索后固定。`L_future-semantic` 在未来伪标签质量和投影链被验证前保持关闭；这不是放弃原始模块，而是防止噪声掩盖其贡献。

训练日程：先训练当前帧 PriorOcc-4D/SGDM，加载其权重训练 occupancy forecasting；再解冻 SemanticInjector 与 SGDM 做联合微调。每新增一个研究模块均保留 checkpoint 与固定随机种子，避免全系统同时变动导致归因失败。

## 7. 分阶段开发流程与验收

| 阶段 | 实现范围 | 验收标准 |
|---|---|---|
| P0：协议 | forecast info、future GT、future evaluator、static-copy baseline | token 同 scene；horizon 无误；future GT/mask/类别合法；评测能分别报告 1/2/3 s。 |
| P1：PriorOcc-4D 地基 | 多帧 SemanticInjector、每帧 SGDM、历史 BEV/semantic 对齐、当前占用 | SGDM 4D 路径确实收到 `sem_logits`；`seg_logits_history` 有正确形状；关闭 injector 时回到 FlashOCC 4D 行为。 |
| P2：未来预测骨架 | Direct future head、future loss、B0 | 单 batch 过拟合；future slot 与 future token 一一对应；无未来泄漏。 |
| P3：语义运动基础 | semantic BEV、动静/逐类 mask、MotionEncoder、SemanticMotionAttention、PerClassCombiner | image semantic、semantic BEV、mask 三联图与 BEV 网格一致；动态/静态类别映射可审计。 |
| P4：核心 SCMF | SCMF、MotionFieldWarper、GRU、gate、自回归 future BEV | zero flow identity、合成平移、梯度、长时递推稳定；得到 E6+SCMF。 |
| P5：语义时间闭环 | future semantic predictor、语义反馈、mask-aware consistency | future semantic 使用真实 future camera supervision或明确 BEV 定义；consistency 不作用于动态/遮挡区域。 |
| P6：论文级实验 | 全部消融、三 seed、效率/可视化/失败分析 | 结论由均值±标准差、动态区与长时表现支持，不能只报告最佳单次结果。 |

## 8. 实验矩阵：围绕 `nextstep` 的每一层语义贡献

主线顺序保留 `nextstep` 的 E0–E9：

| 组别 | 对比 | 回答的问题 |
|---|---|---|
| E-Flash4D | 原生 FlashOCC 4D | 历史 BEV 融合本身能做到什么？ |
| E-Flash4D+Sem / E1 | 加多帧 PriorOcc + SGDM | 语义注入对时序主干的增量？ |
| E2 | 加 direct future head | 预测任务的必要基线？ |
| E3–E6 | 依次加分离、MotionEncoder、语义 query、逐类 combiner | 语义如何逐级参与运动推理？ |
| E6+SCMF | 加语义条件运动场 + warp + GRU | 核心 SCMF 是否超过无语义运动模型？ |
| E7 | 加 future semantic | 未来语义条件链是否有增益？ |
| E8 | 加 mask-aware consistency | 跨帧语义约束是否有增益？ |
| E9 | 固定超参后的完整方法 | 完整“语义先验时空化”结果。 |

必须额外运行的因果消融：

- D1：从 MotionEncoder 去掉 `semantic_bev`；
- D2：用随机/可学习固定 query 替换语义类别 query；
- D3：用二值 dynamic mask 替换逐类 mask；
- D4：SCMF 去掉 `semantic_bev`，但保持参数量相等；这是最关键对照；
- D5：去 future semantic；D6：去 consistency；
- C1/C2/C3：分别去 SCMF、去 GRU、固定 gate；
- C4/C5：SCMF 深度与各输入条件消融；
- 每个核心实验报告 3 seed 的 1/2/3 s、平均、GMO/GSO、参数量、FLOPs、延迟。

成功判据：完整模型必须相对等参数“无语义 SCMF”在多 seed 上呈稳定正增益，且动态类别或长时 horizon 的改善更明显；否则只能说明运动模块有效，不能说明语义条件化是核心原因。

## 9. 配置、文件清单与测试

计划新增/修改的主代码树文件：

```text
tools/create_4d_forecast_infos.py
projects/mmdet3d_plugin/datasets/nuscenes_4d_forecast_dataset.py
projects/mmdet3d_plugin/datasets/pipelines/loading_future_occ.py
projects/mmdet3d_plugin/datasets/pipelines/loading_temporal_seg2d.py
projects/mmdet3d_plugin/datasets/pipelines/__init__.py
projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py
projects/mmdet3d_plugin/models/model_utils/scmf.py
projects/mmdet3d_plugin/models/model_utils/future_semantic.py
projects/mmdet3d_plugin/models/model_utils/sem_consistency.py
projects/mmdet3d_plugin/models/model_utils/__init__.py
projects/mmdet3d_plugin/models/detectors/priorocc_4d.py
projects/mmdet3d_plugin/models/detectors/__init__.py
projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py
```

配置以 `priorocc-r50.py` 的 `semantic_injector` 和 `depthnet_cfg` 原样为起点，并显式设置 `sequential=True`、3 帧历史、future offsets/horizons、动态类别映射、各模块开关和损失权重。

最少测试集：数据索引/scene 边界、BDA flip、时序 2D 标签顺序、SGDM 三元返回、zero-flow identity、已知平移 warp、`B=1/2` 前反传、future evaluator、无未来图像/标签泄漏。每个阶段保存一张固定样本的“2D semantic—semantic BEV—mask—motion field—future occupancy—future semantic”可视化。

## 10. 风险与论文表述

- 不承诺 `nextstep.md` 中的预估 mIoU、FLOPs 或 SOTA；它们应写成待验证假设，而非结论。
- 不将 GT occupancy 输入方法与 camera-only 方法混为同一排名；也不混用 IoU、mIoU、mIoU_f。
- 未来语义伪标签与语义 BEV 投影是最高风险点；若其监督不可靠，可保留为完整方法的后续阶段，但论文主张必须对应实际启用的模块。
- 4D 历史融合解决 ego motion，不等于对象未来运动；SCMF 的价值正是显式建模此残余对象运动，必须用无语义 flow 和无 warp 对照证明。

最终论文的一句话主张建议保持为：**PriorOcc-4D 将 PriorOcc 的 2D 语义先验时空化：语义不仅增强单帧占用特征，还通过动静分离、类别驱动运动推理、SCMF、未来语义和一致性约束，显式驱动 camera-only 4D occupancy forecasting。**
