# PriorOcc-4D 开发执行手册（代码与无数据训练链路优先）

一句话总结：本项目通过将 PriorOcc 的 2D 语义先验扩展为多帧 SGDM、语义动静解耦、语义条件运动场与递推未来预测，实现更稳定的动态场景时空建模，最终得到一个面向未来 1/2/3 秒 occupancy 与语义占据预测的 4D 占据网络。

一句话创新点：创新性在于把 2D 语义先验从单帧深度辅助提升为贯穿时序融合、运动场生成、逐类动态建模和未来占据预测的显式条件，使占据网络能够按语义类别学习不同的时空演化规律。

本手册交给负责实现的 Agent。它是 [`PRIOROCC_4D_DEVELOPMENT_PLAN.md`](PRIOROCC_4D_DEVELOPMENT_PLAN.md) 的**代码优先执行版本**：最终研究范围完整保留 `nextstep.md` 的“语义先验时空化”——多帧 PriorOcc+SGDM、动静分离、语义运动编码、语义类别 attention、SCMF、GRU、逐类组合、未来语义和语义一致性；但当前本地没有数据集，首要目标是先让代码、配置、合成 batch 的前向/反向/优化/推理/评测链路全部跑通。

不要等待数据集，也不要先写依赖真实 nuScenes 文件的大型训练流程。真实数据接入和租服务器训练只能在“无数据绿灯”通过后进行。

## 0. 范围、原则与成功定义

### 唯一代码树

只修改工作区根目录下的 `projects/`、`tools/`、`doc/` 和 `projects/configs/priorocc/`。不要修改已有 `BEVDepth4DOCC`、`BEVDet4D` 的默认行为；新任务使用新类 `PriorOcc4D`，保证现有 FlashOCC/PriorOcc 配置仍可用。

### 代码优先原则

1. 先实现最小可训练的模型接口和合成验证工具，再实现真实 dataset/index。
2. 每增加一个模块，先通过独立的 shape、有限值、梯度和行为单测，再接入 Detector。
3. 每个阶段都必须能从零构建 config、跑一个 optimizer step，并保留开关回退到上一阶段。
4. 不以“没有数据”作为跳过验证的理由：GT `.npz`、2D PNG 和 info 可以在临时目录中合成，模型输入可由固定随机种子生成。
5. 只在工具测试中使用 future pose/图像/GT；模型输入只允许历史 `t-2,t-1,t`，防止未来泄漏。

### 无数据绿灯（租服务器前必须全部通过）

- `python -m compileall projects/mmdet3d_plugin tools` 成功；
- 4D config 能导入 plugin、构建 `PriorOcc4D`；
- 合成 pipeline 能产出 3 帧历史和 3 个 future 的字段；
- 合成 batch 可完成 forward、所有 loss、backward、optimizer step、`simple_test()`；
- 固定 toy batch 连续优化后总 loss 和未来 occupancy loss 明显下降，且无 NaN/Inf；
- SGDM 三个历史帧均收到非空 `sem_logits`；
- flow、gate、future logits、语义 BEV 不发生恒零/恒常数/单类别塌缩；
- future evaluator 能在临时 `.npz` 上分别输出 1/2/3 s 指标。

## 1. 已有代码：必须复用

| 位置 | 必须复用的能力 | 4D 实现要求 |
|---|---|---|
| `projects/configs/priorocc/priorocc-r50.py` | `SemanticInjector` 与 SGDM：`use_semantic_gating=True`、`sem_channels=17` | 新 config 从此复制语义/SGDM 配置，不重新发明 SGDM。 |
| `models/detectors/bevdet.py` | 单帧 `SemanticInjector → sem_logits → view_transformer(..., sem_logits=...)`，兼容三元返回 | 这是 4D 每帧 `prepare_bev_feat` 的模板。 |
| `models/detectors/bevdet4d.py` | `prepare_inputs()`、`shift_feature()`、历史 BEV 融合 | 只复用，不重写 ego-motion 坐标变换。 |
| `models/detectors/bevdet_occ.py` | `BEVDepth4DOCC`、深度损失、`BEVOCCHead2D` | `PriorOcc4D` 继承/参考它，future 复用同一个 C2H head。 |
| `datasets/pipelines/loading.py` | 当前 occupancy `labels.npz` 的读取与 BDA flip | future loader 使用同一字段/flip 语义。 |
| `tools/integration_test.py`、`debug_semantic_injector.py` | 合成 batch、构建 plugin、forward/backward 的脚本风格 | 新工具保持同样的 root-path、`argparse`、打印和非零退出码习惯。 |
| `tools/debug_data_pipeline.py` | 检查 dataset/pipeline 字段与 `DataContainer` | 新 pipeline 验证工具输出同样的 key、shape、dtype、有效比例。 |

已知缺口：`BEVDet4D.prepare_bev_feat()` 当前没有运行 `SemanticInjector`，没有传 `sem_logits` 给 SGDM，也没有返回历史语义；当前 dataset/evaluator 没有 future GT。这些都是新代码应解决的，不是 config 能自动解决的。

## 2. 推荐文件与接口总览

先建立以下文件骨架和注册，再分阶段填充。允许合并小型模块，但不得把大量运算塞进 detector。

```text
projects/mmdet3d_plugin/models/detectors/priorocc_4d.py
projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py
projects/mmdet3d_plugin/models/model_utils/scmf.py
projects/mmdet3d_plugin/models/model_utils/future_semantic.py
projects/mmdet3d_plugin/models/model_utils/sem_consistency.py
projects/mmdet3d_plugin/datasets/nuscenes_4d_forecast_dataset.py
projects/mmdet3d_plugin/datasets/pipelines/loading_future_occ.py
projects/mmdet3d_plugin/datasets/pipelines/loading_temporal_seg2d.py
projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py
tools/verify_priorocc_4d.py
tools/verify_priorocc_4d_pipeline.py
tools/diagnose_priorocc_4d_training.py
```

需要同时更新相应的 `__init__.py`，否则 registry/config build 测试会失败。

模型内部统一返回 `FeatureBundle`（`dict` 即可）：

```text
fused_bev:           (B, 256, 200, 200)
aligned_bev_history: (B, 3, 64, 200, 200)       # t-2,t-1,t，已对齐到 t
seg_logits_history:  (B, 3, 6, 17, fH, fW)
depth_history:       (B, 3, 6, D,  fH, fW)
semantic_bev:        (B, 17, 200, 200)
visibility:          (B, 1,  200, 200)
```

不要默默猜 shape。所有模块构造函数显式接收 `num_future`、`num_semantic_classes`、`bev_channels`、`raw_bev_channels`，并在 forward 做有信息的断言。

## 3. 开发顺序：先跑通训练链路，再连接真实数据

### Phase A：独立模块与合成工具地基

先实现 `dyn_sta_decoder.py` 和 `scmf.py` 中不依赖 dataset 的模块：

1. `warp_feature(bev, flow_cells)`：输入 `(B,C,H,W)` 与 `(B,2,H,W)`，flow 单位为 BEV cell，内部换算为 `grid_sample(..., align_corners=True)` 的归一化偏移。
2. `SemanticDynStaSeparator`：输入 `(B,17,H,W)` 语义 BEV 和 visibility，输出 `dyn_mask`、`sta_mask`、`per_cls_masks`；动态/静态类别 ID 必须从 config 提供，不可硬编码。
3. `SemanticMotionFeatureEncoder`：输入三个对齐 raw BEV 和 semantic BEV，构建 `delta_1/delta_2/accel`，输出 `motion_feat`。
4. `SemanticMotionAttention`：以逐类 semantic mask 池化得到 query，BEV 为 K/V、motion 为 bias；避免将类别 token 复制为完整 `200×200` 图后再做无意义卷积。
5. `SemanticConditionedMotionField`（SCMF）：输入 `fused_bev + motion_feat + semantic_bev + attention feature`，输出 `(B,T,2,H,W)` flow。flow head 最后一层权重/偏置置零，限制为 `tanh()*max_flow_cells`。
6. `SCMFEnhancedPredictor`：按 future step 自回归 warp 上一步 BEV、GRU 更新状态、预测 delta/gate，输出所有 future BEV。gate 初始化偏向 warp 支路，但不可全固定。
7. `PerClassDeltaCombiner`：用逐类 mask 在静态支路与动态预测支路间组合；允许使用 group head 控制参数量，但必须支持二值 mask 对照。

先用 `tools/verify_priorocc_4d.py --stage modules` 验证。工具必须包含：

- `zero-flow`: 输出与输入 `allclose`；
- `known-translation`: 一个亮点向已定义方向移动一格，检查方向与边界；
- `mask-partition`: `dyn_mask + sta_mask` 在可见位置近似 1、范围均在 `[0,1]`；
- `finite-and-grad`: 每个模块输出/梯度均有限，至少一个参数梯度非零；
- `semantic-sensitivity`: 固定 BEV/motion、改变一个区域的 semantic class，SCMF flow 或 per-class gate 应发生可检测变化；否则说明语义通路被断开；
- `identity-init`: SCMF 初始 flow 接近零，future feature 不应 NaN/爆炸。

### Phase B：4D SGDM 与当前帧训练闭环

新增 `PriorOcc4D(BEVDepth4DOCC)`，但先只启用当前 occupancy loss 和历史 2D semantic loss。具体实现：

1. 基于 `BEVDet4D.prepare_inputs()` 拆出 3 帧图像/几何；
2. 每一帧都执行 `image_encoder → SemanticInjector`；
3. 严格按 `BEVDet.extract_img_feat()` 调用 view transformer：`view_transformer(inputs, sem_logits=seg_logits)`，处理 `(bev, depth, refined_sem_logits)`；
4. 用继承的 `shift_feature()` 对齐历史 raw BEV，保留 `aligned_bev_history`，按既有顺序 concat 后经 `bev_encoder()` 得到 `fused_bev`；
5. `SemanticBEVProjector` 必须使用 LSS view transformer 的 frustum/geometry/BEV pooling helper，把当前 `softmax(seg_logits)` 与 depth probability 投影到 `semantic_bev`；禁止独立手写一套投影公式；
6. `forward_train()` 先计算 `loss_depth`、`loss_occ_current` 和三帧 `loss_2d_seg_history`；`simple_test()` 先保证当前预测可返回。

`tools/verify_priorocc_4d.py --stage sgdm-current --device cuda` 必须：hook 三帧 SemanticInjector 和 view transformer，断言三次 `sem_logits` 都非空；打印 FeatureBundle；执行 `forward_train → sum(loss) → backward → AdamW.step()` 两次。

### Phase C：未来 occupancy 的最小训练闭环

在不依赖真实 dataset 的条件下加入 future labels 接口和直接多步 occupancy：

```text
future_voxel_semantics: (B,3,200,200,16)
future_mask_camera:     (B,3,200,200,16)
future_mask_lidar:      (B,3,200,200,16)
gt_semantic_2d_history: (B,3,6,H,W)
```

1. 先实现轻量 `DirectFutureOccupancyHead`，由 `fused_bev` 产生三步 future BEV/occ；这是 E2 的训练基线和后续 SCMF 的回退路径。
2. 每一个 slot 用共享 `BEVOCCHead2D` 和自己的 `future_*[:, k]` 计算 loss，loss key 固定为 `loss_occ_future_1s/2s/3s`，权重由 config 的 `[1.0,0.7,0.5]` 控制。
3. `simple_test()` 必须返回 `pred_occ_current`、按 1/2/3 s 顺序的 `pred_occ_future` 和 `horizons_sec`。
4. 为保留原计划的最终路径，Direct head 的接口要与 `SCMFEnhancedPredictor` 相同：都返回 `list[future_bev]`；后续可用 config switch 替换，而不是重写 loss/eval。

运行 `tools/verify_priorocc_4d.py --stage forecast-smoke --steps 2`。断言改变 `future_voxel_semantics[:,1]` 只改变 2 s loss；future 预测长度为 3；后向时 future head 参数有非零梯度。

### Phase D：完整 `nextstep` 运动主线

按以下顺序接入，**不要删除任何最终模块**：

1. `SemanticDynStaSeparator + SemanticMotionFeatureEncoder`（E3/E4）；
2. `SemanticMotionAttention`（E5）；
3. `PerClassDeltaCombiner`（E6）；
4. `SCMF + MotionFieldWarper + GRU + gate`（E6+SCMF）；
5. 可切换 direct / auto-regressive future rollout，默认 SCMF 自回归；
6. `FutureSemanticPredictor`（E7）：若宣称 future **2D** semantic，必须通过 future camera calibration/rendering 得到 camera logits 并用 future pseudo labels 监督；否则名称必须为 future **BEV** semantic auxiliary，不能混淆；
7. `SemanticTemporalConsistency`（E8）：仅在 `visible ∩ high-confidence ∩ static` 区域对齐计算，排除动态、遮挡及新出现区域。

每一步先在 synthetic test 运行，再连接真实 pipeline。配置必须能关闭新模块，复现前一阶段输出/接口。

### Phase E：真实数据接入（仅在无数据绿灯后）

1. 完成/核验已有 `tools/create_4d_forecast_infos.py`，以 2 Hz 的 `+2/+4/+6` 作为 1/2/3 s 默认 offset；执行 `--verify-only`。
2. 新增 `NuScenes4DOccForecastDataset`、`LoadFutureOccGTFromFiles`、`LoadTemporalSemanticSeg2D`、`LoadFutureSemanticSeg2D`（E7 时启用）。
3. 通过临时合成目录先测试 loader：`labels.npz` 含 semantics/masks，伪标签为 PNG；验证字段、BDA flip、路径异常、时间/scene 边界。
4. 真实数据到位后只先跑 1 个 batch、20 个样本和 100–300 iter smoke run；通过后才启动完整多卡训练。

## 4. 必须实现的无数据验证工具

工具遵循现有 `tools/integration_test.py`、`tools/debug_*` 风格：可直接从仓库根目录运行，使用 `argparse`，导入 plugin，打印可读表格；失败抛异常/返回非零退出码，不以 warning 伪装通过。

### 4.1 `tools/verify_priorocc_4d.py`

这是主验证入口，建议参数：

```bash
python tools/verify_priorocc_4d.py \
  --config projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py \
  --stage all --device cuda --batch-size 1 --seed 3407 --steps 2
```

`--stage` 支持 `modules`、`build`、`sgdm-current`、`forecast-smoke`、`toy-overfit`、`inference`、`all`。需实现：

- `build_synthetic_batch()`：生成 `(B, 18, 3, 256,704)` 历史图像和符合 3 帧格式的内外参/BDA；相机几何必须是有限且可逆的合理矩阵，不能只凑 shape。生成 current/future `(B,3,200,200,16)` target，2D semantic target 和 depth target。
- 固定种子、可选 `--tiny`（允许 config 中将输入/BEV 尺寸缩小，仅用于新模块测试）与全分辨率 smoke；全分辨率运行失败时报告显存，不可静默跳过。
- config build、hook SGDM 参数、FeatureBundle shape/finiteness、loss key、backward、`clip_grad_norm_`、optimizer step、参数是否改变。
- 打印每层参数数、显存峰值、loss、梯度范数；有 CUDA 时执行 `torch.cuda.synchronize()` 后计时。

合成 target 不要完全随机。采用固定的占用块、不同 horizon 的平移车辆块、静态道路/建筑块和相应 2D 类别区域，使 future slot 和运动模块有可检查的信号。可用随机图像特征，但 target 图案与 seed 固定。

### 4.2 `tools/diagnose_priorocc_4d_training.py`

输入 `--stats-json`（由主验证工具每 step 写出）或 checkpoint + synthetic batch，输出 JSON/文本诊断，并在阈值触发时退出失败。每个 step 至少记录：

| 诊断 | 失败条件（建议默认） | 意义 |
|---|---|---|
| loss/grad 有限性 | 任一 NaN/Inf | 数值已失效。 |
| loss 下降 | toy 训练 20 step 后 future total 不低于初始的 90% | 模型/梯度/target routing 可能断路。 |
| 参数更新 | trainable 参数变化比例为 0 | optimizer 或 loss 未连接。 |
| 梯度覆盖 | SemanticInjector、SCMF、GRU、occ head 任一启用模块 grad norm 恒为 0 | 模块未参与训练。 |
| logits 熵 | 所有 future voxel 平均熵接近 0，且同一类占比 >99.5% | 单类别塌缩。 |
| future 多样性 | 三个 horizon 的 logits/argmax 完全相同 | 预测头或时间条件未生效。 |
| flow | mean abs 长期约 0 且 flow-head grad 非零后仍不变；或大量值贴近 max | 恒等塌缩/饱和爆炸。 |
| gate | 均值长期 `<0.01` 或 `>0.99`，标准差近 0 | 单支路塌缩。 |
| semantic BEV | visibility 有效区域内方差约 0 或全零 | 投影/语义通路断开。 |
| mask | dynamic/static overlap 过大或有效区和不接近 1 | 类别 mask 逻辑错误。 |

阈值应当可通过 CLI/config 调整。初始 zero-flow 是正确初始化，故仅在优化若干 step 后结合 flow-head 梯度判断“恒等塌缩”，不能在 step 0 报错。

### 4.3 `tools/verify_priorocc_4d_pipeline.py`

此工具不需要 nuScenes：用 `tempfile.TemporaryDirectory()` 创建 3 个 history、3 个 future 的小型伪 `labels.npz` 与 2D PNG，再构造最小 `results` dict，直接调用新 loader/pipeline。参数：`--test loader|flip|missing|all`。

必须验证：

- 每个 future 的 labels/masks stack 顺序、shape、dtype；
- `flip_dx/flip_dy` 同时作用于 semantics 和两个 mask；
- 缺失 labels key/file、错误类别、错误 shape 均抛出带路径/token 的 `RuntimeError`；
- temporal 2D label 的帧/相机排序与 `PrepareImageInputs` 一致；
- 模拟 `Collect3D`/collate 后字段仍可供 `forward_train()` 读取。

### 4.4 `tools/verify_priorocc_4d_evaluator.py`

用临时 future GT 和构造预测测试 `evaluate_forecast()`：三个 horizon 的 perfect prediction 应给对应满分；只破坏 2 s prediction 时只能降低 2 s 指标与 average；current prediction 不能影响 future average。这能在没有真实数据时验证 token/slot/evaluator 对齐。

## 5. 训练工具与 toy-overfit 规范

`verify_priorocc_4d.py --stage toy-overfit --steps 20` 是租服务器前最重要的检查。推荐流程：

1. 固定一个 `B=1` synthetic batch；先 freeze image backbone/view transformer，只训练 future head、motion modules、occ head，避免小机器因全模型反向过慢。
2. 连续 20 step 的 AdamW（低学习率可配），记录每步总 loss/current/future 分项、梯度、flow/gate/logits 统计；保存到 `work_dirs/priorocc4d_smoke/toy_stats.json`。
3. 先要求 future loss 有明显下降；若没有，逐级执行 `--stage forecast-smoke`、loss routing test、检查目标类别/ignore mask、检查 occ head logits layout。
4. 再解冻 SemanticInjector、SCMF、GRU；要求这些已启用模块至少一次梯度非零且参数更新。
5. 最后运行 `--stage inference`，确定 `simple_test()` 输出三个合理 dtype/shape 的 future occupancy，不调用任何 GT。

不要把“固定 toy batch 过拟合”当成性能证据；它只验证训练链路可学习和无明显塌缩。真实泛化只能在数据/服务器到位后评估。

## 6. 配置要求

新增 `projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py`。必须：

- 以 `priorocc-r50.py` 复制 `semantic_injector` 和 SGDM `depthnet_cfg`；
- 3 帧历史（当前+2 adjacent），BEV encoder `numC_input=numC_Trans*3`；
- 明确 `forecast_cfg.num_future=3`、`horizons_sec=[1,2,3]`、`future_loss_weights=[1,.7,.5]`、动态/静态类别映射、flow 限制、所有模块开关和损失权重；
- 初期 `enable_future_semantic=False`、`enable_semantic_consistency=False`，但配置和模块骨架必须存在；Phase D/E 后打开；
- 提供 `smoke_test=True` 或单独 `priorocc-4d-r50-stgdm-scmf-smoke.py`，仅降低 batch/workers/迭代数，不改变模型 tensor 定义，防止“smoke 能跑而正式配置形状不同”。

## 7. 真实数据到位后的最小流程

真实数据不是本轮阻塞项。到服务器后按此顺序，不跳级：

```bash
python tools/create_4d_forecast_infos.py --root-path data/nuscenes --verify-only --max-samples 20
python tools/verify_priorocc_4d_pipeline.py --config projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py --use-real-sample
python tools/verify_priorocc_4d.py --config projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py --stage all --device cuda
./tools/dist_train.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 1 --cfg-options runner.max_iters=100
./tools/dist_test.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py <checkpoint> 1 --eval mIoU
```

首次真实训练必须查看诊断 JSON、三步 future 可视化和 evaluator slot 对齐，再扩展到多卡、完整 epoch、E0–E9 与 D1–D6/C1–C5 消融。

## 8. Agent 每次交付必须报告

每完成一个可独立阶段，报告：

1. 修改/新增文件；
2. 新接口及关键 tensor shape；
3. 已运行命令与完整通过/失败结果；
4. 无数据绿灯清单的当前状态；
5. 尚未验证的真实数据假设；
6. 不能以“预期能运行”替代真实工具输出。

禁止事项：不读取 future image/semantic/pose 作为模型输入；不把 0.5/1/1.5 s 错报为 1/2/3 s；不静默吞掉缺失标签；不另建 BEV 坐标系；不将 IoU、mIoU、mIoU_f 混为可比指标；不把 synthetic overfit 当成科研结果。
