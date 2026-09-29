# PriorOcc-4D 启动指南

> 本文件包含 PriorOcc-4D 全流程启动指令：SegFormer 2D 语义先验生成 → 占用率 GT 准备 → 预测信息生成 → 训练启动 → 评估，以及所有可调参数。

---

## 0. 环境与数据目录

### 环境

```bash
conda activate flash
# Python 3.9, PyTorch 1.10.0+cu111, mmcv 1.5.3
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 数据目录结构

```
data/nuscenes/
├── samples/                        # nuScenes 原始图像
├── sweeps/                         # nuScenes 原始图像
├── gts/{scene_name}/{token}/       # 占用率 GT (labels.npz) ← 外部下载
├── seg_2d_labels/samples/CAM_*/    # SegFormer 2D 语义伪标签 (*.png) ← 脚本生成
├── bevdetv2-nuscenes_infos_train.pkl   # BEVDet 信息文件 ← 脚本生成
├── bevdetv2-nuscenes_infos_val.pkl
├── bevdetv2-nuscenes_infos_train_forecast.pkl  # 4D 预测信息 ← 脚本生成
├── bevdetv2-nuscenes_infos_val_forecast.pkl
└── ...
```

---

## 1. SegFormer 2D 语义先验生成

使用预训练 SegFormer-B2 (Cityscapes) 对 nuScenes 全部相机图像推理，生成 17 类语义伪标签 PNG。

### 安装依赖

```bash
pip install transformers pillow tqdm opencv-python
```

### 生成命令

```bash
# 完整 trainval
python tools/generate_2d_seg_labels.py \
    --data-root data/nuscenes \
    --output-dir data/nuscenes/seg_2d_labels \
    --split trainval \
    --device cuda:0 \
    --batch-size 16 \
    --num-workers 8

# mini 子集
python tools/generate_2d_seg_labels.py \
    --data-root data/nuscenes \
    --output-dir data/nuscenes/seg_2d_labels \
    --split mini \
    --device cuda:0
```

### 输出格式

- 单通道 uint8 PNG，像素值 0–16 为语义类别，255 为忽略
- Cityscapes 19 类 → nuScenes 17 类映射在脚本内完成
- 路径镜像原图结构：`samples/CAM_FRONT/n008-xxx.png`

### 可视化验证

```bash
python tools/visualize_seg_label.py \
    --label data/nuscenes/seg_2d_labels/samples/CAM_FRONT/n008-xxx.png \
    --output vis.png
```

### 调试工具

```bash
python tools/debug_loading_seg2d.py        # 验证 LoadSemanticSeg2D 管道
python tools/debug_seg2d_nan.py data/nuscenes/seg_2d_labels  # 检查全 255 标签
```

---

## 2. 占用率 GT 准备

占用率 GT (`labels.npz`) 由外部提供，本仓库不生成。需手动下载后放置到正确目录。

### 2a. 下载 GT

从 [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) 下载 `gts` 文件夹，解压至：

```
data/nuscenes/gts/{scene_name}/{token}/labels.npz
```

每个 `labels.npz` 包含：
- `semantics`: `(Dx, Dy, Dz)` uint8 语义标签
- `mask_lidar`: `(Dx, Dy, Dz)` bool 激光雷达可见掩码
- `mask_camera`: `(Dx, Dy, Dz)` bool 相机可见掩码

### 2b. 生成 nuScenes 信息文件 (BEVDet 格式)

```bash
# trainval
python tools/create_data_bevdet.py \
    --root-path data/nuscenes \
    --version v1.0-trainval \
    --extra-tag bevdetv2-nuscenes

# mini
python tools/create_data_bevdet.py \
    --root-path data/nuscenes \
    --version v1.0-mini \
    --extra-tag bevdetv2-nuscenes-mini
```

输出：
- `data/nuscenes/bevdetv2-nuscenes_infos_train.pkl`
- `data/nuscenes/bevdetv2-nuscenes_infos_val.pkl`

每个 info 条目包含 `occ_path` 字段指向对应的 `gts/{scene}/{token}` 目录。

---

## 3. 4D 预测信息生成

从 BEVDet 信息文件生成带有 `forecast` 字典的 4D 预测信息文件。

```bash
# 完整生成
python tools/create_4d_forecast_infos.py --root-path data/nuscenes

# 限量调试（只保留前 20 个有效锚点帧）
python tools/create_4d_forecast_infos.py --root-path data/nuscenes --max-samples 20

# 仅验证已有 _forecast.pkl
python tools/create_4d_forecast_infos.py --root-path data/nuscenes --verify-only
```

### forecast 字典内容

每个有效锚点帧附加 `forecast` 字典：

| 字段 | 说明 |
|---|---|
| `history_indices` | `[idx_t-2, idx_t-1, idx_t]` — 历史帧索引（含当前帧） |
| `history_tokens` | 对应的 sample token |
| `future_indices` | `[idx_t+2, idx_t+4, idx_t+6]` — 未来帧索引 |
| `future_tokens` | 对应的 sample token |
| `horizons_sec` | `[1.0, 2.0, 3.0]` — 预测时间跨度 |
| `future_occ_paths` | 3 个目录路径，每个含 `labels.npz` |

> nuScenes 关键帧频率为 2Hz，因此 1/2/3 秒未来对应 +2/+4/+6 关键帧偏移。

---

## 4. 训练启动

### 4a. 单卡训练

```bash
python tools/train.py \
    projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py \
    --work-dir work_dirs/priorocc-4d
```

### 4b. 多卡分布式训练

```bash
# 8 卡
bash tools/dist_train.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 8

# 4 卡 + 自定义 work_dir
bash tools/dist_train.sh \
    projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 4 \
    --work-dir work_dirs/priorocc-4d

# 1 卡（调试）
bash tools/dist_train.sh \
    projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 1 \
    --cfg-options runner.max_epochs=1 data.samples_per_gpu=1
```

### 4c. SLURM 集群

```bash
bash tools/slurm_train.sh \
    <partition> <job_name> \
    projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py \
    work_dirs/priorocc-4d
```

### 4d. 从断点恢复

```bash
# 自动恢复（从最新 checkpoint）
bash tools/dist_train.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 8 \
    --auto-resume

# 指定 checkpoint
bash tools/dist_train.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 8 \
    --resume-from work_dirs/priorocc-4d/latest.pth
```

### 4e. 无 GPU 最小验证（本地预检）

```bash
python3 -m compileall -q projects/mmdet3d_plugin tools
python3 tools/audit_priorocc_4d_static.py --fail-on HIGH
python3 tools/verify_priorocc_4d_pipeline.py
python3 tools/verify_priorocc_4d_evaluator.py
```

这些检查不依赖 GPU，也不依赖 torch/mmcv 运行时，主要用于确认代码语法、4D config、future label 合约和 evaluator horizon 对齐没有明显错误。真正的 forward/backward smoke test 仍需放到有 PyTorch/CUDA 的服务器上运行。

### 4f. 预训练权重

训练前需下载 BEVDet 预训练权重至 `ckpts/` 目录：

```
ckpts/bevdet-r50-cbgs.pth
```

配置中 `load_from = "ckpts/bevdet-r50-cbgs.pth"` 会自动加载。

---

## 5. 评估与测试

### 5a. 训练中评估

评估在训练过程中自动执行（`evaluation = dict(interval=2, start=2)`），每 2 个 epoch 从第 2 个 epoch 开始评估。

### 5b. 独立测试

```bash
# 单卡测试
python tools/test.py \
    projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py \
    work_dirs/priorocc-4d/latest.pth \
    --eval mIoU

# 多卡测试
bash tools/dist_test.sh \
    projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py \
    work_dirs/priorocc-4d/latest.pth 8 \
    --eval mIoU
```

### 5c. 评估指标

`NuScenes4DOccForecastDataset.evaluate` 输出分 horizon mIoU：

| 指标 | 说明 |
|---|---|
| `mIoU_1s` | 1 秒未来预测 mIoU |
| `mIoU_2s` | 2 秒未来预测 mIoU |
| `mIoU_3s` | 3 秒未来预测 mIoU |
| `mIoU_avg` | 三个 horizon 平均 mIoU |

> 当前帧占用率 mIoU 沿用基类 `NuScenesDatasetOccpancy` 的标准 `mIoU`。

---

## 6. 模块验证工具

验证 PriorOcc-4D 各阶段模块正确性（CPU 可运行，无需数据集）。

```bash
# 单阶段验证
conda run -n flash python tools/verify_priorocc_4d.py --stage modules
conda run -n flash python tools/verify_priorocc_4d.py --stage build
conda run -n flash python tools/verify_priorocc_4d.py --stage sgdm-current
conda run -n flash python tools/verify_priorocc_4d.py --stage forecast-smoke
conda run -n flash python tools/verify_priorocc_4d.py --stage motion-pipeline
conda run -n flash python tools/verify_priorocc_4d.py --stage semantic-closure

# 全部验证
conda run -n flash python tools/verify_priorocc_4d.py --stage all
```

### 验证阶段说明

| 阶段 | 验证内容 |
|---|---|
| `modules` | 6 个运动模块单元测试（warp、mask、梯度、语义敏感性、恒等初始化） |
| `build` | 完整 PriorOcc-4D 模型构建 |
| `sgdm-current` | 3 帧 SGDM 当前帧 forward_train |
| `forecast-smoke` | 未来占用率预测 smoke test |
| `motion-pipeline` | 完整 SCMF 管道（8 个损失 + 4 模块梯度检查 + 回退） |
| `semantic-closure` | P5 语义闭环（loss_sem_consistency + loss_future_semantic + 回退） |

---

## 7. 可调参数

### 7a. 模块启用/禁用开关

通过 `--cfg-options model.<flag>=True/False` 在命令行覆盖：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `enable_dyn_sta_decoder` | `True` | 动静分离解码器 |
| `enable_motion_encoder` | `True` | 运动特征编码器 |
| `enable_semantic_attention` | `True` | 语义运动注意力 |
| `enable_scmf` | `True` | 语义条件运动场（SCMF） |
| `enable_future_prediction` | `True` | 未来占用率预测 |
| `enable_future_semantic` | `False` | 未来 BEV 语义辅助头；默认关闭，等 future semantic 标签接入后再打开 |
| `enable_semantic_consistency` | `True` | 语义时序一致性损失 |

**示例 — 消融实验关闭 SCMF，使用直接预测头：**

```bash
bash tools/dist_train.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 8 \
    --cfg-options \
        model.enable_scmf=False \
        model.enable_future_semantic=False \
        model.enable_semantic_consistency=False \
        model.future_predictor=dict\(type=\'DirectFutureOccupancyHead\',bev_channels=256,num_future=3,hidden_dim=256\)
```

### 7b. 损失权重

| 损失项 | 权重 | 参数位置 | 说明 |
|---|---|---|---|
| `loss_occ` (当前帧) | `1.0` | `occ_head.loss_occ` | CrossEntropy, class_balance=True |
| `loss_depth` | `3.0` | `img_view_transformer.loss_depth_weight` | 深度监督 |
| `loss_2d_seg` | `0.3` | `semantic_injector.loss_2d_seg` | 2D 语义分割辅助损失 |
| `loss_2d_seg_history_{i}` | `0.3` | 代码内 × `loss_weight` | 历史帧 2D 语义损失 |
| `loss_occ_future_{k}s` | `1.0 / 0.7 / 0.5` | `future_loss_weights` | 未来帧占用率损失（1s/2s/3s） |
| `loss_sem_consistency` | `0.05` | 代码内硬编码 | 语义时序一致性（对称 KL） |
| `loss_future_semantic_{k}s` | `0.1` | 代码内硬编码 | 未来 BEV 语义辅助损失 |

**调整示例：**

```bash
# 加大未来预测权重
--cfg-options model.future_loss_weights="[1.5,1.0,0.5]"

# 调整未来 loss weights 需同时改 forecast_cfg
--cfg-options model.future_loss_weights="[1.5,1.0,0.5]" \
             forecast_cfg.future_loss_weights="[1.5,1.0,0.5]"
```

> 注意：`loss_sem_consistency`（×0.05）和 `loss_future_semantic`（×0.1）的权重目前在 `priorocc_4d.py` 代码中硬编码，如需调整需修改源码。

### 7c. 预测时间跨度

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_future` | `3` | 预测未来帧数 |
| `horizons_sec` | `[1.0, 2.0, 3.0]` | 每帧对应秒数 |
| `future_loss_weights` | `[1.0, 0.7, 0.5]` | 每帧损失权重 |
| `max_flow_cells` | `5.0` | BEV 网格中最大运动流（格） |

> 修改 `num_future` 或 `horizons_sec` 需同步修改 `create_4d_forecast_infos.py` 中的 `FUTURE_OFFSETS` 和 `HORIZONS_SEC`，并重新生成 forecast pkl。

### 7d. 多帧配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `multi_adj_frame_id_cfg` | `(1, 3, 1)` | `range(1,3,1)` = [1,2]，即 2 历史帧 + 1 当前帧 = 3 帧 |
| `num_adj` | `2` | 历史帧数量 |
| `align_after_view_transformation` | `False` | 是否在 BEV 变换后对齐 |

### 7e. 语义类别配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_semantic_classes` | `17` | 语义类别数（16 类 + 空闲） |
| `dynamic_class_ids` | `[0,1,2,3,4,5,6,7,8,9,10]` | 11 个动态类别 ID |
| `static_class_ids` | `[11,12,13,14,15,16]` | 6 个静态类别 ID |
| `sem_consistency.conf_threshold` | `0.5` | 一致性损失的高置信度阈值 |

### 7f. 优化器与训练计划

| 参数 | 默认值 | 说明 |
|---|---|---|
| `optimizer` | `AdamW, lr=1e-4, weight_decay=1e-2` | 优化器 |
| `optimizer_config` | `grad_clip max_norm=5` | 梯度裁剪 |
| `lr_config` | `step, warmup=linear, warmup_iters=200, warmup_ratio=0.001, step=[14]` | 学习率策略 |
| `runner` | `EpochBasedRunner, max_epochs=30` | 训练轮数 |
| `samples_per_gpu` | `4` | 每卡 batch size |
| `workers_per_gpu` | `4` | 数据加载线程 |
| `checkpoint_config` | `interval=1, max_keep_ckpts=5` | checkpoint 保存 |
| `evaluation` | `interval=2, start=2` | 评估间隔 |
| `load_from` | `ckpts/bevdet-r50-cbgs.pth` | 预训练权重 |

**调整示例：**

```bash
# 降低学习率 + 减小 batch size
--cfg-options optimizer.lr=5e-5 data.samples_per_gpu=2

# 减少 epoch
--cfg-options runner.max_epochs=20

# 开启 FP16 混合精度（需取消配置中 fp16 注释）
--cfg-options fp16=dict\(loss_scale=\'dynamic\'\)
```

### 7g. BEV 网格与图像配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `grid_config.x` | `[-40, 40, 0.4]` | X 轴范围与分辨率 |
| `grid_config.y` | `[-40, 40, 0.4]` | Y 轴范围与分辨率 |
| `grid_config.z` | `[-1, 5.4, 6.4]` | Z 轴范围与分辨率 |
| `grid_config.depth` | `[1.0, 45.0, 0.5]` | 深度范围与分辨率 |
| `data_config.input_size` | `(256, 704)` | 图像输入尺寸 |
| `data_config.src_size` | `(900, 1600)` | 原始图像尺寸 |
| `bda_aug_conf` | `rot=(0,0), scale=(1,1), flip_dx=0.5, flip_dy=0.5` | BEV 数据增强 |

---

## 8. 完整流程速查（从零到训练）

```bash
# 0. 环境
conda activate flash
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 1. 下载 nuScenes 数据集 + 占用率 GT
#    将 gts/ 解压至 data/nuscenes/gts/
#    将 BEVDet 预训练权重放至 ckpts/bevdet-r50-cbgs.pth

# 2. 生成 BEVDet 信息文件
python tools/create_data_bevdet.py --root-path data/nuscenes --version v1.0-trainval

# 3. 生成 SegFormer 2D 语义伪标签
python tools/generate_2d_seg_labels.py \
    --data-root data/nuscenes \
    --output-dir data/nuscenes/seg_2d_labels \
    --split trainval --device cuda:0

# 4. 生成 4D 预测信息
python tools/create_4d_forecast_infos.py --root-path data/nuscenes

# 5. 验证模块（CPU 可运行）
conda run -n flash python tools/verify_priorocc_4d.py --stage all

# 6. 启动训练
bash tools/dist_train.sh projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py 8
```

---

## 9. 关键文件索引

| 文件 | 说明 |
|---|---|
| `tools/generate_2d_seg_labels.py` | SegFormer 2D 语义伪标签生成 |
| `tools/create_data_bevdet.py` | BEVDet nuScenes 信息文件生成 |
| `tools/create_4d_forecast_infos.py` | 4D 预测信息文件生成 |
| `tools/train.py` | 训练入口 |
| `tools/dist_train.sh` | 分布式训练启动脚本 |
| `tools/test.py` | 测试/评估入口 |
| `tools/verify_priorocc_4d.py` | 模块验证工具 |
| `projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py` | 主配置文件 |
| `projects/configs/priorocc/priorocc-r50.py` | 基础配置（继承） |
| `projects/mmdet3d_plugin/models/detectors/priorocc_4d.py` | PriorOcc-4D 检测器 |
| `projects/mmdet3d_plugin/datasets/nuscenes_4d_forecast_dataset.py` | 4D 预测数据集 |
| `projects/mmdet3d_plugin/datasets/pipelines/loading_future_occ.py` | 未来占用率 GT 加载 |
| `projects/mmdet3d_plugin/datasets/pipelines/loading_temporal_seg2d.py` | 时序 2D 语义加载 |
| `projects/mmdet3d_plugin/datasets/pipelines/loading_seg2d.py` | 单帧 2D 语义加载 |
| `projects/mmdet3d_plugin/models/model_utils/future_semantic.py` | 未来语义预测头 |
| `projects/mmdet3d_plugin/models/model_utils/sem_consistency.py` | 语义一致性损失 |
| `projects/mmdet3d_plugin/models/model_utils/scmf.py` | SCMF 运动场 + 未来预测器 |
| `projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py` | 动静分离 + 运动编码 |
