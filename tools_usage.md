# PriorOcc Tools Usage Guide

本指南详细介绍了 `tools` 文件夹及其子文件夹 `tools/analysis_tools` 中的工具、用途、详细参数说明及启动代码。

> **注意：** 本列表仅列出主要工具，已忽略所有 debug 脚本。

## 1. 训练与测试 (Training & Testing)

用于模型训练、评估和测试的核心脚本。

### 1.1 `train.py`
**用途**：单机单卡或多卡训练模型。

**详细参数**：
- `config`: (必需) 训练配置文件的路径。
- `--work-dir`: (可选) 保存日志和模型的目录。如果不指定，则默认为配置文件的文件名对应的 `work_dirs/` 目录。
- `--resume-from`: (可选) 从指定的 checkpoint 恢复训练（加载权重和优化器状态）。
- `--auto-resume`: (可选, flag) 自动从最新 checkpoint 恢复训练。
- `--validate`: (可选, flag) 在训练过程中每隔一定 epoch 执行验证。
- `--gpus` / `--gpu-ids`: (可选) 指定使用的 GPU 数量或 ID（非分布式训练时使用）。
- `--seed`: (可选, int) 随机种子，默认为 0。
- `--deterministic`: (可选, flag) 设置 CUDNN 为确定性模式（重现性）。
- `--cfg-options`: (可选) 覆盖配置文件中的设置，格式为 `key=value`。
- `--launcher`: (可选) 任务启动器，支持 `none`, `pytorch`, `slurm`, `mpi`。默认为 `none`。

**启动代码**：
```bash
# 单机单卡，覆盖 work_dir
python tools/train.py projects/configs/priorocc/priorocc-r50.py --work-dir work_dirs/priorocc-r50

# 单机多卡 (使用 torch.distributed.launch)
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py projects/configs/priorocc/priorocc-r50.py --launcher pytorch
```

### 1.2 `test.py`
**用途**：单机单卡或多卡测试模型，评估指标。

**详细参数**：
- `config`: (必需) 测试配置文件的路径。
- `checkpoint`: (必需) 模型权重文件 (.pth)。
- `--out`: (可选) 输出结果文件路径 (.pkl 或 .pickle)。
- `--fuse-conv-bn`: (可选, flag) 融合 Conv 和 BN 层以加速推理。
- `--format-only`: (可选, flag) 只格式化输出结果，不进行评估。
- `--eval`: (可选) 评估指标，如 `mAP`, `mIoU`, `bbox`, `segm` 等。
- `--show`: (可选, flag) 显示可视化结果。
- `--show-dir`: (可选) 保存可视化结果的目录。
- `--gpu-collect`: (可选, flag) 使用 GPU 收集结果（分布式测试时推荐）。
- `--tmpdir`: (可选) 用于收集结果的临时目录。
- `--cfg-options`: (可选) 覆盖配置文件中的设置。
- `--eval-options`: (可选) 传递给 dataset.evaluate() 的额外参数。
- `--launcher`: (可选) 任务启动器，支持 `none`, `pytorch`, `slurm`, `mpi`。

**启动代码**：
```bash
# 单机测试并评估 mIoU
python tools/test.py projects/configs/priorocc/priorocc-r50.py work_dirs/priorocc-r50/latest.pth --eval mIoU

# 保存预测结果 (pkl)
python tools/test.py projects/configs/priorocc/priorocc-r50.py work_dirs/priorocc-r50/latest.pth --out results.pkl
```

### 1.3 `dist_train.sh` & `dist_test.sh`
**用途**：封装好的分布式训练/测试脚本。

**参数**：
- `CONFIG`: 配置文件路径。
- `GPUS`/`CHECKPOINT`: GPU 数量（train）或 Checkpoint 路径（test）。
- `[optional arguments]`: 传递给 `train.py` 或 `test.py` 的其他参数。

**启动代码**：
```bash
# 8卡训练
bash tools/dist_train.sh projects/configs/priorocc/priorocc-r50.py 8

# 8卡测试
bash tools/dist_test.sh projects/configs/priorocc/priorocc-r50.py work_dirs/priorocc-r50/latest.pth 8 --eval mIoU
```

### 1.4 `find_best_miou.py`
**用途**：自动扫描训练日志，找出 mIoU 最好的 epoch。

**详细参数**：
- `--config`: (必需) 配置文件路径。
- `--work-dir`: (必需) 包含 checkpoints 的工作目录。
- `--start`: (可选, int) 起始 epoch，默认 10。
- `--end`: (可选, int) 结束 epoch，默认 24。
- `--gpus`: (可选, int) 用于评估的 GPU 数量，默认 2。
- `--eval`: (可选, str) 评估指标键名，默认 `mAP`（对于 Occupancy 任务应设为 `mIoU`）。

**启动代码**：
```bash
python tools/find_best_miou.py --config projects/configs/priorocc/priorocc-r50.py --work-dir work_dirs/priorocc-r50 --start 1 --end 24 --gpus 2 --eval mIoU
```

---

## 2. 数据准备 (Data Preparation)

用于生成数据集信息文件和处理标签。

### 2.1 `create_data_bevdet.py`
**用途**：为 nuScenes 数据集生成 BEVDet 格式的数据信息文件 (.pkl)。

**详细参数**：
- `--root-path`: (可选) 数据集根目录，默认 `data/nuscenes`。
- `--version`: (可选) 数据集版本，默认 `v1.0-trainval`。支持 `v1.0-mini`, `v1.0-test`, `v1.0-trainval`。
- `--extra-tag`: (可选) 输出文件名的额外标签，默认 `bevdetv2-nuscenes`。

**启动代码**：
```bash
# 生成 v1.0-trainval 数据集信息
python tools/create_data_bevdet.py --root-path ./data/nuscenes --version v1.0-trainval --extra-tag bevdetv2-nuscenes

# 生成 v1.0-mini 数据集信息
python tools/create_data_bevdet.py --root-path ./data/nuscenes --version v1.0-mini --extra-tag bevdetv2-nuscenes-mini
```

### 2.2 `generate_2d_seg_labels.py`
**用途**：基于预训练的 SegFormer 模型，为 nuScenes 数据集生成 2D 语义分割伪标签。

**详细参数**：
- `--data-root`: (可选) 数据集根目录，默认 `data/nuscenes`。
- `--output-dir`: (可选) 输出目录，默认 `data/nuscenes/seg_2d_labels`。
- `--split`: (可选) 数据集划分，默认 `trainval`。
- `--device`: (可选) 推理设备，默认 `cuda:0`。
- `--skip-existing`: (可选, flag) 跳过已存在的文件。
- `--batch-size`: (可选, int) 批量大小，默认 32。
- `--num-workers`: (可选, int) 数据加载进程数，默认 8。

**启动代码**：
```bash
python tools/generate_2d_seg_labels.py --data-root data/nuscenes --output-dir data/nuscenes/seg_2d_labels --split trainval
```

### 2.3 `visualize_seg_label.py`
**用途**：可视化 2D 语义分割标签 (PNG 格式)，将灰度标签转换为彩色图。

**详细参数**：
- `--label`: (必需) 输入的灰度标签 PNG 文件路径。
- `--output`: (可选) 输出的彩色 PNG 文件路径，默认 `label_colored.png`。

**启动代码**：
```bash
python tools/visualize_seg_label.py --label /path/to/label.png --output label_vis.png
```

---

## 3. 可视化 (Visualization)

用于可视化 3D Occupancy 预测结果。

### 3.1 `vis_occ.py`
**用途**：可视化 BEV 结果并保存预测结果为 .npz 文件。

**详细参数**：
- `--config`: (必需) 配置文件路径。
- `--weights`: (必需) 模型权重路径。
- `--viz-dir`: (必需) 输出结果的目录。
- `--override`: (可选) 覆盖配置参数。
- `--draw-sem-gt`: (可选, flag) 绘制语义分割真值（Ground Truth）。
- `--draw-pano-gt`: (可选, flag) 绘制全景分割真值。
- `--surround-view-img`: (可选, flag) 保存环视图片。
- `--surround-pano-gt`: (可选, flag) 保存环视全景真值。
- `--use-mini`: (可选, flag) 使用 mini 数据集信息文件。
- `--ann-file`: (可选) 自定义注解文件路径。
- `--num-samples`: (可选, int) 可视化的样本数量。
- `--save-npz`: (可选, flag) 将预测结果保存为 .npz 文件（**推荐开启**，用于后续 3D 可视化）。

**启动代码**：
```bash
# 可视化前 5 个样本并保存 .npz
python tools/vis_occ.py --config projects/configs/priorocc/priorocc-r50.py --weights work_dirs/priorocc-r50/latest.pth --viz-dir vis_results --save-npz --use-mini --num-samples 5
```

### 3.2 `vis_occ_cam_projection.py`
**用途**：将 3D Occupancy 投影回 2D 相机视角进行验证。

**详细参数**：
- `pred_dir`: (必需) 包含预测结果 .npz 文件的目录。
- `--root_path`: (可选) nuScenes 数据根目录，默认 `data/nuscenes`。
- `--info-file`: (可选) 数据信息文件路径。
- `--save-path`: (可选) 结果保存路径，默认 `vis_cam`。
- `--use-mini`: (可选, flag) 使用 mini 数据集信息。
- `--num-samples`: (可选, int) 处理的样本数量，默认 5。

**启动代码**：
```bash
# 需要先运行 vis_occ.py 生成 npz 文件
python tools/vis_occ_cam_projection.py vis_results/npz --save-path vis_cam_proj --use-mini
```

### 3.3 `vis_occ_matplotlib.py`
**用途**：使用 Matplotlib 进行 3D 可视化（支持 Headless 环境，无 GUI）。

**详细参数**：
- `pred_dir`: (必需) 包含预测结果 .npz 文件的目录。
- `--save-path`: (可选) 结果保存路径，默认 `./vis_3d_matplotlib`。
- `--num-samples`: (可选, int) 处理的样本数量，默认 5。
- `--max-points`: (可选, int) 渲染的最大点数（下采样以提高速度），默认 30000。
- `--dpi`: (可选, int) 图片 DPI，默认 150。
- `--multi-view`: (可选, flag) 生成多视角图（前/侧/顶/3D）。
- `--bev`: (可选, flag) 仅生成 BEV 鸟瞰图。

**启动代码**：
```bash
# 生成 3D 散点图
python tools/vis_occ_matplotlib.py vis_results/npz --save-path vis_matplotlib_3d

# 生成 BEV 鸟瞰图
python tools/vis_occ_matplotlib.py vis_results/npz --save-path vis_matplotlib_bev --bev
```

### 3.4 `vis_occ_open3d_headless.py`
**用途**：使用 Open3D 进行高质量 3D 渲染（支持 Headless 环境，强制使用 CPU 渲染以避免 Docker 问题）。

**详细参数**：
- `pred_dir`: (必需) 包含预测结果 .npz 文件的目录。
- `--save-path`: (可选) 结果保存路径，默认 `./vis_3d_open3d`。
- `--num-samples`: (可选, int) 处理的样本数量，默认 5。
- `--multi-view`: (可选, flag) 生成多视角图。

**启动代码**：
```bash
python tools/vis_occ_open3d_headless.py vis_results/npz --save-path vis_open3d --multi-view
```

---

## 4. 模型分析 (Model Analysis & Benchmarking)

用于计算模型参数量、FLOPs、日志分析及性能测试。

### 4.1 `count_model_params.py`
**用途**：统计模型的参数量和 FLOPs。

**详细参数**：
- `config`: (必需) 配置文件路径。
- `--checkpoint`: (可选) 权重文件路径（用于加载特定权重，虽然参数量通常与权重无关，但某些动态结构可能需要）。
- `--detail`: (可选, flag) 显示各子模块的详细参数量占比。

**启动代码**：
```bash
python tools/count_model_params.py projects/configs/priorocc/priorocc-r50.py --detail
```

### 4.2 `analysis_tools/analyze_logs.py`
**用途**：解析训练日志，绘制 loss/mAP 曲线，计算训练时间。

**详细参数**：
- `task`: (必需) 任务类型，支持 `plot_curve`, `cal_train_time`。
- `json_logs`: (必需) 日志文件路径（一个或多个）。
- `--keys`: (可选) 绘制曲线的指标键名，默认 `mAP_0.25`。
- `--title`: (可选) 图表标题。
- `--legend`: (可选) 图例名称。
- `--out`: (可选) 保存图片的文件路径。
- `--interval`: (可选, int) 这里指 epoch 间隔。

**启动代码**：
```bash
# 绘制 loss 曲线
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/priorocc-r50/latest.log.json --keys loss --out loss_curve.png

# 计算每个 epoch 平均训练时间
python tools/analysis_tools/analyze_logs.py cal_train_time work_dirs/priorocc-r50/latest.log.json
```

### 4.3 `analysis_tools/benchmark.py`
**用途**：测试模型推理速度 (FPS)。

**详细参数**：
- `config`: (必需) 配置文件路径。
- `checkpoint`: (必需) 权重文件路径。
- `--samples`: (可选, int) 测试样本数量，默认 500。
- `--log-interval`: (可选, int) 日志打印间隔，默认 50。
- `--fuse-conv-bn`: (可选, flag) 融合 Conv 和 BN。
- `--w_pano`: (可选, flag) 是否包含全景分割头。
- `--no-acceleration`: (可选, flag) 禁用预计算加速。

**启动代码**：
```bash
python tools/analysis_tools/benchmark.py projects/configs/priorocc/priorocc-r50.py work_dirs/priorocc-r50/latest.pth --samples 500
```

---

## 5. 模型部署 (Model Deployment)

用于模型转换。

### 5.1 `export_onnx.py`
**用途**：将 PyTorch 模型导出为 ONNX 格式。

**详细参数**：
- `config`: (必需) 部署配置文件路径。
- `checkpoint`: (必需) 模型权重文件。
- `work_dir`: (必需) 工作目录。
- `--prefix`: (可选) 输出文件名前缀，默认 `bevdet`。
- `--fp16` / `--int8`: (可选, flag) 量化模式。
- `--calib_num`: (可选, int) 校准样本数量。

**启动代码**：
```bash
python tools/export_onnx.py projects/configs/priorocc/priorocc-r50.py work_dirs/priorocc-r50/latest.pth work_dirs/onnx_export
```
