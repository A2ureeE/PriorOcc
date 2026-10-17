# FlashOCC / PriorOcc-4D 工具使用手册

本目录包含数据准备、训练测试、验证诊断、可视化与分析的全部脚本。
本手册给出**每个工具的用途与典型用法**；更详尽的参数表见仓库根目录
[`tools_usage.md`](../tools_usage.md)。多模态相关的新工具（不确定性可视化、
多模态冒烟测试）在本手册有完整详解。

> 约定：所有命令在仓库根目录执行；Python 用 `flash` conda 环境
> （`conda activate flash`）。可视化类工具均为 headless（Agg / CPU 渲染），
> **无需 GPU 即可出图**。

---

## 0. 快速导航（按目的选工具）

| 我想… | 用这个 |
|---|---|
| 生成训练所需的 info / 伪标签 | `create_data_bevdet.py`、`generate_2d_seg_labels.py`、`create_4d_forecast_infos.py` |
| 训练 / 评测模型 | `train.py`、`test.py`、`dist_train.sh`、`dist_test.sh` |
| 训练前验证链路不崩 | `verify_priorocc_4d.py`、`smoke_test_multimodal.py` |
| 看 3D/BEV 占用预测 | `vis_occ_matplotlib.py`（CPU）、`vis_occ_open3d_headless.py` |
| **看多模态预测的不确定性** | **`vis_multimodal_uncertainty.py`（CPU）** |
| 统计参数量 / FLOPs / FPS | `count_model_params.py`、`count_params_flops.py`、`plot_fps_performance.py` |
| 找最佳 checkpoint | `find_best_miou.py` |

---

## 1. 工具全景表

### 1.1 数据准备

| 脚本 | 用途 | 典型命令 |
|---|---|---|
| `create_data_bevdet.py` | 生成 BEVDet info pkl | `--root-path data/nuscenes --version v1.0-trainval` |
| `generate_2d_seg_labels.py` | SegFormer 生成 2D 语义伪标签 | `--data-root data/nuscenes --output-dir data/nuscenes/seg_2d_labels --split trainval --device cuda:0` |
| `create_4d_forecast_infos.py` | 生成 4D 预测 info（历史3帧+未来1/2/3s） | `--root-path data/nuscenes`（调试加 `--version v1.0-mini --max-samples 8`） |
| `update_data_coords.py` | 更新数据坐标系 | 见 `tools_usage.md` |

### 1.2 训练与测试

| 脚本 | 用途 | 典型命令 |
|---|---|---|
| `train.py` | 单卡训练 | `python tools/train.py $CFG --work-dir work_dirs/xxx` |
| `test.py` | 评测 | `python tools/test.py $CFG ckpt.pth --eval mIoU` |
| `dist_train.sh` / `dist_test.sh` | 多卡训练/评测 | `bash tools/dist_train.sh $CFG 8 --work-dir ...` |
| `slurm_train.sh` / `slurm_test.sh` | SLURM 集群训练/评测 | 见 `tools_usage.md` |

### 1.3 验证与诊断

| 脚本 | 用途 | 典型命令 |
|---|---|---|
| `verify_priorocc_4d.py` | 分阶段验证（build/motion-pipeline/minimal-chain 等） | `--stage minimal-chain --config $CFG --stats-out work_dirs/mc_stats.json` |
| `audit_priorocc_4d_static.py` | 静态代码审计 | `--fail-on HIGH` |
| `diagnose_priorocc_4d_training.py` | 训练塌缩诊断（读 minimal-chain stats） | `--stats-json work_dirs/mc_stats.json` |
| **`smoke_test_multimodal.py`** | **多模态 CPU 冒烟测试（模块/模型/评测三层）** | `python tools/smoke_test_multimodal.py --full --device cpu` |
| `integration_test.py` | 端到端集成测试 | 见 `tools_usage.md` |

### 1.4 可视化

| 脚本 | 用途 | 运行环境 |
|---|---|---|
| `vis_occ_matplotlib.py` | matplotlib 3D OCC（headless） | CPU |
| `vis_occ_open3d_headless.py` / `_elevated.py` | Open3D 3D/抬高视角渲染 | CPU |
| `vis_occ_3d_headless.py` | headless 3D | CPU |
| `vis_occ_cam_projection.py` | 占用投影回相机视图 | CPU |
| `visualize_seg_label.py` | 2D 语义标签可视化 | CPU |
| **`vis_multimodal_uncertainty.py`** | **多模态预测不确定性热力图（见 §2）** | **CPU** |

### 1.5 分析与调试

| 脚本 | 用途 |
|---|---|
| `count_model_params.py` / `count_params_flops.py` | 参数量 / FLOPs 统计 |
| `plot_fps_performance.py` | FPS-性能曲线 |
| `find_best_miou.py` | 从日志找最佳 mIoU checkpoint |
| `test_sgdm.py` / `test_bidirectional_sgdm.py` / `test_loss_2d_seg.py` | 单模块单元自测 |
| `debug_*.py` | 数据/模块排错脚本（见 `tools_usage.md`） |
| `export_onnx.py` / `convert_bevdet_to_TRT.py` | 部署导出 |

---

## 2. 多模态不确定性可视化 `vis_multimodal_uncertainty.py`

### 2.1 它解决什么问题

多模态 PriorOcc4D（见 [`PRIOROCC_4D_MULTIMODAL_README.md`](../PRIOROCC_4D_MULTIMODAL_README.md)）
对未来给出 K 个候选占用（`occ_future_modes`）+ 模式概率（`mode_probs`）。
**K 个模式在哪些体素上"意见分歧"，就是模型的预测不确定性**——这是确定性
baseline 永远给不出的信息。本工具把这种分歧渲染成热力图。

### 2.2 三档不确定性度量

| 度量 | 定义 | 需要的数据 | 模型改动 |
|---|---|---|---|
| **A1 mode agreement** | 逐体素 K 个 argmax 类别的众数占比 ∈[1/K,1]；1=全一致 | argmax 标签 | 无 |
| **A2 occupied variance** | 逐体素 occupied/free 二值在 K 模式间的方差 ∈[0,0.25]；安全向（"这里到底有没有东西"） | argmax 标签 | 无 |
| **B predictive entropy** | π 加权融合分布 `p(c\|v)=Σ_k π_k·softmax_k(v)` 的归一化熵 ∈[0,1] | 各模式 softmax 概率 | 需导出概率 |

### 2.3 三种数据源（任选其一）

| 模式 | 命令 | 适用 |
|---|---|---|
| **自测（合成）** | `--self-test` | 无 GPU/无权重时验证管线（CPU） |
| **离线 NPZ** | `--modes-npz PATH` | 推理后存档，离线反复出图（CPU） |
| **在线推理** | `--config C --checkpoint P [--sample-index i]` | 有训练好的多模态权重时（需 GPU） |

### 2.4 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--self-test` | 合成 K=3 数据跑通全管线 | off |
| `--modes-npz` | 输入 NPZ：`mode_labels`(K,T,Dx,Dy,Dz) 必需；`mode_probs`(K,T,...,C)、`mode_pi`(T,K)、`gt`(T,...)、`horizons_sec`(T,) 可选 | — |
| `--config` / `--checkpoint` / `--sample-index` | 在线推理三件套 | — |
| `--output-dir` | 输出目录 | `work_dirs/mm_uncertainty` |
| `--metric` | `all`/`agreement`/`occupancy`/`entropy` | `all` |
| `--export-npz` | 推理模式下把模式存档为 NPZ（供后续离线出图） | — |

### 2.5 示例

```bash
# 1) CPU 自测（无需任何权重/数据，验证管线）
python tools/vis_multimodal_uncertainty.py --self-test \
    --output-dir work_dirs/mm_uncertainty_demo

# 2) 离线出图（先在某处推理导出 npz，再离线渲染）
python tools/vis_multimodal_uncertainty.py \
    --modes-npz work_dirs/mm_out/sample0_modes.npz \
    --output-dir work_dirs/mm_uncertainty

# 3) 从训练好的多模态 checkpoint 在线推理并出图（+ 存档 npz）
python tools/vis_multimodal_uncertainty.py \
    --config projects/configs/priorocc/priorocc-4d-r50-mmodal.py \
    --checkpoint work_dirs/priorocc-4d-mmodal/latest.pth \
    --sample-index 0 --export-npz work_dirs/mm_out/sample0_modes.npz \
    --output-dir work_dirs/mm_uncertainty
```

### 2.6 输出解读

每个 horizon 一张 PNG（`uncertainty_00/01/02.png` = t+1/2/3s），两行：
- **第一行**：K 个模式各自的 BEV 语义俯视图（含各自 π）。
- **第二行**：mode agreement（绿=确定/红=不确定）、occupied variance
  （黑=确定/亮=不确定）、predictive entropy（有概率时）、GT（若有）。

**判读要点**：高不确定区域应集中在**运动物体的未来位置**；若静态背景也高
不确定，说明模式未正确分化（检查训练是否塌缩）。不确定性应随 horizon
增大（误差累积）——自测数据即呈现该趋势（t+1s→t+3s 分歧体素占比递增）。

> 注：自测用整图 `np.roll` 制造模式分歧，故合成数据里静态块也被标记为
> 不确定；真实训练后的模型静态区应一致（低不确定），该简化仅用于验证管线。

---

## 3. 多模态冒烟测试 `smoke_test_multimodal.py`

训练前在 **CPU** 上验证多模态链路正确性（无需 GPU/真实数据）。

```bash
python tools/smoke_test_multimodal.py            # Part A(模块) + Part C(评测)，秒级
python tools/smoke_test_multimodal.py --full     # + Part B(模型级 WTA/梯度/收敛)
```

| 层 | 验证内容 |
|---|---|
| A 模块级 | flow/future 形状、近恒等初始化、破对称、MM-SMP 恒等、mode_logits、正则+多样性梯度、端到端梯度 |
| B 模型级 | 损失键完整（WTA+mode_cls+mode_div+继承键）、梯度到达新参数、**WTA winner 跟随 GT**、simple_test 多模态输出、收敛 |
| C 评测级 | deployed mIoU + best-of-K oracle（上界性）+ mode_selection_acc |

---

## 4. 推荐工作流（多模态）

```
1. 数据：create_data_bevdet → generate_2d_seg_labels → create_4d_forecast_infos
2. 训练前：smoke_test_multimodal.py --full        （CPU 冒烟）
3. 训练：  train.py（多模态 config）               （GPU）
4. 评测：  test.py --eval mIoU                     （deployed + best-of-K）
5. 可视化：vis_multimodal_uncertainty.py --config/--checkpoint   （不确定性图）
           vis_occ_matplotlib.py                                 （占用 3D 图）
6. 分析：  count_params_flops.py / plot_fps_performance.py / find_best_miou.py
```
