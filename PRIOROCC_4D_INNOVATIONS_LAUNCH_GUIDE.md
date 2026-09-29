# PriorOcc-4D 创新版训练启动指南（SMP + 连续性补洞 + SDP）

本指南针对**新配置** `projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py`
（在已审计 baseline `priorocc-4d-r50-stgdm-scmf.py` 上开启三项创新）。
方法与公式见 [`PRIOROCC_4D_INNOVATIONS_README.md`](PRIOROCC_4D_INNOVATIONS_README.md)；
基础环境/数据细节见 [`PRIOROCC_4D_LAUNCH_GUIDE.md`](PRIOROCC_4D_LAUNCH_GUIDE.md)。

---

## 0. 环境

```bash
conda activate flash                 # Python 3.9, PyTorch 1.10+cu111, mmcv 1.5.3, mmdet3d 1.0.0rc4
cd /home/azure/learning/FlashOCC
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

> **GPU 注意**：本机 `NVIDIA GeForce RTX 5060`（sm_120）与 `torch 1.10+cu111`（最高支持 sm_86）
> **不兼容**，`torch.cuda` 实际不可用，验证工具会自动回退 CPU。要真正 GPU 训练，需在
> **兼容 GPU（sm≤86，如 A100/3090/V100）** 的机器上跑，或升级到支持 sm_120 的 PyTorch 构建。

---

## 1. 需要的数据集/文件

| 数据 | 路径 | 来源 |
|---|---|---|
| nuScenes 图像 | `data/nuscenes/samples/`, `sweeps/` | nuScenes 官方 (v1.0-trainval) |
| 3D 占用 GT | `data/nuscenes/gts/{scene}/{token}/labels.npz` | 外部 CVPR2023-3D-Occupancy-Prediction（Occ3D 风格），含 `semantics/mask_lidar/mask_camera` |
| 2D 语义伪标签 | `data/nuscenes/seg_2d_labels/samples/CAM_*/*.png` | 本仓 `tools/generate_2d_seg_labels.py`（SegFormer-B2 cityscapes） |
| BEVDet info | `data/nuscenes/bevdetv2-nuscenes_infos_{train,val}.pkl` | 本仓 `tools/create_data_bevdet.py` |
| **4D 预测 info** | `data/nuscenes/bevdetv2-nuscenes_infos_{train,val}_forecast.pkl` | 本仓 `tools/create_4d_forecast_infos.py`（**训练前必须生成**） |
| 预训练权重 | `ckpts/bevdet-r50-cbgs.pth` | config `load_from` |

> `_forecast.pkl` 只是把已有占用 GT 在时间轴上串成"历史 3 帧 + 未来 1/2/3s"（nuScenes 2Hz，
> `HISTORY_OFFSETS=[2,1]`、`FUTURE_OFFSETS=[2,4,6]`），**不需要 Cam4DOcc**，未来 GT 就是未来
> token 的 `labels.npz`。

---

## 2. 数据处理链（按顺序，一次即可）

```bash
# (1) 生成 BEVDet info pkl
python tools/create_data_bevdet.py --root-path data/nuscenes --version v1.0-trainval

# (2) 生成 2D 语义伪标签（SegFormer，需 GPU；一次性）
python tools/generate_2d_seg_labels.py \
    --data-root data/nuscenes --output-dir data/nuscenes/seg_2d_labels \
    --split trainval --device cuda:0

# (3) 生成 4D 预测 info（串联历史/未来帧 + 定位 labels.npz）
python tools/create_4d_forecast_infos.py --root-path data/nuscenes
#   mini 调试： --version v1.0-mini --max-samples 8
#   仅校验已生成的： --verify-only
```

---

## 3. 训练前最小链路验证（强烈建议先跑）

一次跑通"数据→前向→反向→optimizer.step→收敛"，提前暴露 loss 不收敛 / 维度不匹配 / 梯度断链：

```bash
# 无 GPU 静态自检
python -m compileall -q projects/mmdet3d_plugin tools
python tools/audit_priorocc_4d_static.py --fail-on HIGH

CFG=projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py

# 模块/构建/全链路（合成张量）
python tools/verify_priorocc_4d.py --stage build           --config $CFG
python tools/verify_priorocc_4d.py --stage motion-pipeline --config $CFG

# 最小链路：optimizer.step 循环 + 收敛断言 + 输出 diagnose 所需 stats
#   数据已就绪 → 去掉 --no-real-data 用真实 batch；否则合成回退
python tools/verify_priorocc_4d.py --stage minimal-chain --config $CFG \
    --iters 5 --warmup 2 --overfit-lr 1e-3 \
    --no-real-data --stats-out work_dirs/mc_stats.json

# 训练塌缩诊断（消费上一步的 stats）
python tools/diagnose_priorocc_4d_training.py --stats-json work_dirs/mc_stats.json \
    --max-mask-overlap 0.3 --min-future-loss-drop 0.0     # 合成数据才放宽这两项

# 向后兼容：baseline 旧配置应逐位不变
python tools/verify_priorocc_4d.py --stage all \
    --config projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py
```

**本次实测（CPU/合成）**：`build / motion-pipeline / minimal-chain / diagnose / baseline --stage all`
**全部 PASS**；`minimal-chain` 收敛 `16.49 → 11.81`，梯度到达 SMP 与 SDP.table。

`minimal-chain` 通过标准：所有 loss 有限、维度正确、`loss_total` 下降、SMP/SDP 有梯度、参数无 NaN。

---

## 4. 启动训练

```bash
CFG=projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill.py

# 单卡
python tools/train.py $CFG --work-dir work_dirs/priorocc-4d-innov

# 多卡（分布式）
bash tools/dist_train.sh $CFG 8 --work-dir work_dirs/priorocc-4d-innov

# SLURM
bash tools/slurm_train.sh <partition> <jobname> $CFG work_dirs/priorocc-4d-innov

# 断点续训
python tools/train.py $CFG --work-dir work_dirs/priorocc-4d-innov --auto-resume
```

**先短跑冒烟**（数据就绪后，1 epoch / 小 batch 确认无误再全量）：
```bash
bash tools/dist_train.sh $CFG 1 --cfg-options runner.max_epochs=1 data.samples_per_gpu=1
```

---

## 5. 新增开关、损失与评测

### 5.1 新增/关键 config 开关（均可 `--cfg-options` 覆盖做消融）
| 开关 | 默认 | 作用 |
|---|---|---|
| `model.enable_motion_prior` | True | SMP 每类运动基 + 正则 |
| `model.motion_prior_loss_weights` | `{static_flow:0.05, rigid_smooth:0.02, nonrigid_bound:0.02}` | SMP 损失权重 |
| `model.enable_sem_continuity` | True | 语义连续性补洞损失 |
| `model.sem_continuity_loss_weights` | `{cont_2d:0.05, cont_bev:0.05, cont_occ:0.0}` | 连续性损失权重 |
| `model.continuity_apply_occ` | False | 是否对 occ logits 也做连续性（更重） |
| `model.future_predictor.use_warp_validity` | True | warp 有效性门控（空洞回退残差） |
| `model.img_view_transformer.depthnet_cfg.use_semantic_depth_prior` | True | SDP 语义深度先验 |
| `...depthnet_cfg.sdp_weight / sdp_temperature` | 1.0 / 1.0 | SDP 强度/温度 |

**逐项消融示例**：
```bash
python tools/train.py $CFG --cfg-options model.enable_motion_prior=False      # 去 SMP
python tools/train.py $CFG --cfg-options model.enable_sem_continuity=False    # 去连续性
python tools/train.py $CFG --cfg-options model.future_predictor.use_warp_validity=False  # 去补洞门控
python tools/train.py $CFG --cfg-options model.img_view_transformer.depthnet_cfg.use_semantic_depth_prior=False  # 去 SDP
```

### 5.2 新增损失项（训练日志会出现）
`loss_static_flow`、`loss_rigid_smooth`、`loss_nonrigid_bound`、
`loss_sem_continuity_2d`、`loss_sem_continuity_bev`（`loss_sem_continuity_occ` 仅当
`continuity_apply_occ=True`）。它们与既有 `loss_depth / loss_occ / loss_2d_seg_history_* /
loss_occ_future_{1,2,3}s / loss_sem_consistency` 一起反传。

### 5.3 评测
训练中自动评测（`evaluation=dict(interval=2, start=2)`）；或独立：
```bash
python tools/test.py $CFG work_dirs/priorocc-4d-innov/latest.pth --eval mIoU
```
指标：`mIoU_1.0s / mIoU_2.0s / mIoU_3.0s / mIoU_avg`（`NuScenes4DOccForecastDataset.evaluate`）。

---

## 6. 常见问题

- **`_forecast.pkl not found`**：先跑第 2 步 (3) `create_4d_forecast_infos.py`。
- **`minimal-chain` 报 real data unavailable**：正常——数据未就绪时自动回退合成；数据就绪后去掉
  `--no-real-data` 即用真实 batch。
- **CUDA 不可用 / sm_120 警告**：见第 0 节 GPU 注意；验证会在 CPU 跑，训练需换兼容 GPU 或升级 torch。
- **diagnose 报 `mask overlap too high`**：合成数据下 `semantic_bev` 近均匀属正常，按第 3 节放宽阈值；
  真实数据不应触发。
- **想保留 baseline 完全不变**：所有新 flag 默认在旧 config 中为关；新创新只在新 config 开启，
  已用 `--stage all`（旧 config）验证逐位一致。
