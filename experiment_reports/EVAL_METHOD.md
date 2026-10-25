# 评测方法（Test Protocol）

> 本文件说明论文结果是怎么测出来的，脚本在 `tools/eval/` 与 `tools/stage1/`，
> 结果数值见 `RESULTS_perclass_ep10.md` / `FINAL_MODEL_ep10.md` / `DATA_INVENTORY.md`。

## 1. 三口径（同一次推理、同一帧集、同一 mask，公平可比）

由 `tools/eval/eval_b_baseline.py` 一次推理同时产出 7 段：

| 口径 | 含义 | 作用 |
|---|---|---|
| **CURRENT** | 当前帧占用 mIoU | 感知基线（“预测代价”参考） |
| **MODEL_1s/2s/3s** | 模型预测的未来 1/2/3s 占用 IoU | 被测系统 |
| **B_1s/2s/3s** | 把**模型自身当前帧预测冻结**当作未来预测（static-copy） | **公平平凡基线** |

**核心指标 Δ(MODEL − B)**：在同一帧、同一 `mask_camera`、同一 `Metric_mIoU` 下，运动建模相对“啥也不动直接拷贝”的**净增益**。按动态子集（类 1..10）取 IoU 求差即 `动态 Δ`——这是“语义先验时空化”主张的因果判据（绝对 mIoU 会被静态主导稀释，不能用）。

## 2. 数据与口径纪律

- 集合：nuScenes **全量 4818 有效 val**（forecast 版，含未来帧 GT），`shuffle=False`。
- 权重：一律 `epoch_N_ema.pth`（EMA 稳定高约 +2；混用 EMA/非 EMA 会造数个百分点假差）。
- 掩码：`use_image_mask=True`（`mask_camera`），未来帧用其**自身 GT mask**，保证与 MODEL 同口径。
- 类序：0=others，**1..10 动态类**（barrier,bicycle,bus,car,construction_vehicle,motorcycle,pedestrian,traffic_cone,trailer,truck），11..16 静态面。
- **子集值（600/300 帧）与全量值绝不可互比、也不可用固定偏移换算**；跨口径只并列展示、不同框作差。

## 3. 指标实现

nuScenes-OCC / Occ3D 的 `Metric_mIoU(num_classes=18, use_image_mask=True)`（`add_batch(pred, gt_sem, mask_lidar, mask_camera)`）。逐类 IoU 由解析脚本 `tools/stage1/perclass_report.py --emit --section k`（k=0..6 对应上面 7 段）从评测原始输出抽取。

## 4. 脚本清单（均已入库）

| 脚本 | 用途 |
|---|---|
| `tools/eval/eval_b_baseline.py` | 三口径主评测（CURRENT/MODEL/B），全量或 `--limit N` 子集 |
| `tools/eval/eval_occ_standalone.py` | 单模型占用评测 |
| `tools/eval/eval_static_baseline.py` | 纯 CPU 的 GT 版 static-copy（不打断训练；注意含感知误差，非公平基线） |
| `tools/eval/eval_ep21_same4818.py` | 单帧 ep21 在同 4818 上的当前帧对标 |
| `tools/stage1/perclass_report.py` | 逐段/逐类解析 |
| `tools/stage1/*_eval_guard.sh` `finalp1d_best_scan.sh` `finalp3_ep11_guard.sh` | ckpt 边界自动“暂停→三口径评测→续训/取 best”守护 |

## 5. 复现评测

```bash
# 全量 4818·EMA，三口径
python tools/eval/eval_b_baseline.py \
  released_ckpts/final-p1-depth_phase2_ep10_ema.pth \
  --cfg projects/configs/priorocc/priorocc-4d-r50-final-p1-depth.py
# 解析逐段（0..6）
for k in 0 1 2 3 4 5 6; do python tools/stage1/perclass_report.py <out.log> --emit --section $k; done
```

## 6. 结果落点（已提交）

- 逐类明细（MODEL/B/Δ，三时程）：`experiment_reports/RESULTS_perclass_ep10.md`
- 分阶段汇总 + 负结果：`experiment_reports/DATA_INVENTORY.md §K`
- 训练配方 + 复现命令：`experiment_reports/FINAL_MODEL_ep10.md`
- 方法/公式：`experiment_reports/METHODOLOGY.md`；外部 SOTA：`BASELINES_AND_SOTA.md`
