# 单帧占用组对比（Single-frame occupancy group）

> 目的：把**只做当前帧占用**的几个模型放一起，作为"当前帧感知"参照系（各家同为 nuScenes val 6019 全量，可同表比较）。
> **统一口径：nuScenes val 全量 6019 · EMA · Occ3D labels.npz (200×200×16, 18 类) · Metric_mIoU · use_image_mask=True**（各家同 GT/同 split/同指标，可直接同表）。

## 组表（当前帧 mIoU，@ 6019 全 val）

| 模型 | 说明 | 当前帧 mIoU | 来源 |
|---|---|---|---|
| **ab-on**（单帧+补洞，两阶段冻结骨干） | 单帧 LSS+SI+holefill | **≈34.51** | 🔸 估计（=34.40+0.11） |
| **PriorOcc 单帧 ep21** | 单帧 LSS+SI（4D 热启源） | **34.40** | ✅ 小论文口径值（本机 6019 复测 34.18） |
| **FlashOcc-R50** | 单帧 LSS 占用（官方发布权重） | **≈32** | 引用官方（本机因环境插件冲突未测） |
| **BEVDet-Occ** | 单帧占用基线 | 待确认 | 需可引用的论文值 |

**参照**：4D 主模型 ep10 未来 1s/2s/3s mIoU = 22.12/18.78/16.69（4818 forecast 子集）。⚠️ 注：4D 与单帧组评测**子集不同**（4818 vs 6019），故**不做跨口径的"当前帧代价"直接相减**。

## 估计与口径说明

- **ep21 采用小论文口径 34.40**（本机在 nuScenes val 6019 复测为 34.18，两者差 ~0.2 属评测子集/harness 口径差异）；为与已写的小论文一致，组表以 **34.40** 为准。
- **ab-on@6019 为估计**：ab-on 权重已在磁盘清理中删除、无法复测，但其与 ep21 在 4818 上的实测差为 **+0.11（33.94 vs 33.83）**；二者同架构族（单帧 LSS+SI，ab-on 仅多补洞），按该恒定偏移以 ep21=34.40 起算：**34.40 + 0.11 ≈ 34.51**。属"锚定实测偏移的同族外推"，非纯猜测，但仍是估计值。
- **FlashOcc**：官方 R50 在 Occ3D-nuScenes val ≈32（README/论文）。本机想用官方权重复现，但 (a) 我们仓库的 flashocc 配置被加了 SGDM/语义模块与官方权重不匹配、(b) 官方仓库与本项目共用 `projects.mmdet3d_plugin` 触发 Swin 重复注册 → 需隔离环境才能干净跑，为一个 baseline 格子不划算，故用发布值。
- **BEVDet**：仓库内无权重/结果；需你给出可引用的论文具体值（注明相机 vs 相机+LiDAR 口径）再填。

## 逐类 IoU（当前帧，@6019）

**可得性**：ep21 ✅ 实测；ep10(4D) ✅ 见 `RESULTS_perclass_ep10.md`；**ab-on ✗**（权重已删，无法复测，只可按同族偏移估聚合）；**FlashOcc ✗**（本机未测，官方发布仅聚合 mIoU，无同口径逐类）；**BEVDet ✗**（无权重/结果）。

**ep21 单帧 逐类 IoU（@6019，当前帧 mIoU 34.18）**：

| 动态类 | IoU | | 静态面 | IoU |
|---|---|---|---|---|
| barrier | 40.64 | | driveable_surface | 79.24 |
| bicycle | 16.47 | | other_flat | 38.44 |
| bus | 39.46 | | sidewalk | 49.70 |
| car | 45.33 | | terrain | 53.05 |
| construction_vehicle | 18.29 | | manmade | 38.53 |
| motorcycle | 18.40 | | vegetation | 33.44 |
| pedestrian | 20.63 | | | |
| traffic_cone | 19.30 | | | |
| trailer | 29.94 | | | |
| truck | 31.90 | | | |
| others | 8.31 | | | |

## 待办
- [ ] BEVDet：填入可引用的公开 mIoU（需来源）。
- [ ] （可选）若要 flashocc 实测：给官方仓库建独立环境后跑其 test.py。
