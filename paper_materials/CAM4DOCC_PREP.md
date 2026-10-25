# Cam4DOcc 评测准备：缺口分析与执行方案

> 目标：把 PriorOcc-4D 的结果放到 Cam4DOcc 官方协议上，得到可与 OCFNet / PowerBEV-3D 等直接对比的数字。
> 状态：**仓库已就绪、适配器可写；唯一硬阻塞是 GT 下载不可达。**
> 时间：2026-10-23

---

## 1. 两套协议的事实对照（已实测核实）

| 项 | 本仓库（当前） | Cam4DOcc 官方 |
|---|---|---|
| GT 来源 | **Occ3D-nuScenes**（与 UniOcc ICCV25 的 `NuScenes-via-Occ3D-2Hz` **逐体素完全一致**，仅类 ID 重排） | **nuScenes-Occupancy v0.1**（OpenOccupancy，更稠密，含 noise 清理） |
| 体素 | 0.512 m | 0.2 m |
| 网格 | 200×200×16 | 512×512×40 |
| 物理范围 | **[-51.2, 51.2] × [-5, 3] m** | **[-51.2, 51.2] × [-5, 3] m（完全相同）** |
| 类 | 0=others, 1..16=语义, 17=free | 0=empty/noise, 1..16=语义 |
| 指标 | mIoU（17 类均值） | IoU_c / IoU_f / IoU~_f（完成度 IoU + 语义 IoU） |
| 未来步 | 1s / 2s / 3s | Nf=4 @2Hz → 0.5 / 1.0 / 1.5 / 2.0 s |
| split | val_forecast.pkl（4818 有效） | test_capacity=5119 序列 |

### 1.1 已核实的关键事实

1. **同物理范围**：我们的 occ 网格和 Cam4DOcc 的 `pc_range` 完全一致（`[-51.2,-51.2,-5,51.2,51.2,3]`），只是分辨率不同（0.512m vs 0.2m）。
   → **上采样映射是纯整数索引变换，无坐标重投影风险**。
2. **我们的 GT = UniOcc 的 Occ3D 导出**（实测同一 `sample_token` 的数组：逐类体素数完全对应，`mask_camera` 求和相同 108,100）。
   → 我们当前 21.33/17.64/15.09 的数字应描述为 **"UniOcc/FlashOcc 网格（Occ3D GT，200×200×16 @0.512m, 102.4×102.4m）"**，
     既不是标准 Occ3D（[-40,40]@0.4m），也不是 Cam4DOcc（0.2m）。
3. **类 ID 映射**（我们 → nuScenes-Occupancy）：1..16 大概率恒等（需 GT 到手后验证）；我们 17(free) ↔ 对方 0(empty)；我们 0(others) 对应对方 noise（评测时忽略）。
4. **GMO/GSO 分组**：GMO = 我们索引 1..10（barrier..truck），GSO = 11..16 —— 与 Cam4DOcc 的定义一致，可直接聚合。

---

## 2. 缺口清单

| # | 需要的东西 | 状态 | 说明 |
|---|---|---|---|
| 1 | Cam4DOcc 代码库 | ✅ 已克隆 | `/root/priorocc/Cam4DOcc`（37MB，含 `metric_util.py` 的 SSCMetrics） |
| 2 | OpenOccupancy infos pkl（val 105MB） | ✅ 可下载 | GitHub releases 可访问（`val_pkl` tag） |
| 3 | **nuScenes-Occupancy v0.1 GT（5GB 压缩 / 130GB 解压）** | ❌ **硬阻塞** | Google Drive 返回 503；Baidu 需客户端；HF 无镜像 |
| 4 | 网格适配器（200×200×16 → 512×512×40） | ⬜ 待写 | 纯 `np.repeat`/索引映射，无风险 |
| 5 | 类 ID / GMO-GSO 映射 | ⬜ 待写 | 恒等 + 分组聚合 |
| 6 | 评测包装（用他们的 SSCMetrics） | ⬜ 待写 | 可直接复用 `metric_util.py` |
| 7 | 预测导出（我们模型 → 他们的 npz 格式） | ⬜ 待写 | 需 GPU 推理一次 val |
| 8 | 步长对齐 | ⚠️ 部分 | 我们 1s/2s ↔ 他们 1.0s/2.0s；他们 0.5/1.5s 我们没有；我们 3s 他们不评 |

**磁盘**：剩余 84GB。仅 val（6019 帧 × 10.5MB ≈ 63GB）勉强够，但解压峰值会超；**建议只取 600 帧子集（6.3GB）**做官方协议对照。

---

## 3. 三条可行路线（按成本排序）

### 路线 A：Cam4DOcc 官方协议（目标形态，被 GT 阻塞）

```
1. 找 nuScenes-Occupancy v0.1 的可达镜像（OpenDataLab / 百度网盘 / 作者邮箱）
2. 只解压 val 中我们需要的 token（或取 600 帧子集）
3. 跑 tools/cam4docc/export_occ512.py：我们的模型预测 → 上采样 512×512×40 → npz
4. 跑 tools/cam4docc/eval_cam4docc.py：用 SSCMetrics 算 IoU_c / IoU_f(1.0s, 2.0s)
5. 论文报：「Cam4DOcc protocol, 600-frame val subset, upsampled predictions」
```

### 路线 B：UniOcc 官方协议（当前零成本，推荐作为论文主口径）

```
我们的 GT 与 UniOcc 的 NuScenes-via-Occ3D-2Hz 逐体素一致
→ 直接以 UniOcc（ICCV 2025）的 Occ3D-2Hz split 报告
→ 需要：从 HF 下载 NuScenes-via-Occ3D-2Hz-val（~5.2GB）核对 sample_token 集合
→ 价值：口径有公开论文背书；但 UniOcc 上目前少有相机端到端方法报数，横向对比有限
```

### 路线 C：只做协议自洽（零新增依赖，保底）

```
论文里明确写清网格（200×200×16 @0.512m, 102.4×102.4m, Occ3D GT）
所有对比只用「同协议重跑」的 B 档 + 内部消融
SOTA 表用「不同协议，仅作参考」标注
```

---

## 4. 现在就能做的准备（不依赖 GT）

### 4.1 网格适配器（可立即实现并单测）

映射关系（同物理范围，纯索引）：
- x: 目标 512 格 / 源 200 格；`src = floor((dst + 0.5) * 200 / 512)`
- y: 同上
- z: 目标 40 格 / 源 16 格；`src = floor((dst + 0.5) * 16 / 40)`

### 4.2 类映射与 GMO/GSO 聚合

```python
OUR_GMO = list(range(1, 11))    # barrier..truck
OUR_GSO = list(range(11, 17))   # driveable..vegetation
OUR_FREE = 17
# 到 nuScenes-Occupancy: 1..16 恒等；free(17) -> empty(0)
```

### 4.3 评测包装

复用 `Cam4DOcc/projects/occ_plugin/utils/metric_util.py` 的 `SSCMetrics`：
- `add_batch(pred, gt, nonsurface)` → completion IoU（= IoU_c）
- `get_stats()['iou_ssc_mean']` → 语义 IoU 均值

### 4.4 预测导出（需 GPU，等训练结束）

- 用 `eval_b_baseline.py` 的推理路径，把 `occ_future[k]` 存成 npz：
  `pred_512 = upsample_occ(pred_200)`（int8）
- 每帧存 `{sample_token, pred_1s_512, pred_2s_512}`，供评测脚本读取

---

## 5. 待用户决策 / 需要资源

| 问题 | 选项 |
|---|---|
| nuScenes-Occupancy GT 从哪拿？ | (a) 用户用百度网盘下载后上传（5GB）；(b) 找 OpenDataLab 镜像；(c) 放弃官方协议走路线 B/C |
| 评测规模 | 全 val（63GB，磁盘紧张）还是 600 帧子集（6.3GB） |
| 论文主口径 | 路线 A（Cam4DOcc）/ B（UniOcc）/ C（自洽，SOTA 仅参考） |

**建议**：主口径用 **B（UniOcc/Occ3D-2Hz，与当前 GT 完全一致，零成本、有 ICCV25 背书）**；
若审稿要求 Cam4DOcc，再用路线 A 补 600 帧子集（需用户协助拿到 5GB GT 包）。

---

## 6. 已就绪资产索引

| 资产 | 路径 |
|---|---|
| Cam4DOcc 代码 | `/root/priorocc/Cam4DOcc`（含 metric_util.py / cam4docc_dataset.py） |
| 开放 Occupancy infos（val 105MB） | `https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl` |
| nuScenes-Occupancy 规格 | v0.1: 5GB 压缩 / 130GB 解压；0.2m；512×512×40；类 0-16 |
| 我们的 GT | `/root/autodl-tmp/dataforpriorocc/nuscenes/gts/<scene>/<token>/labels.npz` |
| 我们的评测脚本 | `/root/autodl-tmp/dataforpriorocc/eval_b_baseline.py` |
| UniOcc Occ3D-2Hz-val | `https://huggingface.co/datasets/tasl-lab/uniocc` → `NuScenes-via-Occ3D-2Hz-val` |
