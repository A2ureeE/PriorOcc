# 单帧占用组对比（Single-frame occupancy group）

> 目的：把**只做当前帧占用**的几个模型放一起，作为"当前帧感知"参照系，
> 用来量化 4D 模型（ep10 当前帧 30.58）相对纯单帧的**预测代价 (perception cost of forecasting)**。
> 统一口径：**nuScenes val 全量 4818 · EMA · Occ3D labels.npz (200×200×16, 18 类) · Metric_mIoU · use_image_mask=True**（各家同 GT 同 split，可直接同表）。

## 组表

| 模型 | 类型/说明 | 当前帧 mIoU | 来源 |
|---|---|---|---|
| **ab-on ep4_ema** | 冻结阶段练的**单帧 + 补洞(holefill)**，后作两阶段冻结骨干 | **33.94** | ✅ 本机实测 |
| **PriorOcc 单帧 ep21** | 单帧 LSS + SemanticInjector（4D 的热启源） | **33.83** | ✅ 本机实测 |
| **FlashOcc-R50** | 单帧 LSS 占用（官方发布权重，相机端到端） | ≈32（README 档） | ⚠️ 本机重评受阻，见下 |
| **BEVDet-Occ** | 单帧占用基线（BEVDet 图→BEV→occ） | 待确认 | ❌ 本机无该权重/结果 |

**参照**：4D 主模型 ep10 当前帧 = **30.58**（未来 1s/2s/3s = 22.12/18.78/16.69）。→ 4D 相对单帧的当前帧代价约 **−3.4**。

## 说明与状态

- **ab-on(33.94) vs ep21(33.83)**：同为单帧，ab-on 多了补洞(holefill)监督、略高 +0.11（近噪声）；两者一起说明"单帧当前帧天花板 ≈ 34"。
- **FlashOcc**：本机想按同口径重评，但其测试 pipeline 含 `LoadPointsFromFile`，而数据盘 `samples/LIDAR_TOP` 已清理 → 报 `FileNotFoundError`。
  **补测两选一**（任一约 10–15 min）：① 把 flashocc 测试配置去掉 `LoadPointsFromFile`（预测本就不用点云）后重跑；② 从 `/root/autodl-pub` 重抽 val 场景的 LIDAR_TOP 再跑。当前先记 README 的 R50 ≈32（同口径可比）。
- **BEVDet**：仓库内**没有**已训好的 bevdet 单帧占用权重或评测记录。需你指明：是引用 BEVDet-Occ 论文数字，还是有一份本地权重要测？我不臆造。

## 待办
- [ ] FlashOcc 本机三口径重评（去 LiDAR 依赖）→ 填实测值替换 ≈32。
- [ ] BEVDet：确认来源（论文引用 or 本地权重），再定值。
