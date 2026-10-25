# 论文素材索引（一次性下载用）

> 本目录汇总写 PriorOcc-4D 论文可能用到的全部 `.md`。结果口径：nuScenes val 全量 4818 · EMA · Occ3D(200×200×16,18类) · Metric_mIoU · use_image_mask。主模型 = **final-p1-depth Phase2 epoch_10 (EMA)**。

## 按用途

| 文件 | 内容 | 论文用于 |
|---|---|---|
| `METHODOLOGY.md` | 问题定义、架构、各模块公式（SI/SCMF/PCF/加权）、消融设计 | **Method** |
| `FINAL_MODEL_ep10.md` | 终模型怎么练的：配置链、Phase1/2/3、命令、结果、权重位置、失败尝试 | Method/实现细节、附录 |
| `DATA_INVENTORY.md` | 全部实验臂（A–K）数据清单与结论 | **Experiments / 消融表来源** |
| `ABLATION.md` | 消融矩阵 + 手上权重能跑的对照 + 总对比表 | **Ablation 章节** |
| `RESULTS_perclass_ep10.md` | 主模型逐类 IoU（MODEL vs B档静态复制 vs Δ） | 动态子集证据、failure case |
| `RESULTS_ablation_extra.md` | C 架构对照(SMP/AR) + FlashOcc 基线(补测状态) | 架构消融行 |
| `SINGLEFRAME_GROUP.md` | 单帧占用组（ep21/ab-on/flashocc/bevdet）→ 量化预测代价 | Baseline 对比表 |
| `BASELINES_AND_SOTA.md` | 外部方法与 SOTA 参照 | Related Work / 主对比表 |
| `EVAL_METHOD.md` | 三口径评测协议、EMA纪律、复现命令 | 实验设置/可复现附录 |
| `CAM4DOCC_PREP.md` | Cam4DOcc 官方协议对接（网格/类映射/资产） | 跨协议对比计划 |
| `future_head_redesign.md` | 未来头设计推演（含 attention 方向） | 讨论/未来工作 |
| `README.md` | 仓库原有说明 | — |
| `CLEANUP.md` | 磁盘清理命令（运维，非论文） | 附录/工程 |

## 核心结论速览（全量4818·EMA）
- 主模型 ep10：当前帧 **30.58**；未来 1s/2s/3s mIoU **22.12/18.78/16.69**；动态 Δ(MODEL−静态复制) **+2.93/+2.67/+2.32**。
- 动态加权(Phase1→2)：动态 Δ 从 +1.1 级 → +2.7~2.9 级；当前帧不掉。逐类加权(Phase3)：无进一步增益（饱和）。
- 架构对照(C)：SMP+自回归旧配方动态 Δ +0.92/−0.29/−0.76（2s/3s负）→ 新配方全正，证明去SMP+非自回归+cap+PCF+加权的作用。
- 单帧组：ep21 33.83 / ab-on 33.94；4D 当前帧代价 ≈ −3.4（换取未来能力）。

## 待补
- FlashOcc 本机三口径重评（现因 LiDAR 清理受阻，方案见 SINGLEFRAME_GROUP.md）。
- BEVDet 单帧基线来源待确认。
- Cam4DOcc 官方协议上的未来预测重跑（资产已就绪）。
