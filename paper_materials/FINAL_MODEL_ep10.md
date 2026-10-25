# 最终模型卡片 — final-p1-depth Phase2 epoch_10 (EMA)

> 本文件说明论文主模型（4D 占用预测）**epoch_10** 是怎么练出来的：数据、配置链、
> 分阶段训练、命令、结果、权重位置、如何复现/评测。结果口径统一为
> **全量 4818 val · EMA 权重**（除非注明）。
> 配套：`DATA_INVENTORY.md`（全部实验臂）、`METHODOLOGY.md`（方法）、
> `BASELINES_AND_SOTA.md`（外部对比）。

---

## 0. 一句话结论

从单帧 PriorOcc（ep21，当前帧 33.83）热启，经 **Phase1（干净 4D 配方）→ Phase2（动态类加权 + lr 衰减）** 续训得到的 **epoch_10**，在**几乎不牺牲当前帧**（30.36→30.58）的前提下，把未来预测做到：未来 mIoU **22.12 / 18.78 / 16.69**（1s/2s/3s），相对静态复制基线（B 档）的**动态 Δ 达 +2.93 / +2.67 / +2.32（三个时程全正）**——这是"语义先验时空化提升动态目标未来预测"这一核心主张的主证据。

## 1. 权重位置（本仓库内，.gitignore 已排除，不上传）

```
released_ckpts/
├── final-p1-depth_phase2_ep10_ema.pth   # ★ 论文主模型（EMA, 228M, 仅权重）
├── final-p1-depth_phase2_ep10.pth       #   epoch_10 原始权重（含优化器状态, 682M, 用于续训）
└── final-p1-depth_phase1_ep8_ema.pth    #   Phase1 加权前终值（用于 boost on/off 消融）
ckpts/
├── priorocc-singleframe-ep21.pth        #   单帧热启源（当前帧 33.83）
└── priorocc-singleframe-ep21_4dcompat.pth  #  4D 兼容版（首层形状失配已过滤, Phase1 实际 load_from）
```

> 评测/报数一律用 `_ema`；对外发布主模型 = `final-p1-depth_phase2_ep10_ema.pth`。

## 2. 配置继承链

```
priorocc-r50.py                             # 单帧基线（含 LiDAR depth 监督默认值）
└─ priorocc-4d-r50-smp-sdp-holefill.py      # 4D 多帧运动链
   └─ priorocc-4d-r50-smp-sdp-holefill-nodepth.py  # 派生（depth 可开关）
      └─ priorocc-4d-r50-final-p1.py        # Phase1：去SMP + PCF + ego对齐 + 非AR + 逐段cap + lr1e-4 step[8]
         └─ priorocc-4d-r50-final-p1-depth.py  # Phase2：加 depth=3.0 + uniform 动态类加权=3.0（仅未来）
            └─ priorocc-4d-r50-final-p1-depth-p3.py  # Phase3：逐类加权探针（已判定无增益，弃用）
```

模型关键部件（方法创新，详见 METHODOLOGY.md）：SemanticInjector（SI 语义注入）、
dyn_sta_decoder（动/静分解 + per-class mask）、SCMF（语义条件运动场，**非自回归 + 逐时程 cap [12,24,36] cell**）、
**PCF（PriorCorrespondenceFlow：用相关测量位移，而非回归猜测）**、future-only 动态类 CE 加权、
LiDAR 投影深度监督（仅训练期，推理纯相机）。

## 3. 数据（不入库，见第 7 节说明）

- nuScenes v1.0-trainval；infos：`bevdetv2-nuscenes_infos_{train,val}_forecast.pkl`（forecast 版含未来帧）。
- 占用 GT：`gts/*/labels.npz`（semantics/mask_lidar/mask_camera，形状 (200,200,16)，类序 0=others，1..10=十个动态类，11..16 静态面）。
- 2D 语义标签：`seg_2d_labels/`（SI 监督）。
- LiDAR：`samples/LIDAR_TOP`（**仅**训练期投影 `gt_depth` 做深度监督；模型输入纯相机）。

## 4. 分阶段训练

### Phase 1 — 干净 4D 配方（ep1–8, lr 1e-4）
配置 `priorocc-4d-r50-final-p1-depth.py` 的 Phase1 前身 `...final-p1.py`；`load_from = ckpts/priorocc-singleframe-ep21_4dcompat.pth`。要点：关 SMP、开 PCF、开 ego 对齐（`align_after_view_transfromation=True`）、非自回归 + 逐段 cap、`loss_depth_weight=3.0`。

结果（ep8 全量 EMA）：当前帧 **30.36** / 未来 **21.17 / 17.85 / 15.86** / 动态 Δ **+1.26 / +1.00 / +0.74**。

### Phase 2 — 动态类加权 + lr 衰减（续训, lr 自动 1e-5）
在 Phase1 基础上 `occ_head.dynamic_class_boost=3.0`（**仅作用于未来 occ 损失**，当前帧保持标准 CE），`step=[8]` 使 ep9 起 lr=1e-5。

结果（全量 EMA，逐轮）：

| | 当前帧 | 未来1s | 未来2s | 未来3s | 动态 Δ |
|---|---|---|---|---|---|
| ep8(=Phase1) | 30.36 | 21.17 | 17.85 | 15.86 | +1.26/+1.00/+0.74 |
| **ep10 ★best** | **30.58** | **22.12** | **18.78** | **16.69** | **+2.93/+2.67/+2.32** |
| ep11 | 30.63 | 22.09 | 18.73 | 16.65 | ≈同 |
| ep12 | 30.49 | 22.05 | 18.73 | 16.65 | ≈同 |

判定：加权红利在 **ep10 一轮内吃完**（ep6→ep10 动态 Δ 从 +1.1 跳到 +2.8，远超 ±0.5 噪声；ep10→ep12 三项变化 ≤0.03，饱和）。**取 ep10 为 best**（未来六项读数严格支配 ep11/12）。

### Phase 3 — 逐类加权探针（弃用，负结果）
`...final-p1-depth-p3.py`：bus/trailer/constr ×5、bicycle/motorcycle/pedestrian ×6、其余 ×3，从 ep10 续训 1 轮。全量 ep11：未来 22.06/18.76/16.65、动态 Δ +2.80/+2.65/+2.28 —— **相对 uniform×3 无提升（噪声内）**。据此停止：加权这根轴已饱和；bus 的负 Δ 是"位移精度/几何"病，加权救不动（需 flow 监督或 attention，另议）。

## 5. 复现命令

```bash
# Phase1（load_from 单帧 ep21）
python tools/train.py projects/configs/priorocc/priorocc-4d-r50-final-p1.py \
  --work-dir <wd> --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True data.samples_per_gpu=2 data.workers_per_gpu=4 runner.max_epochs=8

# Phase2（从 Phase1 ep8 续训，depth+boost 配置）
python tools/train.py projects/configs/priorocc/priorocc-4d-r50-final-p1-depth.py \
  --work-dir <wd> --seed 0 --gpu-id 0 \
  --resume-from <wd>/epoch_8.pth \
  --cfg-options device=cuda cudnn_benchmark=True data.samples_per_gpu=2 data.workers_per_gpu=4 runner.max_epochs=10 load_from=None

# 评测（全量 4818 · EMA，三口径：当前/未来MODEL/静态复制B档）
python <eval_b_baseline.py> released_ckpts/final-p1-depth_phase2_ep10_ema.pth \
  --cfg projects/configs/priorocc/priorocc-4d-r50-final-p1-depth.py
```

## 6. 已知失败尝试（避坑，均已回退）

- **训练提速**：PCF 相关体向量化（441→21 unfold）、历史帧 no_grad、历史帧拼批、horizon 拼批——全部触发 **CUDA OOM**（模型显存 97.6%，未来 occ_head 3×921MB logits 是峰值源），已回退，训练以 ~1.1 s/it 稳定跑完。
- **flow 直接监督**：数据支持（gt_boxes 9 列含 velocity、未来框可按 scene+timestamp 取），但 velocity 帧系/朝向有歧义、且需解决 backward-warp "目标格表示能力"这一与旧 warp 同源的天花板；权衡投入产出不划算，暂缓。

## 7. 磁盘/数据现状

大文件数据（nuScenes、work_dirs）留在 `/root/autodl-tmp/dataforpriorocc`（`data/nuscenes` 软链于此），不入库。清理到 ≤50GB 需删除 nuScenes 训练原生存量（sweeps 80G、train samples 等）与旧 work_dirs——**由维护者手动执行**（见仓库根 `CLEANUP.md`）。
