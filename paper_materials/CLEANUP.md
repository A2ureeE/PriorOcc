# 磁盘清理指南（autodl-tmp 降到 ≤50GB，nuScenes 只留测试集）

> ⚠️ 删除不可逆。本仓库运行环境不允许我执行 `rm`，以下命令请你在 SSH 里**逐条确认后执行**。
> 每条前都给了 `du` 核对大小；先跑 `du` 看数字再删。`/root/autodl-tmp` 当前 ≈165GB。

数据实体在 `/root/autodl-tmp/dataforpriorocc`（`PriorOcc/data/nuscenes` 是它的软链）。
`autodl-pub` 上的 nuScenes 原始分卷（`/autodl-pub/data/nuScenes/.../*.tgz`）**不动**，删掉的都可从那里重新解压。

## 第 1 档：安全、收益大（训练已不用）≈ 减 ~92GB

```bash
# 1) sweeps 80G —— 已证实训练零消费（get_adj_info 按 data_infos 索引取邻帧，不读 sweeps）
du -sh /root/autodl-tmp/dataforpriorocc/nuscenes/sweeps
rm -rf /root/autodl-tmp/dataforpriorocc/nuscenes/sweeps

# 2) can_bus 4.4G + maps 之外的非必要 —— can_bus 模型不用
du -sh /root/autodl-tmp/dataforpriorocc/nuscenes/can_bus
rm -rf /root/autodl-tmp/dataforpriorocc/nuscenes/can_bus

# 3) seg_2d_labels 3.0G —— 仅训练期 SI 监督用；不再训练即可删
du -sh /root/autodl-tmp/dataforpriorocc/nuscenes/seg_2d_labels
rm -rf /root/autodl-tmp/dataforpriorocc/nuscenes/seg_2d_labels

# 4) 训练用 LiDAR 样本 51G 量级 —— 仅 depth 监督用；不再训练即可整目录删
#    （samples/LIDAR_TOP 全部是训练解压来的，评测/推理纯相机不需要）
du -sh /root/autodl-tmp/dataforpriorocc/nuscenes/samples/LIDAR_TOP
rm -rf /root/autodl-tmp/dataforpriorocc/nuscenes/samples/LIDAR_TOP

# 5) 训练 infos 里体积大的中间/子集 pkl（保留 val 两个即可）
rm -f /root/autodl-tmp/dataforpriorocc/nuscenes/bevdetv2-nuscenes_infos_train*.pkl
```

第 1 档做完后应已 `df -h /root/autodl-tmp` 见 <60GB 占用。核对：
```bash
du -sh /root/autodl-tmp
```

## 第 2 档：只留测试集（samples 图片按 val scene 裁剪）≈ 再减 ~30GB

`samples/` 下每个场景一个目录，train 与 val 混在一起。只保留 **val 场景**：

```bash
cd /root/priorocc/PriorOcc
python - <<'PY'
import pickle, os, glob
base='/root/autodl-tmp/dataforpriorocc/nuscenes'
val=pickle.load(open(f'{base}/bevdetv2-nuscenes_infos_val.pkl','rb'))
keep=set()
for info in (val['infos'] if isinstance(val,dict) else val):
    for cam in info.get('cams',{}).values():
        # samples/CAM_FRONT/<scene_token>_<token>.jpg  → 取不到 scene 目录名，改按 token 路径父级
        p=cam['data_path']
        keep.add(os.path.basename(os.path.dirname(p)))  # nuScenes samples 按 token 平铺时这行取 token
print('val 保留条目:', len(keep))
# 删除 samples 下不在 keep 的 token 目录/文件（按你实际 samples 布局调整）
PY
echo "上面脚本仅打印计划；确认布局后再手动 rm。"
```

> nuScenes `samples/CAM_*/` 通常按 `<token>.jpg` 平铺（非按 scene 分目录）。若如此，安全做法：
> 用 val infos 的 token 集合生成保留清单，删除 `samples/CAM_*` 中不在清单的 jpg。
> 若你已不需要重训，最省事：**连 `samples/` 一起删**（纯相机评测才需要图像；若连测试评测也要跑，则只留 val token 的图）。请按需选择。

## 第 3 档：旧 work_dirs（消融已完成则可删，先挪走要留的 EMA）

```bash
# 需要留作消融对照的 EMA（若还想本地评测），先复制到仓库外/别处：
#   smp-warmstart/epoch_15_ema.pth  —— “零训练修复版/从头 run” 历史基线
#   ab-on/epoch_4_ema.pth           —— 两阶段冻结感知来源
#   pcf-v1/epoch_6_ema.pth            —— PCF 单独臂对照
# 主模型与 Phase1/3 已存 PriorOcc/released_ckpts（ep8/ep10）；Phase3 ep11 无增益未留。
du -sh /root/autodl-tmp/dataforpriorocc/work_dirs/*
rm -rf /root/autodl-tmp/dataforpriorocc/work_dirs/{smp-warmstart,ab-on,pcf-v1,final-p1-v1,final-p1-depth-p3-v1}
# 保留 final-p1-depth-v1（剩余 epoch 或已被 released_ckpts 取代，可整目录删）
```

## 收尾核对
```bash
df -h /root/autodl-tmp          # 目标 used ≤ ~50GB
du -sh /root/autodl-tmp/dataforpriorocc/*  # 逐项确认
```
删除后如需再训练/评测：从 `/autodl-pub` 重新解压对应分卷即可（sweeps/LIDAR/val 图按需）。
