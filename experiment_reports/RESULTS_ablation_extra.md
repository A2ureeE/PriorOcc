# 消融补充：架构对照(C) 与 FlashOcc 基线（全量4818·EMA）

## C. with-SMP/自回归旧架构  ablation_smp-warmstart_ep15_ema
（用其自身 SMP/AR 配置加载；三口径）
```
  CURRENT = 32.10 25.20 41.95 47.89
  MODEL_1s = 21.34 11.02 36.07 41.30
  MODEL_2s = 17.14 6.73 32.00 36.61
  MODEL_3s = 14.77 4.93 28.82 33.08
  B_1s = 18.96 10.10 31.61 36.34
  B_2s = 15.12 7.02 26.70 30.89
  B_3s = 13.21 5.69 23.96 27.87
```

## FlashOcc-R50 基线（单帧当前帧占用，无未来预测）
配置 `projects/configs/flashocc/flashocc-r50.py`，权重 flashocc-r50-256x704.pth。
- 当前帧 NA（口径 Metric_mIoU/18类/use_image_mask，与本模型当前帧可比）
