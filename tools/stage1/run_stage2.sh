#!/bin/bash
# 阶段 2：冻结阶段 1 单帧感知 + 只训修好两个 bug 的未来头
#
# 前置：
#   1) 阶段 1 已产出 $D/work_dirs/stage1-holefill/best_ema.pth
#   2) 已用 tools/dryload_check.py 审计过 stage2 配置对该 ckpt 的兼容性
#      （预期：感知 559 个张量全命中，仅运动链 92 个缺失）
D=/root/autodl-tmp/dataforpriorocc
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
echo "[stage2 train] $(date '+%F %T')"
python tools/train.py \
  projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py \
  --work-dir $D/work_dirs/stage2-sep-frozen \
  --seed 0 --gpu-id 0 --validate \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=2 data.workers_per_gpu=4
echo "STAGE2_TRAIN_EXIT=$? $(date '+%F %T')"
