#!/bin/bash
# 阶段 2：冻结感知骨干 + 只训未来头（带两个修复）
# 用法: run_stage2_from.sh <perception_ckpt> <work_dir> [max_epochs]
#
# 配置固定为 priorocc-4d-r50-sep-frozen.py：
#   freeze_perception=True（只训 6.763M 运动/未来分支）
#   sep_perception=True（BEV 编码器只吃当前帧 64 通道 → 感知权重 1:1 继承）
#   max_flow_cells=[12,24,36] + non_autoregressive=True + gate_init_bias=2.0
#   enable_sem_continuity=False
set -u
CKPT=$1
WORKDIR=$2
EPOCHS=${3:-6}
# 显存降级用：SAMPLES_PER_GPU=1 可在 4090 上再降一档（OOM 守护使用）
SPG=${SAMPLES_PER_GPU:-2}
D=/root/autodl-tmp/dataforpriorocc
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
echo "[stage2] $(date '+%F %T') ckpt=$CKPT workdir=$WORKDIR epochs=$EPOCHS spg=$SPG"
python tools/train.py projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py \
  --work-dir "$WORKDIR" --seed 0 --gpu-id 0 --validate \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=$SPG data.workers_per_gpu=4 \
    runner.max_epochs=$EPOCHS load_from="$CKPT"
echo "STAGE2_EXIT=$? $(date '+%F %T')"
