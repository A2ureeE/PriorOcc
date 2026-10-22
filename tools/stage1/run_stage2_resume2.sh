#!/bin/bash
# 阶段 2 续训（显式 --resume-from，不用 --auto-resume：本环境 find_latest_checkpoint
# 返回 None 导致回落到 load_from 并崩溃；也没有 --validate，评测改由独立脚本做）
# 用法: run_stage2_resume2.sh <work_dir> [max_epochs]
set -u
WORKDIR=$1
EPOCHS=${2:-6}
SPG=${SAMPLES_PER_GPU:-2}
D=/root/autodl-tmp/dataforpriorocc
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
CKPT=$(readlink -f "$WORKDIR/latest.pth")
echo "[stage2-resume] $(date '+%F %T') resume_from=$CKPT epochs=$EPOCHS spg=$SPG"
python tools/train.py projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py \
  --work-dir "$WORKDIR" --seed 0 --gpu-id 0 --resume-from "$CKPT" \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=$SPG data.workers_per_gpu=4 \
    runner.max_epochs=$EPOCHS load_from=None
echo "STAGE2_RESUME_EXIT=$? $(date '+%F %T')"
