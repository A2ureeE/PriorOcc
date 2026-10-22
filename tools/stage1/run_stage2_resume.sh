#!/bin/bash
# 阶段 2 续训（去掉 --validate：mmdet 的 runner 内 single_gpu_test 用 result[0]
# 取结果，与 PriorOcc4D 返回 dict 不兼容 → KeyError: 0。评测改由
# eval_b_baseline.py 独立脚本在 ckpt 边界做）
set -u
CKPT=$1
WORKDIR=$2
EPOCHS=${3:-6}
SPG=${SAMPLES_PER_GPU:-2}
D=/root/autodl-tmp/dataforpriorocc
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
echo "[stage2-resume] $(date '+%F %T') workdir=$WORKDIR epochs=$EPOCHS spg=$SPG"
python tools/train.py projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py \
  --work-dir "$WORKDIR" --seed 0 --gpu-id 0 --auto-resume \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=$SPG data.workers_per_gpu=4 \
    runner.max_epochs=$EPOCHS load_from="$CKPT"
echo "STAGE2_RESUME_EXIT=$? $(date '+%F %T')"
