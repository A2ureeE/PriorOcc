#!/bin/bash
# 阶段 1 续训：评测守护暂停训练后用本脚本接上（--auto-resume 从 latest 恢复）
D=/root/autodl-tmp/dataforpriorocc
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
echo "[stage1 resume] $(date '+%F %T')"
python tools/train.py \
  projects/configs/priorocc/priorocc-r50-holefill-full.py \
  --work-dir $D/work_dirs/stage1-holefill \
  --seed 0 --gpu-id 0 --validate --auto-resume \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4
echo "STAGE1_RESUME_EXIT=$? $(date '+%F %T')"
