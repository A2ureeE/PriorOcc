#!/bin/bash
# 阶段 1：补洞版单帧感知训练（全量帧 + ep21 热启）
#
# --validate 必须打开：mmdet 的 train_detector 只在 validate=True 时挂
# EvalHook，否则 config 里的 evaluation 不会触发。ep4/ep8/ep12 各一次
# 全量 4818 帧非 EMA 评测，直接对应验收判据
# "当前帧全量 val mIoU > 33.83（ep21 同池非 EMA）"。
D=/root/autodl-tmp/dataforpriorocc
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
echo "[stage1 train] $(date '+%F %T')"
python tools/train.py \
  projects/configs/priorocc/priorocc-r50-holefill-full.py \
  --work-dir $D/work_dirs/stage1-holefill \
  --seed 0 --gpu-id 0 --validate \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4
echo "STAGE1_TRAIN_EXIT=$? $(date '+%F %T')"
