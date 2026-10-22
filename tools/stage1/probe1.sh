#!/bin/bash
# 阶段 1 探针：全量 28130 帧跑 1 轮，然后立刻做 4 项测量
#   1) 全量 4818 非 EMA（与 ep21 的 33.83 同池同类型）
#   2) 全量 4818 EMA
#   3) 600 帧洞率（EMA / 非 EMA）
#   4) 逐类 diff 报告（连续类 = 背景）
D=/root/autodl-tmp/dataforpriorocc
Q=$D/probe1
W=$D/work_dirs/stage1-probe1
CFG=projects/configs/priorocc/priorocc-r50-holefill-probe1.py
EVAL1=$D/eval_occ_standalone.py
REPORT=/root/priorocc/PriorOcc/tools/stage1/perclass_report.py
HOLE=/root/priorocc/PriorOcc/tools/stage1/hole_rate.py
BASE=/root/priorocc/PriorOcc/experiment_reports/raw_eval/eval_ep21_same4818.log
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
mkdir -p $Q
log(){ echo "$(date '+%F %T') $*" | tee -a $Q/run.log; }
free_gpu(){
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
}

log "probe1: 全量 1 轮训练开始（28130 帧 / batch4 = 7033 iter）"
python tools/train.py $CFG --work-dir $W --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 > $Q/train.log 2>&1
[ -f $W/epoch_1.pth ] || { log "FATAL: 训练未产出 epoch_1.pth"; exit 1; }
free_gpu
log "训练完成，开始评测"

python $EVAL1 $W/epoch_1.pth --cfg $CFG --limit 0 > $Q/ep1_full.log 2>&1
python $EVAL1 $W/epoch_1_ema.pth --cfg $CFG --limit 0 > $Q/ep1ema_full.log 2>&1
# ep21 基线直接复用已有全量日志（只差 loss 权重/max_epochs，推理与模型结构逐位一致）
cp -f $D/stage1_ep0_ep21.log $Q/ep0_ep21_full.log
python $HOLE $W/epoch_1_ema.pth --cfg $CFG --limit 600 > $Q/hole_ep1ema.log 2>&1
python $HOLE ckpts/priorocc-singleframe-ep21.pth --cfg $CFG --limit 600 \
  > $Q/hole_ep21.log 2>&1

{
  echo "######## 阶段 1 探针：全量 1 轮（28130 帧, 7033 iter, cont_occ=2.5）########"
  echo ""
  echo "==== 全量 4818 当前帧 mIoU ===="
  grep -hE "EVAL on|mIoU of [0-9]+ samples" $Q/ep0_ep21_full.log | head -2
  grep -hE "EVAL on|mIoU of [0-9]+ samples" $Q/ep1_full.log | head -2
  grep -hE "EVAL on|mIoU of [0-9]+ samples" $Q/ep1ema_full.log | head -2
  echo ""
  echo "==== 逐类：ep1 非 EMA vs ep21（同池同类型）===="
  python $REPORT $Q/ep1_full.log --baseline $Q/ep0_ep21_full.log
  echo ""
  echo "==== 逐类：ep1 EMA vs ep21 ===="
  python $REPORT $Q/ep1ema_full.log --baseline $Q/ep0_ep21_full.log
  echo ""
  echo "==== 洞率（600 帧）===="
  echo "-- ep21 --";      grep -E "hole_rate" $Q/hole_ep21.log
  echo "-- ep1 EMA --";   grep -E "hole_rate" $Q/hole_ep1ema.log
} > $Q/PROBE1_summary.txt 2>&1

log "完成: $Q/PROBE1_summary.txt"
echo PROBE1_DONE >> $Q/run.log
