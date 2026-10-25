#!/bin/bash
# Phase2 best-scan: 全量 4818·EMA 顺序评 ep10/11/12，三口径(CURRENT/MODEL/B档)
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
W=$D/work_dirs/final-p1-depth-v1
CFG=projects/configs/priorocc/priorocc-4d-r50-final-p1-depth.py
EVALB=$D/eval_b_baseline.py
REPORT=$R/tools/stage1/perclass_report.py
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
log(){ echo "$(date '+%F %T') [best-scan] $*" | tee -a "$LOG"; }

free_gpu(){
  for i in $(seq 1 120); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 5
  done
  log "WARN: GPU 未释放"
}

run_eval(){  # $1=ckpt $2=tag  (无 --limit = 全量4818)
  local out="$O/evb_${2}.log"
  python $EVALB "$1" --cfg $CFG > "$out" 2>&1
  for k in 0 1 2 3 4 5 6; do
    case $k in 0) lab="CURRENT";; 1) lab="MODEL_1s";; 2) lab="MODEL_2s";;
      3) lab="MODEL_3s";; 4) lab="B_1s";; 5) lab="B_2s";; 6) lab="B_3s";; esac
    v=$(python $REPORT "$out" --emit --section $k 2>/dev/null)
    log "$2 $lab mIoU/dyn/sta/cont6 = $v"
  done
}

free_gpu
for N in 10 11 12; do
  if [ ! -f "$W/epoch_${N}_ema.pth" ]; then
    log "epoch_${N}_ema.pth 不存在, 跳过"; continue
  fi
  log "全量 4818 评测 epoch_${N}_ema.pth (Phase2)"
  run_eval "$W/epoch_${N}_ema.pth" "finalp1d_p2_ep${N}_full"
  free_gpu
done
log "BEST-SCAN DONE — ep10/11/12 全量三口径已写入"
