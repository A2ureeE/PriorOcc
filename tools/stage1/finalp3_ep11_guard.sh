#!/bin/bash
# P3 单轮验收: 等 final-p1-depth-p3-v1/epoch_11_ema.pth → 全量4818三口径 → 落日志
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
W=$D/work_dirs/final-p1-depth-p3-v1
CFG=projects/configs/priorocc/priorocc-4d-r50-final-p1-depth-p3.py
EVALB=$D/eval_b_baseline.py
REPORT=$R/tools/stage1/perclass_report.py
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
log(){ echo "$(date '+%F %T') [p3-ep11] $*" | tee -a "$LOG"; }

free_gpu(){
  for i in $(seq 1 120); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 5
  done
  log "WARN: GPU 未释放"
}

log "等待 P3 epoch_11_ema.pth (纯Phase3单轮)"
for i in $(seq 1 1440); do
  [ -f "$W/epoch_11_ema.pth" ] && break
  sleep 30
done
[ -f "$W/epoch_11_ema.pth" ] || { log "P3 ep11 等待超时"; exit 1; }
sleep 40
log "P3 ep11: 暂停训练"
pkill -9 -f "final-p1-depth-p3"
free_gpu
out="$O/evb_finalp3_ep11_full.log"
python $EVALB "$W/epoch_11_ema.pth" --cfg $CFG > "$out" 2>&1
for k in 0 1 2 3 4 5 6; do
  case $k in 0) lab="CURRENT";; 1) lab="MODEL_1s";; 2) lab="MODEL_2s";;
    3) lab="MODEL_3s";; 4) lab="B_1s";; 5) lab="B_2s";; 6) lab="B_3s";; esac
  v=$(python $REPORT "$out" --emit --section $k 2>/dev/null)
  log "finalp3_ep11_full $lab mIoU/dyn/sta/cont6 = $v"
done
log "P3-EP11 DONE — 与 Phase2 ep11 全量对比"
