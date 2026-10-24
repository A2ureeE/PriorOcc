#!/bin/bash
# final-p1-depth Phase 2 评测守护
#   Phase2 = ep9-14 @ lr=1e-5 + 动态类加权(仅未来, boost=3.0)
#   ep10/12: 600帧三口径快测 → 续训
#   ep14:    全量4818评测 → Phase 2 终值
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
log(){ echo "$(date '+%F %T') [finalp1d-p2] $*" | tee -a "$LOG"; }

free_gpu(){
  for i in $(seq 1 120); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 5
  done
  log "WARN: GPU 未释放, 强行继续"
}

run_eval(){  # $1=ckpt $2=tag $3=limit(空=全量)
  local out="$O/evb_${2}.log"
  if [ -n "${3:-}" ]; then
    python $EVALB "$1" --cfg $CFG --limit "$3" > "$out" 2>&1
  else
    python $EVALB "$1" --cfg $CFG > "$out" 2>&1
  fi
  for k in 0 1 2 3 4 5 6; do
    case $k in 0) lab="CURRENT";; 1) lab="MODEL_1s";; 2) lab="MODEL_2s";;
      3) lab="MODEL_3s";; 4) lab="B_1s";; 5) lab="B_2s";; 6) lab="B_3s";; esac
    v=$(python $REPORT "$out" --emit --section $k 2>/dev/null)
    log "$2 $lab mIoU/dyn/sta/cont6 = $v"
  done
}

resume_train(){  # $1=epoch
  export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
  export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  setsid nohup python tools/train.py $CFG \
    --work-dir "$W" --seed 0 --gpu-id 0 --resume-from "$W/epoch_${1}.pth" \
    --cfg-options device=cuda cudnn_benchmark=True \
      data.samples_per_gpu=2 data.workers_per_gpu=4 \
      runner.max_epochs=14 load_from=None \
    >> "$O/train_finalp1depth.log" 2>&1 < /dev/null &
}

for N in 10 12; do
  log "等待 final-p1 epoch_${N}_ema.pth (Phase2)"
  for i in $(seq 1 1440); do
    [ -f "$W/epoch_${N}_ema.pth" ] && break
    sleep 30
  done
  [ -f "$W/epoch_${N}_ema.pth" ] || { log "ep${N} 等待超时"; exit 1; }
  sleep 40
  log "final-p1 ep${N}: 暂停训练"
  pkill -9 -f "final-p1-depth"
  free_gpu
  log "final-p1 ep${N}: 600帧评测"
  run_eval "$W/epoch_${N}_ema.pth" "finalp1d_p2_ep${N}" 600
  log "final-p1 ep${N}: 续训"
  resume_train $N
  sleep 120
done

# ---- ep14: Phase 2 终值 (全量) ----
log "等待 final-p1 epoch_14_ema.pth (Phase 2 最终轮)"
for i in $(seq 1 1440); do
  [ -f "$W/epoch_14_ema.pth" ] && break
  sleep 30
done
[ -f "$W/epoch_14_ema.pth" ] || { log "ep14 等待超时"; exit 1; }
sleep 40
log "final-p1 ep14: 暂停训练 (Phase 2 完成)"
pkill -9 -f "final-p1-depth"
free_gpu
log "final-p1 ep14: 全量 4818 评测"
run_eval "$W/epoch_14_ema.pth" "finalp1d_p2_ep14_full"
log "PHASE2D DONE — Phase 2 全量结论已写入"
