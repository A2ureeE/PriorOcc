#!/bin/bash
# PCF 偶数轮评测守护 + 末轮全量收尾
#   ep2/ep4: 暂停训练 → 600 帧三口径(MODEL/B档/当前帧) → 续训
#   ep6:     暂停训练 → 全量 4818 三口径 → 写最终结论
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
W=/root/autodl-tmp/dataforpriorocc/work_dirs/pcf-v1
CFG=projects/configs/priorocc/priorocc-4d-r50-pcf.py
EVALB=$D/eval_b_baseline.py
REPORT=$R/tools/stage1/perclass_report.py
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
log(){ echo "$(date '+%F %T') [pcf-eval] $*" | tee -a $LOG; }

free_gpu(){
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
}

for N in 2 4; do
  log "等待 epoch_${N}_ema.pth（PCF）"
  for i in $(seq 1 1200); do
    [ -f "$W/epoch_${N}_ema.pth" ] && break
    sleep 30
  done
  [ -f "$W/epoch_${N}_ema.pth" ] || { log "10h 未见 epoch_${N}_ema.pth"; exit 1; }
  sleep 30
  log "PCF ep${N}: 暂停训练"
  pkill -9 -f "pcf[-]v1"
  free_gpu
  log "PCF ep${N}: 600 帧三口径评测"
  python $EVALB $W/epoch_${N}_ema.pth --cfg $CFG --limit 600 \
    > $O/evb_pcf_ep${N}.log 2>&1
  for k in 0 1 2 3 4 5 6; do
    lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
          3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
    v=$(python $REPORT $O/evb_pcf_ep${N}.log --emit --section $k 2>/dev/null)
    log "PCF ep${N} $lab mIoU/dyn/sta/cont6 = $v"
  done
  log "PCF ep${N}: 续训"
  export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
  export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  setsid nohup python tools/train.py $CFG \
    --work-dir $W --seed 0 --gpu-id 0 --resume-from $W/epoch_${N}.pth \
    --cfg-options device=cuda cudnn_benchmark=True \
      data.samples_per_gpu=2 data.workers_per_gpu=4 \
      runner.max_epochs=6 load_from=None \
    >> $O/train_pcf.log 2>&1 < /dev/null &
  sleep 120
done

# ---- ep6: 全量收尾 ----
log "等待 epoch_6_ema.pth（PCF 最终轮）"
for i in $(seq 1 1200); do
  [ -f "$W/epoch_6_ema.pth" ] && break
  sleep 30
done
[ -f "$W/epoch_6_ema.pth" ] || { log "10h 未见 epoch_6_ema.pth"; exit 1; }
sleep 30
pkill -9 -f "pcf[-]v1"
free_gpu
log "PCF ep6: 全量 4818 三口径评测"
python $EVALB $W/epoch_6_ema.pth --cfg $CFG --limit 0 \
  > $O/evb_pcf_ep6_full.log 2>&1
for k in 0 1 2 3 4 5 6; do
  lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
        3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
  v=$(python $REPORT $O/evb_pcf_ep6_full.log --emit --section $k 2>/dev/null)
  log "PCF ep6-full $lab mIoU/dyn/sta/cont6 = $v"
done
{
  echo ""
  echo "==================== PCF 最终结论 $(date '+%F %T') ===================="
  echo "骨干: ep4_ema (4818=33.94) 冻结; PCF: correlation-measured flow + zero-init residual"
  echo "判据: 动态 Δ(M−B档) 1s/2s/3s 全转正; 静态不低于 B 档超 0.2; 当前帧≈33.94"
  grep -hE "mIoU of 4818" $O/evb_pcf_ep6_full.log
  echo "（顺序: 当前帧 / MODEL 1s/2s/3s / B档 1s/2s/3s）"
} | tee -a $LOG
log "==================== PCF DONE ===================="
