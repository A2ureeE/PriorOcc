#!/bin/bash
# 阶段 2 偶数轮评测守护（ep4 边界；ep2 已人工补评，ep6 由 stage2_finish.sh 做全量）
#   轮询 epoch_4.pth → 暂停训练 → 600 帧三口径评测(MODEL/B档) → 写 auto_run.log
#   → 用 --resume-from latest 接上训练
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
W2=$D/work_dirs/stage2-frozen
CFG=projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py
EVALB=$D/eval_b_baseline.py
REPORT=$R/tools/stage1/perclass_report.py
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
log(){ echo "$(date '+%F %T') [even-eval] $*" | tee -a $LOG; }

log "等待 epoch_4.pth（最多 12h）"
for i in $(seq 1 1440); do
  [ -f "$W2/epoch_4_ema.pth" ] && break
  sleep 30
done
[ -f "$W2/epoch_4_ema.pth" ] || { log "12h 未见 epoch_4_ema.pth，退出"; exit 1; }
sleep 30
log "epoch_4 边界：暂停训练"
pkill -9 -f "stage2[-]frozen"
for i in $(seq 1 90); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ -n "$m" ] && [ "$m" -lt 1200 ] && break
  sleep 4
done
log "评 epoch_4_ema（600 帧，MODEL + B档）"
python $EVALB $W2/epoch_4_ema.pth --cfg $CFG --limit 600 > $O/evb_s2_ep4.log 2>&1
for k in 0 1 2 3 4 5 6; do
  lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
        3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
  v=$(python $REPORT $O/evb_s2_ep4.log --emit --section $k 2>/dev/null)
  log "ep4 $lab mIoU/dyn/sta/cont6 = $v"
done
log "续训（--resume-from latest）"
setsid nohup bash $R/tools/stage1/run_stage2_resume2.sh $W2 6 \
  >> $O/train_stage2_resume.log 2>&1 < /dev/null &
sleep 120
log "even-eval 守护完成本轮"
