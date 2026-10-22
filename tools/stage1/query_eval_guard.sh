#!/bin/bash
# Query Future Head 评测守护 + 用户裁定 fallback 规则
#   ep2: 暂停训练 → 600 帧三口径评测 → 与纯 PCF ep2 (dyn 1s=8.95) 比较
#         若 query < 8.95 → 杀掉 query, 从 PCF ep2 续训到 ep6 (fallback)
#         若 query >= 8.95 → 续训 query
#   ep4: 同 ep2 逻辑 (fallback 阈值 = PCF 同轮值, 若无则不 fallback)
#   ep6: 全量 4818 三口径 → 最终结论
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
W=$D/work_dirs/query-v1
PCF_W=$D/work_dirs/pcf-v1
CFG=projects/configs/priorocc/priorocc-4d-r50-query.py
PCF_CFG=projects/configs/priorocc/priorocc-4d-r50-pcf.py
EVALB=$D/eval_b_baseline.py
REPORT=$R/tools/stage1/perclass_report.py
# 用户裁定的 fallback 判据: 纯 PCF ep2 的 MODEL 1s dyn = 8.95
PCF_EP2_DYN_1S=8.95
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
log(){ echo "$(date '+%F %T') [query-eval] $*" | tee -a $LOG; }

free_gpu(){
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
}

resume_query(){
  export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
  export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  setsid nohup python tools/train.py $CFG \
    --work-dir $W --seed 0 --gpu-id 0 --resume-from $W/epoch_${1}.pth \
    --cfg-options device=cuda cudnn_benchmark=True \
      data.samples_per_gpu=2 data.workers_per_gpu=4 \
      runner.max_epochs=6 load_from=None \
    >> $O/train_query.log 2>&1 < /dev/null &
}

fallback_pcf(){
  log "FALLBACK: query ep${1} dyn_1s=${2} < PCF ep2 ${PCF_EP2_DYN_1S} → 回退 PCF 续训"
  log "从 PCF ep2 续训到 ep6 (用户裁定规则)"
  export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
  export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  setsid nohup python tools/train.py $PCF_CFG \
    --work-dir $PCF_W --seed 0 --gpu-id 0 \
    --resume-from $PCF_W/epoch_2.pth \
    --cfg-options device=cuda cudnn_benchmark=True \
      data.samples_per_gpu=2 data.workers_per_gpu=4 \
      runner.max_epochs=6 load_from=None \
    >> $O/train_pcf.log 2>&1 < /dev/null &
  log "PCF 已从 ep2 恢复训练 (fallback), query run 废弃"
  exit 0
}

for N in 2 4; do
  log "等待 epoch_${N}_ema.pth（query）"
  for i in $(seq 1 1200); do
    [ -f "$W/epoch_${N}_ema.pth" ] && break
    sleep 30
  done
  [ -f "$W/epoch_${N}_ema.pth" ] || { log "10h 未见 epoch_${N}_ema.pth"; exit 1; }
  sleep 30
  log "query ep${N}: 暂停训练"
  pkill -9 -f "query[-]v1"
  free_gpu
  log "query ep${N}: 600 帧三口径评测"
  python $EVALB $W/epoch_${N}_ema.pth --cfg $CFG --limit 600 \
    > $O/evb_query_ep${N}.log 2>&1
  DYN1S=""
  for k in 0 1 2 3 4 5 6; do
    lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
          3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
    v=$(python $REPORT $O/evb_query_ep${N}.log --emit --section $k 2>/dev/null)
    log "query ep${N} $lab mIoU/dyn/sta/cont6 = $v"
    # 提取 MODEL 1s 的 dyn (第二个字段)
    if [ "$k" = "1" ] && [ -n "$v" ]; then
      DYN1S=$(echo "$v" | awk '{print $2}')
    fi
  done
  # fallback 判定 (仅 ep2; ep4 的阈值待 ep2 结果出来后手动调整)
  if [ "$N" = "2" ] && [ -n "$DYN1S" ]; then
    BETTER=$(python -c "print(1 if float('${DYN1S}') >= float('${PCF_EP2_DYN_1S}') else 0)" 2>/dev/null)
    if [ "$BETTER" = "0" ]; then
      fallback_pcf $N $DYN1S
    else
      log "query ep${N} dyn_1s=${DYN1S} >= PCF ep2 ${PCF_EP2_DYN_1S} → 续训 query"
    fi
  fi
  log "query ep${N}: 续训"
  resume_query $N
  sleep 120
done

# ---- ep6: 全量收尾 ----
log "等待 epoch_6_ema.pth（query 最终轮）"
for i in $(seq 1 1200); do
  [ -f "$W/epoch_6_ema.pth" ] && break
  sleep 30
done
[ -f "$W/epoch_6_ema.pth" ] || { log "10h 未见 epoch_6_ema.pth"; exit 1; }
sleep 30
log "query ep6: 暂停训练（最终轮）"
pkill -9 -f "query[-]v1"
free_gpu
log "query ep6: 全量 4818 三口径评测"
python $EVALB $W/epoch_6_ema.pth --cfg $CFG \
  > $O/evb_query_ep6_full.log 2>&1
for k in 0 1 2 3 4 5 6; do
  lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
        3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
  v=$(python $REPORT $O/evb_query_ep6_full.log --emit --section $k 2>/dev/null)
  log "query ep6(全量) $lab mIoU/dyn/sta/cont6 = $v"
done
log "QUERY DONE — 全量结论已写入"
