#!/bin/bash
# PCF 收尾守护 + 多尺度 query 自动热启
#   Phase 1: 等 PCF ep4 → 暂停 → 600帧评测 → 续训
#   Phase 2: 等 PCF ep6 → 暂停 → 全量4818评测 → 记录最终PCF数字
#   Phase 3: 自动启动多尺度 query（热启自 PCF ep6）
#   Phase 4: 等 query-ms ep2/ep4 → 600帧评测 → 续训; ep6 → 全量评测
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
PCF_W=$D/work_dirs/pcf-v1
QMS_W=$D/work_dirs/query-ms-v1
PCF_CFG=projects/configs/priorocc/priorocc-4d-r50-pcf.py
QMS_CFG=projects/configs/priorocc/priorocc-4d-r50-query-ms.py
EVALB=$D/eval_b_baseline.py
REPORT=$R/tools/stage1/perclass_report.py
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
log(){ echo "$(date '+%F %T') [finish] $*" | tee -a $LOG; }

free_gpu(){
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
}

eval_600(){
  # $1=ckpt $2=cfg $3=tag
  python $EVALB $1 --cfg $2 --limit 600 > $O/evb_${3}.log 2>&1
  for k in 0 1 2 3 4 5 6; do
    lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
          3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
    v=$(python $REPORT $O/evb_${3}.log --emit --section $k 2>/dev/null)
    log "$3 $lab mIoU/dyn/sta/cont6 = $v"
  done
}

eval_full(){
  # $1=ckpt $2=cfg $3=tag
  python $EVALB $1 --cfg $2 > $O/evb_${3}_full.log 2>&1
  for k in 0 1 2 3 4 5 6; do
    lab=$(case $k in 0) echo "CURRENT";; 1) echo "MODEL 1s";; 2) echo "MODEL 2s";;
          3) echo "MODEL 3s";; 4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
    v=$(python $REPORT $O/evb_${3}_full.log --emit --section $k 2>/dev/null)
    log "$3(全量) $lab mIoU/dyn/sta/cont6 = $v"
  done
}

# ===== Phase 1: PCF ep4 =====
log "Phase1: 等待 PCF epoch_4_ema.pth"
for i in $(seq 1 1200); do
  [ -f "$PCF_W/epoch_4_ema.pth" ] && break
  sleep 30
done
[ -f "$PCF_W/epoch_4_ema.pth" ] || { log "Phase1 超时"; exit 1; }
sleep 30
log "PCF ep4: 暂停训练"
pkill -9 -f "pcf[-]v1"
free_gpu
log "PCF ep4: 600帧评测"
eval_600 $PCF_W/epoch_4_ema.pth $PCF_CFG "PCF ep4"
log "PCF ep4: 续训"
setsid nohup python tools/train.py $PCF_CFG \
  --work-dir $PCF_W --seed 0 --gpu-id 0 \
  --resume-from $PCF_W/epoch_4.pth \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=2 data.workers_per_gpu=4 \
    runner.max_epochs=6 load_from=None \
  >> $O/train_pcf.log 2>&1 < /dev/null &
sleep 120

# ===== Phase 2: PCF ep6 全量 =====
log "Phase2: 等待 PCF epoch_6_ema.pth"
for i in $(seq 1 1200); do
  [ -f "$PCF_W/epoch_6_ema.pth" ] && break
  sleep 30
done
[ -f "$PCF_W/epoch_6_ema.pth" ] || { log "Phase2 超时"; exit 1; }
sleep 30
log "PCF ep6: 暂停训练（最终轮）"
pkill -9 -f "pcf[-]v1"
free_gpu
log "PCF ep6: 全量 4818 评测"
eval_full $PCF_W/epoch_6_ema.pth $PCF_CFG "PCF ep6"
log "PCF 最终结论已记录"

# ===== Phase 3: 启动多尺度 query =====
log "Phase3: 启动多尺度 query（热启自 PCF ep6）"
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
setsid nohup python tools/train.py $QMS_CFG \
  --work-dir $QMS_W --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=2 data.workers_per_gpu=4 \
    runner.max_epochs=6 load_from=$PCF_W/epoch_6.pth \
  > $O/train_query_ms.log 2>&1 < /dev/null &
sleep 180
# 确认启动成功
if ps aux | grep "train[.]py" | grep -v grep | grep -q "query-ms"; then
  log "多尺度 query 已启动 (work_dir=$QMS_W)"
else
  log "ERROR: 多尺度 query 启动失败, 检查 $O/train_query_ms.log"
  exit 1
fi

# ===== Phase 4: query-ms 偶数轮评测 =====
for N in 2 4; do
  log "等待 query-ms epoch_${N}_ema.pth"
  for i in $(seq 1 1200); do
    [ -f "$QMS_W/epoch_${N}_ema.pth" ] && break
    sleep 30
  done
  [ -f "$QMS_W/epoch_${N}_ema.pth" ] || { log "query-ms ep${N} 超时"; exit 1; }
  sleep 30
  log "query-ms ep${N}: 暂停训练"
  pkill -9 -f "query[-]ms"
  free_gpu
  log "query-ms ep${N}: 600帧评测"
  eval_600 $QMS_W/epoch_${N}_ema.pth $QMS_CFG "query-ms ep${N}"
  log "query-ms ep${N}: 续训"
  setsid nohup python tools/train.py $QMS_CFG \
    --work-dir $QMS_W --seed 0 --gpu-id 0 \
    --resume-from $QMS_W/epoch_${N}.pth \
    --cfg-options device=cuda cudnn_benchmark=True \
      data.samples_per_gpu=2 data.workers_per_gpu=4 \
      runner.max_epochs=6 load_from=None \
    >> $O/train_query_ms.log 2>&1 < /dev/null &
  sleep 120
done

# ===== Phase 5: query-ms ep6 全量 =====
log "等待 query-ms epoch_6_ema.pth（最终轮）"
for i in $(seq 1 1200); do
  [ -f "$QMS_W/epoch_6_ema.pth" ] && break
  sleep 30
done
[ -f "$QMS_W/epoch_6_ema.pth" ] || { log "query-ms ep6 超时"; exit 1; }
sleep 30
log "query-ms ep6: 暂停训练（最终轮）"
pkill -9 -f "query[-]ms"
free_gpu
log "query-ms ep6: 全量 4818 评测"
eval_full $QMS_W/epoch_6_ema.pth $QMS_CFG "query-ms ep6"
log "ALL DONE — PCF 6轮 + 多尺度 query 6轮 全部完成"
