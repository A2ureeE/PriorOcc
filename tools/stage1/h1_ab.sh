#!/bin/bash
# H1 快速验证：补洞开启后，单帧当前帧的连续类（背景）IoU 是否真的涨
#
# A/B 两臂完全同源：同子集(2000 帧, stride14)、同 seed、同 lr(1e-4 fixed)、
# 同 2000 iter、同评测帧(4818 池 stride8 的 600 帧, EMA ep4)。唯一差异：
#   A: model.enable_sem_continuity=False
#   B: 补洞三层开启，cont_2d/cont_bev=0.05 + cont_occ=2.5（按实测量级对齐）
D=/root/autodl-tmp/dataforpriorocc
Q=$D/quick_hyp
CFG=projects/configs/priorocc/priorocc-r50-holefill-sub2k.py
EVAL1=$D/eval_occ_standalone.py
REPORT=/root/priorocc/PriorOcc/tools/stage1/perclass_report.py
LIMIT=600
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
mkdir -p $Q
log(){ echo "$(date '+%F %T') $*" | tee -a $Q/h1_run.log; }
abort(){ log "FATAL: $*"; echo H1_FAILED >> $Q/h1_run.log; exit 1; }
need(){ [ -f "$1" ] || abort "缺少 $1"; }
free_gpu(){
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
}

log "H1 baseline: ep21 @${LIMIT}"
python $EVAL1 ckpts/priorocc-singleframe-ep21.pth --cfg $CFG --limit $LIMIT \
  > $Q/h1_base_ep21.log 2>&1

log "H1-A(off) 训练"
python tools/train.py $CFG --work-dir $D/work_dirs/h1-off \
  --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 \
    model.enable_sem_continuity=False > $Q/h1_off_train.log 2>&1
need $D/work_dirs/h1-off/epoch_4_ema.pth
free_gpu
python $EVAL1 $D/work_dirs/h1-off/epoch_4_ema.pth --cfg $CFG --limit $LIMIT \
  > $Q/h1_off_ep4ema.log 2>&1
log "H1-A(off) 完成"

log "H1-B(on) 训练"
python tools/train.py $CFG --work-dir $D/work_dirs/h1-on \
  --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 > $Q/h1_on_train.log 2>&1
need $D/work_dirs/h1-on/epoch_4_ema.pth
free_gpu
python $EVAL1 $D/work_dirs/h1-on/epoch_4_ema.pth --cfg $CFG --limit $LIMIT \
  > $Q/h1_on_ep4ema.log 2>&1
log "H1-B(on) 完成"

{
  echo "######## H1: 补洞 A/B（2000帧子集短训 2000 iter, 评测 600 帧 EMA ep4）########"
  echo "==== 基线 ep21（未训练）===="
  python $REPORT $Q/h1_base_ep21.log
  echo ""
  echo "==== A 臂 off vs 基线（负值=短训漂移）===="
  python $REPORT $Q/h1_off_ep4ema.log --baseline $Q/h1_base_ep21.log
  echo ""
  echo "==== B 臂 on vs 基线 ===="
  python $REPORT $Q/h1_on_ep4ema.log --baseline $Q/h1_base_ep21.log
  echo ""
  echo "==== 关键：B(on) − A(off) = 补洞净效应 ===="
  python $REPORT $Q/h1_on_ep4ema.log --baseline $Q/h1_off_ep4ema.log
} > $Q/H1_summary.txt 2>&1
log "H1 完成: $Q/H1_summary.txt"
echo H1_DONE >> $Q/h1_run.log
