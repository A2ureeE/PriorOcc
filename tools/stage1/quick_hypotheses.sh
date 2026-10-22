#!/bin/bash
# 两个假设的小子集快速验证（串行，单卡）
#
#  H1 补洞是否值得：ep21 + 2000 帧 4 轮短训，A/B 两臂只差补洞 loss
#      A 臂: model.enable_sem_continuity=False
#      B 臂: cont_2d/cont_bev=0.05 + cont_occ=2.5（按实测量级对齐）
#      两臂用同 seed / 同子集 / 同 lr，评测同一 602 帧（4818 池 stride 8，EMA）
#  H2 两个 bug 修复是否有效：ep15_ema + 2000 帧 2 轮短训，一次推理同时出
#      MODEL 与 B 档（冻结自身当前帧），看 2s/3s 动态 Δ 是否由负转正
#
# 用法: setsid nohup bash tools/stage1/quick_hypotheses.sh > /dev/null 2>&1 &
D=/root/autodl-tmp/dataforpriorocc
Q=$D/quick_hyp
CFG_H1=projects/configs/priorocc/priorocc-r50-holefill-sub2k.py
CFG_H2=projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-fix-sub2k.py
EVAL1=$D/eval_occ_standalone.py
EVALB=$D/eval_b_baseline.py
REPORT=/root/priorocc/PriorOcc/tools/stage1/perclass_report.py
LIMIT=600
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc
mkdir -p $Q
log(){ echo "$(date '+%F %T') $*" | tee -a $Q/run.log; }
abort(){ log "FATAL: $*"; echo QUICK_HYP_FAILED >> $Q/run.log; exit 1; }
need(){ [ -f "$1" ] || abort "缺少 $1"; }
free_gpu(){
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
}

# ---------------- H1 ----------------
log "H1 baseline: ep21 @${LIMIT}"
python $EVAL1 ckpts/priorocc-singleframe-ep21.pth --cfg $CFG_H1 --limit $LIMIT \
  > $Q/h1_base_ep21.log 2>&1
log "H1-A(off) 训练开始"
python tools/train.py $CFG_H1 --work-dir $D/work_dirs/h1-off \
  --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 \
    model.enable_sem_continuity=False > $Q/h1_off_train.log 2>&1
need $D/work_dirs/h1-off/epoch_4_ema.pth
free_gpu
python $EVAL1 $D/work_dirs/h1-off/epoch_4_ema.pth --cfg $CFG_H1 --limit $LIMIT \
  > $Q/h1_off_ep4ema.log 2>&1
log "H1-A(off) 完成"
log "H1-B(on) 训练开始"
python tools/train.py $CFG_H1 --work-dir $D/work_dirs/h1-on \
  --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 > $Q/h1_on_train.log 2>&1
need $D/work_dirs/h1-on/epoch_4_ema.pth
free_gpu
python $EVAL1 $D/work_dirs/h1-on/epoch_4_ema.pth --cfg $CFG_H1 --limit $LIMIT \
  > $Q/h1_on_ep4ema.log 2>&1
log "H1-B(on) 完成"

# ---------------- H2 ----------------
log "H2 baseline: ep15_ema"
python $EVALB $D/work_dirs/smp-warmstart/epoch_15_ema.pth --cfg $CFG_H2 \
  --limit $LIMIT > $Q/h2_base_ep15ema.log 2>&1
log "H2 训练开始"
python tools/train.py $CFG_H2 --work-dir $D/work_dirs/h2-fixsub \
  --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=2 data.workers_per_gpu=4 > $Q/h2_train.log 2>&1
need $D/work_dirs/h2-fixsub/epoch_2_ema.pth
free_gpu
python $EVALB $D/work_dirs/h2-fixsub/epoch_2_ema.pth --cfg $CFG_H2 --limit $LIMIT \
  > $Q/h2_ep2ema.log 2>&1
log "H2 完成"

# ---------------- 汇总 ----------------
{
  echo "################ H1: 补洞 on/off（同子集 602 帧, EMA ep4）################"
  echo "---- 基线 ep21 ----"
  python $REPORT $Q/h1_base_ep21.log
  echo ""
  echo "---- A 臂 off vs 基线（负值=训练漂移）----"
  python $REPORT $Q/h1_off_ep4ema.log --baseline $Q/h1_base_ep21.log
  echo ""
  echo "---- B 臂 on vs 基线 ----"
  python $REPORT $Q/h1_on_ep4ema.log --baseline $Q/h1_base_ep21.log
  echo ""
  echo "---- 关键：B(on) vs A(off)，差值 = 补洞净效应 ----"
  python $REPORT $Q/h1_on_ep4ema.log --baseline $Q/h1_off_ep4ema.log
} > $Q/H1_summary.txt 2>&1

{
  echo "################ H2: 两 bug 修复 vs B档（同子集 602 帧, EMA）################"
  for tag in base:ep15_ema fix:ep2_ema_fix; do
    name=${tag%%:*}; f=$Q/h2_${name}_*.log
    echo "---- $name ----"
    for s in 0 1 2 3 4 5 6; do
      case $s in
        0) lbl=CURRENT;; 1) lbl="MODEL 1s";; 2) lbl="MODEL 2s";;
        3) lbl="MODEL 3s";; 4) lbl="B档   1s";; 5) lbl="B档   2s";;
        6) lbl="B档   3s";;
      esac
      v=$(python $REPORT $f --emit --section $s 2>/dev/null)
      echo "$lbl  mIoU/dyn/sta/cont6: $v"
    done
  done
} > $Q/H2_summary.txt 2>&1

log "全部完成: $Q/H1_summary.txt  $Q/H2_summary.txt"
echo QUICK_HYP_DONE >> $Q/run.log
