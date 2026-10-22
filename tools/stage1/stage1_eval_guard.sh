#!/bin/bash
# 阶段 1 评测守护：
#   - 每轮边界暂停训练，对 epoch_N_ema.pth 做 300 帧跨场景分层快测
#     （固定子集、固定顺序 => 轮间可比），记录"当前帧 mIoU / 动态 / 静态 /
#      连续类(背景)均值"四个数，并跟踪 best；（逐轮是为了尽早看到补洞对
#      背景连续类的作用，一次快测约 3 min，占总时长 <5%）
#   - 到 ep12 训练结束后，对 best EMA 与 ep12 EMA 各做一次全量 4818 帧评测，
#     并与单帧 ep21 基线（同池 4818、非 EMA）做逐类对比，输出
#     stage1_vs_ep21_perclass.txt —— 直接回答"背景连续类是否增长"。
#
# 用法: setsid nohup bash tools/stage1/stage1_eval_guard.sh > /dev/null 2>&1 &
D=/root/autodl-tmp/dataforpriorocc
W=$D/work_dirs/stage1-holefill
CFG=projects/configs/priorocc/priorocc-r50-holefill-full.py
EVAL=$D/eval_occ_standalone.py
REPORT=/root/priorocc/PriorOcc/tools/stage1/perclass_report.py
RESUME=/root/priorocc/PriorOcc/tools/stage1/run_stage1_resume.sh
BASE=$D/ep21_same4818_ref.log
CL=$D/stage1_miou_curve.log
WL=$D/stage1_guard.log
FL=$D/stage1_full_eval.log
PL=$D/stage1_vs_ep21_perclass.txt
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=/root/priorocc/PriorOcc PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /root/priorocc/PriorOcc

log(){ echo "$(date '+%F %T') $*" >> $WL; }
wait_free(){
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
  log "WARN: 显存 4 分钟内未归零, 继续"
}

# ep21 基线（同池 4818、非 EMA）留档供逐类对比
[ -f "$BASE" ] || cp -f /root/priorocc/PriorOcc/experiment_reports/raw_eval/eval_ep21_same4818.log $BASE
[ -f "$CL" ] || echo "# epoch  mIoU dyn static cont6   (EMA, 300-frame fixed subset)" > $CL
log "stage1_eval_guard 启动 (逐轮快测 + 末轮全量 + 背景类对比)"
BEST=-1; BEST_EP=0

NEXT=1
while [ $NEXT -le 12 ]; do
  sleep 120
  [ -f "$W/epoch_$NEXT.pth" ] || continue
  sleep 25
  log "epoch_$NEXT 边界: 暂停训练"
  pkill -9 -f "stage1-holefill --seed"
  wait_free
  RAW=$D/stage1_eval_ep${NEXT}.log
  python $EVAL "$W/epoch_${NEXT}_ema.pth" --cfg $CFG --limit 300 > $RAW 2>&1
  v=$(python $REPORT $RAW --emit 2>/dev/null)
  cur=$(echo "$v" | awk '{print $1}')
  if [ -z "$cur" ]; then
    log "ERROR ep$NEXT 快测无输出 (见 $RAW), 直接续训"
  else
    echo "ep$NEXT $v" >> $CL
    log "quick-eval ep$NEXT -> $v"
    if awk -v a="$cur" -v b="$BEST" 'BEGIN{exit !(a>b)}'; then
      BEST=$cur; BEST_EP=$NEXT
      cp -f "$W/epoch_${NEXT}_ema.pth" "$W/best_ema.pth"
      log "新 best: ep$NEXT ($cur) -> best_ema.pth"
    fi
  fi
  if [ "$NEXT" -ge 12 ]; then
    log "已到 ep12: 训练结束, 开始全量评测"
    break
  fi
  setsid nohup bash $RESUME >> $D/train_stage1.log 2>&1 < /dev/null &
  log "已续训, PREV=$NEXT"
  NEXT=$((NEXT + 1))
done

# ---- 全量 4818 帧裁决 + 背景类对比 ----
[ -f "$W/best_ema.pth" ] || cp -f "$W/epoch_12_ema.pth" "$W/best_ema.pth"
log "全量评测 best_ema (ep$BEST_EP)"
wait_free
echo "==== best_ema (from ep$BEST_EP) full 4818 ====" >> $FL
python $EVAL "$W/best_ema.pth" --cfg $CFG --limit 0 > $D/stage1_full_best_ema.log 2>&1
python $REPORT $D/stage1_full_best_ema.log --baseline $BASE >> $PL 2>&1
if [ "$BEST_EP" != "12" ]; then
  log "全量评测 ep12_ema 对照"
  echo "==== epoch_12_ema full 4818 ====" >> $FL
  python $EVAL "$W/epoch_12_ema.pth" --cfg $CFG --limit 0 > $D/stage1_full_ep12_ema.log 2>&1
  echo "" >> $PL
  python $REPORT $D/stage1_full_ep12_ema.log --baseline $BASE >> $PL 2>&1
fi
grep -hE "IoU = |mIoU of " $D/stage1_full_best_ema.log >> $FL
log "stage1_eval_guard 完成 (对比报告: $PL)"
echo "STAGE1_GUARD_DONE" >> $WL
