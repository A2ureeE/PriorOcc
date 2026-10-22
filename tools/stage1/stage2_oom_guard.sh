#!/bin/bash
# 阶段 2 的 OOM / 崩溃守护
#
# 背景：阶段 2（4D + 冻结感知 + 只训未来头）是唯一没在 GPU 上跑过的新配置，
# 若在交接点 OOM，主流水线会 FATAL 退出、后面 14h 训练不会启动。
# 本守护只做两件事：
#   1) 检测 train_stage2.log 里的 "out of memory"，且没有存活训练进程、
#      且 work_dir 没有 epoch_1.pth 时 → 用 samples_per_gpu=1 重新拉起；
#   2) 若主流水线已死、但阶段 2 训练被本守护救活并跑完，则补做
#      评测 + 结论落盘（与主流水线 Step4c/Step5 同口径）。
# 不做任何破坏性操作；正常路径（主流水线成功）下它会自行退出。
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
log(){ echo "$(date '+%F %T') [oom-guard] $*" | tee -a $LOG; }

alive_train(){ pgrep -f "train[.]py.*sep-frozen" > /dev/null 2>&1 && return 0 || return 1; }

# 1) 等阶段 2 日志出现（最多 10h）
for i in $(seq 1 1200); do
  [ -f "$O/train_stage2.log" ] && break
  sleep 30
done
[ -f "$O/train_stage2.log" ] || { log "10h 内未等到阶段2启动，守护退出"; exit 0; }
sleep 120

RESCUED=0
if grep -qiE "out of memory|CUDA out of memory" "$O/train_stage2.log" \
   && ! alive_train && [ ! -f "$W2/epoch_1.pth" ]; then
  log "检测到阶段2 OOM，改用 samples_per_gpu=1 重新拉起"
  SAMPLES_PER_GPU=1 bash tools/stage1/run_stage2_from.sh \
    "$(grep -oE '/[^ ]*\.pth' $O/train_stage2.log | head -1)" $W2 6 \
    >> $O/train_stage2_retry.log 2>&1
  RESCUED=1
fi

# 正常路径：主流水线在跑/已完成 → 退出
if [ "$RESCUED" -eq 0 ]; then
  log "阶段2 未 OOM（或已被主流水线完成），守护退出"
  exit 0
fi

# 2) 救援路径：等训练跑完，补评测与结论
for i in $(seq 1 2400); do
  [ -f "$W2/epoch_6_ema.pth" ] && break
  sleep 30
done
if [ ! -f "$W2/epoch_6_ema.pth" ]; then
  log "救援训练未产出 epoch_6_ema.pth，停止（详见 train_stage2_retry.log）"
  exit 1
fi
for i in $(seq 1 90); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ -n "$m" ] && [ "$m" -lt 1200 ] && break
  sleep 4
done
log "救援训练完成，评测全量 4818 三口径"
python $EVALB $W2/epoch_6_ema.pth --cfg $CFG --limit 0 > $O/eval_stage2_b.log 2>&1
for k in 1 2 3 4 5 6; do
  lab=$(case $k in 1) echo "MODEL 1s";; 2) echo "MODEL 2s";; 3) echo "MODEL 3s";;
        4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
  v=$(python $REPORT $O/eval_stage2_b.log --emit --section $k 2>/dev/null)
  log "RESCUE $lab mIoU/dyn/sta/cont6 = $v"
done
{
  echo ""
  echo "==================== 最终结论（OOM 救援路径）$(date '+%F %T') ===================="
  echo "阶段2 以 samples_per_gpu=1 重跑完成（原 batch2 OOM）"
  grep -hE "mIoU of 4818" $O/eval_stage2_b.log
  echo "（顺序: 当前帧 / 未来1s / 2s / 3s / B档1s / B档2s / B档3s）"
} >> $LOG
log "救援路径完成"
