#!/bin/bash
# 阶段 2 收尾：等 epoch_6_ema.pth → 全量 4818 三口径评测 → 逐 horizon 汇总 → 结论落盘
# （主流水线 auto_pipeline.sh 在“停 ep8”时已 FATAL 退出，故由本脚本接管收尾）
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
log(){ echo "$(date '+%F %T') [finish] $*" | tee -a $LOG; }

log "等待阶段2 产出 epoch_6_ema.pth（最多 20h）"
for i in $(seq 1 2400); do
  [ -f "$W2/epoch_6_ema.pth" ] && break
  sleep 30
done
if [ ! -f "$W2/epoch_6_ema.pth" ]; then
  log "20h 内未见 epoch_6_ema.pth，停止（详见 $O/train_stage2.log）"
  exit 1
fi
for i in $(seq 1 90); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ -n "$m" ] && [ "$m" -lt 1200 ] && break
  sleep 4
done
log "阶段2 训练完成，全量 4818 三口径评测"
python $EVALB $W2/epoch_6_ema.pth --cfg $CFG --limit 0 > $O/eval_stage2_b.log 2>&1
for k in 1 2 3 4 5 6; do
  lab=$(case $k in 1) echo "MODEL 1s";; 2) echo "MODEL 2s";; 3) echo "MODEL 3s";;
        4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
  v=$(python $REPORT $O/eval_stage2_b.log --emit --section $k 2>/dev/null)
  log "$lab mIoU/dyn/sta/cont6 = $v"
done
{
  echo ""
  echo "==================== 最终结论 $(date '+%F %T') ===================="
  echo "EP21 基线(4818, 非EMA) 33.83 ; 补洞骨干 ep4_ema(4818) 33.94 (+0.11, 连续类 +0.19)"
  echo "阶段2: 冻结该感知骨干, 只训未来头, caps[12,24,36] + non_autoregressive"
  echo "阶段2 最终(EMA ep6) 全量 4818:"
  grep -hE "mIoU of 4818" $O/eval_stage2_b.log
  echo "(顺序: 当前帧 / 未来1s / 2s / 3s / B档1s / B档2s / B档3s)"
  echo "验收: 2s/3s 动态 > B档; 整体 >= 21.34/17.14/14.77; 当前帧 ~33.94(冻结不动)"
  echo "原始日志: $O/"
} | tee -a $LOG
log "==================== STAGE2 FINISH DONE ===================="
echo STAGE2_FINISH_DONE >> $O/run.done
