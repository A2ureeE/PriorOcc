#!/bin/bash
# PriorOcc-4D 全自动流水线（补洞 A/B → 分支训练 → 结论）
#
# 决策链（任务书）：
#   Step1  A/B 各 1 轮（ep21 热启, lr=1e-5, clip 5, warmup 200, 全量帧）
#          A: 补洞 ON (cont_2d/cont_bev=0.05, cont_occ=0)  B: 补洞 OFF
#   Step2  全量 4818 当前帧评测 A/B（非 EMA 为主判据，EMA 作佐证）
#   Step3  判定：A.mIoU ≥ B.mIoU 且 A 小类(ped/bike/moto)均值 ≥ B → 有用
#   Step4  分支一：A 配置续到 8 轮 → SF8 → 冻结感知训未来头 6 轮
#          分支二：跳过补洞，直接 ep21 冻结感知训未来头 6 轮
#   Step5  评测 + 结论落 experiment_reports/auto_run.log
#
# 用法: setsid nohup bash tools/stage1/auto_pipeline.sh > /dev/null 2>&1 &
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
LOG=$R/experiment_reports/auto_run.log
A_CFG=projects/configs/priorocc/priorocc-r50-holefill-ab-A.py
B_CFG=projects/configs/priorocc/priorocc-r50-holefill-ab-B.py
S2_CFG=projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py
EP21=$R/ckpts/priorocc-singleframe-ep21.pth
EVAL1=$D/eval_occ_standalone.py
EVALB=$D/eval_b_baseline.py
JUDGE=$R/tools/stage1/ab_judge.py
REPORT=$R/tools/stage1/perclass_report.py
WA=$D/work_dirs/ab-on
WB=$D/work_dirs/ab-off
W2=$D/work_dirs/stage2-frozen
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
mkdir -p $O

log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
die(){ log "FATAL: $*"; log "=== 流水线安全停止（未启动任何长训练）==="; exit 1; }
free_gpu(){
  for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0
    sleep 4
  done
  log "WARN: 显存 6 分钟未归零，继续"
}
need(){ [ -f "$1" ] || die "缺少产物 $1 （见 $2）"; }

# ---------------- Step 0 定位 ----------------
log "==================== AUTO PIPELINE START ===================="
log "[Step0] EP21=$EP21"
[ -f "$EP21" ] || die "EP21 不存在: $EP21"
[ -f projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-nodepth.py ] \
  || die "基础配置缺失"
[ -f "$A_CFG" ] && [ -f "$B_CFG" ] && [ -f "$S2_CFG" ] || die "A/B/S2 配置缺失"
log "[Step0] 基础配置 OK; A=$A_CFG B=$B_CFG S2=$S2_CFG"
log "[Step0] 预算: A/B 各 1 轮约 53min 训练 + 14min 评测；分支一另加 7 轮(6.2h)"
log "[Step0] 4D 阶段2: 6 轮 × ~2.2h + 3 次 runner 内全量评测"

# ---------------- Step 1 A 臂 ----------------
log "[Step1-A] 训练开始（1 轮, lr 1e-5, 补洞 ON, cont_occ=0）"
python tools/train.py $A_CFG --work-dir $WA --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 \
    runner.max_epochs=1 > $O/train_A.log 2>&1
need $WA/epoch_1.pth $O/train_A.log
need $WA/epoch_1_ema.pth $O/train_A.log
free_gpu
log "[Step1-A] 训练完成，评测全量 4818（非 EMA + EMA）"
python $EVAL1 $WA/epoch_1.pth --cfg $A_CFG --limit 0 > $O/eval_A_ep1.log 2>&1
python $EVAL1 $WA/epoch_1_ema.pth --cfg $A_CFG --limit 0 > $O/eval_A_ep1ema.log 2>&1
log "[Step1-A] $(grep -hE 'mIoU of 4818' $O/eval_A_ep1.log | tail -1) (非EMA)"
log "[Step1-A] $(grep -hE 'mIoU of 4818' $O/eval_A_ep1ema.log | tail -1) (EMA)"

# ---------------- Step 1 B 臂 ----------------
free_gpu
log "[Step1-B] 训练开始（1 轮, lr 1e-5, 补洞 OFF）"
python tools/train.py $B_CFG --work-dir $WB --seed 0 --gpu-id 0 \
  --cfg-options device=cuda cudnn_benchmark=True \
    data.samples_per_gpu=4 data.workers_per_gpu=4 \
    runner.max_epochs=1 > $O/train_B.log 2>&1
need $WB/epoch_1.pth $O/train_B.log
need $WB/epoch_1_ema.pth $O/train_B.log
free_gpu
log "[Step1-B] 训练完成，评测全量 4818（非 EMA + EMA）"
python $EVAL1 $WB/epoch_1.pth --cfg $B_CFG --limit 0 > $O/eval_B_ep1.log 2>&1
python $EVAL1 $WB/epoch_1_ema.pth --cfg $B_CFG --limit 0 > $O/eval_B_ep1ema.log 2>&1
log "[Step1-B] $(grep -hE 'mIoU of 4818' $O/eval_B_ep1.log | tail -1) (非EMA)"
log "[Step1-B] $(grep -hE 'mIoU of 4818' $O/eval_B_ep1ema.log | tail -1) (EMA)"

# ---------------- Step 2/3 判定 ----------------
log "[Step2] 判定（判据: A.mIoU>=B.mIoU 且 A 小类均值>=B，容差 0.05）"
python $JUDGE --a $O/eval_A_ep1.log --b $O/eval_B_ep1.log \
  --a-ema $O/eval_A_ep1ema.log --b-ema $O/eval_B_ep1ema.log \
  > $O/judge.log 2>&1
JV=$?
cat $O/judge.log | tee -a $LOG
if [ $JV -eq 3 ]; then die "判定解析失败（$O/judge.log）"; fi
if [ $JV -eq 0 ]; then
  BRANCH=holefill_useful
else
  BRANCH=holefill_useless
fi
log "[Step3] BRANCH=$BRANCH"
# 逐类 diff 落日志（小类重点）
python $REPORT $O/eval_A_ep1.log --baseline $O/eval_B_ep1.log | tee -a $LOG

# ---------------- Step 4 分支 ----------------
if [ "$BRANCH" = "holefill_useful" ]; then
  log "[Step4-1a] 补洞有用 → A 配置续到 8 轮产出 SF8（从 ep1 续，不重跑第 1 轮）"
  python tools/train.py $A_CFG --work-dir $WA --seed 0 --gpu-id 0 \
    --auto-resume --cfg-options device=cuda cudnn_benchmark=True \
      data.samples_per_gpu=4 data.workers_per_gpu=4 \
      runner.max_epochs=8 > $O/train_A8.log 2>&1
  need $WA/epoch_8_ema.pth $O/train_A8.log
  SF8=$WA/epoch_8_ema.pth
  free_gpu
  log "[Step4-1b] SF8=$SF8，评测全量 4818"
  python $EVAL1 $SF8 --cfg $A_CFG --limit 0 > $O/eval_SF8.log 2>&1
  log "[Step4-1b] SF8 $(grep -hE 'mIoU of 4818' $O/eval_SF8.log | tail -1)"
  python $REPORT $O/eval_SF8.log >> $LOG 2>&1
  S2_CKPT=$SF8
else
  log "[Step4-2] 补洞无用 → 跳过补洞训练，直接用 EP21 冻结感知训未来头"
  S2_CKPT=$EP21
fi

# ---------------- Step 4c 阶段2：冻结感知 + 未来头 ----------------
free_gpu
log "[Step4c] 阶段2 开始: load_from=$S2_CKPT, 6 轮, caps[12,24,36]+non-AR"
bash tools/stage1/run_stage2_from.sh "$S2_CKPT" $W2 6 > $O/train_stage2.log 2>&1
need $W2/epoch_6_ema.pth $O/train_stage2.log
free_gpu
log "[Step4c] 阶段2 完成，全量 4818 三口径评测（MODEL / B档 / 当前帧）"
python $EVALB $W2/epoch_6_ema.pth --cfg $S2_CFG --limit 0 > $O/eval_stage2_b.log 2>&1
log "[Step4c] 当前帧 $(grep -hE 'mIoU of 4818' $O/eval_stage2_b.log | head -1)"
for k in 1 2 3 4 5 6; do
  lab=$(case $k in 1) echo "MODEL 1s";; 2) echo "MODEL 2s";; 3) echo "MODEL 3s";;
        4) echo "B档   1s";; 5) echo "B档   2s";; 6) echo "B档   3s";; esac)
  v=$(python $REPORT $O/eval_stage2_b.log --emit --section $k 2>/dev/null)
  log "[Step4c] $lab mIoU/dyn/sta/cont6 = $v"
done

# ---------------- Step 5 结论 ----------------
{
  echo ""
  echo "==================== 最终结论 $(date '+%F %T') ===================="
  echo "EP21 基线（4818 同池, 非 EMA）: 33.83"
  echo "A 臂(补洞ON) ep1: $(grep -hE 'mIoU of 4818' $O/eval_A_ep1.log | tail -1)"
  echo "B 臂(补洞OFF) ep1: $(grep -hE 'mIoU of 4818' $O/eval_B_ep1.log | tail -1)"
  grep -hE "delta\(A-B\)|VERDICT|REASON" $O/judge.log
  echo "判定分支: $BRANCH"
  echo "阶段2 骨干: $S2_CKPT"
  echo "阶段2 最终（EMA ep6）四 horizon:"
  grep -hE "mIoU of 4818" $O/eval_stage2_b.log
  echo "（顺序: 当前帧 / 未来1s / 2s / 3s / B档1s / B档2s / B档3s）"
  echo "原始日志: $O/"
} | tee -a $LOG
log "==================== AUTO PIPELINE DONE ===================="
echo AUTO_PIPELINE_DONE >> $O/run.done
