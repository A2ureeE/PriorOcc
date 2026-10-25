#!/bin/bash
# 无人值守收尾：等 C(smp ep15)评测完→解析→跑 flashocc 当前帧评测→解析→
# 写 RESULTS_ablation_extra.md→汇总所有论文 md 到 paper_materials/ + 索引。
set -u
D=/root/autodl-tmp/dataforpriorocc
R=/root/priorocc/PriorOcc
O=$D/auto
ER=$R/experiment_reports
CFG_SMP=projects/configs/priorocc/priorocc-4d-r50-smp-sdp-holefill-nodepth.py
CFG_FLASH=projects/configs/flashocc/flashocc-r50.py
REPORT=$R/tools/stage1/perclass_report.py
LOGC=$O/evb_ablation_smp_ep15_full.log
LOGF=$O/evb_flashocc_r50.log
export OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $R
st(){ echo "$(date '+%F %T') [finalize] $*"; }

free_gpu(){ for i in $(seq 1 240); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ -n "$m" ] && [ "$m" -lt 1200 ] && return 0; sleep 10; done; }

# 1) 等 C 评测完成（7 个 mIoU 块 或 无评测进程）
st "等 C(smp ep15) 评测完成"
for i in $(seq 1 240); do
  n=$(grep -acE "mIoU of 4818 samples" "$LOGC" 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 7 ] && break
  p=$(ps aux|grep -E "eval_b_baseline|eval_occ"|grep -v grep|wc -l)
  # 进程没了但块<7 说明可能异常，最多再等一轮
  [ "$p" -eq 0 ] && [ "${n:-0}" -lt 7 ] && { sleep 20; n=$(grep -acE "mIoU of 4818 samples" "$LOGC"||echo 0); [ "$n" -lt 7 ] && { st "C 评测异常退出(块=$n)，继续但标注"; break; }; }
  sleep 20
done
st "C 完成，解析三口径"
parse(){ # $1=log -> 打印 sec0..6
  for k in 0 1 2 3 4 5 6; do
    lab=$(case $k in 0) echo CURRENT;; 1) echo MODEL_1s;; 2) echo MODEL_2s;; 3) echo MODEL_3s;; 4) echo B_1s;; 5) echo B_2s;; 6) echo B_3s;; esac)
    v=$(python $REPORT "$1" --emit --section $k 2>/dev/null); echo "  $lab = ${v:-NA}"
  done
}

# 2) 跑 flashocc 当前帧评测
free_gpu
st "跑 flashocc-r50 当前帧评测 (eval_occ_standalone)"
python tools/eval/eval_occ_standalone.py \
  /root/autodl-tmp/dataforpriorocc/nuscenes/flashocc-r50-256x704.pth \
  --cfg $CFG_FLASH > "$LOGF" 2>&1 || st "flashocc 评测报错，见 $LOGF"
FREE_MI=$(grep -aoE "mIoU of [0-9]+ samples: [0-9.]+" "$LOGF" | tail -1)
st "flashocc 完成: ${FREE_MI:-NA}"

# 3) 写 RESULTS_ablation_extra.md
{
  echo "# 消融补充：架构对照(C) 与 FlashOcc 基线（全量4818·EMA）"
  echo ""
  echo "## C. with-SMP/自回归旧架构  ablation_smp-warmstart_ep15_ema"
  echo "（用其自身 SMP/AR 配置加载；三口径）"
  echo '```'
  parse "$LOGC"
  echo '```'
  echo ""
  echo "## FlashOcc-R50 基线（单帧当前帧占用，无未来预测）"
  echo "配置 \`$CFG_FLASH\`，权重 flashocc-r50-256x704.pth。"
  echo "- 当前帧 ${FREE_MI:-NA}（口径 Metric_mIoU/18类/use_image_mask，与本模型当前帧可比）"
} > $ER/RESULTS_ablation_extra.md
st "写 RESULTS_ablation_extra.md"

# 4) 汇总论文 md 到 paper_materials/
mkdir -p $R/paper_materials
for f in METHODOLOGY.md FINAL_MODEL_ep10.md DATA_INVENTORY.md ABLATION.md \
         RESULTS_perclass_ep10.md RESULTS_ablation_extra.md EVAL_METHOD.md \
         BASELINES_AND_SOTA.md CAM4DOCC_PREP.md; do
  [ -f "$ER/$f" ] && cp -a "$ER/$f" "$R/paper_materials/$f"
done
st "paper_materials 汇总完成"
st "ALLDONE"
