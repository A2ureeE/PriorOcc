#!/bin/bash
# 批量测试脚本：测试 epoch 14-25 的 mIoU 和 benchmark FPS
# 只保存关键结果

cd /root/learning/PriorOcc

# 创建输出目录
OUTPUT_DIR="work_dirs/test3/eval_results"
mkdir -p $OUTPUT_DIR

# 配置
CHECKPOINT_DIR="work_dirs/test3"
CONFIG="projects/configs/flashocc/flashocc-r50.py"

# 创建启用fp16的临时配置文件（用于benchmark）
CONFIG_FP16="work_dirs/test3/flashocc-r50-fp16.py"
cat > $CONFIG_FP16 << 'EOF'
_base_ = ['../../../projects/configs/flashocc/flashocc-r50.py']
fp16 = dict(loss_scale=512.)
EOF

echo "=============================================="
echo "批量测试脚本启动"
echo "测试范围: Epoch 14 - 25"
echo "输出目录: ${OUTPUT_DIR}"
echo "=============================================="

# 遍历 epoch 14 到 25 的 checkpoint
for epoch in $(seq 14 25); do
    CHECKPOINT="${CHECKPOINT_DIR}/epoch_${epoch}_ema.pth"
    
    # 检查checkpoint是否存在
    if [ -f "$CHECKPOINT" ]; then
        OUTPUT_FILE="${OUTPUT_DIR}/epoch_${epoch}_results.txt"
        
        echo "Testing Epoch ${epoch}..."
        
        # 创建输出文件头
        {
            echo "Epoch ${epoch} 测试结果"
            echo "时间: $(date)"
            echo ""
            echo "========== mIoU =========="
        } > $OUTPUT_FILE
        
        # 运行mIoU测试，只提取关键结果
        PYTHONPATH=. python tools/test.py \
            $CONFIG \
            $CHECKPOINT \
            --eval mIoU \
            2>&1 | grep -E "(per class IoU|IoU =|mIoU of|'mIoU')" >> $OUTPUT_FILE
        
        {
            echo ""
            echo "========== Benchmark (FP16) =========="
        } >> $OUTPUT_FILE
        
        # 运行benchmark测试，只提取FPS结果
        PYTHONPATH=. python tools/analysis_tools/benchmark.py \
            $CONFIG_FP16 \
            $CHECKPOINT \
            --samples 200 \
            --log-interval 50 \
            --fuse-conv-bn \
            2>&1 | grep -E "(fps:|Overall)" >> $OUTPUT_FILE
        
        echo "Epoch ${epoch} 完成 -> ${OUTPUT_FILE}"
    fi
done

echo ""
echo "所有测试完成！结果保存在: ${OUTPUT_DIR}"
