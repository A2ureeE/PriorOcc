#!/bin/bash
# 批量测试脚本：测试14*25=350轮的mIoU和benchmark帧数
# 每轮测试结果保存到单独的txt文件中
# 使用fp16模式

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
echo "测试范围: 14*25=350轮"
echo "输出目录: ${OUTPUT_DIR}"
echo "=============================================="

# 遍历所有epoch checkpoint (1-350轮, 对应14*25)
for epoch in $(seq 1 350); do
    CHECKPOINT="${CHECKPOINT_DIR}/epoch_${epoch}_ema.pth"
    
    # 检查checkpoint是否存在
    if [ -f "$CHECKPOINT" ]; then
        OUTPUT_FILE="${OUTPUT_DIR}/epoch_${epoch}_results.txt"
        
        echo "=============================================="
        echo "Testing Epoch ${epoch}..."
        echo "Checkpoint: ${CHECKPOINT}"
        echo "Output: ${OUTPUT_FILE}"
        echo "=============================================="
        
        # 创建输出文件头
        {
            echo "=============================================="
            echo "Epoch ${epoch} 测试结果"
            echo "=============================================="
            echo "开始时间: $(date)"
            echo "Checkpoint: ${CHECKPOINT}"
            echo "Config: ${CONFIG}"
            echo ""
            echo "========== mIoU 评估结果 =========="
        } > $OUTPUT_FILE
        
        # 运行测试：mIoU评估 (使用FP32保证精度)
        PYTHONPATH=. python tools/test.py \
            $CONFIG \
            $CHECKPOINT \
            --eval mIoU \
            2>&1 | tee -a $OUTPUT_FILE
        
        {
            echo ""
            echo "========== Benchmark 性能测试 =========="
        } >> $OUTPUT_FILE
        
        # 运行benchmark测试 (使用fp16配置)
        PYTHONPATH=. python tools/analysis_tools/benchmark.py \
            $CONFIG_FP16 \
            $CHECKPOINT \
            --samples 200 \
            --log-interval 50 \
            --fuse-conv-bn \
            2>&1 | tee -a $OUTPUT_FILE
        
        # 添加结束信息
        {
            echo ""
            echo "=============================================="
            echo "结束时间: $(date)"
            echo "=============================================="
        } >> $OUTPUT_FILE
        
        echo ""
        echo "Epoch ${epoch} 测试完成！结果已保存到: ${OUTPUT_FILE}"
        echo ""
    fi
done

echo "=============================================="
echo "所有测试完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "=============================================="

# 汇总所有mIoU结果
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
echo "生成结果汇总..."
{
    echo "=============================================="
    echo "mIoU 结果汇总"
    echo "生成时间: $(date)"
    echo "=============================================="
    echo ""
    
    for epoch in $(seq 1 350); do
        RESULT_FILE="${OUTPUT_DIR}/epoch_${epoch}_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            echo "=== Epoch ${epoch} ==="
            grep -E "(mIoU|fps:|Overall)" $RESULT_FILE
            echo ""
        fi
    done
} > $SUMMARY_FILE

echo "汇总结果已保存到: ${SUMMARY_FILE}"
