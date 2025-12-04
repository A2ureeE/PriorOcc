#!/usr/bin/env python3
"""
调试脚本：用于验证伪标签生成脚本的关键逻辑（不一定需要 GPU 或 transformers）

功能：
- 若环境支持 transformers，会用少量真实图像（或示例）做一次推理流程（不保存结果）
- 若 transformers 不可用，则模拟模型输出并测试后处理（映射、保存路径创建等）

目的：快速发现路径拼接、映射、保存等逻辑的 bug
"""
import os
import os.path as osp
import sys
import tempfile
import numpy as np
import cv2


def simulate_mapping_test():
    """模拟一个小的预测并测试映射与保存逻辑"""
    print('运行模拟测试：映射与保存逻辑')
    # 生成随机 cityscapes 类别图（小尺寸）
    h, w = 64, 128
    rng = np.random.RandomState(0)
    pred = rng.randint(0, 19, size=(h, w), dtype=np.uint8)

    # 使用与生成脚本相同的映射表（重复实现以避免依赖）
    CITYSCAPES_TO_NUSCENES = {
        0: 11, 1: 13, 2: 15, 3: 15, 4: 1, 5: 15, 6: 8, 7: 8, 8: 16, 9: 14,
        10: 255, 11: 7, 12: 7, 13: 4, 14: 10, 15: 3, 16: 9, 17: 6, 18: 2
    }

    mapped = np.full_like(pred, 255, dtype=np.uint8)
    for src, tgt in CITYSCAPES_TO_NUSCENES.items():
        mapped[pred == src] = tgt

    # 保存到临时目录
    tmpdir = tempfile.mkdtemp(prefix='debug_seg_')
    out_path = osp.join(tmpdir, 'samples', 'CAM_FRONT')
    os.makedirs(out_path, exist_ok=True)
    save_path = osp.join(out_path, 'debug.png')
    cv2.imwrite(save_path, mapped)

    print('Saved debug label to', save_path)
    print('Unique labels in mapped:', np.unique(mapped))


def run_with_transformers_example():
    """如果 transformers 可用，尝试构建模型并对示例图片做单次前向（不保存）"""
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except Exception as e:
        print('transformers not available:', e)
        return False

    print('transformers available — 测试加载小模型')
    model_name = 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024'
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    except Exception as e:
        print('下载或加载模型失败:', e)
        return False

    # 生成随机图像作为输入
    from PIL import Image
    import numpy as np
    img = (np.random.rand(256, 512, 3) * 255).astype('uint8')
    img_pil = Image.fromarray(img)

    inputs = processor(images=img_pil, return_tensors='pt')
    with __import__('torch').no_grad():
        outputs = model(**{k: v for k, v in inputs.items()})
    logits = outputs.logits
    pred = logits.argmax(dim=1).cpu().numpy()[0]
    print('Transformers 推理形状:', pred.shape)
    return True


def main():
    print('debug_generate_2d_seg_labels: 开始')

    ok = run_with_transformers_example()
    if not ok:
        print('Transformer 测试不可用，使用模拟流程进行测试')
        simulate_mapping_test()
    else:
        print('Transformer 测试成功，随后仍执行映射/保存逻辑测试')
        simulate_mapping_test()

    print('debug_generate_2d_seg_labels: 结束')


if __name__ == '__main__':
    main()
