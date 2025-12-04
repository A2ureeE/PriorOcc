#!/usr/bin/env python3
"""
生成 2D 语义伪标签的脚本（基于 SegFormer）

说明：
- 使用 HuggingFace 的 SegFormer 预训练模型（Cityscapes）对 nuScenes 图像做语义分割
- 将 Cityscapes 类别映射到 nuScenes 17 类集合
- 输出为单通道 PNG（uint8），像素值为类别索引，255 表示 ignore

用法示例：
    python tools/generate_2d_seg_labels.py \
        --data-root data/nuscenes \
        --output-dir data/nuscenes/seg_2d_labels \
        --split trainval \
        --device cuda:0

中文注释已添加以便理解每一步。
"""
import os
import os.path as osp
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import cv2


# Cityscapes -> nuScenes 类别映射（近似映射）
# 源类别索引来自 Cityscapes (0-18)，目标为 nuScenes 0-16，255 忽略
CITYSCAPES_TO_NUSCENES = {
    0: 11,    # road -> driveable_surface
    1: 13,    # sidewalk -> sidewalk
    2: 15,    # building -> manmade
    3: 15,    # wall -> manmade
    4: 1,     # fence -> barrier
    5: 15,    # pole -> manmade
    6: 8,     # traffic light -> traffic_cone (近似)
    7: 8,     # traffic sign -> traffic_cone (近似)
    8: 16,    # vegetation -> vegetation
    9: 14,    # terrain -> terrain
    10: 255,  # sky -> ignore
    11: 7,    # person -> pedestrian
    12: 7,    # rider -> pedestrian
    13: 4,    # car -> car
    14: 10,   # truck -> truck
    15: 3,    # bus -> bus
    16: 9,    # train -> trailer (近似)
    17: 6,    # motorcycle -> motorcycle
    18: 2,    # bicycle -> bicycle
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 2D semantic pseudo labels')
    parser.add_argument('--data-root', type=str, default='data/nuscenes',
                        help='nuScenes data root')
    parser.add_argument('--output-dir', type=str, default='data/nuscenes/seg_2d_labels',
                        help='Output directory for 2D labels')
    parser.add_argument('--split', type=str, default='trainval',
                        choices=['train', 'val', 'trainval'],
                        help='Data split to process')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (cuda:0 or cpu)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip already generated labels')
    return parser.parse_args()


def build_segformer(device='cuda:0'):
    """尝试构建 SegFormer 模型；若依赖缺失则抛出错误。

    返回 (model, processor)
    """
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except Exception as e:
        raise ImportError('缺少 transformers 库，请先执行: pip install transformers') from e

    model_name = 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024'
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    import torch
    model.to(torch.device(device))
    model.eval()
    return model, processor


def inference_segformer(model, processor, image_path, device='cuda:0'):
    """对单张图像进行分割推理，返回 Cityscapes 类别预测 (H, W)"""
    from PIL import Image
    import torch

    image = Image.open(image_path).convert('RGB')
    W, H = image.size

    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, num_classes, h, w)
    pred = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)

    # Resize 回原始尺寸
    pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    return pred


def map_cityscapes_to_nuscenes(pred):
    """将 Cityscapes 的预测映射到 nuScenes 类别集合"""
    mapped = np.full_like(pred, 255, dtype=np.uint8)
    for src, tgt in CITYSCAPES_TO_NUSCENES.items():
        mapped[pred == src] = tgt
    return mapped


def collect_image_list_from_info(data_root, split):
    """从 bevdetv2 info 文件中收集需要处理的图像路径"""
    splits = ['train', 'val'] if split == 'trainval' else [split]
    image_list = []
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for sp in splits:
        info_path = osp.join(data_root, f'bevdetv2-nuscenes_infos_{sp}.pkl')
        if not osp.exists(info_path):
            print(f'[Warning] info file not found: {info_path}')
            continue
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        infos = data.get('infos', data)
        for info in infos:
            for cam in cam_names:
                cam_info = info['cams'][cam]
                img_path = cam_info['data_path']
                image_list.append(img_path)

    return image_list


def collect_image_list_from_folder(data_root):
    """备选：从 samples 文件夹直接读取图片列表"""
    samples_dir = osp.join(data_root, 'samples')
    if not osp.exists(samples_dir):
        return []
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    image_list = []
    for cam in cam_names:
        cam_dir = osp.join(samples_dir, cam)
        if not osp.exists(cam_dir):
            continue
        for fname in os.listdir(cam_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(osp.join(cam_dir, fname))
    return image_list


def main():
    args = parse_args()

    print('开始 2D 语义伪标签生成')
    print('data root:', args.data_root)
    print('output dir:', args.output_dir)

    # 尝试构建模型，若失败提示用户安装依赖
    try:
        model, processor = build_segformer(args.device)
    except Exception as e:
        print('模型构建失败:', e)
        print('请先安装依赖: pip install transformers pillow torch')
        return

    # 收集图像列表
    image_list = collect_image_list_from_info(args.data_root, args.split)
    if len(image_list) == 0:
        print('未找到 info 文件或 info 中无图片，尝试从 samples 文件夹读取')
        image_list = collect_image_list_from_folder(args.data_root)

    if len(image_list) == 0:
        print('未找到任何图像，退出')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    for img_path in tqdm(image_list, desc='Processing'):
        try:
            # 规范化路径（若为相对 path）
            if not osp.isabs(img_path):
                # 一般 info 中存的是相对 data_root 的路径或绝对路径
                candidate = osp.join(args.data_root, img_path)
                if osp.exists(candidate):
                    img_path_full = candidate
                elif osp.exists(osp.join(args.data_root, 'samples', img_path)):
                    img_path_full = osp.join(args.data_root, 'samples', img_path)
                else:
                    img_path_full = img_path
            else:
                img_path_full = img_path

            # 构建输出路径
            # 例如: data/nuscenes/samples/CAM_FRONT/xxx.jpg -> output_dir/samples/CAM_FRONT/xxx.png
            if img_path_full.startswith(args.data_root):
                rel = osp.relpath(img_path_full, args.data_root)
            else:
                # 若是绝对路径且不在 data_root 下，直接使用文件名
                rel = osp.basename(img_path_full)

            out_path = osp.join(args.output_dir, rel)
            out_path = out_path.rsplit('.', 1)[0] + '.png'

            if args.skip_existing and osp.exists(out_path):
                skipped += 1
                continue

            os.makedirs(osp.dirname(out_path), exist_ok=True)

            pred = inference_segformer(model, processor, img_path_full, args.device)
            mapped = map_cityscapes_to_nuscenes(pred)
            cv2.imwrite(out_path, mapped)
            processed += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print('\n[Error] failed to process', img_path, e)
            continue

    print('Done. processed=%d skipped=%d failed=%d' % (processed, skipped, failed))


if __name__ == '__main__':
    main()
