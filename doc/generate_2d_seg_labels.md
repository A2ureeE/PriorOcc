# 生成 2D 语义伪标签（SegFormer） 使用说明

本文档说明如何使用 `tools/generate_2d_seg_labels.py` 为 nuScenes 数据生成 2D 语义伪标签，并介绍输出格式与注意事项。

## 环境准备

- 推荐 Python 3.8+
- 安装必要依赖：

```bash
pip install transformers pillow torch torchvision tqdm opencv-python
```

（若使用 GPU，请确保已安装对应的 CUDA 版本的 `torch`）

## 脚本位置

`tools/generate_2d_seg_labels.py`

## 基本用法

```bash
python tools/generate_2d_seg_labels.py \
    --data-root data/nuscenes \
    --output-dir data/nuscenes/seg_2d_labels \
    --split trainval \
    --device cuda:0
```

参数说明：

- `--data-root`：nuScenes 数据根目录（包含 `samples/` 和 info 文件）
- `--output-dir`：输出伪标签目录（脚本会在此目录下按原结构写入 PNG）
- `--split`：`train` / `val` / `trainval`（默认为 `trainval`）
- `--device`：模型运行设备，例如 `cuda:0` 或 `cpu`

## 输出格式

- 输出为单通道 PNG（uint8），像素值为类别索引（0-16），255 表示忽略像素。
- 输出路径保持相对结构，例如：

```
data/nuscenes/seg_2d_labels/samples/CAM_FRONT/n008-2018-xxx.png
```

## 类别映射

脚本使用 Cityscapes 预训练模型（SegFormer），并将 Cityscapes 的类别映射到 nuScenes 17 类集合（为近似映射，见脚本中 `CITYSCAPES_TO_NUSCENES`）。

## 调试与常见问题

- 若遇到 `ImportError`，请先安装 `transformers`、`pillow`、`torch`。
- 若图像路径未找到，请确认 `bevdetv2-nuscenes_infos_{split}.pkl` 存在或 `data_root/samples/` 的目录结构是完整的。

## 示例：只处理前 100 张图（手动修改脚本或使用工具筛选）

可在脚本中加入计数逻辑，或先用 `tools/debug_generate_2d_seg_labels.py` 进行小规模测试。

---

# 将 2D 伪标签接入训练 Pipeline

本节说明如何将生成的 2D 语义伪标签接入 FlashOCC 训练流程，使 `SemanticInjector` 模块能够使用这些标签进行监督学习。

## 整体流程

```
生成伪标签 -> LoadSemanticSeg2D 加载 -> SemanticInjector 使用 -> loss_2d_seg 计算损失
```

## LoadSemanticSeg2D 模块

### 文件位置

`projects/mmdet3d_plugin/datasets/pipelines/loading_seg2d.py`

### 功能说明

- 从预生成的 PNG 文件中读取每个相机视角的 2D 语义分割标签
- 自动根据图像路径构建对应的伪标签路径
- 支持与图像相同的几何变换（resize, crop, flip）
- 输出 `gt_semantic_2d`，shape 为 `(N_views, H, W)`

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seg_prefix` | str | `'data/nuscenes/seg_2d_labels'` | 伪标签文件根目录 |
| `num_classes` | int | 17 | 语义类别数量 |
| `ignore_index` | int | 255 | 忽略像素的索引值 |
| `to_float32` | bool | False | 是否转换为 float32 |

## 配置文件修改

在 `projects/configs/flashocc/flashocc-r50.py` 中添加 `LoadSemanticSeg2D` 到训练 pipeline：

```python
train_pipeline = [
    dict(type='PrepareImageInputs', ...),
    dict(type='LoadAnnotationsBEVDepth', ...),
    # 添加 2D 语义标签加载（在 PrepareImageInputs 之后）
    dict(type='LoadSemanticSeg2D',
         seg_prefix='data/nuscenes/seg_2d_labels',
         num_classes=17,
         ignore_index=255),
    # ... 其他 pipeline 步骤
]
```

**注意**：`LoadSemanticSeg2D` 应该放在 `PrepareImageInputs` 之后，以便获取图像变换参数。

## 数据流说明

1. **加载阶段**：`LoadSemanticSeg2D` 读取伪标签并存入 `results['gt_semantic_2d']`

2. **收集阶段**：在 `Collect3D` 中添加 `'gt_semantic_2d'` 到 keys 列表：
   ```python
   dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_semantic_2d'])
   ```

3. **训练阶段**：`BEVDetOCC.forward_train()` 接收 `gt_semantic_2d` 并传给 `loss_2d_seg()` 计算损失

## 路径映射规则

| 原始图像路径 | 伪标签路径 |
|-------------|-----------|
| `data/nuscenes/samples/CAM_FRONT/xxx.jpg` | `seg_prefix/samples/CAM_FRONT/xxx.png` |
| `/abs/path/samples/CAM_BACK/yyy.jpg` | `seg_prefix/samples/CAM_BACK/yyy.png` |

## 调试验证

可使用以下脚本验证 `LoadSemanticSeg2D` 是否正确加载：

```python
# tools/debug_loading_seg2d.py
from projects.mmdet3d_plugin.datasets.pipelines import LoadSemanticSeg2D

loader = LoadSemanticSeg2D(seg_prefix='data/nuscenes/seg_2d_labels')

# 模拟 results 字典
results = {
    'cams': {
        'CAM_FRONT': {'data_path': 'data/nuscenes/samples/CAM_FRONT/xxx.jpg'},
        'CAM_FRONT_RIGHT': {'data_path': 'data/nuscenes/samples/CAM_FRONT_RIGHT/xxx.jpg'},
        # ... 其他相机
    }
}

results = loader(results)
print('gt_semantic_2d shape:', results['gt_semantic_2d'].shape)
print('Unique labels:', np.unique(results['gt_semantic_2d']))
```

## 常见问题

### Q: 伪标签文件不存在怎么办？
A: `LoadSemanticSeg2D` 会自动用 `ignore_index` (255) 填充，不会报错。

### Q: 如何确保标签与图像对齐？
A: 确保伪标签生成时使用相同的文件名结构，且 `LoadSemanticSeg2D` 放在 `PrepareImageInputs` 之后以获取变换参数。

### Q: 如何处理不同分辨率的标签？
A: `LoadSemanticSeg2D` 会自动根据 crop 参数进行裁剪，使用最近邻插值保持类别边界清晰。

---

如需我帮你把生成的伪标签接入训练 pipeline（实现 `LoadSemanticSeg2D`），我可以继续实现并更新配置。 
