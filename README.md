# PriorOcc: Enhancing Occupancy Prediction with Explicit 2D Semantic Priors

[English](README.md) | [简体中文](README_zh-CN.md)

**PriorOcc** is an enhanced 3D occupancy prediction framework based on [FlashOCC](https://github.com/Yzichen/FlashOCC). It introduces an **Explicit 2D Semantic Prior Injection** mechanism ("Scheme C") to guide the learning of 3D voxel representations using dense 2D semantic supervision.

## 🚀 Key Features

- **SemanticInjector Module**: A lightweight plugin module that injects 2D semantic priors into the BEV pipeline.
  - **SegHead**: Generates 2D semantic logits from image features.
  - **FusionLayer**: Fuses the semantic logits back into the image features to enrich them with semantic context.
- **2D Semantic Supervision**: Incorporates an auxiliary 2D segmentation loss (`loss_2d_seg`) to explicitly supervise the backbone.
- **Pseudo-Label Generation**: Provides tools to generate 2D semantic pseudo-labels from nuScenes images using pretrained segmentation models (e.g., SegFormer).
- **Seamless Integration**: Fully integrated into the BEVDet/FlashOCC training pipeline.

## 🛠️ Installation

Please refer to the original [FlashOCC Installation Guide](doc/install.md) for environment setup.

**Additional Requirements:**
```bash
pip install transformers  # For pseudo-label generation
```

## 📊 Data Preparation

### 1. Generate 2D Semantic Labels
PriorOcc requires 2D semantic segmentation labels for the nuScenes dataset. We provide a script to generate these using a pretrained SegFormer model.

```bash
# Generate labels (saved to data/nuscenes/seg_2d_labels)
python tools/generate_2d_seg_labels.py
```

For more details, see [doc/generate_2d_seg_labels.md](doc/generate_2d_seg_labels.md).

### 2. Update Data Info
Ensure your dataset info files are generated as per standard FlashOCC/BEVDet procedures.

## 🏃‍♂️ Training

To train PriorOcc with the ResNet-50 backbone:

```bash
./tools/dist_train.sh projects/configs/flashocc/flashocc-r50.py 8
```

The configuration has been updated to:
1. Load the generated 2D semantic labels via `LoadSemanticSeg2D`.
2. Initialize the `SemanticInjector`.
3. Compute and optimize `loss_2d_seg`.

## 🧪 Verification

We provide an integration test script to verify the pipeline and tensor shapes before training:

```bash
python tools/integration_test.py
```

## 🏗️ Model Architecture

The **SemanticInjector** is inserted after the image backbone/neck and before the view transformer.

1. **Input**: Multi-view image features $F_{img}$.
2. **SegHead**: $S_{logits} = \text{SegHead}(F_{img})$.
3. **Loss**: $L_{2d} = \text{CrossEntropy}(S_{logits}, Y_{gt})$.
4. **Fusion**: $F_{enhanced} = \text{Fusion}(F_{img}, S_{logits})$.
5. **Output**: Enhanced features passed to the View Transformer.

## 🙏 Acknowledgements

This project is built upon the excellent work of [FlashOCC](https://github.com/Yzichen/FlashOCC) by Yzichen. We thank the authors for their contribution to the community.

- **FlashOCC**: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin.
