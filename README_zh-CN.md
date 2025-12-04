# PriorOcc: 基于显式 2D 语义先验增强的 Occupancy 预测

[English](README.md) | [简体中文](README_zh-CN.md)

**PriorOcc** 是一个基于 [FlashOCC](https://github.com/Yzichen/FlashOCC) 的增强型 3D 占据预测框架。它引入了 **显式 2D 语义先验注入** 机制（"Scheme C"），利用密集的 2D 语义监督来指导 3D 体素表示的学习。

## 🚀 主要特性

- **SemanticInjector 模块**：一个轻量级的插件模块，用于将 2D 语义先验注入到 BEV 流程中。
  - **SegHead**：从图像特征生成 2D 语义 logits。
  - **FusionLayer**：将语义 logits 融合回图像特征中，以丰富其语义上下文。
- **2D 语义监督**：引入辅助的 2D 分割损失 (`loss_2d_seg`) 来显式监督主干网络。
- **伪标签生成**：提供工具使用预训练分割模型（如 SegFormer）从 nuScenes 图像生成 2D 语义伪标签。
- **无缝集成**：完全集成到 BEVDet/FlashOCC 训练流程中。

## 🛠️ 安装

环境设置请参考原始 [FlashOCC 安装指南](doc/install.md)。

**额外依赖：**
```bash
pip install transformers  # 用于伪标签生成
```

## 📊 数据准备

### 1. 生成 2D 语义标签
PriorOcc 需要 nuScenes 数据集的 2D 语义分割标签。我们提供了一个脚本，使用预训练的 SegFormer 模型生成这些标签。

```bash
# 生成标签（保存到 data/nuscenes/seg_2d_labels）
python tools/generate_2d_seg_labels.py
```

更多详情请参阅 [doc/generate_2d_seg_labels.md](doc/generate_2d_seg_labels.md)。

### 2. 更新数据信息
确保按照标准的 FlashOCC/BEVDet 流程生成数据集信息文件。

## 🏃‍♂️ 训练

使用 ResNet-50 主干网络训练 PriorOcc：

```bash
./tools/dist_train.sh projects/configs/flashocc/flashocc-r50.py 8
```

配置已更新为：
1. 通过 `LoadSemanticSeg2D` 加载生成的 2D 语义标签。
2. 初始化 `SemanticInjector`。
3. 计算并优化 `loss_2d_seg`。

## 🧪 验证

我们提供了一个集成测试脚本，用于在训练前验证流程和张量形状：

```bash
python tools/integration_test.py
```

## 🏗️ 模型架构

**SemanticInjector** 插入在图像主干/颈部之后，视图变换器之前。

1. **输入**：多视图图像特征 $F_{img}$。
2. **SegHead**：$S_{logits} = \text{SegHead}(F_{img})$。
3. **损失**：$L_{2d} = \text{CrossEntropy}(S_{logits}, Y_{gt})$。
4. **融合**：$F_{enhanced} = \text{Fusion}(F_{img}, S_{logits})$。
5. **输出**：传递给视图变换器的增强特征。

## 🙏 致谢

本项目基于 Yzichen 的杰出工作 [FlashOCC](https://github.com/Yzichen/FlashOCC) 构建。感谢作者对社区的贡献。

- **FlashOCC**: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin.
