# Language Self-Gating (LSG)

## 概述

Language Self-Gating (LSG) 是一种结合语义锚点与物理高度先验对 3D 体素特征进行软门控的模块。
目标是抑制语义/物理不一致的位置响应，使占据预测更加遵循物理与语义先验。

## 设计要点

- Projection: 对原始体素特征 (B,C,D,H,W) 使用 1x1x1 Conv3d 投影到低维空间。
- Semantic Anchors: 一组可以用文本嵌入（如 CLIP）替换的语义向量，用于与投影特征做相似度计算。
- Physical Bias: 针对高度切片 D 的可学习偏置，用于注入高度先验（例如地面更可能出现车辆/行人）。
- Gate: 通过相似度与物理偏置构造 gate logits，经过 Sigmoid 得到门控值，最后按位置抑制原始特征。

## 使用与集成

1. 将 `LanguageSelfGating` 插入到 BEV 流程中 C2H（channel-to-height）之后、OCC Head 之前。

```
Backbone -> Neck -> SemanticInjector -> ViewTransformer -> C2H -> [LanguageSelfGating] -> OCC Head
```

2. 初始化时可以选择使用 CLIP 文本嵌入替换 `anchors` 参数：
   - 离线计算文本嵌入并保存为 `anchors.pth`。
   - 在模型初始化时加载并用 `model.anchors.data.copy_(loaded)` 替换随机初始化。

## 调试

提供了 `tools/debug_language_self_gating.py` 用于在 CPU 上快速测试前向与反向传播。

```bash
python tools/debug_language_self_gating.py
```

## 后续改进方向

- 使用 CLIP 文本嵌入替换语义锚点，并对齐类别表。
- 在门控计算中引入空间上下文（轻量卷积或 Transformer 局部互信息）。
- 将 gate map 可视化为热力图，用于分析模块行为。

## 联系

如需我继续将该模块完整集成到 `BEVDetOCC` 的 forward 流程并添加配置，请回复“集成 LSG”，我会继续实施。