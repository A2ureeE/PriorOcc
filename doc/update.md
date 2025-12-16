Project Specification: FlashOcc 架构改进方案 (Scheme C)1. 项目背景与目标基础模型: FlashOcc (基于 BEVDet/BEVFormer 的高速 3D Occupancy 预测网络)。当前痛点:原始 FlashOcc 依赖 Channel-to-Height (C2H) 模块从 2D BEV 特征恢复 3D 结构，存在语义与几何特征纠缠的问题。在长尾物体（小物体）和纹理相似类别（如路面 vs 人行道）上 mIoU 较低。存在“深度鬼影”导致的 False Positives。改进目标:采用 方案 C：显式 2D 语义先验注入 (Explicit 2D Semantic Prior Injection)。通过引入一个轻量级的 2D 语义分割分支，将明确的语义信息作为先验（Prior）注入到 BEV 特征生成之前的阶段，从而辅助 C2H 模块更精准地进行 3D 预测。约束条件:保持 Real-time 推理速度（ROS 落地导向）。不使用 Open-Vocabulary (OpenOcc) 方案，仅做 Closed-Set (e.g., nuScenes 17 classes)。推理阶段不得包含 CLIP 大模型，仅使用轻量级 CNN Head。模型结构需支持 TensorRT 导出。2. 系统架构设计2.1 整体数据流系统将采用 Shared Backbone + Multi-Head 架构：Image Backbone: 输入环视图像，输出多尺度特征 $F_{img}$。Branch A (New - 2D Semantic Prior):输入: $F_{img}$操作: 轻量级卷积头 (SegHead)输出: 2D 语义 Logits $S_{seg}$ (Channel数 = 类别数)Feature Fusion (Core):操作: Concat(F_{img}, S_{seg}) -> 1x1 Conv -> F_{fused}目的: 将 $S_{seg}$ 作为强语义通道嵌入特征。Branch B (Original - 3D Occupancy):输入: $F_{fused}$操作: LSS / View Transformer -> BEV Encoder -> C2H Head输出: 3D Occupancy GridBranch C (Optional/Future - Traffic Light):输入: $F_{img}$ 或 $S_{seg}$操作: 2D Classification Head输出: 红绿灯状态 (End-to-End 逻辑输入)2.2 关键模块细节A. 语义注入模块 (SemanticInjector)我们需要实现一个 nn.Module，包含以下步骤：SegHead: 简单的 FCN 结构（如 Conv3x3 -> BN -> ReLU -> Conv1x1），将 Backbone 特征映射为 $N_{class}$ 通道的 Logits。FusionLayer: 将原始特征和 Logits 在 Channel 维度拼接，然后通过一个 Conv1x1 降维或融合，恢复到 LSS 需要的输入维度。B. 损失函数 (Loss Function)训练时需要增加辅助 Loss：$$L_{total} = L_{occ} + \lambda_{seg} \cdot L_{2d\_seg}$$$L_{2d\_seg}$: 标准 CrossEntropy Loss。GT 来源: 使用数据集提供的 2D 分割标签，或者离线使用 CLIP/SAM 生成的伪标签。3. Implementation Request (给 Claude 的具体指令)请基于 PyTorch (以及 mmdetection3d/BEVDet 代码风格)，辅助我完成以下代码修改：任务 1: 定义 SemanticInjector 类创建一个新的文件或类，实现 2D 分割头和融合逻辑。输入: Backbone 输出的 Feature Map (B, C, H, W)。输出:feat_fused: 融合后的特征，用于送入 LSS。seg_logits: 用于计算 2D 辅助 Loss。要求: 使用标准的 nn.Conv2d, nn.BatchNorm2d 以确保 TensorRT 兼容性。任务 2: 修改模型 Forward 流程在 FlashOcc 的主模型类中（通常是 BEVDet 或 FlashOcc 类）：在 extract_img_feat 之后，插入 SemanticInjector。将融合后的特征传给 view_transformer。保留 seg_logits 以便后续计算 Loss。任务 3: Loss 计算修改 Loss 计算模块：接收 seg_logits 和 gt_semantic_2d。计算 2D 分割 Loss 并加权融合到总 Loss 中。注意处理 GT 和 Logits 的分辨率对齐问题（通常需要 interpolate）。4. 部署注意事项 (Deployment Context)Clip Free: 再次强调，推理代码中不能出现 CLIP 模型。2D 分割头是随 Backbone 一起端到端训练的。Export: 新增的层必须支持 ONNX 导出。避免使用复杂的 Python 控制流。


代码结构修改已完成。现在我们需要进入“配置与集成”阶段，以确保这些新模块能被 mmdetection3d 框架正确识别并运行。

请协助我完成以下后续步骤：

### 1. 模块注册 (Module Registration)
请检查并告诉我如何修改 `projects/mmdet3d_plugin/models/dense_heads/__init__.py`，确保 `SemanticInjector` 被正确导入。
同时，请确认 `semantic_injector.py` 中的类定义是否已加上 `@HEADS.register_module()` 装饰器。

### 2. 配置文件修改 (Config Modification)
我使用的配置文件是 `projects/configs/flashocc/flashocc-r50-256x704.py` (假设)。
请给出一个具体的代码片段，展示如何在配置文件的 `model` 字典中添加 `semantic_injector` 的配置项。
* 参数需匹配 `__init__`：`in_channels=256`, `num_classes=17`, `out_channels=256`。

### 3. 数据流与 Loss 的鲁棒性处理 (Critical)
目前的数据流水线 (Data Pipeline) 可能尚未包含 `gt_semantic_2d` 数据。为了防止代码一运行就因为找不到 GT 而报错崩溃，请帮我**修改 Task 3 中的 loss_2d_seg 计算逻辑**：
* **增加空值检查**：如果 `gt_semantic_2d` 为 `None`，请打印一条 Warning 并返回 0 loss，而不是让程序崩溃。这样我可以先验证网络的前向传播（Forward）是否正常，之后再补充数据预处理。

### 4. 编写 Debug 脚本
请为我编写一个简单的 `debug_run.py` 脚本：
* 加载配置文件。
* 构建模型。
* 生成伪造的输入数据（Dummy Input tensors）。
* 运行一次 `forward_train`。
* **目标**：验证 `SemanticInjector` 的 tensor 拼接维度是否匹配，以及显存是否正常。