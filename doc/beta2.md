改进意见
完善自门控机制：
角色： 你是一位 3D 计算机视觉和自动驾驶领域的专家研究工程师，特别熟悉 MMDetection3D、BEVDet 和 FlashOcc 的架构。
背景 (Context)： 我正在进行一个名为 "PriorOcc" 的研究项目。你将为我修改代码，但是以下内容仅供参考，你可以提出自己的见解并且按照你自己的见解进行修改，不需要经过我的同意，最后输出一个文档告诉每一个升级部分的原理、原因、数学原理和优点
基线 (Baseline)： FlashOcc (ResNet50 + LSS View Transformer + Occupancy Head)。
现状： 我已经在主干网络（Backbone）上添加了一个轻量级的辅助 2D 语义头 (SemanticInjector) 来预测 2D 语义分割 Logits。
问题： 目前这个 2D 头仅仅用于计算辅助损失（多任务学习）。审稿人可能会认为这种创新太微不足道（Trivial）。
目标： 我曾经在项目中写过一个叫做LSG的门控机制，我想将其升级为 "语义门控深度模块 (SGDM)"。不仅仅是计算 Loss，我要将 2D 语义 Logits 反哺（Feed back）给 LSS 的 DepthNet，从而显式地引导这种不适定的（Ill-posed）2D 到 3D 深度估计过程。
任务要求： 请帮我修改代码（PyTorch），以实现以下两个核心机制：
语义门控深度估计 (交互机制)
目标是基于语义先验作为条件来调节深度分布。
数学逻辑： D(u,v)=Φ(Fimg (u,v)⊕Gating(Ssem (u,v)))
实现细节仅供参考，请你选择你认为最佳方案：
修改 LSSViewTransformer 中的 DepthNet，使其接受两个输入：img_features（来自主干）和 sem_logits（来自我的 SemanticInjector）。
融合策略： 将 img_features 与 softmax(sem_logits) 进行拼接（Concatenate）。
门控 (可选但推荐)： 应用轻量级的 "SE-Block" 风格注意力机制，即利用语义特征在预测深度之前对图像特征进行重加权（Re-weighting）。
约束： 保持计算开销最小化（轻量级）。
在执行验证的时候，严禁输入任何2D检测数据，仅仅使用NuSance数据集数据

更改Loss：
基于 Focal Loss 的鲁棒训练 (替换 CrossEntropy)
目标是提升语义先验的质量，特别是针对对深度估计至关重要的“困难”样本（如小物体、边界），且无需手动设置阈值。
逻辑： 在辅助 2D 语义头中采用 Focal Loss。这将自动降低简单背景样本的权重，并将梯度集中在困难或分类错误的像素上。
实现：
在配置文件的 loss_2d_seg 部分，将 CrossEntropyLoss 替换为 FocalLoss。
确保 SemanticInjector 或 Loss 包装器能正确处理 Focal Loss 的输入格式。
参数： 建议典型的 gamma (如 2.0) 和 alpha 值用于类别平衡。
执行动作：
分析代码修改位置。
提供修改后的 DepthNet 代码。
提供更新后的 Config 片段，展示如何在 MMDetection3D 格式下将 Loss 换成 Focal Loss。
利用全局验证脚本，验证流程全通不会报错，不会出现梯度爆炸或者训练不熟练

其他：
输出的时候请展示测试集Loss，我上一次训练的时候发现到了第16轮就达到了最佳效果，我是否应该在后面的步骤降低学习率？
保留的权重文件应该保留十轮而不是现在的五轮，至少保留从第14轮开始的原始权重文件
从每轮训练都需要进行测试集测试，测试的脚本在test.py，并且打印mIou，在保存权重的时候自动保存一个最佳权重文件（以最大平均mIou来进行选择）
训练完成后保存一张Loss的曲线图，包含Training和validation两个曲线