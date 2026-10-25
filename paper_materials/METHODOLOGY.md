# PriorOcc-4D 方法论（供写作 agent 使用）

> 目标：提供论文 Method 章节所需的全部公式、模块定义、设计动机。写作 agent 可直接引用公式编号和模块名称。
> 本文与 DATA_INVENTORY.md（实验数据）和 BASELINES_AND_SOTA.md（外部对比）配套。
> 最后更新：2026-10-23

---

## 1. 问题定义

给定历史 $T_{\text{obs}}=3$ 帧环视相机图像 $\{I_{t-2}, I_{t-1}, I_t\}$（6 相机 @ 2Hz），预测未来 $T_{\text{fut}}=3$ 个时刻的 3D 语义占用网格：

$$\{O_{t+1}, O_{t+2}, O_{t+3}\} = f_\theta(I_{t-2}, I_{t-1}, I_t)$$

其中 $O_{t+\Delta} \in \{0,\dots,17\}^{H \times W \times Z}$，$H{=}W{=}200$，$Z{=}16$，体素分辨率 0.512m，感知范围 $[-51.2, 51.2]\,\text{m} \times [-5, 3]\,\text{m}$。

**核心创新**：语义先验时空化（Semantic Prior Spatiotemporalization）——将单帧 PriorOcc 的 2D 语义先验从"回答*哪里有什么*"扩展为"回答*它将怎么动*"。

---

## 2. 整体架构

$$\underbrace{\{I_{t-i}\}_{i=0}^{2}}_{\text{3帧相机}} \xrightarrow{\text{Backbone}} \underbrace{F_i}_{\text{图像特征}} \xrightarrow{\text{SI}} \underbrace{\tilde{F}_i}_{\text{语义增强}} \xrightarrow{\text{LSS}} \underbrace{B_i}_{\text{BEV 特征}} \xrightarrow{\text{Encoder}} \underbrace{\tilde{B}}_{\text{融合 BEV}} \xrightarrow{\text{Motion}} \underbrace{\hat{O}_{t+\Delta}}_{\text{未来占用}}$$

模块图（各组件独立可关/可消融）：

```
3 帧相机图 → Backbone(ResNet-50) → SI 语义注入 → LSS 视角变换(含DepthNet) → BEV 特征
                                                                            ↓
                                                          3帧拼接(192ch) → BEV Encoder → 融合 BEV
                                                                            ↓
                                              ┌── dyn_sta_decoder → per_cls_masks + dyn_mask
                                              ├── SemanticMotionAttention → 运动特征
                                              ├── SCMF(语义条件流场) + PCF(测量流) → 光流
                                              └── SCMFEnhancedPredictor(非AR warp) → 未来 BEV
                                                                            ↓
                                                          occ_head → O_hat_{t+1}, O_hat_{t+2}, O_hat_{t+3}
```

---

## 3. 各模块公式

### 3.1 SemanticInjector（SI，语义先验注入）

**动机**：将 2D 语义分割的类别信息注入图像特征，使动态物体在特征空间可分。

$$\mathbf{s} = \text{SegHead}(F) \in \mathbb{R}^{C_{\text{cls}} \times H' \times W'}$$

$$\tilde{F} = \text{Fusion}([\,F \,\|\, \mathbf{s}\,])$$

其中 $\text{SegHead}$ 为 $3{\times}3\,\text{Conv} \to \text{BN} \to \text{ReLU} \to 1{\times}1\,\text{Conv}$，$\text{Fusion}$ 为 $1{\times}1\,\text{Conv} \to \text{BN} \to \text{ReLU}$，$\|$ 表示通道拼接。$\mathbf{s}$ 同时由 2D 语义分割交叉熵 $\mathcal{L}_{\text{seg}}$ 监督。

**关键作用**：使后续 correlation（PCF）能跨帧匹配同类物体——SI 让 car 的特征在特征空间与 building 可分，correlation 因此能"认出"同一辆车。

### 3.2 SemanticMotionAttention（SMA，语义运动注意力）

$$M_{\text{dyn}}, M_{\text{sta}}, \{M_c\}_{c=1}^{C_{\text{cls}}} = \text{Decoder}(\tilde{B}, \text{motion\_feat})$$

$$\mathbf{a} = \text{Attn}\bigl(Q=\text{pool}(\{M_c\}),\ K=\tilde{B},\ V=\tilde{B}\bigr)$$

其中 $M_c$ 为第 $c$ 类的空间掩码，$M_{\text{dyn}} = \sum_{c \in \mathcal{C}_{\text{dyn}}} M_c$ 为动态区域掩码。

### 3.3 SCMF（Semantic-Conditioned Motion Field，语义条件流场）

**语义条件化**：流预测头以逐类语义掩码为条件输入，对不同语义区域输出不同的流。

$$\mathbf{f}_{\text{raw}} = \text{ConvGRU}(\tilde{B}, \mathbf{a}, \{M_c\})$$

$$\mathbf{f}_k = \tanh(\mathbf{f}_{\text{raw},k}) \cdot \boldsymbol{\tau}_k$$

其中 $\boldsymbol{\tau}_k$ 为第 $k$ 个 horizon 的位移上限（cells），设为 $[\tau_1, \tau_2, \tau_3] = [12, 24, 36]$（对应 6.1m / 12.2m / 18.4m，覆盖 GT 位移 p90 = 4.4m / 7.2m / 9.2m）。

**非自回归推演**：每个 horizon 独立 warp 同一个当前 BEV（消除自回归链上的误差累积）：

$$B^{\text{warp}}_k = \mathcal{W}(\tilde{B},\ \mathbf{f}_k)$$

其中 $\mathcal{W}$ 为 backward 双线性 warp。

### 3.4 PriorCorrespondenceFlow（PCF，先验对应流）

**动机**：SCMF 的流是回归"猜"出来的；PCF 改为从历史帧"测量"位移。

**Ego 对齐**：先将历史 BEV 特征做 ego 运动补偿（消除自车位移，只留物体运动）：

$$B_{\text{hist}}^{\text{align}} = \mathcal{T}_{\text{ego}}(B_{\text{hist}})$$

**相关测量**：在 ego 对齐后的当前帧与历史帧之间构造余弦相似度 cost-volume：

$$\text{corr}(\mathbf{p}, \delta) = \frac{\hat{B}_{\text{cur}}(\mathbf{p}) \cdot \hat{B}_{\text{hist}}(\mathbf{p} + \delta)}{\|\hat{B}_{\text{cur}}(\mathbf{p})\| \,\|\hat{B}_{\text{hist}}(\mathbf{p}+\delta)\|},\quad \delta \in \{-r,\dots,r\}^2$$

其中 $\hat{\cdot}$ 为归一化后的特征，$r$ 为搜索半径（10 cells）。

**Soft-argmax 位移**：

$$\mathbf{v}(\mathbf{p}) = \frac{1}{\Delta t_{\text{lag}}} \sum_{\delta} \delta \cdot \text{softmax}_\delta\!\left(\frac{\text{corr}(\mathbf{p}, \delta)}{\tau}\Big)$$

其中 $\tau$ 为温度（0.1，低温度 → 尖锐分布 → 亚像素精度），$\Delta t_{\text{lag}}=1\,\text{s}$ 为历史帧时间距离。

**先验融合**：

$$\mathbf{f}^{\text{PCF}}_k = M_{\text{dyn}} \odot \bigl(\mathbf{v} \cdot \Delta t_k + \mathbf{r}_k\bigr)$$

其中 $\mathbf{r}_k$ 为第 $k$ 个 horizon 的可学习残差修正（**零初始化**，起步 = 纯测量）。$M_{\text{dyn}} \odot$ 为软门控（$\odot$ 为元素乘）。

**总光流**：

$$\mathbf{f}_k = \underbrace{\tanh(\mathbf{f}^{\text{SCMF}}_{\text{raw},k}) \cdot \boldsymbol{\tau}_k}_{\text{SCMF 回归}} + \underbrace{M_{\text{dyn}} \odot (\mathbf{v} \cdot \Delta t_k + \mathbf{r}_k)}_{\text{PCF 测量}}$$

### 3.5 SCMFEnhancedPredictor（未来 BEV 预测器）

**门控融合**：

$$\alpha = \sigma(\text{Gate}(\tilde{B}, \mathbf{f}))$$

$$\hat{B}_{t+\Delta t_k} = \alpha \cdot \underbrace{\mathcal{W}(\tilde{B}, \mathbf{f}_k)}_{\text{warp（执行位移）}} + (1-\alpha) \cdot \underbrace{(\tilde{B} + \text{GRU}(\tilde{B}))}_{\text{残差（微调）}}$$

其中 $\sigma$ 为 sigmoid，$\alpha$ 初始化偏 warp 分支（$\text{bias}=2.0$，$\alpha \approx 0.88$）。

**参数量**：SCMF + Predictor 共 ~3.7M 参数。

### 3.6 语义连续性损失（补洞）

$$\mathcal{L}_{\text{cont}} = \lambda_{\text{2D}} \mathcal{L}^{\text{2D}}_{\text{TV}} + \lambda_{\text{BEV}} \mathcal{L}^{\text{BEV}}_{\text{TV}} + \lambda_{\text{occ}} \mathcal{L}^{\text{occ}}_{\text{TV}}$$

其中 $\mathcal{L}_{\text{TV}}$ 为总变差（Total Variation），应用于连续类（路面、人行道、地形、建筑、植被）的预测。

### 3.7 训练目标

$$\mathcal{L} = \mathcal{L}_{\text{occ}} + \sum_{k=1}^{3} w_k \mathcal{L}^{\text{future}}_{\text{occ},k} + \sum_{i=0}^{2} \lambda_{\text{seg}} \mathcal{L}^{\text{2D}}_{\text{seg},i} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{cont}} \mathcal{L}_{\text{cont}}$$

其中 $w_k = [1.0, 0.7, 0.5]$，$\lambda_{\text{depth}} = 3.0$（与 FlashOcc 单帧 baseline 一致），$\lambda_{\text{seg}} = 0.3$。

**类别平衡**（动态类加权，Phase 2）：

$$\text{CE}(c) = w_c \cdot \text{CE}_{\text{std}}(c),\quad w_c = \begin{cases} \beta & c \in \mathcal{C}_{\text{dyn}} \\ 1 & \text{otherwise} \end{cases},\quad \beta = 3.0$$

---

## 4. 先验使用全景（论文卖点组织）

| 环节 | 先验的角色 | 公式中的体现 |
|---|---|---|
| SI | 让动态物体在特征空间可分 | $\tilde{F} = \text{Fusion}([F \| \mathbf{s}])$ |
| PCF | correlation 能"认出"同类物体 → 测准位移 | $\text{corr}(\mathbf{p}, \delta)$ 使用 SI 增强后的特征 |
| SCMF | 流头知道"哪里是动态区" | $\mathbf{f}_{\text{raw}} = \text{ConvGRU}(\tilde{B}, \mathbf{a}, \{M_c\})$ |
| SMA | 逐类语义掩码池化 query | $Q = \text{pool}(\{M_c\})$ |
| PCF 融合 | $M_{\text{dyn}}$ 门控只让动态区产生测量流 | $M_{\text{dyn}} \odot (\mathbf{v} \cdot \Delta t_k + \mathbf{r}_k)$ |
| 动态类加权 | 梯度预算从饱和的静态类重分配给欠学的动态类 | $w_c = \beta$ for $c \in \mathcal{C}_{\text{dyn}}$ |

**一句话**：先验在管线中被使用 6 处，覆盖特征层（SI）、注意力层（SMA）、流场层（SCMF 条件化 + PCF 门控）、测量层（correlation 可分性）、损失层（动态类加权）。

---

## 5. 设计取舍（论文 Discussion 素材）

| 被否方案 | 否决原因 | 证据 |
|---|---|---|
| SMP（语义运动先验）| rigid_smooth 奖励全局平移退化解，基矩阵收缩车辆位移至 34% | 逐类 IoU 指纹 + ckpt 基矩阵审计 |
| dyn_boost（特征幅值放大）| 模型学到负值（选择抑制），幅值 ≠ 注意力 | dyn_boost = −0.136 |
| 自回归推演 | 误差沿链累积 | H2 消融：非AR 单独把 2s/3s 动态 Δ 从负转正 |
| 全局 flow cap=5 | GT p90 位移 8.6~17.9 cells，物理不够 | 配置审查 |
| 硬门控流场 | backward warp 需要目标位置的流非零，源位置掩码破坏取样 | 合成单测 |
| PCF 直接叠加到已训 SCMF | 已放大的 SCMF 流 + PCF 测量 = 双重计费（超调） | C 变体全线回退 |

---

## 6. 消融实验设计（论文 Table 素材）

| 消融臂 | 关闭什么 | 验证什么 | 预期（基于已验证数据） |
|---|---|---|---|
| Full | — | — | — |
| w/o SI | SemanticInjector | 特征层先验 | correlation 匹配精度下降 → PCF 增益减少 |
| w/o PCF | PriorCorrespondenceFlow | 测量 vs 回归 | 1s 动态 IoU 下降 ~0.8 |
| w/o SCMF 条件化 | 语义掩码条件输入 | 流场层先验 | 流预测精度下降 |
| w/o 动态类加权 | dynamic_class_boost | 损失层先验 | 动态类 IoU 下降 |
| w/o 非AR | non_autoregressive=False | 2s/3s 消融 | 2s/3s 动态 Δ 从正转负 |
| w/o depth 监督 | loss_depth_weight=0 | 训练期辅助监督 | 当前帧 mIoU 降 ~2 |

---

## 7. 代码位置索引（写作 agent 可查实现）

| 模块 | 文件 | 类名 |
|---|---|---|
| 检测器 | `projects/mmdet3d_plugin/models/detectors/priorocc_4d.py` | `PriorOcc4D` |
| SI | `projects/mmdet3d_plugin/models/model_utils/semantic_injector.py` | `SemanticInjector` |
| PCF | `projects/mmdet3d_plugin/models/model_utils/prior_correspondence.py` | `PriorCorrespondenceFlow` |
| SCMF | `projects/mmdet3d_plugin/models/model_utils/scmf.py` | `SCMFEnhancedPredictor` |
| SMA | `projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py` | `SemanticMotionAttention` |
| 动静分解 | `projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py` | `PerClassDeltaCombiner` |
| 补洞 | `projects/mmdet3d_plugin/models/model_utils/semantic_continuity.py` | `SemanticContinuityLoss` |
| Query 头（备选） | `projects/mmdet3d_plugin/models/model_utils/query_future_head.py` | `SemanticQueryFutureHead` |
| 类别加权 | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py` | `BEVOCCHead2D` (dynamic_class_boost) |
