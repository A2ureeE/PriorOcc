# PriorOcc-4D 多模态扩展：语义引导的多模态占用预测（Semantic-guided Multimodal Occupancy Forecasting）

本篇是**多模态方向的技术设计文档**（动机 + 方法 + 公式 + 评测协议 + 消融矩阵）。
基础框架（SMP/SDP/连续性补洞/SCMF）见 [`PRIOROCC_4D_INNOVATIONS_README.md`](PRIOROCC_4D_INNOVATIONS_README.md)，
整体架构见 [`README.md`](README.md)。

---

## 0. 一句话动机

> **未来本质是多模态的**：前车可能直行/左转/刹车。确定性预测（单一 flow + CE argmax 监督）
> 会把多个可能未来**平均成一个模糊解**（mode-averaging），这正是长时（3s）与动态物体
> 精度骤降的根源。本方向把 SCMF 的单一运动场升级为 **K 个模式**，并用**语义先验决定
> "这个场景可能走哪个模式"**——语义不仅回答"那里有什么/它怎么动"，还回答
> "**它可能怎么动（的不确定性结构）**"。

**差异化 claim**（与通用多模态轨迹预测的区别）：语义类别决定运动的多模态结构——
车辆多模态、行人有界、静态单模态。多模态性被约束在**语义允许的运动子空间内**，
这是单模态 SMP（"语义约束运动空间"）在概率维度的自然延伸。

---

## 1. 与确定性 baseline 的关系

| 组件 | 确定性（baseline） | 多模态（本方向） | 兼容性 |
|---|---|---|---|
| 运动场生成 | `SemanticConditionedMotionField` → flow `(B,T,2,H,W)` | `MultimodalSemanticConditionedMotionField` → flow `(B,K,T,2,H,W)` | 新类，旧类不动 |
| 运动先验 | `SemanticMotionPrior`（每组 1 个 2×2 基） | `SemanticMultimodalMotionPrior`（每组 K 个模式基 + 语义模式 logits） | 新类，旧类不动 |
| 未来预测 | `SCMFEnhancedPredictor`（1 条自回归链） | `MultimodalSCMFEnhancedPredictor`（K 条链，ConvGRU 参数共享） | 新类，旧类不动 |
| 训练损失 | 逐 horizon CE | **WTA** winner CE + 模式分类 CE + 多样性正则 | 损失键 `loss_occ_future_{t+1}s` **不变** |
| 推理 | 直接解码 | 语义模式概率选优解码 + 全模式输出 | 新增键，旧键不变 |
| 评测 | mIoU@1/2/3s | deployed mIoU + **best-of-K oracle** + 模式选择准确率 | 新增键，旧键不变 |

`enable_multimodal=False`（默认）时走确定性路径，已审计 baseline 逐位一致。

---

## 2. 架构与张量流

```
fused_bev (B,256,H,W) ─┐
motion_feat ───────────┤
attn_feat ─────────────┼─► MultimodalSCMF ─► flow_raw (B,K,T,2,H,W)
semantic_bev (B,17,H,W)┘                       │
                                               ▼
                          SemanticMultimodalMotionPrior
                          ├─ forward: 每模式独立施加语义运动基 → flow (B,K,T,2,H,W)
                          ├─ mode_logits(per_cls_masks) → π_logits (B,K)
                          └─ reg losses（每模式静态/刚体/行人）+ diversity
                                               │
                                               ▼
                     MultimodalSCMFEnhancedPredictor（K 条链，GRU 共享）
                          future_bevs (B,K,T,C,H,W)
                                               │
                    ┌──────────────────────────┴─────────────────────────┐
                    ▼ 训练                                                    ▼ 推理
          WTA: no_grad 对每模式 k 打分                                  π = softmax(mode_logits)
          k* = argmin_k Σ_t w_t·CE(decode(fut[k,t]), GT_t)              k_sel = argmax π
          winner 链带梯度解码 → loss_occ_future_{t+1}s                   deployed = decode(fut[k_sel])
          CE(π_logits, k*) → loss_mode_cls                              全模式解码 → occ_future_modes
          -mean|fut_i-fut_j| → loss_mode_div                            （供 best-of-K 评测）
```

关键形状：`flow (B,K=3,T=3,2,H,W)`；`future_bevs (B,K,T,256,H,W)`；
`mode_logits (B,K)`；`occ_future_modes (K,T)`。

---

## 3. 方法与公式

记号：`φ_k` = 模式 k 的 flow `(B,T,2,H,W)`；`m` = 每类语义概率 `(B,C,H,W)`；
`M ∈ {0,1}^{G×C}` = 组成员矩阵；`K` = 模式数（默认 3）；`T` = 未来步数（3）。

### 3.1 MultimodalSCMF（K 模式运动场生成）

```
x      = Decoder( Concat[fused_bev, motion_feat, attn_feat, semantic_bev] )
raw    = MotionHead_1x1(x)                       # (B, K·T·2, H, W)
φ      = tanh(raw) · max_flow_cells              # (B, K, T, 2, H, W)
```

- **近恒等 + 破对称初始化**（关键设计）：权重零初始化；模式 k 的 bias 置为
  `(k − (K−1)/2)·atanh(Δ/max_flow)`，其中 `Δ=mode_bias_cells=0.05`。
  初始 `|φ| ≤ 0.05·(K−1)/2` cells（200×200@0.4m 网格上 ≈ 2cm，warp 近似恒等），
  但 **K 个模式从第 0 步起就不同** → WTA 梯度可分化，避免对称死锁。

### 3.2 SemanticMultimodalMotionPrior（K 模式语义运动先验）

**前向**（每模式独立施加组混合基，恒等初始化）：
```
p_g  = Σ_{c∈g} m_c                                  # (B,G,H,W)
p̃_g  = p_g / (Σ_g p_g + ε)
A_{g,k} ∈ R^{2×2}                                   # 每组 K 个模式基, init I
A_mix[k] = Σ_g p̃_g · A_{g,k}                        # (B,K,H,W,2,2)
φ_out[k] = (1−α)·φ[k] + α·(A_mix[k] @ φ[k])         # α = sigmoid(alpha_logit)
```
恒等性：`A_{g,k}=I, ∀k` ⇒ `φ_out = φ`（初始不扰动任何模式）。

**语义模式概率**（核心：语义组成 → 模式分布）：
```
c̄    = mean_{H,W}(m)                                # (B,C) 场景级类别分布
p̃_g  = normalize_g(einsum('bc,gc->bg', c̄, M))       # (B,G)
z    = einsum('bg,gk->bk', p̃_g, W_mode)             # W_mode ∈ R^{G×K} 可学习, init 0
π    = softmax(z)                                   # (B,K) 模式概率
```
`W_mode=0` ⇒ 初始均匀。训练后：车辆多的场景 → 车辆组主导的模式概率分布；
静态场景 → 单模态（概率集中）。**语义是多模态结构的驱动者**。

**正则损失**（单模态 SMP 三正则推广到所有模式——"静态区在任何模式下都不动"）：
```
L_static   = Σ_k Σ p_sta·(|φx_k|+|φy_k|) / (K·T·Σ p_sta + ε)
L_rigid    = 每模式置信度加权 TV，跨模式求和归一
L_nonrigid = Σ_k Σ p_ped·relu(‖φ_k‖−τ)² / (K·T·Σ p_ped + ε)
```

**多样性正则**（anti-collapse）：
```
L_div = − mean_{i<j} |φ_i − φ_j|      # 最小化它 = 最大化模式分离
```
初始各模式有 bias 差异 ⇒ `L_div` 从第 0 步就有非零梯度（不会因对称而失效）。

### 3.3 MultimodalSCMFEnhancedPredictor（K 链自回归）

模式 k 独立递推（ConvGRU/gate 参数跨模式**共享**，hidden 分离）：
```
coarse_k[t] = warp(prev[t−1], φ_k[t])               # + validity 门控（继承创新②）
delta_k[t]  = Proj(ConvGRU(prev[t−1], h_k))
future_k[t] = α_k·coarse_k[t] + (1−α_k)·(prev[t−1]+delta_k[t])
```
参数效率：相对确定性预测器**零额外参数**（只是跑 K 次链）；计算量 ×K（BEV 级 conv，可控）。

### 3.4 WTA 训练损失（Winner-Take-All）

```
# 选择（no_grad, per-sample）
score[b,k] = Σ_t w_t · CE_masked( decode(fut[b,k,t]), GT_t[b] )
k*[b]      = argmin_k score[b,k]

# 反传（带梯度，只过 winner 链）
L_occ_future = Σ_t w_t · CE( decode(fut[b,k*,t]), GT_t )     # 键: loss_occ_future_{t+1}s
L_mode_cls   = CE( z[b], k*[b].detach() )                     # 键: loss_mode_cls, 权重 0.5
L_mode_div   = −mean_{i<j}|φ_i−φ_j|                           # 键: loss_mode_div, 权重 0.05
```

- WTA 是轨迹预测领域抑制 mode-averaging 的标准做法：只有最接近 GT 的模式收到梯度，
  不同样本/不同 batch 落在不同模式上 → 模式自然分化。
- `L_mode_cls` 把 WTA winner 作为伪标签教语义选模式——**推理时模式选择能力的来源**。
- `L_mode_div` 防止 K 个模式塌缩成一个（多模态最常见的失败模式）。

### 3.5 损失总表（在原总损失上新增 2 项）

| 损失键 | 公式核心 | 默认权重 | 来源 |
|---|---|---|---|
| `loss_occ_future_{1,2,3}s` | **WTA winner** 链的占用 CE（时间衰减 1/0.7/0.5） | 同 baseline | MultimodalPredictor |
| `loss_mode_cls` | CE(语义模式 logits, WTA winner) | 0.5 | MM-SMP `mode_logits` |
| `loss_mode_div` | −mean 两两模式 L1（anti-collapse） | 0.05 | MM-SMP `diversity_loss` |
| `loss_static_flow / rigid_smooth / nonrigid_bound` | 三正则作用于**所有模式** | 0.05/0.02/0.02 | MM-SMP（继承） |
| 其余（depth/occ/2d_seg/consistency/continuity） | 不变 | 不变 | 继承 |

---

## 4. 评测协议

`NuScenes4DOccForecastDataset.evaluate` 现报告三组指标（多模态结果时）：

| 指标 | 含义 | 说明 |
|---|---|---|
| `mIoU_{1,2,3}s` / `mIoU_avg` | **deployed**：语义模式选择（argmax π）后的预测 | 与确定性方法**公平可比**的主指标 |
| `mIoU_bestofK_{1,2,3}s` / `mIoU_bestofK_avg` | **oracle**：per-sample 选 mIoU 最优模式 | 多模态覆盖能力的**上限**，须与 deployed 分开报告 |
| `mode_selection_acc` | deployed 选择 == oracle 选择的比例 | 模式概率学的准不准（分析指标） |

> 论文表述注意：best-of-K 是 oracle 上界，不能当主结果排名；主结果用 deployed mIoU，
> best-of-K 用于展示"多模态假设集覆盖了真实未来"（gap = 模式选择的学习空间）。

---

## 5. 代码与文件索引

```
projects/mmdet3d_plugin/models/
  model_utils/
    scmf.py                            # + MultimodalSemanticConditionedMotionField
                                       # + MultimodalSCMFEnhancedPredictor（新增类，旧类不动）
    semantic_motion_prior.py           # + SemanticMultimodalMotionPrior（新增类）
  detectors/
    priorocc_4d.py                     # + enable_multimodal / num_modes
                                       # + _per_sample_occ_ce / _multimodal_future_losses (WTA)
                                       # + _multimodal_simple_test（模式选择+全模式输出）
datasets/
  nuscenes_4d_forecast_dataset.py      # evaluate: + best-of-K oracle + mode_selection_acc
                                       #   （顺带修复 count_miou 返回 dict 的遗留 bug）
projects/configs/priorocc/
  priorocc-4d-r50-mmodal.py            # 主配置（继承 smp-sdp-holefill，替换三模块）
tools/
  smoke_test_multimodal.py             # CPU 冒烟测试（模块/模型/评测三层）
```

**消融开关**（`--cfg-options`）：
```bash
model.enable_multimodal=False            # 回退确定性（对照）
model.num_modes=1                        # K=1（消融模式数）
model.num_modes=5                        # 模式数扫描
model.mode_cls_loss_weight=0.0           # 去模式监督（检验语义选择贡献）
model.mode_div_loss_weight=0.0           # 去多样性（检验塌缩）
model.motion_prior=None                  # 去多模态 SMP（检验语义模式先验）
```

**核心消融矩阵**（论文 D 系列扩展）：
| 编号 | 对比 | 回答的问题 |
|---|---|---|
| M1 | multimodal vs deterministic（等训练预算） | 多模态本身值多少？ |
| M2 | K=1/3/5 扫描 | 模式数敏感性 |
| M3 | 去 `L_mode_cls`（随机选模式） | 语义模式选择的贡献（**核心**：证明语义驱动） |
| M4 | 去 `L_mode_div`（+模式两两相似度统计） | 塌缩是否发生 |
| M5 | mode_logits 用随机初始化固定 query（无语义） | 语义 vs 免费多模态（**核心**） |
| M6 | 按 horizon 拆分增益（1s/2s/3s） | 多模态增益是否集中在长时 |
| M7 | 动态类（GMO）vs 静态类拆分 | 增益是否集中在动态物体 |

M3/M5 直接回答 reviewer 的核心质疑"语义先验在多模态里的独立贡献"。

---

## 6. 冒烟测试（CPU，合成数据）

```bash
python tools/smoke_test_multimodal.py             # Part A(模块) + Part C(评测)，秒级
python tools/smoke_test_multimodal.py --full      # + Part B(模型级 WTA/梯度/收敛)
```

覆盖（`--full`）与**实测结果（CPU，合成 batch）**：
- **A 模块级（7 项 PASS）**：flow/future 形状；近恒等初始化（|flow| ≤ Δ(K−1)/2）；
  破对称（模式两两不同）；MM-SMP 恒等（A=I ⇒ 不扰动）；mode_logits 归一/语义敏感；
  正则+多样性有限且梯度到达 `group_modes`/`group_mode_logits`；SCMF→SMP→Predictor 端到端梯度。
- **B 模型级（PASS）**：三模块确为多模态类；损失键完整（`loss_occ_future_{1,2,3}s` +
  `loss_mode_cls` + `loss_mode_div` + 5 个继承键）且全部有限；梯度到达全部新参数；
  **WTA 机制验证**（关键）：用 hook 放大模式分离后，手动复现打分规则验证
  **winner 精确跟随 GT 模式**（K=3 三例全对），且 `loss_mode_cls` 数值区分 winner
  （0.68/1.81/1.11）；`simple_test` 返回 deployed + K×T 全模式 + mode_probs（和为 1）；
  `_per_sample_occ_ce` 数值对照手算；**收敛** `loss 23.26 → 14.08`（overfit 合成 batch）。
  > 注：严格初始化下 K 条链 decode 重叠 ≈99.99%，winner 由噪声决定——这是**训练前预期状态**
  > （模式分离是训练要学的东西），测试用 hook 放大分离度验证机制本身的正确性。
- **C 评测级（PASS）**：fake GT + 假多模态结果走通 evaluate；
  deployed(坏模式) mIoU_avg=**35.85** < best-of-K **100.00**（oracle 上界性）；
  `mode_selection_acc` 正确。

**回归验证**：确定性配置 `smp-sdp-holefill` 在新代码下 build PASS 且参数量
**59,485,223（与改动前完全一致）**；minimal-chain 回归 PASS。多模态配置参数量
59,488,351（**仅 +3,128**：K 模式基 + 模式 logits + motion_head 通道扩展）。

**顺带修复的遗留 bug**：`NuScenes4DOccForecastDataset.evaluate` 原实现把
`count_miou()` 返回的 dict 直接 append 进 `miou_values`（`np.mean(dict)` 必崩）——
原评测代码从未真正跑通过；已改为取 `['mIoU']` 标量。

---

## 7. 训练入口

```bash
CFG=projects/configs/priorocc/priorocc-4d-r50-mmodal.py
python tools/train.py $CFG --work-dir work_dirs/priorocc-4d-mmodal
# 评测（报告 deployed + best-of-K + selection acc）
python tools/test.py $CFG work_dirs/priorocc-4d-mmodal/latest.pth --eval mIoU
```

前置数据准备与 baseline 相同（`_forecast.pkl` 等，见主 README 第 7 节）。
显存注意：WTA 选择阶段对 K×T 个解码走 no_grad、反传只过 winner 链，
训练峰值显存约为确定性版本的 1.3–1.6×（K=3）；推理按 deployed 模式解码，
延迟与确定性几乎持平（仅多 K 次 flow/BEV 链）。

---

## 8. 风险与验证状态

| 风险 | 缓解 | 状态 |
|---|---|---|
| 模式塌缩 | bias 破对称 + L_div + WTA | 冒烟通过；真实训练需监控 `L_div` 与模式两两相似度 |
| 模式选择学不准（deployed≈随机） | L_mode_cls + M3/M5 消融定位 | 待真实训练验证 |
| best-of-K 被误当主结果 | 评测协议强制分开报告 | 已实现 |
| 显存 | no_grad 选择 + winner-only 反传 | 冒烟通过；真实 batch 待测 |
| nuScenes 未来实际高度确定 → 增益小 | M6/M7 按 horizon/动态类拆分增益定位 | **待验证（关键风险）** |

> 诚实声明：冒烟测试只证明**链路正确**（形状/梯度/损失/收敛/评测），不证明方法有效。
> 方法的真实增益依赖完整训练后的 M1/M3/M5/M6/M7 消融；若 nuScenes 3s 内多模态性
> 不足导致增益 <0.5 mIoU，应按预案转向占用流场方向（见评估讨论）。
