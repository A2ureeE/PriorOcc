"""
Language Self-Gating (LSG)

中文说明：
LSG 模块用于对 3D 体素（voxel）特征进行语义与物理联合的软门控，抑制不符合物理常识或语义先验的响应。

输入/输出：
- 输入 x: Tensor, 形状 (B, C, D, H, W)
- 输出 gated_x: Tensor, 同形状 (B, C, D, H, W)
- 可选输出 gate_map: Tensor, 形状 (B, D, H, W)

设计要点：
1. 使用 1x1x1 的 Conv3d 将输入投影到较低维的特征空间（便于与锚点匹配）。
2. 定义若干语义锚点（可用 CLIP 文本嵌入替换），计算锚点与体素投影特征的相似度。
3. 采用 Z 轴（高度）先验的可学习偏置（physical bias），与相似度合并后通过 Sigmoid 形成门控值。
4. 使用门控值对原始体素特征进行缩放（抑制或保持）。

该实现为轻量可训练版本，便于在 CPU 上单元测试与后续集成。

"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageSelfGating(nn.Module):
    """Language Self-Gating 模块

    参数：
    - in_channels: 输入通道数（体素特征通道数）
    - proj_channels: 投影通道数（默认为 in_channels // 2）
    - num_anchors: 语义锚点数量（默认 6，可根据类别或文本锚点调整）
    - grid_D: 体素的高度切片数 D（用于 physical bias 的尺寸），若为 None 则动态使用输入的 D
    - scale: 可选的门控缩放系数（控制门控对特征的影响）

    前向：
    - x: (B, C, D, H, W)
    返回：
    - gated_x: (B, C, D, H, W)
    - gate_map: (B, D, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        proj_channels: Optional[int] = None,
        num_anchors: int = 6,
        grid_D: Optional[int] = None,
        scale: float = 1.0,
    ):
        super().__init__()
        if proj_channels is None:
            proj_channels = max(8, in_channels // 2)
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        self.num_anchors = num_anchors
        self.grid_D = grid_D
        self.scale = scale

        # 1x1x1 投影，用于将原始特征投影到较低维空间便于相似度计算
        self.projector = nn.Conv3d(in_channels, proj_channels, kernel_size=1)

        # 语义锚点（随机初始化，可用 CLIP 文本嵌入替换）
        # 形状 (num_anchors, proj_channels)
        self.anchors = nn.Parameter(torch.randn(num_anchors, proj_channels) * 0.1)

        # 物理偏置：针对高度 D 的可学习偏置，如果 grid_D 未知则在 forward 中动态广播
        if grid_D is not None:
            # (D,)
            self.register_parameter('physical_bias', nn.Parameter(torch.zeros(grid_D)))
        else:
            # 延迟创建为 buffer/parameter 在 forward
            self.physical_bias = None

        # 可学习缩放系数（标量），用于控制门控影响强度
        self.gate_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        """前向：
        支持两种输入格式：
          - 3D 体素：x (B, C, D, H, W)
          - 2D BEV：x (B, C, H, W)（内部自动扩展为 D=1）

        返回 gated_x, gate_map
        """
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            is_2d = False
        elif x.dim() == 4:
            # 2D 输入，扩展为 D=1
            B, C, H, W = x.shape
            D = 1
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
            is_2d = True
        else:
            raise AssertionError('输入应为 (B,C,D,H,W) 或 (B,C,H,W)')

        # 如果 physical_bias 未提前设置，则在第一次 forward 时初始化
        if self.physical_bias is None:
            # 初始化为与高度成反比的偏置（较低高度通常更可能为地面语义）
            # 这里使用缓慢递减初始化示例：bias[z] = -z / D
            bias = -torch.arange(D, dtype=torch.float32, device=x.device) / max(1.0, D - 1)
            self.register_parameter('physical_bias', nn.Parameter(bias))

        # 投影
        proj = self.projector(x)  # (B, P, D, H, W)

        # 归一化通道方向（L2），避免幅值差异过大
        proj_norm = proj / (proj.norm(p=2, dim=1, keepdim=True) + 1e-6)  # (B, P, D, H, W)

        # 归一化锚点
        anchors = self.anchors  # (A, P)
        anchors_norm = anchors / (anchors.norm(p=2, dim=1, keepdim=True) + 1e-6)  # (A, P)

        # 计算相似度：einsum 以高效得到 (B, A, D, H, W)
        # proj_norm: (B, P, D, H, W) -> (B, D, H, W, P) for einsum convenience
        sim = torch.einsum('bpdhw,ap->badhw', proj_norm, anchors_norm)  # (B, A, D, H, W)

        # 聚合锚点相似度（取最大或加权和）。这里使用最大以突出最相关锚点
        max_sim, _ = sim.max(dim=1)  # (B, D, H, W)

        # 将 physical_bias 加入，需要根据当前 D 进行适配
        pb = self.physical_bias  # (grid_D,) 或者动态初始化后的 (D,)
        if pb.shape[0] != D:
            # 如果 physical_bias 维度与当前 D 不匹配，使用插值或者取子集/扩展
            if D == 1:
                # 2D 输入情况，取 physical_bias 的均值作为单一偏置
                pb = pb.mean().view(1)
            else:
                # 使用线性插值对齐到新的 D
                pb = F.interpolate(
                    pb.view(1, 1, -1),  # (1, 1, grid_D)
                    size=D,
                    mode='linear',
                    align_corners=True
                ).view(D)
        pb = pb.view(1, D, 1, 1)
        gate_logits = self.scale * (max_sim + pb)

        # Sigmoid 得到门控值 0..1
        gate_map = torch.sigmoid(gate_logits)  # (B, D, H, W)

        # 应用门控到原始特征（按位置抑制）
        gate_expand = gate_map.unsqueeze(1)  # (B,1,D,H,W)

        # 我们使用抑制形式：gated = x * (1 - gate)，当 gate 接近 1 时抑制显著
        gated = x * (1.0 - gate_expand)

        # 如果原始输入是 2D，则恢复维度
        if is_2d:
            gated = gated.squeeze(2)  # (B, C, H, W)
            gate_map = gate_map.squeeze(1) if gate_map.dim() == 4 else gate_map.squeeze(1)

        return gated, gate_map


if __name__ == '__main__':
    # 简要的模块级单测（可在 tools 中运行）
    m = LanguageSelfGating(in_channels=256, proj_channels=128, num_anchors=6)
    x = torch.randn(2, 256, 8, 10, 10)
    gx, gm = m(x)
    print('gated:', gx.shape, 'gate map:', gm.shape)
