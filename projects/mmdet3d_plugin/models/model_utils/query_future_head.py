"""Semantic Query Future Head: vectorized cross-attention with velocity bias.

Uses F.unfold to extract all window patches at once, then computes attention
scores and value aggregation via broadcasting — no Python loops over offsets.

For each horizon k with time dt_k:
    query = bev + horizon_embed + prior_proj(per_cls_masks)
    score(B, A) = Q(B)·K(A)/sqrt(d) + w_vel*(-||B-A-v(A)·dt||²) + w_sem*dyn(A)
    future_k = bev + zero_init_out(Σ softmax(score) · V)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class SemanticQueryFutureHead(nn.Module):
    """Vectorized windowed cross-attention future BEV predictor."""

    def __init__(self,
                 bev_channels=256,
                 num_future=3,
                 hidden_dim=64,
                 window_radius=10,
                 num_semantic_classes=17,
                 horizons_sec=(1.0, 2.0, 3.0),
                 vel_bias_init=2.0,
                 downsample=2):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_future = num_future
        self.hidden_dim = hidden_dim
        self.window_radius = window_radius
        self.num_semantic_classes = num_semantic_classes
        self.horizons_sec = list(horizons_sec)
        self.downsample = downsample
        self.K = (2 * window_radius + 1) ** 2

        self.horizon_embed = nn.Embedding(num_future, bev_channels)
        self.prior_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), num_semantic_classes, bev_channels // 2,
                1, bias=False),
            build_norm_layer(dict(type='BN'), bev_channels // 2)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                dict(type='Conv2d'), bev_channels // 2, bev_channels,
                1, bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True))
        self.q_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), bev_channels, hidden_dim, 1, bias=False),
            build_norm_layer(dict(type='BN'), hidden_dim)[1])
        self.k_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), bev_channels, hidden_dim, 1, bias=False),
            build_norm_layer(dict(type='BN'), hidden_dim)[1])
        self.v_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), bev_channels, hidden_dim, 1, bias=False),
            build_norm_layer(dict(type='BN'), hidden_dim)[1])
        self.vel_bias_logit = nn.Parameter(torch.tensor(float(vel_bias_init)))
        self.sem_bias_logit = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Conv2d(hidden_dim, bev_channels, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Pre-compute offset grids (registered as buffers for device transfer)
        R = window_radius
        dh = torch.arange(2 * R + 1).repeat_interleave(2 * R + 1) - R  # (K,)
        dw = torch.arange(2 * R + 1).repeat(2 * R + 1) - R  # (K,)
        self.register_buffer('offset_h', dh.float())  # (K,)
        self.register_buffer('offset_w', dw.float())  # (K,)

    def forward(self, bev, per_cls_masks, dyn_mask, velocity=None):
        """Compute future BEVs via vectorized cross-attention.

        Args:
            bev: (B, C, H, W) current BEV feature.
            per_cls_masks: (B, C_sem, H, W) per-class semantic probabilities.
            dyn_mask: (B, 1, H, W) dynamic region probability.
            velocity: (B, 2, H, W) PCF velocity field or None.

        Returns:
            future_bevs: (B, T, C, H, W)
        """
        B, C, H, W = bev.shape
        R = self.window_radius
        K = self.K
        d = self.hidden_dim

        # Downsample for memory efficiency: at 200x200, the unfold tensor
        # (B, d, K, H, W) would be ~9GB per projection → OOM on 24GB GPU.
        # At 100x100 (2x coarser), it's ~2.3GB. R=10 at coarse resolution
        # covers ±10m (equiv to R=20 at original), sufficient for all
        # horizons (3s p90 displacement = 17.9 cells ≈ 9m).
        ds = self.downsample
        if ds > 1 and (H % ds == 0) and (W % ds == 0):
            bev_a = F.avg_pool2d(bev, ds)
            masks_a = F.avg_pool2d(per_cls_masks, ds)
            dyn_a = F.avg_pool2d(dyn_mask, ds)
            vel_a = F.avg_pool2d(velocity, ds) / ds \
                if velocity is not None else None
            # velocity / ds: cells are ds× larger at coarse resolution,
            # so the same physical velocity has ds× smaller cell value
        else:
            bev_a, masks_a = bev, per_cls_masks
            dyn_a, vel_a = dyn_mask, velocity
        Ha, Wa = bev_a.shape[2], bev_a.shape[3]

        # Project K, V once
        k = self.k_proj(bev_a)   # (B, d, Ha, Wa)
        v = self.v_proj(bev_a)   # (B, d, Ha, Wa)

        # Unfold: extract all (2R+1)² patches at once
        # k_unfold: (B, d*K, Ha*Wa) → (B, d, K, Ha, Wa)
        k_unf = F.unfold(k, kernel_size=2 * R + 1, padding=R, stride=1)
        k_unf = k_unf.view(B, d, K, Ha, Wa)
        v_unf = F.unfold(v, kernel_size=2 * R + 1, padding=R, stride=1)
        v_unf = v_unf.view(B, d, K, Ha, Wa)

        # Unfold dyn_mask for semantic bias: (B, 1, K, Ha, Wa)
        mask_unf = F.unfold(dyn_a, kernel_size=2 * R + 1, padding=R, stride=1)
        mask_unf = mask_unf.view(B, 1, K, Ha, Wa)

        # Unfold velocity for velocity-matching bias
        if vel_a is not None:
            vel_h_unf = F.unfold(
                vel_a[:, 0:1], kernel_size=2 * R + 1, padding=R, stride=1)
            vel_h_unf = vel_h_unf.view(B, 1, K, Ha, Wa)
            vel_w_unf = F.unfold(
                vel_a[:, 1:2], kernel_size=2 * R + 1, padding=R, stride=1)
            vel_w_unf = vel_w_unf.view(B, 1, K, Ha, Wa)

        # Semantic prior conditioning
        prior = self.prior_proj(masks_a)

        # Bias weights
        vel_w = torch.sigmoid(self.vel_bias_logit)
        sem_w = torch.sigmoid(self.sem_bias_logit)

        # Semantic bias (shared across horizons): (B, 1, K, H, W)
        sem_bias = sem_w * mask_unf

        # Offset grids reshaped for broadcasting: (1, 1, K, 1, 1)
        off_h = self.offset_h.view(1, 1, K, 1, 1)
        off_w = self.offset_w.view(1, 1, K, 1, 1)

        future_bevs = []
        for t in range(self.num_future):
            dt = self.horizons_sec[t]

            # Query
            h_emb = self.horizon_embed(
                torch.tensor(t, device=bev.device))
            h_map = h_emb.view(1, C, 1, 1).expand(B, C, Ha, Wa)
            q_input = bev_a + h_map + prior
            q = self.q_proj(q_input)  # (B, d, Ha, Wa)

            # Vectorized score: (B, K, Ha, Wa)
            q_exp = q.unsqueeze(2)  # (B, d, 1, Ha, Wa)
            score = (q_exp * k_unf).sum(dim=1) * self.scale  # (B, K, Ha, Wa)

            # Add semantic bias: (B, K, Ha, Wa)
            score = score + sem_bias.squeeze(1)

            # Add velocity-matching bias
            if vel_a is not None:
                # gap = offset + v * dt (per key position)
                gap_h = off_h + vel_h_unf * dt  # (B, 1, K, Ha, Wa)
                gap_w = off_w + vel_w_unf * dt  # (B, 1, K, Ha, Wa)
                vel_bias = -(gap_h ** 2 + gap_w ** 2)  # (B, 1, K, Ha, Wa)
                score = score + vel_w * vel_bias.squeeze(1)

            # Softmax over K
            weights = F.softmax(score, dim=1)  # (B, K, Ha, Wa)

            # Vectorized value aggregation:
            w_exp = weights.unsqueeze(1)  # (B, 1, K, Ha, Wa)
            output = (w_exp * v_unf).sum(dim=2)  # (B, d, Ha, Wa)

            # Project to full channels then upsample to original resolution
            projected = self.out_proj(output)  # (B, C, Ha, Wa), zero-init
            if Ha != H or Wa != W:
                projected = F.interpolate(
                    projected, size=(H, W), mode='bilinear',
                    align_corners=False)

            # Residual (zero-init → copy at start)
            future_t = bev + projected
            future_bevs.append(future_t)

        return torch.stack(future_bevs, dim=1)  # (B, T, C, H, W)
