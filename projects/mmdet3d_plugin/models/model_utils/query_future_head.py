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
                 vel_bias_init=2.0):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_future = num_future
        self.hidden_dim = hidden_dim
        self.window_radius = window_radius
        self.num_semantic_classes = num_semantic_classes
        self.horizons_sec = list(horizons_sec)
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

        # Project K, V once
        k = self.k_proj(bev)   # (B, d, H, W)
        v = self.v_proj(bev)   # (B, d, H, W)

        # Unfold: extract all (2R+1)² patches at once
        # k_unfold: (B, d*K, H*W) → (B, d, K, H, W)
        k_unf = F.unfold(k, kernel_size=2 * R + 1, padding=R, stride=1)
        k_unf = k_unf.view(B, d, K, H, W)
        v_unf = F.unfold(v, kernel_size=2 * R + 1, padding=R, stride=1)
        v_unf = v_unf.view(B, d, K, H, W)

        # Unfold dyn_mask for semantic bias: (B, 1, K, H, W)
        mask_unf = F.unfold(dyn_mask, kernel_size=2 * R + 1, padding=R, stride=1)
        mask_unf = mask_unf.view(B, 1, K, H, W)

        # Unfold velocity for velocity-matching bias
        if velocity is not None:
            vel_h_unf = F.unfold(
                velocity[:, 0:1], kernel_size=2 * R + 1, padding=R, stride=1)
            vel_h_unf = vel_h_unf.view(B, 1, K, H, W)
            vel_w_unf = F.unfold(
                velocity[:, 1:2], kernel_size=2 * R + 1, padding=R, stride=1)
            vel_w_unf = vel_w_unf.view(B, 1, K, H, W)

        # Semantic prior conditioning
        prior = self.prior_proj(per_cls_masks)

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
            h_map = h_emb.view(1, C, 1, 1).expand(B, C, H, W)
            q_input = bev + h_map + prior
            q = self.q_proj(q_input)  # (B, d, H, W)

            # Vectorized score: (B, K, H, W)
            # score[b,k,h,w] = sum_d q[b,d,h,w] * k_unf[b,d,k,h,w] * scale
            q_exp = q.unsqueeze(2)  # (B, d, 1, H, W)
            score = (q_exp * k_unf).sum(dim=1) * self.scale  # (B, K, H, W)

            # Add semantic bias: (B, K, H, W)
            score = score + sem_bias.squeeze(1)

            # Add velocity-matching bias
            if velocity is not None:
                # gap = offset + v * dt (per key position)
                gap_h = off_h + vel_h_unf * dt  # (B, 1, K, H, W)
                gap_w = off_w + vel_w_unf * dt  # (B, 1, K, H, W)
                vel_bias = -(gap_h ** 2 + gap_w ** 2)  # (B, 1, K, H, W)
                score = score + vel_w * vel_bias.squeeze(1)

            # Softmax over K
            weights = F.softmax(score, dim=1)  # (B, K, H, W)

            # Vectorized value aggregation:
            # output[b,d,h,w] = sum_k weights[b,k,h,w] * v_unf[b,d,k,h,w]
            w_exp = weights.unsqueeze(1)  # (B, 1, K, H, W)
            output = (w_exp * v_unf).sum(dim=2)  # (B, d, H, W)

            # Residual (zero-init → copy at start)
            future_t = bev + self.out_proj(output)
            future_bevs.append(future_t)

        return torch.stack(future_bevs, dim=1)  # (B, T, C, H, W)
