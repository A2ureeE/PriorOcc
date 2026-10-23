"""Semantic Query Future Head: multi-scale vectorized cross-attention.

Per-horizon resolution and window radius:
    1s: full resolution (200x200), small window (R=5) — precision for small disp
    2s: half resolution (100x100), medium window (R=9) — range for medium disp
    3s: half resolution (100x100), medium window (R=9) — range for large disp

Horizons sharing the same (downsample, radius) share one unfold tensor,
keeping peak memory manageable: 1s unfold ~4.9GB + shared 2s/3s ~3.7GB.

Velocity-matching bias: bias(B, A) = -||B - A - v(A)*dt||^2
    (query at B attends to source A whose velocity delivers content to B)
Zero-init output -> starts as exact copy of current BEV (B-baseline).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class SemanticQueryFutureHead(nn.Module):
    """Multi-scale windowed cross-attention future BEV predictor.

    Args:
        bev_channels: input/output BEV feature channels (256).
        num_future: number of future timesteps (3).
        hidden_dim: attention Q/K/V dimension (64).
        window_radius: int or list of int per horizon. Window radius in
            BEV cells at the (possibly downsampled) resolution.
        num_semantic_classes: number of semantic classes (17).
        horizons_sec: future horizon times for velocity extrapolation.
        vel_bias_init: initial velocity bias logit (sigmoid-gated).
        downsample: int or list of int per horizon. Spatial downsample
            factor before attention (1 = full resolution).
    """

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
        self.num_semantic_classes = num_semantic_classes
        self.horizons_sec = list(horizons_sec)

        # Accept int or per-horizon list for radius and downsample
        if isinstance(window_radius, (list, tuple)):
            assert len(window_radius) == num_future
            self.per_horizon_R = list(window_radius)
        else:
            self.per_horizon_R = [window_radius] * num_future
        if isinstance(downsample, (list, tuple)):
            assert len(downsample) == num_future
            self.per_horizon_ds = list(downsample)
        else:
            self.per_horizon_ds = [downsample] * num_future
        # Backward-compat single values
        self.window_radius = self.per_horizon_R[0]
        self.downsample = self.per_horizon_ds[0]

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

        # Pre-compute offset grids per unique radius (registered buffers
        # so they move with .cuda(); do NOT keep separate CPU copies)
        for R in set(self.per_horizon_R):
            dh = torch.arange(2 * R + 1).repeat_interleave(2 * R + 1) - R
            dw = torch.arange(2 * R + 1).repeat(2 * R + 1) - R
            self.register_buffer(f'off_h_{R}', dh.float())
            self.register_buffer(f'off_w_{R}', dw.float())

        # Group horizons by (ds, R) for shared unfold
        self._groups = []
        for t in range(num_future):
            key = (self.per_horizon_ds[t], self.per_horizon_R[t])
            if self._groups and self._groups[-1][0] == key:
                self._groups[-1][1].append(t)
            else:
                self._groups.append((key, [t]))

    def forward(self, bev, per_cls_masks, dyn_mask, velocity=None):
        """Compute future BEVs via multi-scale cross-attention.

        Args:
            bev: (B, C, H, W) current BEV feature.
            per_cls_masks: (B, C_sem, H, W) per-class semantic probabilities.
            dyn_mask: (B, 1, H, W) dynamic region probability.
            velocity: (B, 2, H, W) PCF velocity field or None.

        Returns:
            future_bevs: (B, T, C, H, W)
        """
        B, C, H, W = bev.shape
        d = self.hidden_dim

        vel_w = torch.sigmoid(self.vel_bias_logit)
        sem_w = torch.sigmoid(self.sem_bias_logit)

        future_bevs = [None] * self.num_future

        for (ds, R), horizons in self._groups:
            K = (2 * R + 1) ** 2

            # Downsample inputs for this group
            if ds > 1 and (H % ds == 0) and (W % ds == 0):
                bev_a = F.avg_pool2d(bev, ds)
                masks_a = F.avg_pool2d(per_cls_masks, ds)
                dyn_a = F.avg_pool2d(dyn_mask, ds)
                vel_a = F.avg_pool2d(velocity, ds) / ds \
                    if velocity is not None else None
            else:
                bev_a, masks_a = bev, per_cls_masks
                dyn_a, vel_a = dyn_mask, velocity
            Ha, Wa = bev_a.shape[2], bev_a.shape[3]

            # Project K, V once per group
            k = self.k_proj(bev_a)
            v = self.v_proj(bev_a)

            # Unfold: (B, d, K, Ha, Wa)
            k_unf = F.unfold(
                k, kernel_size=2 * R + 1, padding=R, stride=1)
            k_unf = k_unf.view(B, d, K, Ha, Wa)
            v_unf = F.unfold(
                v, kernel_size=2 * R + 1, padding=R, stride=1)
            v_unf = v_unf.view(B, d, K, Ha, Wa)
            mask_unf = F.unfold(
                dyn_a, kernel_size=2 * R + 1, padding=R, stride=1)
            mask_unf = mask_unf.view(B, 1, K, Ha, Wa)

            if vel_a is not None:
                vel_h_unf = F.unfold(
                    vel_a[:, 0:1], kernel_size=2 * R + 1,
                    padding=R, stride=1).view(B, 1, K, Ha, Wa)
                vel_w_unf = F.unfold(
                    vel_a[:, 1:2], kernel_size=2 * R + 1,
                    padding=R, stride=1).view(B, 1, K, Ha, Wa)

            # Prior conditioning (shared within group)
            prior = self.prior_proj(masks_a)

            # Semantic bias (shared within group)
            sem_bias = sem_w * mask_unf

            # Offsets for this radius (read from registered buffers,
            # which follow the module's device)
            off_h = getattr(self, f'off_h_{R}')
            off_w = getattr(self, f'off_w_{R}')
            off_h = off_h.view(1, 1, K, 1, 1)
            off_w = off_w.view(1, 1, K, 1, 1)

            for t in horizons:
                dt = self.horizons_sec[t]

                # Query
                h_emb = self.horizon_embed(
                    torch.tensor(t, device=bev.device))
                h_map = h_emb.view(1, C, 1, 1).expand(B, C, Ha, Wa)
                q = self.q_proj(bev_a + h_map + prior)

                # Score: (B, K, Ha, Wa)
                score = (q.unsqueeze(2) * k_unf).sum(dim=1) * self.scale
                score = score + sem_bias.squeeze(1)

                # Velocity-matching bias
                if vel_a is not None:
                    gap_h = off_h + vel_h_unf * dt
                    gap_w = off_w + vel_w_unf * dt
                    vel_bias = -(gap_h ** 2 + gap_w ** 2)
                    score = score + vel_w * vel_bias.squeeze(1)

                weights = F.softmax(score, dim=1)
                output = (weights.unsqueeze(1) * v_unf).sum(dim=2)

                # Project then upsample to original resolution
                projected = self.out_proj(output)
                if Ha != H or Wa != W:
                    projected = F.interpolate(
                        projected, size=(H, W), mode='bilinear',
                        align_corners=False)

                future_bevs[t] = bev + projected

        return torch.stack(future_bevs, dim=1)
