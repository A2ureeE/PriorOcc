"""Semantic Query Future Head: cross-attention with velocity-matching bias.

Replaces backward warp (warp + GRU + gate) with windowed cross-attention
where the PCF-measured velocity DIRECTLY guides WHERE to look.

Core formula (velocity-matching bias):
    For query at B, key at A (within window):
        v(A) = PCF-measured velocity at A
        D(A) = A + v(A) * dt          <- where the content at A will arrive
        bias(B, A) = -||B - D(A)||^2  <- high when A's content arrives at B

    When the car at A has velocity that exactly takes it to B:
        bias = 0 (maximum attention)
    When the velocity takes it elsewhere:
        bias < 0 (reduced attention)

This is the correct use of velocity: it tells the attention WHERE content
WILL ARRIVE, so the query at the arrival position knows WHERE to look back.
No propagation needed — the bias is computed locally from the velocity at
the key position (source) and the offset to the query position (destination).

Also includes semantic prior (dyn_mask) as a secondary attention bias.
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
    """Windowed cross-attention with velocity-matching bias.

    For each horizon k with time dt_k:
        query = bev + horizon_embed + semantic_proj(per_cls_masks)
        score(B, A) = Q(B)·K(A)/sqrt(d)
                      + w_vel * (-||B - A - v(A)·dt_k||^2)   <- velocity
                      + w_sem * dyn_mask(A)                   <- semantic
        future_k = bev + zero_init_out(sum softmax(score) · V)

    Args:
        bev_channels: input/output BEV feature channels (256).
        num_future: number of future timesteps (3).
        hidden_dim: attention Q/K/V dimension (64).
        window_radius: attention window radius in BEV cells (10).
        num_semantic_classes: number of semantic classes (17).
        horizons_sec: future horizon times for velocity extrapolation.
        vel_bias_init: initial velocity bias logit (0=sigmoid(0)=0.5;
            use 2.0 for sigmoid(2)=0.88 strong initial trust in velocity).
    """

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

        # Per-horizon embedding
        self.horizon_embed = nn.Embedding(num_future, bev_channels)

        # Semantic prior projection for query conditioning
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

        # Q, K, V projections
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

        # Velocity-matching bias weight (learnable, sigmoid-gated positive)
        # Init to 2.0 -> sigmoid(2)=0.88 -> strong initial trust in velocity
        self.vel_bias_logit = nn.Parameter(torch.tensor(float(vel_bias_init)))

        # Semantic attention bias weight (learnable, sigmoid-gated positive)
        self.sem_bias_logit = nn.Parameter(torch.tensor(1.0))

        # Output projection: ZERO-INIT -> starts as exact copy (B-baseline)
        self.out_proj = nn.Conv2d(hidden_dim, bev_channels, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, bev, per_cls_masks, dyn_mask, velocity=None):
        """Compute future BEVs via velocity-guided cross-attention.

        Args:
            bev: (B, C, H, W) current BEV feature.
            per_cls_masks: (B, C_sem, H, W) per-class semantic probabilities.
            dyn_mask: (B, 1, H, W) dynamic region probability.
            velocity: (B, 2, H, W) PCF-measured velocity field
                (dim 0 = v_h in H/cells per sec, dim 1 = v_w in W/cells per
                sec). None = no velocity guidance (pure content attention).

        Returns:
            future_bevs: (B, T, C, H, W) future BEV features.
        """
        B, C, H, W = bev.shape
        R = self.window_radius

        # Project K, V once (shared across horizons)
        k = self.k_proj(bev)
        v = self.v_proj(bev)

        # Pad for windowed access
        k_pad = F.pad(k, (R, R, R, R), mode='constant', value=0)
        v_pad = F.pad(v, (R, R, R, R), mode='constant', value=0)
        mask_pad = F.pad(dyn_mask, (R, R, R, R), mode='constant', value=0)
        if velocity is not None:
            vel_pad = F.pad(velocity, (R, R, R, R), mode='constant', value=0)
        else:
            vel_pad = None

        # Semantic prior conditioning (shared across horizons)
        prior = self.prior_proj(per_cls_masks)

        # Bias weights (sigmoid-gated to stay positive)
        vel_w = torch.sigmoid(self.vel_bias_logit)
        sem_w = torch.sigmoid(self.sem_bias_logit)

        future_bevs = []
        for t in range(self.num_future):
            dt = self.horizons_sec[t]

            # Query: bev + horizon embedding + semantic prior
            h_emb = self.horizon_embed(
                torch.tensor(t, device=bev.device))
            h_map = h_emb.view(1, C, 1, 1).expand(B, C, H, W)
            q_input = bev + h_map + prior
            q = self.q_proj(q_input)

            # Compute attention scores with velocity-matching bias
            scores = []
            for dh in range(2 * R + 1):
                for dw in range(2 * R + 1):
                    k_patch = k_pad[:, :, dh:dh + H, dw:dw + W]
                    score = (q * k_patch).sum(dim=1) * self.scale

                    # Semantic bias: attend more to dynamic regions
                    dyn_patch = mask_pad[:, 0, dh:dh + H, dw:dw + W]
                    score = score + sem_w * dyn_patch

                    # Velocity-matching bias:
                    # Key at A = query_pos + (dh-R, dw-R)
                    # v(A) = velocity at A
                    # Predicted arrival: D(A) = A + v(A)*dt
                    # Bias = -||query_pos - D(A)||^2
                    #      = -||(dh-R) + v_h*dt, (dw-R) + v_w*dt||^2
                    # Maximum (0) when (dh-R, dw-R) = (-v_h*dt, -v_w*dt)
                    # i.e., the key is at the source whose velocity
                    # exactly delivers content to the query position.
                    if vel_pad is not None:
                        v_h = vel_pad[:, 0, dh:dh + H, dw:dw + W]
                        v_w = vel_pad[:, 1, dh:dh + H, dw:dw + W]
                        off_h = float(dh - R)
                        off_w = float(dw - R)
                        # gap between "where the key's velocity takes it"
                        # and "where the query is"
                        gap_h = off_h + v_h * dt
                        gap_w = off_w + v_w * dt
                        score = score - vel_w * (gap_h ** 2 + gap_w ** 2)

                    scores.append(score)

            scores = torch.stack(scores, dim=1)  # (B, K, H, W)
            weights = F.softmax(scores, dim=1)   # (B, K, H, W)

            # Aggregate values (weighted sum over window)
            output = torch.zeros(
                B, self.hidden_dim, H, W,
                device=bev.device, dtype=bev.dtype)
            idx = 0
            for dh in range(2 * R + 1):
                for dw in range(2 * R + 1):
                    v_patch = v_pad[:, :, dh:dh + H, dw:dw + W]
                    output = output + weights[:, idx:idx + 1] * v_patch
                    idx += 1

            # Residual: zero-init -> starts as exact copy of current BEV
            future_t = bev + self.out_proj(output)
            future_bevs.append(future_t)

        return torch.stack(future_bevs, dim=1)  # (B, T, C, H, W)
