import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS

from .dyn_sta_decoder import warp_feature


class ConvGRU2D(nn.Module):
    """Conv2d-based GRU for spatial feature maps.

    Standard GRU gates with Conv2d instead of Linear, enabling spatial
    hidden state propagation over (B, C, H, W) feature maps.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True)
        self.conv_h = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            bias=True)
        self.hidden_channels = hidden_channels

    def forward(self, x, h):
        """Args:
            x: (B, C_in, H, W) input feature.
            h: (B, C_hidden, H, W) hidden state.

        Returns:
            h_new: (B, C_hidden, H, W) updated hidden state.
        """
        combined = torch.cat([x, h], dim=1)
        zr = self.conv_zr(combined)
        z, r = torch.sigmoid(zr[:, :self.hidden_channels]), \
            torch.sigmoid(zr[:, self.hidden_channels:])
        h_candidate = torch.tanh(
            self.conv_h(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * h_candidate
        return h_new


@NECKS.register_module()
class SemanticConditionedMotionField(nn.Module):
    """SCMF: Semantic-Conditioned Motion Field generator.

    Predicts per-horizon BEV optical flow conditioned on fused BEV, motion
    features, semantic BEV, and attention features. Last layer is
    zero-initialized so initial flow is zero (identity warp).

    Args:
        bev_channels: feature channels for BEV/motion/attention inputs.
        num_semantic_classes: number of semantic classes.
        num_future: number of future timesteps.
        max_flow_cells: max flow magnitude in BEV cells.
        hidden_dim: hidden dimension in decoder.
        norm_cfg: normalization config.
    """

    def __init__(self,
                 bev_channels=256,
                 num_semantic_classes=17,
                 num_future=3,
                 max_flow_cells=5.0,
                 hidden_dim=256,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_semantic_classes = num_semantic_classes
        self.num_future = num_future
        self.max_flow_cells = max_flow_cells

        in_channels = bev_channels * 3 + num_semantic_classes
        self.decoder = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), in_channels, hidden_dim, 3, padding=1,
                bias=False),
            build_norm_layer(norm_cfg, hidden_dim)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                dict(type='Conv2d'), hidden_dim, hidden_dim, 3, padding=1,
                bias=False),
            build_norm_layer(norm_cfg, hidden_dim)[1],
            nn.ReLU(inplace=True))
        self.motion_head = nn.Conv2d(
            hidden_dim, num_future * 2, kernel_size=1, bias=True)

        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)

    def forward(self, fused_bev, motion_feat, semantic_bev, attention_feature):
        """Args:
            fused_bev: (B, C, H, W).
            motion_feat: (B, C, H, W).
            semantic_bev: (B, C_sem, H, W).
            attention_feature: (B, C, H, W).

        Returns:
            flow: (B, T, 2, H, W) flow in BEV cell units.
        """
        B, C, H, W = fused_bev.shape
        assert motion_feat.shape == (B, C, H, W)
        assert attention_feature.shape == (B, C, H, W)
        assert semantic_bev.shape == (B, self.num_semantic_classes, H, W)

        x = torch.cat(
            [fused_bev, motion_feat, attention_feature, semantic_bev], dim=1)
        x = self.decoder(x)
        raw_flow = self.motion_head(x)
        flow = torch.tanh(raw_flow) * self.max_flow_cells
        flow = flow.view(B, self.num_future, 2, H, W)
        return flow


@NECKS.register_module()
class MotionFieldWarper(nn.Module):
    """Warp BEV feature using a single-timestep flow field.

    Args:
        direction: 'forward' (content moves with flow) or 'backward'.
    """

    def __init__(self, direction='forward'):
        super().__init__()
        self.direction = direction

    def forward(self, bev, flow, return_validity=False):
        """Args:
            bev: (B, C, H, W).
            flow: (B, 2, H, W) single timestep flow in BEV cell units.
            return_validity: if True, also return the (B, 1, H, W) hole mask.

        Returns:
            warped: (B, C, H, W).
            validity: (B, 1, H, W) (only when return_validity=True).
        """
        if self.direction == 'forward':
            return warp_feature(bev, flow, return_validity=return_validity)
        else:
            return warp_feature(bev, -flow, return_validity=return_validity)


@NECKS.register_module()
class SCMFEnhancedPredictor(nn.Module):
    """Auto-regressive future BEV predictor using SCMF flow + ConvGRU + gate.

    Per horizon step k:
        coarse_k  = warp(bev_{k-1}, flow_k)
        refined_k = bev_{k-1} + GRU_delta_k
        future_k  = alpha_k * coarse_k + (1 - alpha_k) * refined_k

    Args:
        bev_channels: BEV feature channels.
        num_future: number of future timesteps.
        hidden_dim: GRU hidden dimension.
    """

    def __init__(self,
                 bev_channels=256,
                 num_future=3,
                 hidden_dim=256,
                 use_warp_validity=False):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_future = num_future
        self.hidden_dim = hidden_dim
        self.use_warp_validity = use_warp_validity

        self.warper = MotionFieldWarper(direction='forward')
        self.conv_gru = ConvGRU2D(bev_channels, hidden_dim)
        self.gru_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), hidden_dim, bev_channels, 1, bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True))
        gate_in = bev_channels * 3 + (1 if use_warp_validity else 0)
        self.gate = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), gate_in, bev_channels, 1,
                bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, bev_current, flow):
        """Args:
            bev_current: (B, C, H, W) current (t=0) BEV feature.
            flow: (B, T, 2, H, W) per-horizon flow from SCMF.

        Returns:
            future_bevs: (B, T, C, H, W) future BEV features.
        """
        B, C, H, W = bev_current.shape
        assert flow.shape == (B, self.num_future, 2, H, W), \
            f"flow shape {flow.shape} != ({B}, {self.num_future}, 2, {H}, {W})"

        prev_bev = bev_current
        hx = torch.zeros(
            B, self.hidden_dim, H, W,
            device=bev_current.device, dtype=bev_current.dtype)

        future_bevs = []
        for k in range(self.num_future):
            flow_k = flow[:, k]
            if self.use_warp_validity:
                coarse_k, validity_k = self.warper(
                    prev_bev, flow_k, return_validity=True)
            else:
                coarse_k = self.warper(prev_bev, flow_k)
            hx = self.conv_gru(prev_bev, hx)
            delta_k = self.gru_proj(hx)
            refined_k = prev_bev + delta_k
            gate_feats = [prev_bev, coarse_k, delta_k]
            if self.use_warp_validity:
                gate_feats.append(validity_k)
            alpha = self.gate(torch.cat(gate_feats, dim=1))
            future_k = alpha * coarse_k + (1 - alpha) * refined_k
            future_bevs.append(future_k)
            prev_bev = future_k

        future_bevs = torch.stack(future_bevs, dim=1)
        return future_bevs


@NECKS.register_module()
class DirectFutureOccupancyHead(nn.Module):
    """Lightweight direct future BEV predictor (Phase C baseline).

    Produces T future BEV features as residuals on current BEV. Each step
    has its own conv block; last layer is zero-initialized so initial
    future = current (identity, stable).

    Interface matches SCMFEnhancedPredictor: forward(bev_current, flow=None).
    The flow argument is ignored, enabling config-level swap without changing
    call sites.

    Args:
        bev_channels: BEV feature channels.
        num_future: number of future timesteps.
        hidden_dim: hidden dimension in per-step decoder.
    """

    def __init__(self,
                 bev_channels=256,
                 num_future=3,
                 hidden_dim=256):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_future = num_future

        self.future_convs = nn.ModuleList([
            nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d'),
                    bev_channels,
                    hidden_dim,
                    3,
                    padding=1,
                    bias=False),
                build_norm_layer(dict(type='BN'), hidden_dim)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, bev_channels, 1, bias=True))
            for _ in range(num_future)
        ])

        for conv in self.future_convs:
            nn.init.zeros_(conv[-1].weight)
            nn.init.zeros_(conv[-1].bias)

    def forward(self, bev_current, flow=None):
        """Args:
            bev_current: (B, C, H, W) current BEV feature.
            flow: ignored (interface compat with SCMFEnhancedPredictor).

        Returns:
            future_bevs: (B, T, C, H, W) future BEV features.
        """
        future_bevs = []
        for k in range(self.num_future):
            delta = self.future_convs[k](bev_current)
            future_k = bev_current + delta
            future_bevs.append(future_k)
        return torch.stack(future_bevs, dim=1)


@NECKS.register_module()
class MultimodalSemanticConditionedMotionField(nn.Module):
    """K-mode SCMF: predicts K parallel semantic-conditioned motion fields.

    Output flow shape is (B, K, T, 2, H, W) — one motion field per mode,
    so the future is represented as K candidate scene evolutions instead
    of a single deterministic field (which mode-averages multimodal
    futures into a blurry mean).

    Near-identity init: motion_head weights are zero-initialized and the
    bias holds a tiny per-mode offset that breaks the symmetry between
    modes while keeping |flow| <= mode_bias_cells * (K-1)/2 BEV cells
    (a few cm on the 0.4 m grid), i.e. initial warps stay ~identity.

    Args:
        bev_channels: feature channels for BEV/motion/attention inputs.
        num_semantic_classes: number of semantic classes.
        num_future: number of future timesteps (T).
        num_modes: number of motion modes (K).
        max_flow_cells: max flow magnitude in BEV cells.
        hidden_dim: hidden dimension in decoder.
        mode_bias_cells: per-mode initial flow separation (BEV cells).
        norm_cfg: normalization config.
    """

    def __init__(self,
                 bev_channels=256,
                 num_semantic_classes=17,
                 num_future=3,
                 num_modes=3,
                 max_flow_cells=5.0,
                 hidden_dim=256,
                 mode_bias_cells=0.05,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_semantic_classes = num_semantic_classes
        self.num_future = num_future
        self.num_modes = num_modes
        self.max_flow_cells = max_flow_cells
        self.mode_bias_cells = mode_bias_cells

        in_channels = bev_channels * 3 + num_semantic_classes
        self.decoder = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), in_channels, hidden_dim, 3, padding=1,
                bias=False),
            build_norm_layer(norm_cfg, hidden_dim)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                dict(type='Conv2d'), hidden_dim, hidden_dim, 3, padding=1,
                bias=False),
            build_norm_layer(norm_cfg, hidden_dim)[1],
            nn.ReLU(inplace=True))
        self.motion_head = nn.Conv2d(
            hidden_dim, num_modes * num_future * 2, kernel_size=1, bias=True)

        nn.init.zeros_(self.motion_head.weight)
        # Symmetry-breaking bias: mode k starts at a small constant offset
        # (k - (K-1)/2) * mode_bias_cells so that WTA gradients diverge from
        # step 1. Invert tanh*max_flow to place the offset in raw-logit space.
        with torch.no_grad():
            target = min(mode_bias_cells / max(max_flow_cells, 1e-6), 0.9)
            raw_off = math.atanh(target)
            bias = self.motion_head.bias.view(num_modes, num_future, 2)
            bias.zero_()
            for k in range(num_modes):
                bias[k, :, :] = (k - (num_modes - 1) / 2.0) * raw_off

    def forward(self, fused_bev, motion_feat, semantic_bev, attention_feature):
        """Args:
            fused_bev: (B, C, H, W).
            motion_feat: (B, C, H, W).
            semantic_bev: (B, C_sem, H, W).
            attention_feature: (B, C, H, W).

        Returns:
            flow: (B, K, T, 2, H, W) flow in BEV cell units, one per mode.
        """
        B, C, H, W = fused_bev.shape
        assert motion_feat.shape == (B, C, H, W)
        assert attention_feature.shape == (B, C, H, W)
        assert semantic_bev.shape == (B, self.num_semantic_classes, H, W)

        x = torch.cat(
            [fused_bev, motion_feat, attention_feature, semantic_bev], dim=1)
        x = self.decoder(x)
        raw_flow = self.motion_head(x)  # (B, K*T*2, H, W)
        flow = torch.tanh(raw_flow) * self.max_flow_cells
        flow = flow.view(B, self.num_modes, self.num_future, 2, H, W)
        return flow


@NECKS.register_module()
class MultimodalSCMFEnhancedPredictor(nn.Module):
    """K autoregressive future-BEV chains with shared ConvGRU weights.

    Each mode k runs its own auto-regressive chain
        coarse  = warp(prev, flow_k)
        refined = prev + GRU_delta
        future  = alpha * coarse + (1 - alpha) * refined
    with a separate hidden state, while ConvGRU/gate parameters are shared
    across modes (parameter-efficient multimodality).

    Args:
        bev_channels: BEV feature channels.
        num_future: number of future timesteps (T).
        num_modes: number of motion modes (K).
        hidden_dim: GRU hidden dimension.
        use_warp_validity: feed warp validity mask into the gate.
    """

    def __init__(self,
                 bev_channels=256,
                 num_future=3,
                 num_modes=3,
                 hidden_dim=256,
                 use_warp_validity=False):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_future = num_future
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.use_warp_validity = use_warp_validity

        self.warper = MotionFieldWarper(direction='forward')
        self.conv_gru = ConvGRU2D(bev_channels, hidden_dim)
        self.gru_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), hidden_dim, bev_channels, 1, bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True))
        gate_in = bev_channels * 3 + (1 if use_warp_validity else 0)
        self.gate = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), gate_in, bev_channels, 1, bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

    def _run_chain(self, bev_current, flow_mode):
        """Run one auto-regressive chain for a single mode.

        Args:
            bev_current: (B, C, H, W).
            flow_mode: (B, T, 2, H, W) flow of this mode.

        Returns:
            chain: (B, T, C, H, W) future BEV features of this mode.
        """
        B, C, H, W = bev_current.shape
        T = flow_mode.shape[1]
        prev_bev = bev_current
        hx = torch.zeros(
            B, self.hidden_dim, H, W,
            device=bev_current.device, dtype=bev_current.dtype)
        chain = []
        for t in range(T):
            flow_t = flow_mode[:, t]
            if self.use_warp_validity:
                coarse_t, validity_t = self.warper(
                    prev_bev, flow_t, return_validity=True)
            else:
                coarse_t = self.warper(prev_bev, flow_t)
            hx = self.conv_gru(prev_bev, hx)
            delta_t = self.gru_proj(hx)
            refined_t = prev_bev + delta_t
            gate_feats = [prev_bev, coarse_t, delta_t]
            if self.use_warp_validity:
                gate_feats.append(validity_t)
            alpha = self.gate(torch.cat(gate_feats, dim=1))
            future_t = alpha * coarse_t + (1 - alpha) * refined_t
            chain.append(future_t)
            prev_bev = future_t
        return torch.stack(chain, dim=1)

    def forward(self, bev_current, flow):
        """Args:
            bev_current: (B, C, H, W) current (t=0) BEV feature.
            flow: (B, K, T, 2, H, W) per-mode per-horizon flow from the
                multimodal SCMF.

        Returns:
            future_bevs: (B, K, T, C, H, W) future BEV features per mode.
        """
        assert flow.dim() == 6, \
            f"multimodal predictor expects 6-dim flow, got {flow.dim()}"
        B, K, T, two, H, W = flow.shape
        assert K == self.num_modes, \
            f"flow modes {K} != num_modes {self.num_modes}"
        assert T == self.num_future, \
            f"flow horizons {T} != num_future {self.num_future}"
        assert two == 2
        assert bev_current.shape == (B, self.bev_channels, H, W)

        chains = [self._run_chain(bev_current, flow[:, k]) for k in range(K)]
        return torch.stack(chains, dim=1)  # (B, K, T, C, H, W)
