"""Prior-gated Correspondence Flow (PCF).

Measures per-pixel dynamic displacement by correlating current-frame BEV
features with ego-aligned history BEV features (RAFT-style cost volume),
then extrapolates to future horizons using constant velocity. The result
is gated by the dynamic mask from the SI prior so only dynamic regions
get the measured displacement; static regions keep the global (SCMF) flow.

Design principle (user's request): use prior knowledge to tell the model
WHERE dynamic objects are, and let it MEASURE how they move from history,
rather than hypothesizing (regressing) a displacement from scratch.

The only learnable parameters are a per-horizon residual correction
(zero-initialized, so the module starts as a pure measurement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS


@NECKS.register_module()
class PriorCorrespondenceFlow(nn.Module):
    """Prior-gated correspondence-measured flow for dynamic objects.

    flow_add[:, k] = dyn_mask * (v_measured * dt_k + residual_k)

    where v_measured comes from soft-argmax over a correlation volume
    between current and ego-aligned history BEV features.

    Args:
        bev_channels: channels of raw (pre-encoder) BEV features (64).
        num_future: number of future timesteps (3).
        search_radius: correlation search radius in BEV cells (10).
        temperature: softmax temperature for soft-argmax (0.1; lower=sharper).
        horizons_sec: future horizon times in seconds ((1, 2, 3)).
        history_lag: temporal distance of the history frame used (1.0s).
        history_index: index into bev_feat_list for the history frame
            (0=key/current, 1=0.5s back, 2=1.0s back).
    """

    def __init__(self,
                 bev_channels=64,
                 num_future=3,
                 search_radius=10,
                 temperature=0.1,
                 horizons_sec=(1.0, 2.0, 3.0),
                 history_lag=1.0,
                 history_index=2,
                 gate_mode='none'):
        """
        gate_mode:
            'none'  — no explicit gating. After ego alignment, correlation
                      naturally gives ~0 displacement for static content,
                      so the flow is self-gating. Recommended (backward
                      warping needs flow at DESTINATION positions, and a
                      source-side hard gate breaks this).
            'soft'  — g * flow with g = 0.2 + 0.8 * dyn_mask. Suppresses
                      noise in static regions without fully blocking.
            'hard'  — g * flow with g = dyn_mask. NOTE: breaks backward
                      warping at destination positions outside the mask.
        """
        super().__init__()
        self.bev_channels = bev_channels
        self.num_future = num_future
        self.search_radius = search_radius
        self.temperature = temperature
        self.horizons_sec = list(horizons_sec)
        self.history_lag = history_lag
        self.history_index = history_index
        self.gate_mode = gate_mode

        # Learned residual per horizon (zero-init: start = pure measurement)
        # 2 channels per horizon: (dx, dy)
        self.residual = nn.Conv2d(bev_channels, num_future * 2, 1, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def _correlation_displacement(self, bev_cur, bev_hist):
        """Soft-argmax displacement from correlation volume.

        Args:
            bev_cur: (B, C, H, W) current BEV features.
            bev_hist: (B, C, H, W) ego-aligned history BEV features.

        Returns:
            disp_h, disp_w: (B, 1, H, W) measured displacement (history→current)
                in BEV cells. Positive = content moved in +H/+W direction.
        """
        B, C, H, W = bev_cur.shape
        r = self.search_radius
        K = (2 * r + 1) ** 2

        # Normalize features to unit norm per pixel (cosine similarity).
        # This puts all pixels on the same scale [-1, 1], so the softmax
        # temperature is meaningful across different feature magnitudes.
        # Real feature matches → similarity ~0.8–1.0; noise → ~0.0–0.2.
        cur_n = F.normalize(bev_cur, dim=1, eps=1e-8)
        hist_n = F.normalize(bev_hist, dim=1, eps=1e-8)

        # Pad history for windowed correlation
        padded = F.pad(hist_n, (r, r, r, r), mode='constant', value=0)

        # Correlation volume (cosine similarity): corr[b, k, h, w] ∈ [-1, 1]
        # Vectorized: unfold in W dim per kernel row (21 iterations instead of 441)
        # Memory per row: B × C × (2r+1) × H × W × 4B ≈ 215 MB (freed after each)
        corr_rows = []
        cur_exp = cur_n.unsqueeze(2)  # (B, C, 1, H, W) for broadcasting
        for dh in range(2 * r + 1):
            hist_row = padded[:, :, dh:dh + H, :]  # (B, C, H, W+2r)
            # Unfold W dim: all 2r+1 column shifts at once
            hist_unf = F.unfold(
                hist_row, kernel_size=(1, 2 * r + 1), stride=1)
            hist_unf = hist_unf.view(B, C, 2 * r + 1, H, W)
            # Corr for all dw in this row: (B, 2r+1, H, W)
            corr_row = (cur_exp * hist_unf).sum(dim=1)
            corr_rows.append(corr_row)
        corr = torch.cat(corr_rows, dim=1)  # (B, K, H, W)

        # Softmax over K (temperature controls sharpness)
        weights = F.softmax(corr / self.temperature, dim=1)  # (B, K, H, W)

        # Offset values (zero-based index → actual cell offset)
        # k = dh_idx * (2r+1) + dw_idx → dh = dh_idx - r, dw = dw_idx - r
        # dh_vals[k] and dw_vals[k] give the actual (dh, dw) offset for index k
        # dh_vals: [−r,...,r, −r,...,r, ...] (row-major)
        # dw_vals: [−r,...,−r, ..., r,...,r] (column within row)
        # k = dh_idx * (2r+1) + dw_idx (outer=dh/H, inner=dw/W)
        # dh_vals[k] = k // (2r+1) - r  → repeat_interleave (outer var)
        # dw_vals[k] = k % (2r+1) - r  → repeat (inner var)
        dh_vals = torch.arange(
            2 * r + 1, device=bev_cur.device,
            dtype=bev_cur.dtype).repeat_interleave(2 * r + 1) - r  # (K,)
        dw_vals = torch.arange(
            2 * r + 1, device=bev_cur.device,
            dtype=bev_cur.dtype).repeat(2 * r + 1) - r  # (K,)

        # Soft-argmax: weighted average of offsets
        # dh_avg[b, 0, h, w] = sum_k weights[b,k,h,w] * dh_vals[k]
        # This is the average offset in history where the best match is found.
        # Positive dh_avg = match is at a HIGHER H position in history.
        dh_avg = (weights * dh_vals.view(1, K, 1, 1)).sum(dim=1, keepdim=True)
        dw_avg = (weights * dw_vals.view(1, K, 1, 1)).sum(dim=1, keepdim=True)

        # Displacement (history → current):
        # corr offset (dh, dw) means: history position (h+dh, w+dw) matches current (h, w).
        # The content moved FROM history (h+dh, w+dw) TO current (h, w).
        # So displacement = current − history = (h−(h+dh), w−(w+dw)) = (−dh, −dw).
        # Positive displacement = content moved in +H/+W direction.
        disp_h = -dh_avg  # (B, 1, H, W)
        disp_w = -dw_avg  # (B, 1, H, W)

        return disp_h, disp_w

    def forward(self, bev_cur, bev_hist, dyn_mask):
        """Compute prior-gated correspondence flow addition.

        Args:
            bev_cur: (B, C, H, W) current frame raw BEV (64-ch, pre-encoder).
            bev_hist: (B, C, H, W) ego-aligned history BEV (64-ch, pre-encoder).
            dyn_mask: (B, 1, H, W) dynamic region probability from SI prior.

        Returns:
            flow_add: (B, T, 2, H, W) flow addition (added to SCMF flow).
                flow_add[:, k, 0] = dx (W dim), flow_add[:, k, 1] = dy (H dim).
        """
        if bev_hist is None or self.history_lag <= 0:
            B = bev_cur.shape[0]
            H, W = bev_cur.shape[-2:]
            return bev_cur.new_zeros(B, self.num_future, 2, H, W)

        B, C, H, W = bev_cur.shape

        # Measure displacement from correlation (in cells)
        disp_h, disp_w = self._correlation_displacement(bev_cur, bev_hist)
        # disp_h, disp_w: (B, 1, H, W), positive = moved in +H/+W direction

        # Velocity (cells per second)
        v_h = disp_h / self.history_lag  # (B, 1, H, W)
        v_w = disp_w / self.history_lag  # (B, 1, H, W)

        # Learned residual (zero-init: at start, flow = pure measurement)
        residual = self.residual(bev_cur)  # (B, T*2, H, W)
        residual = residual.view(B, self.num_future, 2, H, W)

        # Compute per-horizon flow addition
        # Flow convention (from warp_feature):
        #   flow[:, 0] = dx → W dimension (positive = content moves in +W)
        #   flow[:, 1] = dy → H dimension (positive = content moves in +H)
        # warp_feature uses backward warping: sample_grid = base_grid − flow_norm
        # → output(h,w) = input(h−dy, w−dx), so positive flow moves content
        #   from input to output in the +H/+W direction.
        # The measured velocity (v_h, v_w) is in +H/+W direction (from history
        # to current), and we want to extrapolate it forward (current to future),
        # so the flow is simply velocity × horizon_time (same sign).
        flow_add = torch.zeros(
            B, self.num_future, 2, H, W,
            device=bev_cur.device, dtype=bev_cur.dtype)
        if self.gate_mode == 'hard':
            g = dyn_mask.squeeze(1)  # (B, H, W)
        elif self.gate_mode == 'soft':
            g = 0.2 + 0.8 * dyn_mask.squeeze(1)
        else:  # 'none'
            g = None
        for k in range(self.num_future):
            dt = self.horizons_sec[k]
            dx = v_w.squeeze(1) * dt + residual[:, k, 0]  # (B, H, W)
            dy = v_h.squeeze(1) * dt + residual[:, k, 1]  # (B, H, W)
            if g is not None:
                dx = dx * g
                dy = dy * g
            flow_add[:, k, 0] = dx
            flow_add[:, k, 1] = dy

        # Store velocity for downstream modules (e.g. query future head)
        self._velocity = torch.cat([v_h, v_w], dim=1)  # (B, 2, H, W)

        return flow_add
