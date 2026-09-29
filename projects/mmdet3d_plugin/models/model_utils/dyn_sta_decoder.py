import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS


def warp_feature(bev, flow_cells, return_validity=False):
    """Warp BEV feature using flow in BEV cell units via backward warping.

    Uses grid_sample with align_corners=True (consistent with bevdet4d.py shift_feature).
    Zero flow produces identity (grid unchanged).

    Args:
        bev: (B, C, H, W) BEV feature tensor.
        flow_cells: (B, 2, H, W) flow in BEV cell units.
            flow[:, 0] = dx (x-axis, maps to W dimension)
            flow[:, 1] = dy (y-axis, maps to H dimension)
        return_validity: if True, also return a (B, 1, H, W) mask marking
            cells whose sampling grid stayed in-bounds. Out-of-bounds cells are
            zero-padded by grid_sample, i.e. disocclusion "holes".

    Returns:
        warped: (B, C, H, W) warped BEV feature.
        validity: (B, 1, H, W) in-bounds mask (only when return_validity=True).
    """
    B, C, H, W = bev.shape
    assert flow_cells.shape == (B, 2, H, W), \
        f"flow_cells shape {flow_cells.shape} != ({B}, 2, {H}, {W})"

    if bev.is_floating_point():
        dtype = bev.dtype
    else:
        dtype = bev.float()

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=bev.device, dtype=dtype),
        torch.linspace(-1, 1, W, device=bev.device, dtype=dtype),
        indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

    flow_norm = torch.empty_like(flow_cells)
    flow_norm[:, 0] = flow_cells[:, 0] * (2.0 / max(W - 1, 1))
    flow_norm[:, 1] = flow_cells[:, 1] * (2.0 / max(H - 1, 1))

    sample_grid = base_grid - flow_norm.permute(0, 2, 3, 1)

    warped = F.grid_sample(
        bev, sample_grid, mode='bilinear', padding_mode='zeros',
        align_corners=True)

    if not return_validity:
        return warped

    validity = (
        (sample_grid[..., 0] >= -1) & (sample_grid[..., 0] <= 1) &
        (sample_grid[..., 1] >= -1) & (sample_grid[..., 1] <= 1)
    ).to(dtype).unsqueeze(1)  # (B, 1, H, W)
    return warped, validity


@NECKS.register_module()
class SemanticDynStaSeparator(nn.Module):
    """Partition semantic BEV into dynamic/static masks using config-provided class IDs.

    Args:
        num_semantic_classes: number of semantic classes.
        dynamic_class_ids: list of class indices considered dynamic.
        static_class_ids: list of class indices considered static.
        eps: epsilon for numerical stability.
    """

    def __init__(self,
                 num_semantic_classes=17,
                 dynamic_class_ids=None,
                 static_class_ids=None,
                 eps=1e-8):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.eps = eps
        if dynamic_class_ids is None:
            dynamic_class_ids = list(range(11))
        if static_class_ids is None:
            static_class_ids = list(range(11, num_semantic_classes))
        self.register_buffer(
            'dynamic_class_ids',
            torch.tensor(dynamic_class_ids, dtype=torch.long))
        self.register_buffer(
            'static_class_ids',
            torch.tensor(static_class_ids, dtype=torch.long))

    def forward(self, semantic_bev, visibility):
        """Args:
            semantic_bev: (B, C_sem, H, W) normalized semantic probabilities.
            visibility: (B, 1, H, W) visibility mask.

        Returns:
            dyn_mask: (B, 1, H, W) dynamic probability.
            sta_mask: (B, 1, H, W) static probability.
            per_cls_masks: (B, C_sem, H, W) per-class probabilities.
        """
        B, C_sem, H, W = semantic_bev.shape
        assert C_sem == self.num_semantic_classes, \
            f"semantic_bev channels {C_sem} != num_semantic_classes {self.num_semantic_classes}"
        assert visibility.shape == (B, 1, H, W), \
            f"visibility shape {visibility.shape} != ({B}, 1, {H}, {W})"

        dyn_mask = semantic_bev[:, self.dynamic_class_ids].sum(
            dim=1, keepdim=True).clamp(0, 1)
        sta_mask = semantic_bev[:, self.static_class_ids].sum(
            dim=1, keepdim=True).clamp(0, 1)
        per_cls_masks = semantic_bev

        return dyn_mask, sta_mask, per_cls_masks


@NECKS.register_module()
class SemanticMotionFeatureEncoder(nn.Module):
    """Encode motion features from 3 aligned raw BEV frames + semantic BEV.

    Args:
        raw_bev_channels: channels of raw BEV (numC_Trans, default 64).
        bev_channels: output feature channels (default 256).
        num_semantic_classes: number of semantic classes (default 17).
        norm_cfg: normalization config.
    """

    def __init__(self,
                 raw_bev_channels=64,
                 bev_channels=256,
                 num_semantic_classes=17,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.raw_bev_channels = raw_bev_channels
        self.bev_channels = bev_channels
        self.num_semantic_classes = num_semantic_classes

        self.raw_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), raw_bev_channels, bev_channels, 1,
                bias=False),
            build_norm_layer(norm_cfg, bev_channels)[1],
            nn.ReLU(inplace=True))

        self.sem_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), num_semantic_classes, bev_channels, 1,
                bias=False),
            build_norm_layer(norm_cfg, bev_channels)[1],
            nn.ReLU(inplace=True))

        motion_in = bev_channels * 3 + bev_channels
        self.motion_out = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), motion_in, bev_channels, 1, bias=False),
            build_norm_layer(norm_cfg, bev_channels)[1],
            nn.ReLU(inplace=True))

    def forward(self, aligned_bev_history, semantic_bev):
        """Args:
            aligned_bev_history: (B, 3, C_raw, H, W) ego-aligned t-2, t-1, t.
            semantic_bev: (B, C_sem, H, W) semantic probability BEV.

        Returns:
            motion_feat: (B, C_bev, H, W).
        """
        B, T, C_raw, H, W = aligned_bev_history.shape
        assert T == 3, f"Expected 3 history frames, got {T}"
        assert C_raw == self.raw_bev_channels, \
            f"raw_bev channels {C_raw} != {self.raw_bev_channels}"
        assert semantic_bev.shape == (B, self.num_semantic_classes, H, W), \
            f"semantic_bev shape mismatch"

        proj_t2 = self.raw_proj(aligned_bev_history[:, 0])
        proj_t1 = self.raw_proj(aligned_bev_history[:, 1])
        proj_t0 = self.raw_proj(aligned_bev_history[:, 2])

        delta_1 = proj_t1 - proj_t2
        delta_2 = proj_t0 - proj_t1
        accel = delta_2 - delta_1

        sem_feat = self.sem_proj(semantic_bev)

        motion_input = torch.cat([delta_1, delta_2, accel, sem_feat], dim=1)
        motion_feat = self.motion_out(motion_input)
        return motion_feat


@NECKS.register_module()
class SemanticMotionAttention(nn.Module):
    """Per-class semantic mask pools query from fused BEV.
    BEV serves as K/V, motion feature provides additive bias.
    Uses efficient batched matmul (attention matrix is C_sem x HW, not HW x HW).

    Args:
        bev_channels: input feature channels.
        num_semantic_classes: number of semantic classes.
        num_heads: number of attention heads.
        hidden_dim: hidden dimension for projections.
    """

    def __init__(self,
                 bev_channels=256,
                 num_semantic_classes=17,
                 num_heads=4,
                 hidden_dim=256):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_semantic_classes = num_semantic_classes
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(bev_channels, hidden_dim)
        self.k_proj = nn.Sequential(
            build_conv_layer(dict(type='Conv2d'), bev_channels, hidden_dim, 1,
                             bias=False),
            build_norm_layer(dict(type='BN'), hidden_dim)[1])
        self.v_proj = nn.Sequential(
            build_conv_layer(dict(type='Conv2d'), bev_channels, hidden_dim, 1,
                             bias=False),
            build_norm_layer(dict(type='BN'), hidden_dim)[1])
        self.motion_bias_proj = nn.Linear(bev_channels, num_heads)
        self.out_proj = nn.Sequential(
            build_conv_layer(dict(type='Conv2d'), hidden_dim, bev_channels, 1,
                             bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True))
        self.eps = 1e-8

    def forward(self, fused_bev, motion_feat, per_cls_masks):
        """Args:
            fused_bev: (B, C, H, W).
            motion_feat: (B, C, H, W).
            per_cls_masks: (B, C_sem, H, W) per-class probabilities.

        Returns:
            attn_feat: (B, C, H, W) attention-enhanced feature.
        """
        B, C, H, W = fused_bev.shape
        C_sem = self.num_semantic_classes
        assert motion_feat.shape == (B, C, H, W)
        assert per_cls_masks.shape == (B, C_sem, H, W)

        masks_flat = per_cls_masks.view(B, C_sem, H * W)
        bev_flat = fused_bev.view(B, C, H * W)
        mask_sum = masks_flat.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        cls_query = torch.bmm(masks_flat, bev_flat.transpose(1, 2))
        cls_query = cls_query / mask_sum
        q = self.q_proj(cls_query)
        q = q.view(B, C_sem, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        k = self.k_proj(fused_bev).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v_proj(fused_bev).view(B, self.num_heads, self.head_dim, H * W)

        motion_flat = motion_feat.view(B, C, H * W)
        motion_pooled = torch.bmm(masks_flat, motion_flat.transpose(1, 2))
        motion_pooled = motion_pooled / mask_sum
        bias = self.motion_bias_proj(motion_pooled)
        bias = bias.permute(0, 2, 1).unsqueeze(-1)

        scores = torch.matmul(q, k) * self.scale
        scores = scores + bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.transpose(-1, -2))
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, C_sem, self.hidden_dim)
        out_bev_flat = torch.bmm(out.transpose(1, 2), masks_flat)
        mask_sum_spatial = masks_flat.sum(dim=1).unsqueeze(1)
        out_bev_flat = out_bev_flat / (mask_sum_spatial + self.eps)
        out_bev = out_bev_flat.view(B, self.hidden_dim, H, W)
        attn_feat = self.out_proj(out_bev)
        return attn_feat


@NECKS.register_module()
class PerClassDeltaCombiner(nn.Module):
    """Per-class mask-weighted delta combination with binary mask ablation support.

    Args:
        bev_channels: input/output feature channels.
        num_semantic_classes: number of semantic classes.
        binary_mask: if True, threshold masks to 0/1 (for ablation).
        mask_threshold: threshold for binary mode.
    """

    def __init__(self,
                 bev_channels=256,
                 num_semantic_classes=17,
                 binary_mask=False,
                 mask_threshold=0.5):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_semantic_classes = num_semantic_classes
        self.binary_mask = binary_mask
        self.mask_threshold = mask_threshold

        self.motion_to_cls = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), bev_channels, num_semantic_classes, 1,
                bias=False),
            build_norm_layer(dict(type='BN'), num_semantic_classes)[1],
            nn.ReLU(inplace=True))
        self.cls_to_motion = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'), num_semantic_classes, bev_channels, 1,
                bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True))

    def forward(self, fused_bev, motion_feat, per_cls_masks):
        """Args:
            fused_bev: (B, C, H, W).
            motion_feat: (B, C, H, W).
            per_cls_masks: (B, C_sem, H, W).

        Returns:
            combined_delta: (B, C, H, W).
        """
        B, C, H, W = fused_bev.shape
        assert motion_feat.shape == (B, C, H, W)
        assert per_cls_masks.shape == (B, self.num_semantic_classes, H, W)

        if self.binary_mask:
            masks = (per_cls_masks > self.mask_threshold).float()
        else:
            masks = per_cls_masks

        cls_motion = self.motion_to_cls(motion_feat)
        weighted = cls_motion * masks
        combined_delta = self.cls_to_motion(weighted)
        return combined_delta


@NECKS.register_module()
class SemanticBEVProjector(nn.Module):
    """Project 2D semantic seg_logits into BEV using the view transformer's
    voxel pooling. Reuses get_ego_coor() and voxel_pooling_v2() — no
    independent projection logic.

    Args:
        num_semantic_classes: number of semantic classes.
        eps: epsilon for normalization.
    """

    def __init__(self, num_semantic_classes=17, eps=1e-8):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.eps = eps

    def forward(self, seg_logits, depth_prob, view_transformer, metas_input):
        """Project 2D semantic probabilities into BEV.

        Args:
            seg_logits: (B*N, C_sem, fH, fW) from SemanticInjector.
            depth_prob: (B*N, D, fH, fW) post-softmax depth from view transformer.
            view_transformer: LSSViewTransformer reference (for get_ego_coor,
                voxel_pooling_v2).
            metas_input: (sensor2ego, ego2global, intrins, post_rots,
                post_trans, bda) with shapes (B, N, 4, 4) etc.

        Returns:
            semantic_bev: (B, C_sem, H, W) normalized semantic probabilities.
            visibility: (B, 1, H, W) total depth mass per cell.
        """
        sensor2ego, ego2global, intrins, post_rots, post_trans, bda = \
            metas_input
        B, N = sensor2ego.shape[0], sensor2ego.shape[1]
        C_sem = self.num_semantic_classes

        sem_prob = torch.softmax(seg_logits, dim=1)
        fH, fW = sem_prob.shape[-2], sem_prob.shape[-1]
        sem_prob = sem_prob.view(B, N, C_sem, fH, fW)
        depth_prob = depth_prob.view(B, N, -1, fH, fW)

        coor = view_transformer.get_ego_coor(
            sensor2ego, ego2global, intrins, post_rots, post_trans, bda)

        semantic_bev = view_transformer.voxel_pooling_v2(
            coor, depth_prob, sem_prob)

        ones = torch.ones(
            B, N, 1, fH, fW, device=sem_prob.device, dtype=sem_prob.dtype)
        visibility = view_transformer.voxel_pooling_v2(
            coor, depth_prob, ones)

        semantic_bev = semantic_bev / (visibility + self.eps)
        return semantic_bev, visibility
