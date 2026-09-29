# Copyright (c) Phigent Robotics. All rights reserved.
"""Pure-PyTorch CPU fallback for BEVPoolv2.

Used when CUDA is unavailable (e.g. RTX 5060 sm_120 with PyTorch 1.10).
Implements the same scatter-add semantics as the CUDA extension using
torch.index_add_, with full autograd support via a custom Function.
"""
import torch


class QuickCumsumCPU(torch.autograd.Function):
    r"""CPU BEVPoolv2 implementation using pure PyTorch index_add_."""

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        B, Dz, Dy, Dx, C = bev_feat_shape
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.long()
        ranks_feat = ranks_feat.long()
        ranks_bev = ranks_bev.long()

        depth_flat = depth.flatten()
        feat_flat = feat.reshape(-1, C)

        depth_vals = depth_flat[ranks_depth]
        feat_vals = feat_flat[ranks_feat]
        weighted = depth_vals.unsqueeze(-1) * feat_vals

        bev_flat = torch.zeros(
            B * Dz * Dy * Dx, C, dtype=feat.dtype, device=feat.device)
        bev_flat.index_add_(0, ranks_bev, weighted)

        ctx.save_for_backward(ranks_bev, depth, feat,
                              ranks_depth, ranks_feat)
        ctx.bev_feat_shape = bev_feat_shape
        return bev_flat.view(B, Dz, Dy, Dx, C)

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_depth, ranks_feat = \
            ctx.saved_tensors
        B, Dz, Dy, Dx, C = ctx.bev_feat_shape

        out_grad_flat = out_grad.reshape(B * Dz * Dy * Dx, C)

        grad_weighted = out_grad_flat[ranks_bev]

        depth_flat = depth.flatten()
        feat_flat = feat.reshape(-1, C)
        depth_vals = depth_flat[ranks_depth]
        feat_vals = feat_flat[ranks_feat]

        grad_depth_flat = torch.zeros_like(depth_flat)
        grad_feat_flat = torch.zeros_like(feat_flat)

        grad_depth_vals = (grad_weighted * feat_vals).sum(dim=-1)
        grad_depth_flat.index_add_(0, ranks_depth, grad_depth_vals)

        grad_feat_vals = grad_weighted * depth_vals.unsqueeze(-1)
        grad_feat_flat.index_add_(0, ranks_feat, grad_feat_vals)

        return (grad_depth_flat.view_as(depth),
                grad_feat_flat.view_as(feat),
                None, None, None, None, None, None, None)


def bev_pool_v2_cpu(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                    bev_feat_shape, interval_starts, interval_lengths):
    """CPU wrapper matching the CUDA bev_pool_v2 interface.

    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (N_points, )
        ranks_feat:  (N_points, )
        ranks_bev:   (N_points, )
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        interval_starts: (N_pillar, )
        interval_lengths: (N_pillar, )

    Returns:
        x: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    x = QuickCumsumCPU.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
