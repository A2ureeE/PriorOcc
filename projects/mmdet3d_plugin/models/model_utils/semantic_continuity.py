"""Semantic Continuity Loss.

Background classes such as driveable surface / sidewalk / terrain / manmade /
vegetation are spatially continuous -- they form large connected regions
without holes. This module turns that prior into a confidence-weighted
total-variation (TV) regularizer on the predicted class-probability fields, at
three levels:

  * 2D semantic logits (SemanticInjector output, key frame)
  * BEV semantic probabilities (SemanticBEVProjector output)
  * 3D occupancy logits (optional, off by default)

Smoothing the continuous-class probability field suppresses the "holes"
(isolated low-probability cells inside a large background region) that arise
from sparse-depth projection noise and warp disocclusion.

The TV weights (per-cell confidence of being a continuous class) are detached
so the loss is a pure smoothness term on the probabilities and cannot be
minimized by shrinking the confidence itself. The module has no learnable
parameters; gradients flow back into seg_head / depth_net / occ_head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS


@NECKS.register_module()
class SemanticContinuityLoss(nn.Module):
    """Confidence-weighted TV smoothness on continuous semantic classes.

    Args:
        num_classes: number of 2D/BEV semantic classes (C).
        continuous_class_ids: class ids treated as spatially continuous.
        occ_num_classes: number of occupancy classes (for the optional occ TV).
        edge_aware: reserved for future image-edge-aware weighting.
        eps: numerical stability epsilon.
    """

    def __init__(self,
                 num_classes=17,
                 continuous_class_ids=(11, 12, 13, 14, 15, 16),
                 occ_num_classes=18,
                 edge_aware=False,
                 eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.occ_num_classes = occ_num_classes
        self.edge_aware = edge_aware
        self.eps = eps
        self.register_buffer(
            'continuous_ids',
            torch.tensor(list(continuous_class_ids), dtype=torch.long))

    @staticmethod
    def _weighted_tv(field, weight, eps):
        """Confidence-weighted mean total variation of a probability field.

        Args:
            field: (..., K, d0, d1) class probabilities for K continuous classes,
                spatial dims are the last two.
            weight: (..., 1, d0, d1) detached per-cell confidence.
            eps: stability epsilon.

        Returns:
            scalar weighted TV.
        """
        K = field.shape[-3]
        # field: (..., K, d0, d1); weight: (..., 1, d0, d1)
        dx = field[..., 1:] - field[..., :-1]        # diff along d1 -> (...,K,d0,d1-1)
        dy = field[..., 1:, :] - field[..., :-1, :]  # diff along d0 -> (...,K,d0-1,d1)
        wx = torch.min(weight[..., :-1], weight[..., 1:]).detach()          # (...,1,d0,d1-1)
        wy = torch.min(weight[..., :-1, :], weight[..., 1:, :]).detach()    # (...,1,d0-1,d1)
        num = (wx * dx.abs()).sum() + (wy * dy.abs()).sum()
        den = wx.sum() * K + wy.sum() * K + eps
        return num / den

    def forward(self,
                seg_logits_key=None,
                semantic_bev=None,
                visibility=None,
                occ_logits=None,
                mask_camera=None):
        """Compute continuity losses for whichever tensors are provided.

        Args:
            seg_logits_key: (B*N, C, H, W) key-frame 2D semantic logits.
            semantic_bev: (B, C, Dy, Dx) BEV semantic probabilities.
            visibility: (B, 1, Dy, Dx) BEV visibility / depth mass.
            occ_logits: (B, Dx, Dy, Dz, C_occ) occupancy logits (optional).
            mask_camera: (B, Dx, Dy, Dz) camera-visible mask (optional).

        Returns:
            dict with unweighted scalars: loss_sem_continuity_2d,
            loss_sem_continuity_bev, loss_sem_continuity_occ.
        """
        out = {}
        cont = self.continuous_ids

        # ---- 2D semantic continuity ----
        if seg_logits_key is not None:
            q = torch.softmax(seg_logits_key.float(), dim=1)
            conf = q[:, cont].sum(dim=1, keepdim=True)  # (BN,1,H,W)
            q_cont = q[:, cont].unsqueeze(1)  # (BN,1,K,H,W)
            out['loss_sem_continuity_2d'] = self._weighted_tv(
                q_cont, conf, self.eps)
        else:
            out['loss_sem_continuity_2d'] = None

        # ---- BEV semantic continuity ----
        if semantic_bev is not None:
            s = semantic_bev.float()
            s_cont = s[:, cont]  # (B,K,Dy,Dx)
            conf = s[:, cont].sum(dim=1, keepdim=True)  # (B,1,Dy,Dx)
            if visibility is not None:
                conf = conf * visibility.float()
            out['loss_sem_continuity_bev'] = self._weighted_tv(
                s_cont.unsqueeze(1), conf, self.eps)
        else:
            out['loss_sem_continuity_bev'] = None

        # ---- 3D occupancy continuity (optional) ----
        if occ_logits is not None:
            po = torch.softmax(occ_logits.float(), dim=-1)  # (B,Dx,Dy,Dz,C_occ)
            cont_occ = cont[cont < po.shape[-1]]
            po_cont = po[..., cont_occ]  # (B,Dx,Dy,Dz,K)
            conf = po[..., cont_occ].sum(dim=-1, keepdim=True)  # (B,Dx,Dy,Dz,1)
            if mask_camera is not None:
                conf = conf * mask_camera.float().unsqueeze(-1)
            # TV over Dx (dim1) and Dy (dim2); move K to a leading-ish axis.
            # Rearrange to (..., K, d0, d1) with d0=Dy, d1=Dx per _weighted_tv.
            field = po_cont.permute(0, 3, 4, 2, 1)  # (B,Dz,K,Dy,Dx)
            w = conf.permute(0, 3, 4, 2, 1)         # (B,Dz,1,Dy,Dx)
            out['loss_sem_continuity_occ'] = self._weighted_tv(
                field, w, self.eps)
        else:
            out['loss_sem_continuity_occ'] = None

        return out
