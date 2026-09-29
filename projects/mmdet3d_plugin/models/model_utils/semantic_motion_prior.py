"""Semantic Motion Prior (SMP).

Constrains the SCMF motion-field learning space with semantic category priors:
each semantic group (rigid vehicle / rigid small / non-rigid pedestrian /
static background) owns a learned 2x2 motion basis, mixed per-pixel by class
probability and applied to the raw flow. Also provides the regularization
losses that were missing from the PriorOcc-4D design (L_motion_reg).

Identity at init: group bases are initialized to I and per-class probabilities
are normalized over groups, so the transform is exactly the identity and does
not perturb the audited SCMF baseline until training moves the bases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS

DEFAULT_CLASS_GROUPS = dict(
    rigid_vehicle=[0, 1, 2, 3, 4],
    rigid_small=[5, 6, 7, 9],
    nonrigid_ped=[8],
    static=[10, 11, 12, 13, 14, 15, 16],
)


@NECKS.register_module()
class SemanticMotionPrior(nn.Module):
    """Per-class motion-basis constraint + motion regularization losses.

    Args:
        num_semantic_classes: number of semantic classes (C).
        num_future: number of future timesteps (T) in the flow tensor.
        class_groups: dict mapping group name -> list of class ids.
        static_group: name of the static group (flow ~ 0).
        rigid_groups: names of rigid groups (spatially smooth flow).
        nonrigid_groups: names of non-rigid groups (bounded magnitude).
        basis_mode: 'per_group' (independent 2x2 per group) or 'shared_modes'
            (K shared 2x2 bases combined per group by learned weights).
        num_modes: K, number of shared bases when basis_mode='shared_modes'.
        ped_max_flow_cells: tau, magnitude bound for non-rigid groups.
        residual: if True, blend phi_out = (1-a)*phi + a*transform with a
            learned gate; if False, phi_out = transform.
        eps: numerical stability epsilon.
    """

    def __init__(self,
                 num_semantic_classes=17,
                 num_future=3,
                 class_groups=None,
                 static_group='static',
                 rigid_groups=('rigid_vehicle', 'rigid_small'),
                 nonrigid_groups=('nonrigid_ped',),
                 basis_mode='per_group',
                 num_modes=4,
                 ped_max_flow_cells=2.0,
                 residual=True,
                 eps=1e-6):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.num_future = num_future
        self.basis_mode = basis_mode
        self.num_modes = num_modes
        self.ped_max_flow_cells = ped_max_flow_cells
        self.residual = residual
        self.eps = eps

        if class_groups is None:
            class_groups = DEFAULT_CLASS_GROUPS
        self.group_names = list(class_groups.keys())
        self.num_groups = len(self.group_names)
        self.static_group = static_group
        self.rigid_groups = list(rigid_groups)
        self.nonrigid_groups = list(nonrigid_groups)

        # membership (G, C): 1 if class c belongs to group g.
        membership = torch.zeros(self.num_groups, num_semantic_classes)
        for g, name in enumerate(self.group_names):
            for c in class_groups[name]:
                membership[g, c] = 1.0
        self.register_buffer('membership', membership)

        # Id lists used by the regularization losses (as buffers of class ids).
        self.register_buffer(
            'static_ids',
            torch.tensor(class_groups[static_group], dtype=torch.long))
        rigid_ids = [c for n in rigid_groups for c in class_groups[n]]
        nonrigid_ids = [c for n in nonrigid_groups for c in class_groups[n]]
        self.register_buffer('rigid_ids', torch.tensor(rigid_ids, dtype=torch.long))
        self.register_buffer(
            'nonrigid_ids', torch.tensor(nonrigid_ids, dtype=torch.long))

        # Learned motion bases.
        eye = torch.eye(2)
        if basis_mode == 'per_group':
            self.group_transform = nn.Parameter(eye.expand(self.num_groups, 2, 2).clone())
        elif basis_mode == 'shared_modes':
            self.basis = nn.Parameter(eye.expand(num_modes, 2, 2).clone())
            self.group_mode_logits = nn.Parameter(
                torch.zeros(self.num_groups, num_modes))
        else:
            raise ValueError(f'unknown basis_mode: {basis_mode}')

        # Residual gate logit; sigmoid(0)=0.5, but transform is identity at
        # init so the blend is exactly phi regardless of alpha.
        self.alpha_logit = nn.Parameter(torch.zeros(1))

    def _group_bases(self):
        """Return per-group 2x2 bases A of shape (G, 2, 2)."""
        if self.basis_mode == 'per_group':
            return self.group_transform
        w = torch.softmax(self.group_mode_logits, dim=1)  # (G, K)
        return torch.einsum('gk,kij->gij', w, self.basis)

    def _group_probs(self, per_cls_masks):
        """Raw (un-normalized) per-group probability maps (B, G, H, W)."""
        return torch.einsum('bchw,gc->bghw', per_cls_masks, self.membership)

    def forward(self, flow, per_cls_masks):
        """Apply the semantic motion prior to the flow field.

        Args:
            flow: (B, T, 2, H, W) raw SCMF flow in BEV cells.
            per_cls_masks: (B, C, H, W) per-class semantic probabilities.

        Returns:
            flow_out: (B, T, 2, H, W) constrained flow.
        """
        B, T, two, H, W = flow.shape
        assert two == 2, f'flow must have 2 components, got {two}'
        assert per_cls_masks.shape[1] == self.num_semantic_classes, \
            'per_cls_masks channels != num_semantic_classes'

        A = self._group_bases()  # (G, 2, 2)
        p = self._group_probs(per_cls_masks)  # (B, G, H, W)
        p = p / (p.sum(dim=1, keepdim=True) + self.eps)  # normalize -> identity at init

        # Per-pixel mixed basis A_mix (B, 2, 2, H, W).
        A_mix = torch.einsum('bghw,gij->bijhw', p, A)

        # transform[b,t,i,h,w] = sum_j A_mix[b,i,j,h,w] * flow[b,t,j,h,w]
        transform = (A_mix.unsqueeze(1) * flow.unsqueeze(2)).sum(dim=3)

        if not self.residual:
            return transform
        alpha = torch.sigmoid(self.alpha_logit)
        return (1 - alpha) * flow + alpha * transform

    def regularization_losses(self, flow, per_cls_masks):
        """Motion regularization losses (unweighted scalars).

        Args:
            flow: (B, T, 2, H, W) effective (post-prior) flow.
            per_cls_masks: (B, C, H, W) per-class semantic probabilities.

        Returns:
            dict with keys 'static_flow', 'rigid_smooth', 'nonrigid_bound'.
        """
        B, T = flow.shape[0], flow.shape[1]
        eps = self.eps

        p_sta = per_cls_masks[:, self.static_ids].sum(dim=1, keepdim=True)  # (B,1,H,W)
        p_rig = per_cls_masks[:, self.rigid_ids].sum(dim=1, keepdim=True)
        p_ped = per_cls_masks[:, self.nonrigid_ids].sum(dim=1, keepdim=True)

        # (1) Static regions should not move: fills the missing L_motion_reg.
        abs_flow = flow.abs().sum(dim=2)  # (B, T, H, W)
        static_num = (p_sta * abs_flow).sum()
        static_den = p_sta.sum() * T + eps
        loss_static = static_num / static_den

        # (2) Rigid bodies move coherently: confidence-weighted TV on flow.
        dx = flow[..., :, 1:] - flow[..., :, :-1]  # (B,T,2,H,W-1) diff along W
        dy = flow[..., 1:, :] - flow[..., :-1, :]  # (B,T,2,H-1,W) diff along H
        wx = torch.min(p_rig[..., :-1], p_rig[..., 1:])  # (B,1,H,W-1)
        wy = torch.min(p_rig[..., :-1, :], p_rig[..., 1:, :])  # (B,1,H-1,W)
        rigid_num = (wx * dx.abs()).sum() + (wy * dy.abs()).sum()
        rigid_den = wx.sum() * (T * 2) + wy.sum() * (T * 2) + eps
        loss_rigid = rigid_num / rigid_den

        # (3) Non-rigid (pedestrian) magnitude is bounded by a hinge.
        mag = torch.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2 + eps)  # (B,T,H,W)
        excess = F.relu(mag - self.ped_max_flow_cells) ** 2
        ped_num = (p_ped * excess).sum()
        ped_den = p_ped.sum() * T + eps
        loss_ped = ped_num / ped_den

        return dict(
            static_flow=loss_static,
            rigid_smooth=loss_rigid,
            nonrigid_bound=loss_ped)
