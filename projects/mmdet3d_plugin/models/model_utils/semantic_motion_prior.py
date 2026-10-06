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


@NECKS.register_module()
class SemanticMultimodalMotionPrior(nn.Module):
    """K-mode semantic motion prior (multimodal SMP).

    Extends the single-modal SMP to K parallel motion modes:

    1. Per-group K learned 2x2 mode bases A_{g,k} (init I): each semantic
       group owns K motion-mode bases, so different modes correspond to
       different plausible motions (e.g. turn / straight / brake) while
       each mode still lives in the group's admissible motion subspace.
    2. Semantic mode logits: the global class distribution of the scene
       is pooled to group probabilities, which weight learnable
       per-group mode logits into per-sample mode logits (B, K) —
       "which future mode does this semantic composition imply?".
    3. Regularization losses (static/rigid/nonrigid) are applied to every
       mode so that e.g. static regions do not move under ANY mode.
    4. A diversity loss (negative mean pairwise L1 between mode flows)
       fights mode collapse: minimizing it pushes modes apart.

    Identity at init: A_{g,k} = I for all k, so flow_out == flow for every
    mode; group_mode_logits = 0, so initial mode probabilities are uniform.

    Args:
        num_semantic_classes: number of semantic classes (C).
        num_future: number of future timesteps (T).
        num_modes: number of motion modes (K).
        class_groups: dict mapping group name -> list of class ids.
        static_group: name of the static group (flow ~ 0 in all modes).
        rigid_groups: names of rigid groups (spatially smooth flow).
        nonrigid_groups: names of non-rigid groups (bounded magnitude).
        ped_max_flow_cells: tau, magnitude bound for non-rigid groups.
        residual: blend phi_out = (1-a)*phi + a*transform with learned a.
        eps: numerical stability epsilon.
    """

    def __init__(self,
                 num_semantic_classes=17,
                 num_future=3,
                 num_modes=3,
                 class_groups=None,
                 static_group='static',
                 rigid_groups=('rigid_vehicle', 'rigid_small'),
                 nonrigid_groups=('nonrigid_ped',),
                 ped_max_flow_cells=2.0,
                 residual=True,
                 basis_mode=None,
                 eps=1e-6):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.num_future = num_future
        self.num_modes = num_modes
        self.static_group = static_group
        self.rigid_groups = list(rigid_groups)
        self.nonrigid_groups = list(nonrigid_groups)
        self.ped_max_flow_cells = ped_max_flow_cells
        self.residual = residual
        self.eps = eps
        # basis_mode is accepted (and ignored) for config-inheritance
        # compatibility with the single-modal SemanticMotionPrior.
        self.basis_mode = basis_mode

        if class_groups is None:
            class_groups = DEFAULT_CLASS_GROUPS
        self.group_names = list(class_groups.keys())
        self.num_groups = len(self.group_names)

        membership = torch.zeros(self.num_groups, num_semantic_classes)
        for g, name in enumerate(self.group_names):
            for c in class_groups[name]:
                membership[g, c] = 1.0
        self.register_buffer('membership', membership)

        self.register_buffer(
            'static_ids',
            torch.tensor(class_groups[static_group], dtype=torch.long))
        rigid_ids = [c for n in rigid_groups for c in class_groups[n]]
        nonrigid_ids = [c for n in nonrigid_groups for c in class_groups[n]]
        self.register_buffer(
            'rigid_ids', torch.tensor(rigid_ids, dtype=torch.long))
        self.register_buffer(
            'nonrigid_ids', torch.tensor(nonrigid_ids, dtype=torch.long))

        # (G, K, 2, 2) mode bases, identity at init.
        eye = torch.eye(2)
        self.group_modes = nn.Parameter(
            eye.expand(self.num_groups, num_modes, 2, 2).clone())
        # (G, K) per-group mode logits, zero at init -> uniform mode probs.
        self.group_mode_logits = nn.Parameter(
            torch.zeros(self.num_groups, num_modes))
        self.alpha_logit = nn.Parameter(torch.zeros(1))

    def _group_probs(self, per_cls_masks):
        """Raw (un-normalized) per-group probability maps (B, G, H, W)."""
        return torch.einsum('bchw,gc->bghw', per_cls_masks, self.membership)

    def forward(self, flow, per_cls_masks):
        """Apply per-mode semantic motion bases to a multimodal flow.

        Args:
            flow: (B, K, T, 2, H, W) raw multimodal SCMF flow in BEV cells.
            per_cls_masks: (B, C, H, W) per-class semantic probabilities.

        Returns:
            flow_out: (B, K, T, 2, H, W) constrained per-mode flow.
        """
        assert flow.dim() == 6, \
            f'multimodal SMP expects 6-dim flow, got {flow.dim()}'
        B, K, T, two, H, W = flow.shape
        assert two == 2
        assert K == self.num_modes, \
            f'flow modes {K} != num_modes {self.num_modes}'
        assert per_cls_masks.shape[1] == self.num_semantic_classes

        A = self.group_modes  # (G, K, 2, 2)
        p = self._group_probs(per_cls_masks)  # (B, G, H, W)
        p = p / (p.sum(dim=1, keepdim=True) + self.eps)

        # A_mix[b,k,h,w,i,j] = sum_g p[b,g,h,w] * A[g,k,i,j]
        A_mix = torch.einsum('bghw,gkij->bkhwij', p, A)  # (B,K,H,W,2,2)
        # transform[b,k,t,i,h,w] = sum_j A_mix[b,k,h,w,i,j] * flow[b,k,t,j,h,w]
        transform = torch.einsum(
            'bkhwij,bktjhw->bktihw', A_mix, flow)  # (B,K,T,2,H,W)

        if not self.residual:
            return transform
        alpha = torch.sigmoid(self.alpha_logit)
        return (1 - alpha) * flow + alpha * transform

    def mode_logits(self, per_cls_masks):
        """Semantic mode logits: which mode does this scene imply?

        Global class distribution -> group probabilities -> weighted
        per-group mode logits -> per-sample mode logits.

        Args:
            per_cls_masks: (B, C, H, W) per-class semantic probabilities.

        Returns:
            logits: (B, K) mode logits (softmax over dim=1 -> mode probs).
        """
        cls_prob = per_cls_masks.mean(dim=(2, 3))  # (B, C)
        p_g = torch.einsum('bc,gc->bg', cls_prob, self.membership)  # (B, G)
        p_g = p_g / (p_g.sum(dim=1, keepdim=True) + self.eps)
        logits = torch.einsum('bg,gk->bk', p_g, self.group_mode_logits)
        return logits  # (B, K)

    def regularization_losses(self, flow, per_cls_masks):
        """Motion regularization losses applied to every mode.

        The single-modal formulas (static ~ 0 / rigid smooth TV / pedestrian
        hinge bound) are evaluated per mode and averaged over modes, so the
        semantic motion-space constraint holds under every future mode.

        Args:
            flow: (B, K, T, 2, H, W) effective (post-prior) multimodal flow.
            per_cls_masks: (B, C, H, W) per-class semantic probabilities.

        Returns:
            dict with keys 'static_flow', 'rigid_smooth', 'nonrigid_bound'.
        """
        assert flow.dim() == 6
        B, K, T = flow.shape[0], flow.shape[1], flow.shape[2]
        eps = self.eps

        p_sta = per_cls_masks[:, self.static_ids].sum(
            dim=1, keepdim=True)  # (B,1,H,W)
        p_rig = per_cls_masks[:, self.rigid_ids].sum(dim=1, keepdim=True)
        p_ped = per_cls_masks[:, self.nonrigid_ids].sum(dim=1, keepdim=True)

        # (1) Static regions should not move under ANY mode.
        abs_flow = flow.abs().sum(dim=3)  # (B, K, T, H, W)
        static_num = (p_sta.unsqueeze(1) * abs_flow).sum()
        static_den = p_sta.sum() * K * T + eps
        loss_static = static_num / static_den

        # (2) Rigid bodies move coherently in every mode.
        dx = flow[..., :, 1:] - flow[..., :, :-1]  # (B,K,T,2,H,W-1)
        dy = flow[..., 1:, :] - flow[..., :-1, :]  # (B,K,T,2,H-1,W)
        wx = torch.min(p_rig[..., :-1], p_rig[..., 1:])  # (B,1,H,W-1)
        wy = torch.min(p_rig[..., :-1, :], p_rig[..., 1:, :])  # (B,1,H-1,W)
        rigid_num = (wx.unsqueeze(1).unsqueeze(1) * dx.abs()).sum() + \
            (wy.unsqueeze(1).unsqueeze(1) * dy.abs()).sum()
        rigid_den = wx.sum() * (K * T * 2) + wy.sum() * (K * T * 2) + eps
        loss_rigid = rigid_num / rigid_den

        # (3) Non-rigid (pedestrian) magnitude bounded in every mode.
        mag = torch.sqrt(
            flow[..., 0, :, :] ** 2 + flow[..., 1, :, :] ** 2 + eps)
        excess = F.relu(mag - self.ped_max_flow_cells) ** 2
        ped_num = (p_ped.unsqueeze(1).unsqueeze(1) * excess).sum()
        ped_den = p_ped.sum() * K * T + eps
        loss_ped = ped_num / ped_den

        return dict(
            static_flow=loss_static,
            rigid_smooth=loss_rigid,
            nonrigid_bound=loss_ped)

    def diversity_loss(self, flow):
        """Negative mean pairwise L1 between mode flows (anti-collapse).

        Minimizing this loss maximizes the separation between modes. At init
        the symmetry-breaking mode biases of the multimodal SCMF make this
        term non-zero, so it also provides an initial push against collapse.

        Args:
            flow: (B, K, T, 2, H, W) multimodal flow.

        Returns:
            scalar loss (<= 0; more negative = more diverse modes).
        """
        assert flow.dim() == 6
        K = flow.shape[1]
        if K < 2:
            return flow.sum() * 0.0
        diffs = []
        for i in range(K):
            for j in range(i + 1, K):
                diffs.append((flow[:, i] - flow[:, j]).abs().mean())
        return -torch.stack(diffs).mean()
