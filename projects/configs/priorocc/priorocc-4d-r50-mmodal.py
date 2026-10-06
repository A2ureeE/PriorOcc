# PriorOcc-4D + Semantic-guided Multimodal Occupancy Forecasting.
#
# Inherits the full innovation stack (SMP + hole-filling + SDP) and replaces
# the deterministic single-flow pipeline with a K-mode one:
#   1. MultimodalSemanticConditionedMotionField: K parallel flow fields
#      (B, K, T, 2, H, W) with symmetry-breaking near-identity init.
#   2. SemanticMultimodalMotionPrior: per-group K motion-mode bases +
#      semantic mode logits (which mode does this scene imply?) +
#      per-mode regularizers + anti-collapse diversity loss.
#   3. MultimodalSCMFEnhancedPredictor: K auto-regressive chains with
#      shared ConvGRU weights -> future_bevs (B, K, T, C, H, W).
#   4. WTA training in PriorOcc4D._multimodal_future_losses:
#      winner occ CE (keys stay loss_occ_future_{t+1}s) + mode-classification
#      CE (loss_mode_cls) + diversity (loss_mode_div).
#   5. Evaluation reports deployed (semantic mode selection) mIoU plus
#      best-of-K oracle mIoU and mode_selection_acc.
#
# Ablations (--cfg-options):
#   model.enable_multimodal=False                       # deterministic path
#   model.num_modes=1                                   # K=1 sanity check
#   model.mode_cls_loss_weight=0.0                      # no mode supervision
#   model.mode_div_loss_weight=0.0                      # no anti-collapse
#   model.num_modes=5                                   # mode-count sweep

_base_ = ['./priorocc-4d-r50-smp-sdp-holefill.py']

num_modes = 3

model = dict(
    type='PriorOcc4D',
    enable_multimodal=True,
    num_modes=num_modes,
    mode_cls_loss_weight=0.5,
    mode_div_loss_weight=0.05,

    # --- K-mode SCMF: flow (B, K, T, 2, H, W) ---
    scmf=dict(
        type='MultimodalSemanticConditionedMotionField',
        bev_channels=256,
        num_semantic_classes=17,
        num_future=3,
        num_modes=num_modes,
        max_flow_cells=5.0,
        hidden_dim=256,
        mode_bias_cells=0.05,
    ),

    # --- K auto-regressive chains (shared ConvGRU weights) ---
    future_predictor=dict(
        type='MultimodalSCMFEnhancedPredictor',
        bev_channels=256,
        num_future=3,
        num_modes=num_modes,
        hidden_dim=256,
        use_warp_validity=True,
    ),

    # --- K-mode semantic motion prior ---
    motion_prior=dict(
        type='SemanticMultimodalMotionPrior',
        num_semantic_classes=17,
        num_future=3,
        num_modes=num_modes,
        basis_mode=None,  # unused by the multimodal prior; kept for config
        ped_max_flow_cells=2.0,
    ),
)
