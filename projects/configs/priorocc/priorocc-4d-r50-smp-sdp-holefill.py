# PriorOcc-4D + semantic-prior innovations.
#
# Inherits the audited baseline (priorocc-4d-r50-stgdm-scmf.py) and turns on:
#   1. SemanticMotionPrior (SMP): per-class motion basis + reg losses
#      (loss_static_flow / loss_rigid_smooth / loss_nonrigid_bound).
#   2. Semantic continuity hole-filling: warp-validity gate in the future
#      predictor + SemanticContinuityLoss (2D sem + BEV).
#   3. SGDM upgrade: SemanticDepthPrior (SDP) modulating depth logits.
#
# Every innovation is independently toggleable, e.g.:
#   --cfg-options model.enable_motion_prior=False
#   --cfg-options model.enable_sem_continuity=False
#   --cfg-options model.future_predictor.use_warp_validity=False
#   --cfg-options model.img_view_transformer.depthnet_cfg.use_semantic_depth_prior=False

_base_ = ['./priorocc-4d-r50-stgdm-scmf.py']

model = dict(
    # --- 1. Semantic Motion Prior (rigid / non-rigid motion-space constraint) ---
    enable_motion_prior=True,
    motion_prior=dict(
        type='SemanticMotionPrior',
        num_semantic_classes=17,
        num_future=3,
        basis_mode='per_group',
        ped_max_flow_cells=2.0,
    ),
    motion_prior_loss_weights=dict(
        static_flow=0.05, rigid_smooth=0.02, nonrigid_bound=0.02),

    # --- 2a. Semantic continuity (background hole-filling) ---
    enable_sem_continuity=True,
    sem_continuity=dict(
        type='SemanticContinuityLoss',
        num_classes=17,
        continuous_class_ids=[11, 12, 13, 14, 15, 16],
    ),
    sem_continuity_loss_weights=dict(cont_2d=0.05, cont_bev=0.05, cont_occ=0.0),
    continuity_apply_occ=False,

    # --- 2b. Warp-validity gate (holes fall back to the temporal residual) ---
    future_predictor=dict(
        type='SCMFEnhancedPredictor',
        bev_channels=256,
        num_future=3,
        hidden_dim=256,
        use_warp_validity=True,
    ),

    # --- 3. SGDM upgrade: Semantic Depth Prior ---
    img_view_transformer=dict(
        depthnet_cfg=dict(
            use_semantic_gating=True,
            use_bidirectional_sgdm=False,
            sem_channels=17,
            sgdm_reduction=4,
            depth_feedback_weight=0.3,
            use_semantic_depth_prior=True,
            sdp_weight=1.0,
            sdp_temperature=1.0,
        ),
    ),
)
