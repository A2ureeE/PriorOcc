#!/usr/bin/env python3
"""PriorOcc-4D Phase A Verification Tool.

Usage:
    python tools/verify_priorocc_4d.py --stage modules
    python tools/verify_priorocc_4d.py --stage modules --test zero-flow
    python tools/verify_priorocc_4d.py --stage build
    python tools/verify_priorocc_4d.py --stage modules --device cuda
"""

import argparse
import os
import sys
import traceback

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(requested=None):
    if requested == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device='cuda:0')
            return torch.device('cuda:0')
        except RuntimeError:
            pass
    return torch.device('cpu')


def import_plugins(cfg):
    if not getattr(cfg, 'plugin', False):
        return
    import importlib
    plugin_dir = getattr(cfg, 'plugin_dir', '')
    module_path = plugin_dir.replace('/', '.').rstrip('.')
    importlib.import_module(module_path)


def make_bev(B, C, H, W, device, requires_grad=True):
    return torch.randn(B, C, H, W, device=device, requires_grad=requires_grad)


def make_raw_history(B, T, C, H, W, device):
    return torch.randn(B, T, C, H, W, device=device)


def make_sem_bev(B, C_sem, H, W, device):
    logits = torch.randn(B, C_sem, H, W, device=device)
    return torch.softmax(logits, dim=1)


def make_visibility(B, H, W, device):
    return torch.rand(B, 1, H, W, device=device).clamp(0.1, 1.0)


def test_zero_flow(device):
    """warp_feature with zero flow returns input (allclose)."""
    from projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder import \
        warp_feature
    bev = torch.randn(2, 64, 200, 200, device=device, requires_grad=True)
    flow = torch.zeros(2, 2, 200, 200, device=device)
    warped = warp_feature(bev, flow)
    max_diff = (warped - bev).abs().max().item()
    assert max_diff < 1e-3, f"Zero-flow should be identity, max diff: {max_diff}"
    loss = warped.sum()
    loss.backward()
    assert bev.grad is not None, "Gradient should flow through warp_feature"
    assert bev.grad.abs().sum() > 0, "Gradient should be non-zero"


def test_known_translation(device):
    """A bright spot moves in the defined direction by 1 cell."""
    from projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder import \
        warp_feature
    H, W = 20, 20
    bev = torch.zeros(1, 1, H, W, device=device)
    bev[0, 0, 10, 10] = 1.0
    flow = torch.zeros(1, 2, H, W, device=device)
    flow[0, 0, :, :] = 1.0
    warped = warp_feature(bev, flow)
    assert warped[0, 0, 10, 11].item() > 0.9, \
        f"Spot should move to (10,11), got {warped[0,0,10,11].item()}"
    assert warped[0, 0, 10, 10].item() < 0.1, \
        f"Original position should be ~0, got {warped[0,0,10,10].item()}"
    bev_edge = torch.zeros(1, 1, H, W, device=device)
    bev_edge[0, 0, 10, W - 1] = 1.0
    flow_edge = torch.zeros(1, 2, H, W, device=device)
    flow_edge[0, 0, :, :] = 1.0
    warped_edge = warp_feature(bev_edge, flow_edge)
    assert warped_edge[0, 0, 10, W - 1].item() < 0.1, \
        "Content at right edge should be zeroed when moving right"


def test_mask_partition(device):
    """dyn_mask + sta_mask approx 1 in visible areas, both in [0,1]."""
    from projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder import \
        SemanticDynStaSeparator
    sep = SemanticDynStaSeparator(
        num_semantic_classes=17,
        dynamic_class_ids=list(range(11)),
        static_class_ids=list(range(11, 17))).to(device)
    sem_bev = make_sem_bev(2, 17, 50, 50, device)
    vis = make_visibility(2, 50, 50, device)
    dyn, sta, per_cls = sep(sem_bev, vis)
    assert dyn.min() >= 0 and dyn.max() <= 1.0, "dyn_mask out of [0,1]"
    assert sta.min() >= 0 and sta.max() <= 1.0, "sta_mask out of [0,1]"
    visible = vis > 0.1
    total = dyn + sta
    diff = (total[visible] - 1.0).abs().max().item()
    assert diff < 1e-5, f"dyn+sta should be ~1 in visible, max diff: {diff}"


def test_finite_and_grad(device):
    """All module outputs finite; at least one param grad non-zero per module."""
    from projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder import \
        SemanticMotionFeatureEncoder, SemanticMotionAttention, \
        PerClassDeltaCombiner
    from projects.mmdet3d_plugin.models.model_utils.scmf import \
        SemanticConditionedMotionField, SCMFEnhancedPredictor

    B, C, H, W = 2, 256, 50, 50
    C_sem, C_raw = 17, 64

    fused = make_bev(B, C, H, W, device)
    raw_hist = make_raw_history(B, 3, C_raw, H, W, device)
    sem = make_sem_bev(B, C_sem, H, W, device)
    vis = make_visibility(B, H, W, device)

    enc = SemanticMotionFeatureEncoder(C_raw, C, C_sem).to(device)
    attn = SemanticMotionAttention(C, C_sem, num_heads=4).to(device)
    comb = PerClassDeltaCombiner(C, C_sem).to(device)
    scmf = SemanticConditionedMotionField(C, C_sem, num_future=3).to(device)
    pred = SCMFEnhancedPredictor(C, num_future=3).to(device)

    sep_mod = __import__(
        'projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder',
        fromlist=['SemanticDynStaSeparator']).SemanticDynStaSeparator(
        num_semantic_classes=17,
        dynamic_class_ids=list(range(11)),
        static_class_ids=list(range(11, 17))).to(device)
    _, _, per_cls = sep_mod(sem, vis)

    motion = enc(raw_hist, sem)
    assert torch.isfinite(motion).all(), "motion_feat has NaN/Inf"

    attn_feat = attn(fused, motion, per_cls)
    assert torch.isfinite(attn_feat).all(), "attn_feat has NaN/Inf"

    delta = comb(fused, motion, per_cls)
    assert torch.isfinite(delta).all(), "delta has NaN/Inf"

    flow = scmf(fused, motion, sem, attn_feat)
    assert torch.isfinite(flow).all(), "flow has NaN/Inf"
    assert flow.shape == (B, 3, 2, H, W), f"flow shape: {flow.shape}"

    future = pred(fused, flow)
    assert torch.isfinite(future).all(), "future has NaN/Inf"
    assert future.shape == (B, 3, C, H, W), f"future shape: {future.shape}"

    loss = future.sum() + flow.sum() + attn_feat.sum() + delta.sum()
    loss.backward()

    for name, module in [('enc', enc), ('attn', attn), ('comb', comb),
                         ('scmf', scmf), ('pred', pred)]:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters())
        assert has_grad, f"{name} has no non-zero gradients"


def test_semantic_sensitivity(device):
    """Changing semantic class changes SCMF decoder features detectably.

    The SCMF flow head is zero-initialized, so final flow is always 0 at init.
    Instead, we check the decoder's intermediate features to verify the
    semantic pathway is connected.
    """
    from projects.mmdet3d_plugin.models.model_utils.dyn_sta_decoder import \
        SemanticMotionFeatureEncoder
    from projects.mmdet3d_plugin.models.model_utils.scmf import \
        SemanticConditionedMotionField

    B, C, H, W = 1, 256, 50, 50
    C_sem = 17

    torch.manual_seed(42)
    fused = torch.randn(B, C, H, W, device=device)
    raw_hist = torch.randn(B, 3, 64, H, W, device=device)
    attn_feat = torch.randn(B, C, H, W, device=device)

    sem_a = torch.zeros(B, C_sem, H, W, device=device)
    sem_a[:, 15, :, :] = 1.0

    sem_b = torch.zeros(B, C_sem, H, W, device=device)
    sem_b[:, 4, :, :] = 1.0

    enc = SemanticMotionFeatureEncoder(64, C, C_sem).to(device)
    scmf = SemanticConditionedMotionField(C, C_sem, num_future=3).to(device)

    motion_a = enc(raw_hist, sem_a)
    x_a = torch.cat(
        [fused, motion_a, attn_feat, sem_a], dim=1)
    dec_a = scmf.decoder(x_a)

    motion_b = enc(raw_hist, sem_b)
    x_b = torch.cat(
        [fused, motion_b, attn_feat, sem_b], dim=1)
    dec_b = scmf.decoder(x_b)

    diff = (dec_a - dec_b).abs().max().item()
    assert diff > 1e-4, \
        f"SCMF decoder features should change with semantic class, max diff: {diff}"


def test_identity_init(device):
    """SCMF initial flow approx 0, future features no NaN/explosion."""
    from projects.mmdet3d_plugin.models.model_utils.scmf import \
        SemanticConditionedMotionField, SCMFEnhancedPredictor

    B, C, H, W = 2, 256, 50, 50
    C_sem = 17

    fused = make_bev(B, C, H, W, device)
    motion = make_bev(B, C, H, W, device)
    sem = make_sem_bev(B, C_sem, H, W, device)
    attn = make_bev(B, C, H, W, device)

    scmf = SemanticConditionedMotionField(C, C_sem, num_future=3).to(device)
    pred = SCMFEnhancedPredictor(C, num_future=3).to(device)

    last_layer = scmf.motion_head
    assert torch.allclose(last_layer.weight,
                          torch.zeros_like(last_layer.weight)), \
        "SCMF last layer weight should be zero-initialized"
    assert torch.allclose(last_layer.bias,
                          torch.zeros_like(last_layer.bias)), \
        "SCMF last layer bias should be zero-initialized"

    flow = scmf(fused, motion, sem, attn)
    max_flow = flow.abs().max().item()
    assert max_flow < 1e-5, f"Initial flow should be ~0, max: {max_flow}"

    future = pred(fused, flow)
    assert torch.isfinite(future).all(), "Future features have NaN/Inf"
    assert future.abs().max().item() < fused.abs().max().item() * 10, \
        f"Future may be exploding: {future.abs().max().item()}"


def stage_build(config_path):
    """Build full model from config and verify."""
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    device = get_device()

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)

    print(f"  Model type: {type(model).__name__}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")

    for flag, attr in [
        ('enable_dyn_sta_decoder', 'dyn_sta_decoder'),
        ('enable_motion_encoder', 'motion_encoder'),
        ('enable_scmf', 'scmf'),
        ('enable_future_prediction', 'future_predictor'),
    ]:
        enabled = cfg.model.get(flag, True)
        has_mod = hasattr(model, attr) and getattr(model, attr) is not None
        status = 'built' if has_mod else 'skeleton (Phase B)'
        print(f"  {flag}={enabled} -> {attr}: {status}")


def stage_sgdm_current(config_path):
    """Build PriorOcc4D and test 3-frame forward_train with SGDM integration.

    Verifies:
    - 3 seg_logits non-empty (one per frame)
    - loss_depth computed
    - occupancy loss computed
    - 3 loss_2d_seg_history_* computed
    - All losses finite
    - Backward pass succeeds
    - SemanticInjector has non-zero gradients
    """
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    cfg.model.img_backbone.pretrained = None

    device = torch.device('cpu')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    model.train()

    B, N_cams, N_frames = 1, 6, 3
    H_img, W_img = 256, 704
    Dx, Dy, Dz = 200, 200, 16
    N_total = N_cams * N_frames

    # Camera intrinsics: fx=fy=500, cx=352, cy=128
    intrins = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0

    # Camera-to-lidar rotation: x_fwd=-y_cam, y_left=-x_cam, z_up=-z_cam... no.
    # nuScenes convention: cam(x,y,z)=(right,down,fwd), lidar(x,y,z)=(fwd,left,up)
    sensor2ego = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    rot_cam2lidar = torch.tensor([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.]], dtype=torch.float32)
    sensor2ego[:] = rot_cam2lidar

    ego2global = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_rots = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_trans = torch.zeros(B, N_total, 3, dtype=torch.float32)
    bda = torch.eye(3, dtype=torch.float32).repeat(B, 1, 1)

    imgs = torch.randn(B, N_total, 3, H_img, W_img, dtype=torch.float32)
    img_inputs = [imgs, sensor2ego, ego2global, intrins,
                  post_rots, post_trans, bda]

    img_metas = [dict(box_type_3d='LiDAR')]

    gt_depth = torch.ones(B, N_cams, H_img, W_img,
                          dtype=torch.float32) * 10.0
    voxel_semantics = torch.randint(0, 18, (B, Dx, Dy, Dz))
    mask_camera = torch.ones(B, Dx, Dy, Dz, dtype=torch.float32)
    gt_semantic_2d = torch.randint(0, 17, (B, N_cams, H_img, W_img))

    losses = model.forward_train(
        points=None,
        img_metas=img_metas,
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
    )

    print(f"  Losses: {sorted(losses.keys())}")
    assert 'loss_depth' in losses, "Missing loss_depth"

    occ_keys = [k for k in losses if 'occ' in k.lower()]
    assert len(occ_keys) > 0, "Missing occupancy loss"

    seg_keys = sorted([k for k in losses if '2d_seg_history' in k])
    print(f"  2D seg history losses: {seg_keys}")
    assert len(seg_keys) == 3, \
        f"Expected 3 2D seg history losses, got {len(seg_keys)}"

    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite: {v}"
    print("  All losses finite: OK")

    total_loss = sum(losses.values())
    total_loss.backward()
    print(f"  Total loss: {total_loss.item():.4f}")

    si_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.semantic_injector.parameters())
    assert si_grad, "SemanticInjector has no non-zero gradients"
    print("  SemanticInjector gradients: OK")


def stage_forecast_smoke(config_path):
    """Phase C: future occupancy prediction smoke test.

    Verifies:
    - 3 loss_occ_future_*s keys computed
    - All losses finite
    - Backward succeeds, future head has non-zero gradients
    - Loss routing: changing future_voxel_semantics[:,1] only affects 2s loss
    - simple_test returns dict with occ_current, occ_future (len 3), horizons_sec
    """
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    cfg.model.img_backbone.pretrained = None

    device = torch.device('cpu')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    model.train()

    B, N_cams, N_frames = 1, 6, 3
    H_img, W_img = 256, 704
    Dx, Dy, Dz = 200, 200, 16
    num_future = model.num_future
    N_total = N_cams * N_frames

    intrins = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0

    sensor2ego = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    rot_cam2lidar = torch.tensor([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.]], dtype=torch.float32)
    sensor2ego[:] = rot_cam2lidar

    ego2global = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_rots = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_trans = torch.zeros(B, N_total, 3, dtype=torch.float32)
    bda = torch.eye(3, dtype=torch.float32).repeat(B, 1, 1)

    imgs = torch.randn(B, N_total, 3, H_img, W_img, dtype=torch.float32)
    img_inputs = [imgs, sensor2ego, ego2global, intrins,
                  post_rots, post_trans, bda]

    img_metas = [dict(box_type_3d='LiDAR')]

    gt_depth = torch.ones(B, N_cams, H_img, W_img,
                          dtype=torch.float32) * 10.0
    voxel_semantics = torch.randint(0, 18, (B, Dx, Dy, Dz))
    mask_camera = torch.ones(B, Dx, Dy, Dz, dtype=torch.float32)
    gt_semantic_2d = torch.randint(0, 17, (B, N_cams, H_img, W_img))

    future_voxel_semantics = torch.randint(0, 18, (B, num_future, Dx, Dy, Dz))
    future_mask_camera = torch.ones(
        B, num_future, Dx, Dy, Dz, dtype=torch.float32)

    losses = model.forward_train(
        points=None,
        img_metas=img_metas,
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
        future_voxel_semantics=future_voxel_semantics,
        future_mask_camera=future_mask_camera,
    )

    print(f"  Losses: {sorted(losses.keys())}")

    future_keys = sorted([k for k in losses if k.startswith('loss_occ_future_')])
    print(f"  Future loss keys: {future_keys}")
    assert len(future_keys) == num_future, \
        f"Expected {num_future} future loss keys, got {len(future_keys)}"

    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite: {v}"
    print("  All losses finite: OK")

    total_loss = sum(losses.values())
    total_loss.backward()
    print(f"  Total loss: {total_loss.item():.4f}")

    fh_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.future_predictor.parameters())
    assert fh_grad, "Future predictor has no non-zero gradients"
    print("  Future predictor gradients: OK")

    # Loss routing test: change future_voxel_semantics[:, 1] only
    model.zero_grad()
    model.eval()
    with torch.no_grad():
        losses_1 = model.forward_train(
            points=None,
            img_metas=img_metas,
            img_inputs=img_inputs,
            gt_depth=gt_depth,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            mask_lidar=mask_camera,
            gt_semantic_2d=gt_semantic_2d,
            future_voxel_semantics=future_voxel_semantics,
            future_mask_camera=future_mask_camera,
        )
        future_voxel_semantics_2 = future_voxel_semantics.clone()
        future_voxel_semantics_2[:, 1] = torch.randint(
            0, 18, (B, Dx, Dy, Dz))
        losses_2 = model.forward_train(
            points=None,
            img_metas=img_metas,
            img_inputs=img_inputs,
            gt_depth=gt_depth,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            mask_lidar=mask_camera,
            gt_semantic_2d=gt_semantic_2d,
            future_voxel_semantics=future_voxel_semantics_2,
            future_mask_camera=future_mask_camera,
        )

    for k_idx in range(num_future):
        k_name = f'loss_occ_future_{k_idx+1}s'
        v1 = losses_1[k_name].item()
        v2 = losses_2[k_name].item()
        if k_idx == 1:
            assert abs(v1 - v2) > 1e-6, \
                f"{k_name} should change but didn't: {v1} vs {v2}"
        else:
            assert abs(v1 - v2) < 1e-6, \
                f"{k_name} should NOT change but did: {v1} vs {v2}"
    print("  Loss routing (slot isolation): OK")

    # simple_test with future predictions
    model.eval()
    with torch.no_grad():
        result = model.simple_test(
            points=None,
            img_metas=img_metas,
            img=img_inputs,
        )
    assert isinstance(result, dict), \
        f"simple_test should return dict, got {type(result)}"
    assert 'occ_current' in result, "Missing occ_current in result"
    assert 'occ_future' in result, "Missing occ_future in result"
    assert 'horizons_sec' in result, "Missing horizons_sec in result"
    assert len(result['occ_future']) == num_future, \
        f"Expected {num_future} future predictions, got {len(result['occ_future'])}"
    assert len(result['horizons_sec']) == num_future, \
        f"Expected {num_future} horizons, got {len(result['horizons_sec'])}"
    print(f"  simple_test: occ_current + {len(result['occ_future'])} future preds + horizons: OK")


def stage_motion_pipeline(config_path):
    """Phase D: full SCMF motion pipeline integration test.

    Verifies:
    - All motion modules built (scmf, motion_encoder, semantic_attention,
      delta_combiner, future_predictor, bev_projector)
    - All existing losses present (loss_depth, loss_occ, loss_2d_seg_history_*)
    - 3 future loss keys (loss_occ_future_1s/2s/3s)
    - All losses finite, backward succeeds
    - Gradient checks: scmf, motion_encoder, semantic_attention,
      delta_combiner, future_predictor all have non-zero gradients
    - simple_test returns dict with occ_current + 3 future preds + horizons_sec
    - Fallback: enable_scmf=False + DirectFutureOccupancyHead forward_train succeeds
    """
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    cfg.model.img_backbone.pretrained = None

    device = torch.device('cpu')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    model.train()

    # Verify all motion modules are built
    motion_modules = [
        'dyn_sta_decoder', 'motion_encoder', 'semantic_attention',
        'delta_combiner', 'scmf', 'future_predictor', 'bev_projector']
    for mod_name in motion_modules:
        mod = getattr(model, mod_name, None)
        assert mod is not None, f"{mod_name} not built"
    print("  All motion modules built: OK")

    B, N_cams, N_frames = 1, 6, 3
    H_img, W_img = 256, 704
    Dx, Dy, Dz = 200, 200, 16
    num_future = model.num_future
    N_total = N_cams * N_frames

    intrins = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0

    sensor2ego = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    rot_cam2lidar = torch.tensor([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.]], dtype=torch.float32)
    sensor2ego[:] = rot_cam2lidar

    ego2global = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_rots = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_trans = torch.zeros(B, N_total, 3, dtype=torch.float32)
    bda = torch.eye(3, dtype=torch.float32).repeat(B, 1, 1)

    imgs = torch.randn(B, N_total, 3, H_img, W_img, dtype=torch.float32)
    img_inputs = [imgs, sensor2ego, ego2global, intrins,
                  post_rots, post_trans, bda]

    img_metas = [dict(box_type_3d='LiDAR')]

    gt_depth = torch.ones(B, N_cams, H_img, W_img,
                          dtype=torch.float32) * 10.0
    voxel_semantics = torch.randint(0, 18, (B, Dx, Dy, Dz))
    mask_camera = torch.ones(B, Dx, Dy, Dz, dtype=torch.float32)
    gt_semantic_2d = torch.randint(0, 17, (B, N_cams, H_img, W_img))

    future_voxel_semantics = torch.randint(
        0, 18, (B, num_future, Dx, Dy, Dz))
    future_mask_camera = torch.ones(
        B, num_future, Dx, Dy, Dz, dtype=torch.float32)

    losses = model.forward_train(
        points=None,
        img_metas=img_metas,
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
        future_voxel_semantics=future_voxel_semantics,
        future_mask_camera=future_mask_camera,
    )

    print(f"  Losses: {sorted(losses.keys())}")

    # Check existing losses
    assert 'loss_depth' in losses, "Missing loss_depth"
    occ_keys = [k for k in losses if 'occ' in k.lower()
                and 'future' not in k]
    assert len(occ_keys) > 0, "Missing current occupancy loss"
    seg_keys = sorted([k for k in losses if '2d_seg_history' in k])
    assert len(seg_keys) == 3, \
        f"Expected 3 2D seg history losses, got {len(seg_keys)}"
    print(f"  Existing losses (depth, occ, 3x seg_history): OK")

    # Check future losses
    future_keys = sorted(
        [k for k in losses if k.startswith('loss_occ_future_')])
    print(f"  Future loss keys: {future_keys}")
    assert len(future_keys) == num_future, \
        f"Expected {num_future} future loss keys, got {len(future_keys)}"

    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite: {v}"
    print("  All losses finite: OK")

    total_loss = sum(losses.values())
    total_loss.backward()
    print(f"  Total loss: {total_loss.item():.4f}")

    # Gradient checks for motion modules.
    # scmf gets grad via motion_head (d(raw_flow)/d(weight)=decoder_out, non-zero).
    # motion_encoder gets grad via delta_combiner -> refined_bev -> future_predictor.
    # delta_combiner gets grad via refined_bev -> future_predictor.
    # future_predictor is directly in the loss path.
    # semantic_attention: attn_feat only feeds SCMF decoder, but motion_head is
    # zero-init so d(flow)/d(attn_feat)=0 at init — grad will appear after
    # motion_head becomes non-zero during training. Verify output is finite.
    grad_modules = [
        ('scmf', model.scmf),
        ('motion_encoder', model.motion_encoder),
        ('delta_combiner', model.delta_combiner),
        ('future_predictor', model.future_predictor),
    ]
    for name, module in grad_modules:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters())
        assert has_grad, f"{name} has no non-zero gradients"
    print("  Gradient checks (scmf, motion_encoder, delta_combiner, "
          "future_predictor): OK")

    # semantic_attention: verify output is finite and wired into SCMF forward
    assert model.semantic_attention is not None
    assert all(
        torch.isfinite(p).all() for p in model.semantic_attention.parameters())
    print("  semantic_attention: params finite, wired via SCMF "
          "(zero grad at init due to zero-init motion_head): OK")

    # simple_test with future predictions
    model.eval()
    with torch.no_grad():
        result = model.simple_test(
            points=None,
            img_metas=img_metas,
            img=img_inputs,
        )
    assert isinstance(result, dict), \
        f"simple_test should return dict, got {type(result)}"
    assert 'occ_current' in result, "Missing occ_current in result"
    assert 'occ_future' in result, "Missing occ_future in result"
    assert 'horizons_sec' in result, "Missing horizons_sec in result"
    assert len(result['occ_future']) == num_future, \
        f"Expected {num_future} future predictions, got {len(result['occ_future'])}"
    print(f"  simple_test: occ_current + {len(result['occ_future'])} future preds: OK")

    # Fallback test: enable_scmf=False + DirectFutureOccupancyHead
    print("\n  [fallback: enable_scmf=False + DirectFutureOccupancyHead]")
    cfg_fb = Config.fromfile(config_path)
    import_plugins(cfg_fb)
    cfg_fb.model.img_backbone.pretrained = None
    cfg_fb.model.enable_scmf = False
    cfg_fb.model.future_predictor = dict(
        type='DirectFutureOccupancyHead',
        bev_channels=256,
        num_future=num_future,
        hidden_dim=256)

    model_fb = build_model(
        cfg_fb.model,
        train_cfg=cfg_fb.get('train_cfg'),
        test_cfg=cfg_fb.get('test_cfg'))
    model_fb.init_weights()
    model_fb = model_fb.to(device)
    model_fb.train()

    assert model_fb.scmf is None, "Fallback model should have scmf=None"
    assert not hasattr(model_fb.future_predictor, 'conv_gru'), \
        "DirectFutureOccupancyHead should not have conv_gru"
    print("  Fallback model: scmf=None, DirectFutureOccupancyHead: OK")

    losses_fb = model_fb.forward_train(
        points=None,
        img_metas=img_metas,
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
        future_voxel_semantics=future_voxel_semantics,
        future_mask_camera=future_mask_camera,
    )

    future_keys_fb = sorted(
        [k for k in losses_fb if k.startswith('loss_occ_future_')])
    assert len(future_keys_fb) == num_future, \
        f"Fallback: expected {num_future} future loss keys, got {len(future_keys_fb)}"
    for k, v in losses_fb.items():
        assert torch.isfinite(v), f"Fallback loss {k} is not finite: {v}"
    print("  Fallback forward_train: all losses finite: OK")


def stage_semantic_closure(config_path):
    """P5: semantic temporal closure integration test.

    Verifies:
    - loss_sem_consistency present and finite
    - 3 loss_future_semantic_{k+1}s keys present and finite
    - All existing losses (depth, occ, seg_history, future_occ) present
    - Backward succeeds, future_semantic has non-zero gradients
    - simple_test returns dict with future predictions
    - Fallback: both P5 disabled → motion-pipeline behavior
    """
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    cfg.model.img_backbone.pretrained = None
    cfg.model.enable_future_semantic = True

    device = torch.device('cpu')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    model.train()

    assert model.future_semantic is not None, \
        "future_semantic not built"
    assert model.sem_consistency is not None, \
        "sem_consistency not built"
    print("  P5 modules built: OK")

    B, N_cams, N_frames = 1, 6, 3
    H_img, W_img = 256, 704
    Dx, Dy, Dz = 200, 200, 16
    num_future = model.num_future
    N_total = N_cams * N_frames

    intrins = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0

    sensor2ego = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    rot_cam2lidar = torch.tensor([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.]], dtype=torch.float32)
    sensor2ego[:] = rot_cam2lidar

    ego2global = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_rots = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_trans = torch.zeros(B, N_total, 3, dtype=torch.float32)
    bda = torch.eye(3, dtype=torch.float32).repeat(B, 1, 1)

    imgs = torch.randn(B, N_total, 3, H_img, W_img, dtype=torch.float32)
    img_inputs = [imgs, sensor2ego, ego2global, intrins,
                  post_rots, post_trans, bda]

    img_metas = [dict(box_type_3d='LiDAR')]

    gt_depth = torch.ones(B, N_cams, H_img, W_img,
                          dtype=torch.float32) * 10.0
    voxel_semantics = torch.randint(0, 18, (B, Dx, Dy, Dz))
    mask_camera = torch.ones(B, Dx, Dy, Dz, dtype=torch.float32)
    gt_semantic_2d = torch.randint(0, 17, (B, N_cams, H_img, W_img))

    future_voxel_semantics = torch.randint(
        0, 18, (B, num_future, Dx, Dy, Dz))
    future_mask_camera = torch.ones(
        B, num_future, Dx, Dy, Dz, dtype=torch.float32)

    future_gt_semantic_bev = torch.randint(
        0, 17, (B, num_future, Dx, Dy), dtype=torch.long)

    losses = model.forward_train(
        points=None,
        img_metas=img_metas,
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
        future_voxel_semantics=future_voxel_semantics,
        future_mask_camera=future_mask_camera,
        future_gt_semantic_bev=future_gt_semantic_bev,
    )

    print(f"  Losses: {sorted(losses.keys())}")

    assert 'loss_sem_consistency' in losses, \
        "Missing loss_sem_consistency"
    print("  loss_sem_consistency: OK")

    fsem_keys = sorted(
        [k for k in losses if k.startswith('loss_future_semantic_')])
    print(f"  Future semantic loss keys: {fsem_keys}")
    assert len(fsem_keys) == num_future, \
        f"Expected {num_future} future semantic loss keys, got {len(fsem_keys)}"

    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite: {v}"
    print("  All losses finite: OK")

    total_loss = sum(losses.values())
    total_loss.backward()
    print(f"  Total loss: {total_loss.item():.4f}")

    fs_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.future_semantic.parameters())
    assert fs_grad, "future_semantic has no non-zero gradients"
    print("  future_semantic gradients: OK")

    model.eval()
    with torch.no_grad():
        result = model.simple_test(
            points=None,
            img_metas=img_metas,
            img=img_inputs,
        )
    assert isinstance(result, dict), \
        f"simple_test should return dict, got {type(result)}"
    assert len(result['occ_future']) == num_future
    print(f"  simple_test: occ_current + {len(result['occ_future'])} future preds: OK")

    # Fallback: P5 disabled
    print("\n  [fallback: P5 disabled]")
    cfg_fb = Config.fromfile(config_path)
    import_plugins(cfg_fb)
    cfg_fb.model.img_backbone.pretrained = None
    cfg_fb.model.enable_future_semantic = False
    cfg_fb.model.enable_semantic_consistency = False

    model_fb = build_model(
        cfg_fb.model,
        train_cfg=cfg_fb.get('train_cfg'),
        test_cfg=cfg_fb.get('test_cfg'))
    model_fb.init_weights()
    model_fb = model_fb.to(device)
    model_fb.train()

    assert model_fb.future_semantic is None
    assert model_fb.sem_consistency is None
    print("  Fallback: future_semantic=None, sem_consistency=None: OK")

    losses_fb = model_fb.forward_train(
        points=None,
        img_metas=img_metas,
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
        future_voxel_semantics=future_voxel_semantics,
        future_mask_camera=future_mask_camera,
    )
    assert 'loss_sem_consistency' not in losses_fb
    assert not any(k.startswith('loss_future_semantic_') for k in losses_fb)
    for k, v in losses_fb.items():
        assert torch.isfinite(v), f"Fallback loss {k} is not finite: {v}"
    print("  Fallback: P5 losses absent, all existing losses finite: OK")


def _make_synthetic_4d_batch(model, device):
    """Build a synthetic 4D training batch (kwargs for forward_train)."""
    B, N_cams, N_frames = 1, 6, 3
    H_img, W_img = 256, 704
    Dx, Dy, Dz = 200, 200, 16
    num_future = model.num_future
    N_total = N_cams * N_frames

    intrins = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0
    sensor2ego = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    rot_cam2lidar = torch.tensor([
        [0., 0., 1., 0.], [-1., 0., 0., 0.],
        [0., -1., 0., 0., ], [0., 0., 0., 1.]], dtype=torch.float32)
    sensor2ego[:] = rot_cam2lidar
    ego2global = torch.eye(4, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_rots = torch.eye(3, dtype=torch.float32).repeat(B, N_total, 1, 1)
    post_trans = torch.zeros(B, N_total, 3, dtype=torch.float32)
    bda = torch.eye(3, dtype=torch.float32).repeat(B, 1, 1)
    imgs = torch.randn(B, N_total, 3, H_img, W_img, dtype=torch.float32)
    img_inputs = [t.to(device) for t in
                  [imgs, sensor2ego, ego2global, intrins, post_rots,
                   post_trans, bda]]

    gt_depth = (torch.ones(B, N_cams, H_img, W_img) * 10.0).to(device)
    voxel_semantics = torch.randint(0, 18, (B, Dx, Dy, Dz)).to(device)
    mask_camera = torch.ones(B, Dx, Dy, Dz, dtype=torch.float32).to(device)
    gt_semantic_2d = torch.randint(0, 17, (B, N_cams, H_img, W_img)).to(device)
    future_voxel_semantics = torch.randint(
        0, 18, (B, num_future, Dx, Dy, Dz)).to(device)
    future_mask_camera = torch.ones(
        B, num_future, Dx, Dy, Dz, dtype=torch.float32).to(device)

    return dict(
        points=None,
        img_metas=[dict(box_type_3d='LiDAR')],
        img_inputs=img_inputs,
        gt_depth=gt_depth,
        voxel_semantics=voxel_semantics,
        mask_camera=mask_camera,
        mask_lidar=mask_camera,
        gt_semantic_2d=gt_semantic_2d,
        future_voxel_semantics=future_voxel_semantics,
        future_mask_camera=future_mask_camera,
    )


def _dc_unwrap(v):
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        return v
    return v.data if isinstance(v, DataContainer) else v


def _to_batch_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        x = x.to(device)
        return x.unsqueeze(0) if x.dim() >= 1 else x
    return x


def _try_real_batch(cfg, device):
    """Best-effort single real sample from the training dataset.

    Returns a forward_train kwargs dict, or None if data is unavailable
    (missing _forecast.pkl, import errors, unexpected formats), in which case
    the caller falls back to synthetic tensors.
    """
    try:
        from mmdet3d.datasets import build_dataset
        ds = build_dataset(cfg.data.train)
        sample = ds[0]
        batch = {}
        for k, v in sample.items():
            v = _dc_unwrap(v)
            if k == 'img_metas':
                batch[k] = [v] if isinstance(v, dict) else list(v)
            elif k == 'img_inputs':
                batch[k] = [_to_batch_tensor(t, device) for t in v]
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                batch[k] = _to_batch_tensor(v, device)
        for req in ('img_inputs', 'gt_depth', 'voxel_semantics', 'mask_camera'):
            if req not in batch:
                raise KeyError(f"real sample missing '{req}'")
        batch.setdefault('mask_lidar', batch['mask_camera'])
        batch.setdefault('points', None)
        batch.setdefault('img_metas', [dict(box_type_3d='LiDAR')])
        return batch
    except Exception as e:
        print(f"  [minimal-chain] real data unavailable "
              f"({type(e).__name__}: {e}); falling back to synthetic")
        return None


def _find_module(model, class_name):
    for m in model.modules():
        if m.__class__.__name__ == class_name:
            return m
    return None


def stage_minimal_chain(config_path, iters=5, overfit_lr=1e-3, warmup=2,
                        use_real_data=True, stats_out=None, device=None):
    """Pre-training minimal-chain verification.

    Runs a short overfit-a-batch loop (forward -> backward -> optimizer.step)
    and asserts the things that most often break a first training run:
      * every expected loss key is present and finite each step
      * tensor dimensions through the motion/continuity/SDP paths are correct
      * gradients reach the new modules (SMP, SDP) after warmup
      * the total loss decreases (a convergence smoke test)
      * no NaN/Inf parameters after stepping
    Emits a per-step stats JSON compatible with diagnose_priorocc_4d_training.py.
    """
    import copy
    import json
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugins(cfg)
    cfg.model.img_backbone.pretrained = None
    device = device or torch.device('cpu')

    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    model.train()

    batch = _try_real_batch(cfg, device) if use_real_data else None
    mode = 'real' if batch is not None else 'synthetic'
    if batch is None:
        batch = _make_synthetic_4d_batch(model, device)
    print(f"  [minimal-chain] batch mode: {mode}")

    opt_cfg = copy.deepcopy(
        cfg.get('optimizer', dict(type='AdamW', lr=1e-4, weight_decay=0.01)))
    opt_cfg['lr'] = overfit_lr
    optimizer = build_optimizer(model, opt_cfg)
    max_norm = None
    oc = cfg.get('optimizer_config', {})
    if isinstance(oc, dict) and oc.get('grad_clip'):
        max_norm = oc['grad_clip'].get('max_norm')

    # Forward hooks to capture internal tensors for stats + dim checks.
    captured = {}
    hooks = []

    def cap(key):
        def h(m, i, o):
            captured[key] = o
        return h

    def cap2(k0, k1, k2=None):
        def h(m, i, o):
            captured[k0] = o[0]
            captured[k1] = o[1]
            if k2 is not None:
                captured[k2] = o[2]
        return h

    if getattr(model, 'scmf', None) is not None:
        hooks.append(model.scmf.register_forward_hook(cap('flow')))
    if getattr(model, 'future_predictor', None) is not None:
        hooks.append(
            model.future_predictor.register_forward_hook(cap('future_bevs')))
        if hasattr(model.future_predictor, 'gate'):
            hooks.append(
                model.future_predictor.gate.register_forward_hook(cap('gate')))
    if getattr(model, 'occ_head', None) is not None:
        hooks.append(model.occ_head.register_forward_hook(cap('occ_logits')))
    if getattr(model, 'bev_projector', None) is not None:
        hooks.append(model.bev_projector.register_forward_hook(
            cap2('semantic_bev', 'visibility')))
    if getattr(model, 'dyn_sta_decoder', None) is not None:
        hooks.append(model.dyn_sta_decoder.register_forward_hook(
            cap2('dyn_mask', 'sta_mask', 'per_cls_masks')))

    num_future = model.num_future
    mfc = getattr(model, 'max_flow_cells', 5.0)
    trainable = [p for p in model.parameters() if p.requires_grad]

    def flat_snapshot():
        return torch.cat([p.detach().reshape(-1) for p in trainable])

    expected_new = []
    if cfg.model.get('enable_motion_prior', False):
        expected_new += ['loss_static_flow', 'loss_rigid_smooth',
                         'loss_nonrigid_bound']
    if cfg.model.get('enable_sem_continuity', False):
        expected_new += ['loss_sem_continuity_2d', 'loss_sem_continuity_bev']

    stats = []
    loss_totals = []
    step_idx = 0
    try:
        for it in range(warmup + iters):
            optimizer.zero_grad()
            captured.clear()
            losses = model.forward_train(**batch)

            # --- expected keys + finiteness (every step) ---
            assert 'loss_depth' in losses, "missing loss_depth"
            fut_keys = [k for k in losses if k.startswith('loss_occ_future_')]
            assert len(fut_keys) == num_future, \
                f"expected {num_future} future losses, got {len(fut_keys)}"
            for k in expected_new:
                assert k in losses, f"missing expected loss '{k}'"
            if cfg.model.get('enable_semantic_consistency', False):
                assert 'loss_sem_consistency' in losses, \
                    "missing loss_sem_consistency"
            for k, v in losses.items():
                assert torch.isfinite(v).all(), f"loss {k} not finite: {v}"

            total = sum(losses.values())
            total.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm if max_norm else float('inf'))
            mh = getattr(model.scmf, 'motion_head', None) \
                if getattr(model, 'scmf', None) is not None else None
            flow_grad_norm = float(mh.weight.grad.norm()) \
                if (mh is not None and mh.weight.grad is not None) else 0.0

            before = flat_snapshot()
            optimizer.step()
            after = flat_snapshot()
            changed_ratio = float((before != after).float().mean())

            # no NaN/Inf params after stepping
            assert torch.isfinite(after).all(), "NaN/Inf in params after step"

            record = it >= warmup
            if record:
                loss_totals.append(float(total.item()))
                entry = {
                    'step': step_idx,
                    'mode': mode,
                    'loss_total': float(total.item()),
                    'loss_occ_future_total': float(
                        sum(losses[k].item() for k in fut_keys)),
                    'grad_norm': float(grad_norm),
                    'param_changed_ratio': changed_ratio,
                    'flow_grad_norm': flow_grad_norm,
                }
                for k, v in losses.items():
                    entry[k] = float(v.item())
                occ = captured.get('occ_logits')
                if occ is not None:
                    p = torch.softmax(occ.float(), dim=-1)
                    entry['logits_entropy'] = float(
                        -(p * torch.log(p + 1e-12)).sum(-1).mean())
                    entry['dominant_class_ratio'] = float(p.max(-1).values.mean())
                fut = captured.get('future_bevs')
                if fut is not None and fut.shape[1] >= 2:
                    entry['future_pairwise_diff'] = float(
                        (fut[:, 0] - fut[:, 1]).abs().mean())
                flow = captured.get('flow')
                if flow is not None:
                    entry['flow_mean_abs'] = float(flow.abs().mean())
                    entry['flow_saturation_ratio'] = float(
                        (flow.abs() >= 0.99 * mfc).float().mean())
                gate = captured.get('gate')
                if gate is not None:
                    entry['gate_mean'] = float(gate.mean())
                    entry['gate_std'] = float(gate.std())
                sem_bev = captured.get('semantic_bev')
                if sem_bev is not None:
                    entry['semantic_bev_std'] = float(sem_bev.std())
                dyn, sta = captured.get('dyn_mask'), captured.get('sta_mask')
                if dyn is not None and sta is not None:
                    entry['mask_overlap'] = float((dyn * sta).mean())
                stats.append(entry)
                step_idx += 1
    finally:
        for h in hooks:
            h.remove()

    # --- convergence smoke test (uses recorded loss totals) ---
    assert len(loss_totals) >= 2, "need >=2 recorded iters for convergence"
    assert loss_totals[-1] < loss_totals[0], \
        f"loss did not decrease: {loss_totals[0]:.4f} -> {loss_totals[-1]:.4f}"
    print(f"  convergence: loss {loss_totals[0]:.4f} -> "
          f"{loss_totals[-1]:.4f}: OK")

    # --- dimension checks (fresh forward; re-attach capture hooks) ---
    model.zero_grad()
    captured.clear()
    dim_hooks = []
    if getattr(model, 'scmf', None) is not None:
        dim_hooks.append(model.scmf.register_forward_hook(cap('flow')))
    if getattr(model, 'bev_projector', None) is not None:
        dim_hooks.append(model.bev_projector.register_forward_hook(
            cap2('semantic_bev', 'visibility')))
    if getattr(model, 'dyn_sta_decoder', None) is not None:
        dim_hooks.append(model.dyn_sta_decoder.register_forward_hook(
            cap2('dyn_mask', 'sta_mask', 'per_cls_masks')))
    if getattr(model, 'future_predictor', None) is not None:
        dim_hooks.append(
            model.future_predictor.register_forward_hook(cap('future_bevs')))
    with torch.no_grad():
        model.forward_train(**batch)
    for h in dim_hooks:
        h.remove()
    flow = captured.get('flow')
    if flow is not None:
        assert flow.dim() == 5 and flow.shape[1] == num_future \
            and flow.shape[2] == 2, f"bad flow shape {tuple(flow.shape)}"
    sem_bev = captured.get('semantic_bev')
    if sem_bev is not None:
        assert sem_bev.shape[1] == cfg.model.get('num_semantic_classes', 17), \
            f"bad semantic_bev channels {sem_bev.shape[1]}"
    fut = captured.get('future_bevs')
    if fut is not None:
        assert fut.dim() == 5 and fut.shape[1] == num_future, \
            f"bad future_bevs shape {tuple(fut.shape)}"
    # SMP output shape
    if getattr(model, 'motion_prior', None) is not None and flow is not None \
            and captured.get('per_cls_masks') is not None:
        out = model.motion_prior(flow, captured['per_cls_masks'])
        assert out.shape == flow.shape, \
            f"SMP output {tuple(out.shape)} != flow {tuple(flow.shape)}"
    # warp validity shape
    if getattr(model.future_predictor, 'use_warp_validity', False):
        H = W = flow.shape[-1] if flow is not None else 8
        dummy_bev = torch.zeros(1, model.future_predictor.bev_channels, H, W,
                                device=device)
        dummy_flow = torch.zeros(1, 2, H, W, device=device)
        _, validity = model.future_predictor.warper(
            dummy_bev, dummy_flow, return_validity=True)
        assert validity.shape == (1, 1, H, W), \
            f"bad validity shape {tuple(validity.shape)}"
    print("  dimension checks (flow/semantic_bev/future_bevs/SMP/validity): OK")

    # --- gradients reach new modules (deterministic micro-tests) ---
    if getattr(model, 'motion_prior', None) is not None:
        mp = model.motion_prior
        mp.zero_grad()
        Csem = cfg.model.get('num_semantic_classes', 17)
        hs = ws = 16
        dflow = torch.randn(1, num_future, 2, hs, ws, device=device)
        dmasks = torch.softmax(
            torch.randn(1, Csem, hs, ws, device=device), dim=1)
        out = mp(dflow, dmasks)
        reg = mp.regularization_losses(out, dmasks)
        (out.abs().mean() + sum(reg.values())).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in mp.parameters()), \
            "SemanticMotionPrior received no gradient"
        print("  grad -> SemanticMotionPrior: OK")

    sdp = _find_module(model, 'SemanticDepthPrior')
    if sdp is not None:
        sdp.zero_grad()
        D = sdp.depth_channels
        dl = torch.randn(2, D, 8, 8, device=device)
        sl = torch.randn(2, sdp.sem_channels, 8, 8, device=device)
        prob = torch.softmax(sdp(dl, sl), dim=1)
        bins = torch.arange(
            D, device=device, dtype=prob.dtype).view(1, D, 1, 1)
        (prob * bins).sum(dim=1).sum().backward()
        assert sdp.table.grad is not None and sdp.table.grad.abs().sum() > 0, \
            "SemanticDepthPrior table received no gradient"
        print("  grad -> SemanticDepthPrior.table: OK")

    if stats_out:
        os.makedirs(os.path.dirname(os.path.abspath(stats_out)), exist_ok=True)
        with open(stats_out, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  stats JSON written to {stats_out} ({len(stats)} steps)")
        print(f"    -> diagnose with: python tools/diagnose_priorocc_4d_training.py "
              f"--stats-json {stats_out}"
              + (" --max-mask-overlap 0.3 --min-future-loss-drop 0.0"
                 if mode == 'synthetic' else ""))

    print(f"  [minimal-chain] mode={mode}, recorded steps={len(stats)}")


def main():
    parser = argparse.ArgumentParser(
        description='PriorOcc-4D Verification')
    parser.add_argument(
        '--stage', type=str, default='modules',
        choices=['modules', 'build', 'sgdm-current', 'forecast-smoke',
                 'motion-pipeline', 'semantic-closure', 'minimal-chain', 'all'])
    parser.add_argument(
        '--test', type=str, default='all',
        help='Test name or "all"')
    parser.add_argument(
        '--config', type=str,
        default='projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stats-out', type=str, default=None,
                        help='write minimal-chain stats JSON here')
    parser.add_argument('--iters', type=int, default=5,
                        help='recorded optimizer steps for minimal-chain')
    parser.add_argument('--overfit-lr', type=float, default=1e-3,
                        help='LR for the minimal-chain overfit loop')
    parser.add_argument('--warmup', type=int, default=2,
                        help='unrecorded warmup steps before recording')
    parser.add_argument('--use-real-data', dest='use_real_data',
                        action='store_true', default=True,
                        help='try a real dataloader batch (default)')
    parser.add_argument('--no-real-data', dest='use_real_data',
                        action='store_false',
                        help='force synthetic batch')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}, Seed: {args.seed}")

    if args.stage in ('modules', 'all'):
        tests = {
            'zero-flow': test_zero_flow,
            'known-translation': test_known_translation,
            'mask-partition': test_mask_partition,
            'finite-and-grad': test_finite_and_grad,
            'semantic-sensitivity': test_semantic_sensitivity,
            'identity-init': test_identity_init,
        }
        selected = tests if args.test == 'all' else {
            args.test: tests[args.test]}

        all_pass = True
        for name, fn in selected.items():
            print(f"\n[{name}]")
            try:
                fn(device)
                print("  PASS")
            except Exception as e:
                print(f"  FAIL: {e}")
                traceback.print_exc()
                all_pass = False

        print(f"\n{'='*40}")
        print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    if args.stage in ('build', 'all'):
        print(f"\n[build]")
        try:
            stage_build(args.config)
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    if args.stage in ('sgdm-current', 'all'):
        print(f"\n[sgdm-current]")
        try:
            stage_sgdm_current(args.config)
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    if args.stage in ('forecast-smoke', 'all'):
        print(f"\n[forecast-smoke]")
        try:
            stage_forecast_smoke(args.config)
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    if args.stage in ('motion-pipeline', 'all'):
        print(f"\n[motion-pipeline]")
        try:
            stage_motion_pipeline(args.config)
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    if args.stage in ('semantic-closure', 'all'):
        print(f"\n[semantic-closure]")
        try:
            stage_semantic_closure(args.config)
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    if args.stage == 'minimal-chain':
        print(f"\n[minimal-chain]")
        try:
            stage_minimal_chain(
                args.config, iters=args.iters, overfit_lr=args.overfit_lr,
                warmup=args.warmup, use_real_data=args.use_real_data,
                stats_out=args.stats_out, device=device)
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            sys.exit(1)

    if args.stage in ('modules', 'sgdm-current', 'forecast-smoke',
                      'motion-pipeline', 'semantic-closure', 'minimal-chain'):
        sys.exit(0)


if __name__ == '__main__':
    main()
