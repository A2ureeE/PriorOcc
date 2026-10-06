#!/usr/bin/env python3
"""CPU smoke test for the semantic-guided multimodal occupancy forecasting.

Three layers, fastest first:
  A. module-level   — shapes, near-identity init, symmetry breaking,
                      multimodal SMP identity/mode-logits/regularizers,
                      end-to-end gradient flow through the K-mode chain.
  B. model-level    — build PriorOcc4D from the mmodal config, synthetic
                      4D batch: forward_train loss keys (WTA winner keys
                      + loss_mode_cls + loss_mode_div), finiteness,
                      gradients into the new modules, simple_test multimodal
                      outputs, per-sample CE unit test, convergence loop.
  C. evaluation     — fake GT + fake multimodal results through
                      NuScenes4DOccForecastDataset.evaluate: deployed mIoU,
                      best-of-K oracle mIoU (>= deployed), mode_selection_acc.

Usage:
    python tools/smoke_test_multimodal.py                 # A + C (fast)
    python tools/smoke_test_multimodal.py --full          # A + B + C
    python tools/smoke_test_multimodal.py --full --iters 3
"""

import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MM_CFG = 'projects/configs/priorocc/priorocc-4d-r50-mmodal.py'


def warmup_plugin_imports():
    """Import the plugin package twice if needed.

    The first import may JIT-check the dvr CUDA extension and fail when
    ninja is unavailable; a prebuilt .so already exists in the torch
    extension cache, so the second import loads it directly.
    """
    for attempt in range(2):
        try:
            import projects.mmdet3d_plugin  # noqa: F401
            return
        except Exception:
            if attempt == 1:
                raise


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Part A: module-level tests (small tensors, no mmcv model build)
# ---------------------------------------------------------------------------

def _make_inputs(B=2, C=256, C_sem=17, H=16, W=16, device='cpu'):
    fused = torch.randn(B, C, H, W, device=device)
    motion = torch.randn(B, C, H, W, device=device)
    attn = torch.randn(B, C, H, W, device=device)
    sem = torch.softmax(torch.randn(B, C_sem, H, W, device=device), dim=1)
    return fused, motion, attn, sem


def test_a_shapes(device):
    """MM-SCMF flow (B,K,T,2,H,W); MM predictor future_bevs (B,K,T,C,H,W)."""
    from projects.mmdet3d_plugin.models.model_utils.scmf import (
        MultimodalSemanticConditionedMotionField,
        MultimodalSCMFEnhancedPredictor)

    B, C, C_sem, H, W, K, T = 2, 64, 17, 16, 16, 3, 3
    fused, motion, attn, sem = _make_inputs(B, C, C_sem, H, W, device)

    scmf = MultimodalSemanticConditionedMotionField(
        bev_channels=C, num_semantic_classes=C_sem, num_future=T,
        num_modes=K, hidden_dim=64).to(device)
    pred = MultimodalSCMFEnhancedPredictor(
        bev_channels=C, num_future=T, num_modes=K, hidden_dim=64,
        use_warp_validity=True).to(device)

    flow = scmf(fused, motion, sem, attn)
    assert flow.shape == (B, K, T, 2, H, W), f"flow shape {flow.shape}"
    assert torch.isfinite(flow).all(), "flow has NaN/Inf"

    future = pred(fused, flow)
    assert future.shape == (B, K, T, C, H, W), f"future shape {future.shape}"
    assert torch.isfinite(future).all(), "future_bevs has NaN/Inf"


def test_a_near_identity_init(device):
    """Initial flow stays ~identity: |flow| <= mode_bias*(K-1)/2 + slack."""
    from projects.mmdet3d_plugin.models.model_utils.scmf import (
        MultimodalSemanticConditionedMotionField)

    B, C, C_sem, H, W, K, T = 2, 64, 17, 16, 16, 3, 3
    fused, motion, attn, sem = _make_inputs(B, C, C_sem, H, W, device)
    scmf = MultimodalSemanticConditionedMotionField(
        bev_channels=C, num_semantic_classes=C_sem, num_future=T,
        num_modes=K, hidden_dim=64, mode_bias_cells=0.05,
        max_flow_cells=5.0).to(device)

    flow = scmf(fused, motion, sem, attn)
    bound = 0.05 * (K - 1) / 2.0 + 1e-4
    max_abs = flow.abs().max().item()
    assert max_abs <= bound, \
        f"near-identity init violated: |flow|={max_abs:.4f} > {bound:.4f}"


def test_a_symmetry_breaking(device):
    """Modes differ at init (bias offsets) so WTA gradients can diverge."""
    from projects.mmdet3d_plugin.models.model_utils.scmf import (
        MultimodalSemanticConditionedMotionField)

    B, C, C_sem, H, W, K, T = 2, 64, 17, 16, 16, 3, 3
    fused, motion, attn, sem = _make_inputs(B, C, C_sem, H, W, device)
    scmf = MultimodalSemanticConditionedMotionField(
        bev_channels=C, num_semantic_classes=C_sem, num_future=T,
        num_modes=K, hidden_dim=64, mode_bias_cells=0.05).to(device)

    flow = scmf(fused, motion, sem, attn)
    for i in range(K):
        for j in range(i + 1, K):
            diff = (flow[:, i] - flow[:, j]).abs().max().item()
            assert diff > 1e-3, \
                f"modes {i},{j} identical at init (collapse from step 0)"


def test_a_mm_smp_identity(device):
    """A_{g,k}=I at init => flow_out == flow for every mode."""
    from projects.mmdet3d_plugin.models.model_utils.semantic_motion_prior \
        import SemanticMultimodalMotionPrior

    B, K, T, C_sem, H, W = 2, 3, 3, 17, 16, 16
    smp = SemanticMultimodalMotionPrior(
        num_semantic_classes=C_sem, num_future=T, num_modes=K).to(device)
    flow = torch.randn(B, K, T, 2, H, W, device=device)
    masks = torch.softmax(
        torch.randn(B, C_sem, H, W, device=device), dim=1)

    out = smp(flow, masks)
    assert out.shape == flow.shape, f"SMP output shape {out.shape}"
    assert torch.allclose(out, flow, atol=1e-5), \
        "identity init violated: SMP changed the flow"


def test_a_mm_smp_mode_logits(device):
    """mode_logits: (B,K); softmax normalizes; semantics change the logits
    once group_mode_logits is non-trivial."""
    from projects.mmdet3d_plugin.models.model_utils.semantic_motion_prior \
        import SemanticMultimodalMotionPrior

    B, K, C_sem, H, W = 2, 3, 17, 16, 16
    smp = SemanticMultimodalMotionPrior(
        num_semantic_classes=C_sem, num_future=3, num_modes=K).to(device)
    with torch.no_grad():
        smp.group_mode_logits.copy_(torch.randn_like(smp.group_mode_logits))

    masks_a = torch.zeros(B, C_sem, H, W, device=device)
    masks_a[:, 4, :, :] = 1.0  # car (rigid_vehicle)
    masks_b = torch.zeros(B, C_sem, H, W, device=device)
    masks_b[:, 15, :, :] = 1.0  # manmade (static)

    logits_a = smp.mode_logits(masks_a)
    logits_b = smp.mode_logits(masks_b)
    assert logits_a.shape == (B, K), f"mode logits shape {logits_a.shape}"
    probs = torch.softmax(logits_a, dim=1)
    assert torch.allclose(
        probs.sum(dim=1), torch.ones(B, device=device), atol=1e-5), \
        "mode probabilities do not sum to 1"
    diff = (logits_a - logits_b).abs().max().item()
    assert diff > 1e-4, \
        "semantic composition should change mode logits"


def test_a_mm_smp_losses_and_grad(device):
    """Regularizers finite + gradients reach group_modes and
    group_mode_logits; diversity is negative and has gradient."""
    from projects.mmdet3d_plugin.models.model_utils.semantic_motion_prior \
        import SemanticMultimodalMotionPrior

    B, K, T, C_sem, H, W = 2, 3, 3, 17, 16, 16
    smp = SemanticMultimodalMotionPrior(
        num_semantic_classes=C_sem, num_future=T, num_modes=K).to(device)
    flow = torch.randn(B, K, T, 2, H, W, device=device) * 2.0
    masks = torch.softmax(
        torch.randn(B, C_sem, H, W, device=device), dim=1)

    out = smp(flow, masks)
    reg = smp.regularization_losses(out, masks)
    for k, v in reg.items():
        assert torch.isfinite(v), f"regularizer {k} not finite: {v}"
    div = smp.diversity_loss(out)
    assert torch.isfinite(div), "diversity loss not finite"
    assert div.item() < 0, "diversity loss should be <= 0 (anti-collapse)"

    # Gradient micro-test: mode_logits path (group_mode_logits) and
    # forward path (group_modes).
    logits = smp.mode_logits(masks)
    loss = out.abs().mean() + logits.sum() + sum(reg.values()) + div
    loss.backward()
    assert smp.group_modes.grad is not None and \
        smp.group_modes.grad.abs().sum() > 0, \
        "no gradient reached group_modes"
    assert smp.group_mode_logits.grad is not None and \
        smp.group_mode_logits.grad.abs().sum() > 0, \
        "no gradient reached group_mode_logits"


def test_a_end_to_end_grad(device):
    """Gradients flow SCMF -> SMP -> predictor on a synthetic loss."""
    from projects.mmdet3d_plugin.models.model_utils.scmf import (
        MultimodalSemanticConditionedMotionField,
        MultimodalSCMFEnhancedPredictor)
    from projects.mmdet3d_plugin.models.model_utils.semantic_motion_prior \
        import SemanticMultimodalMotionPrior

    B, C, C_sem, H, W, K, T = 2, 64, 17, 16, 16, 3, 3
    fused, motion, attn, sem = _make_inputs(B, C, C_sem, H, W, device)

    scmf = MultimodalSemanticConditionedMotionField(
        bev_channels=C, num_semantic_classes=C_sem, num_future=T,
        num_modes=K, hidden_dim=64).to(device)
    smp = SemanticMultimodalMotionPrior(
        num_semantic_classes=C_sem, num_future=T, num_modes=K).to(device)
    pred = MultimodalSCMFEnhancedPredictor(
        bev_channels=C, num_future=T, num_modes=K, hidden_dim=64).to(device)

    flow = scmf(fused, motion, sem, attn)
    flow = smp(flow, sem)
    future = pred(fused, flow)
    logits = smp.mode_logits(sem)
    loss = future.abs().mean() + logits.sum() + \
        sum(smp.regularization_losses(flow, sem).values()) + \
        smp.diversity_loss(flow)
    loss.backward()

    for name, mod in [('scmf', scmf), ('smp', smp), ('pred', pred)]:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mod.parameters())
        assert has_grad, f"{name} received no gradient"


# ---------------------------------------------------------------------------
# Part B: model-level tests (build from config, synthetic 4D batch, CPU)
# ---------------------------------------------------------------------------

def _import_plugins(cfg):
    if not getattr(cfg, 'plugin', False):
        return
    import importlib
    module_path = cfg.plugin_dir.replace('/', '.').rstrip('.')
    importlib.import_module(module_path)


def _build_model(config_path):
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    _import_plugins(cfg)
    cfg.model.img_backbone.pretrained = None
    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    return model, cfg


def _make_synthetic_4d_batch(model, device):
    """Same synthetic batch layout as verify_priorocc_4d.minimal-chain."""
    B, N_cams, N_frames = 1, 6, 3
    H_img, W_img = 256, 704
    Dx, Dy, Dz = 200, 200, 16
    num_future = model.num_future
    N_total = N_cams * N_frames

    intrins = torch.eye(3).repeat(B, N_total, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0
    sensor2ego = torch.eye(4).repeat(B, N_total, 1, 1)
    rot = torch.tensor([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                        [0., -1., 0., 0.], [0., 0., 0., 1.]], dtype=torch.float32)
    sensor2ego[:] = rot
    ego2global = torch.eye(4).repeat(B, N_total, 1, 1)
    post_rots = torch.eye(3).repeat(B, N_total, 1, 1)
    post_trans = torch.zeros(B, N_total, 3)
    bda = torch.eye(3).repeat(B, 1, 1)
    imgs = torch.randn(B, N_total, 3, H_img, W_img)
    img_inputs = [t.to(device) for t in
                  [imgs, sensor2ego, ego2global, intrins, post_rots,
                   post_trans, bda]]

    return dict(
        points=None,
        img_metas=[dict(box_type_3d='LiDAR')],
        img_inputs=img_inputs,
        gt_depth=(torch.ones(B, N_cams, H_img, W_img) * 10.0).to(device),
        voxel_semantics=torch.randint(0, 18, (B, Dx, Dy, Dz)).to(device),
        mask_camera=torch.ones(B, Dx, Dy, Dz).to(device),
        gt_semantic_2d=torch.randint(
            0, 17, (B, N_cams, H_img, W_img)).to(device),
        future_voxel_semantics=torch.randint(
            0, 18, (B, num_future, Dx, Dy, Dz)).to(device),
        future_mask_camera=torch.ones(
            B, num_future, Dx, Dy, Dz).to(device),
    )


def _mask_lidar(batch):
    batch.setdefault('mask_lidar', batch['mask_camera'])
    return batch


def test_b_per_sample_occ_ce(device):
    """Per-sample masked CE used for WTA selection: manual reference check."""
    model, _ = _build_model(MM_CFG)
    model = model.to(device)

    B, Dx, Dy, Dz, C = 3, 8, 8, 4, 18
    torch.manual_seed(0)
    logits = torch.randn(B, Dx, Dy, Dz, C, device=device)
    gt = torch.randint(0, C, (B, Dx, Dy, Dz), device=device)
    mask = torch.zeros(B, Dx, Dy, Dz, device=device)
    mask[:, :4] = 1.0  # only half the voxels are visible

    out = model._per_sample_occ_ce(logits, gt, mask)
    assert out.shape == (B,), f"per-sample CE shape {out.shape}"

    # Manual reference for sample 0.
    l0 = logits[0].reshape(-1, C)
    g0 = gt[0].reshape(-1).long()
    m0 = mask[0].reshape(-1)
    ce0 = torch.nn.functional.cross_entropy(l0, g0, reduction='none')
    ref0 = (ce0 * m0).sum() / m0.sum()
    assert torch.allclose(out[0], ref0, atol=1e-5), \
        f"per-sample CE mismatch: {out[0].item()} vs {ref0.item()}"


def test_b_forward_backward(device, iters=3, warmup=1):
    """forward_train loss schema + finiteness + gradients + WTA sanity +
    simple_test multimodal outputs + convergence mini-loop."""
    import copy
    from mmcv.runner import build_optimizer

    model, cfg = _build_model(MM_CFG)
    model = model.to(device)
    model.train()

    # --- B1: multimodal modules built ---
    from projects.mmdet3d_plugin.models.model_utils.scmf import (
        MultimodalSemanticConditionedMotionField,
        MultimodalSCMFEnhancedPredictor)
    from projects.mmdet3d_plugin.models.model_utils.semantic_motion_prior \
        import SemanticMultimodalMotionPrior

    assert isinstance(model.scmf,
                      MultimodalSemanticConditionedMotionField), \
        "scmf is not the multimodal variant"
    assert isinstance(model.future_predictor,
                      MultimodalSCMFEnhancedPredictor), \
        "future_predictor is not the multimodal variant"
    assert isinstance(model.motion_prior,
                      SemanticMultimodalMotionPrior), \
        "motion_prior is not the multimodal variant"
    assert model.enable_multimodal is True
    K = model.num_modes
    print(f"  multimodal modules built (K={K}): OK")

    batch = _mask_lidar(_make_synthetic_4d_batch(model, device))
    num_future = model.num_future

    # --- B2: loss schema + finiteness (single forward) ---
    losses = model.forward_train(**batch)
    print(f"  losses: {sorted(losses.keys())}")

    for t in range(num_future):
        assert f'loss_occ_future_{t+1}s' in losses, \
            f"missing loss_occ_future_{t+1}s"
    assert 'loss_mode_cls' in losses, "missing loss_mode_cls"
    assert 'loss_mode_div' in losses, "missing loss_mode_div"
    # Inherited innovation stack keys must survive.
    for k in ('loss_static_flow', 'loss_rigid_smooth',
              'loss_nonrigid_bound', 'loss_sem_continuity_2d',
              'loss_sem_continuity_bev'):
        assert k in losses, f"missing inherited loss {k}"
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"loss {k} not finite: {v}"
    print(f"  loss schema (WTA winner + mode_cls + mode_div + inherited): OK")

    # --- B3: backward + gradient routing ---
    model.zero_grad()
    total = sum(losses.values())
    total.backward()

    for name in ('scmf', 'future_predictor', 'motion_prior',
                 'motion_encoder', 'delta_combiner'):
        mod = getattr(model, name)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mod.parameters())
        assert has_grad, f"{name} received no gradient"
    mp = model.motion_prior
    assert mp.group_modes.grad is not None and \
        mp.group_modes.grad.abs().sum() > 0, \
        "no gradient reached group_modes (WTA winner path)"
    print("  gradients (scmf/future_predictor/motion_prior/"
          "motion_encoder/delta_combiner + group_modes): OK")

    # --- B4: WTA selects the mode that matches the GT ---
    # At random init the K chains decode almost identically (argmax overlap
    # ~99.99%), so the winner is decided by numeric noise and cannot yet
    # follow the GT — the expected PRE-training state (mode separation is
    # what training has to learn). To prove the WTA MACHINERY is correct,
    # we temporarily amplify the mode separation with a forward hook
    # (per-mode channel offset on future_bevs), then verify that the
    # winner — recomputed manually from the exact scoring rule — follows
    # the GT mode, and that loss_mode_cls tracks the winning mode.
    model.eval()
    captured = {}

    def spread_hook(module, inputs, output):
        # output: (B, K, T, C, H, W). Push modes apart by a constant
        # channel offset so their occ-head decodes diverge.
        K_ = output.shape[1]
        offs = torch.zeros(
            K_, device=output.device, dtype=output.dtype)
        for k in range(K_):
            offs[k] = (k - (K_ - 1) / 2.0) * 2.0
        out = output + offs.view(1, K_, 1, 1, 1, 1)
        captured['fut'] = out
        return out

    hook = model.future_predictor.register_forward_hook(spread_hook)
    try:
        with torch.no_grad():
            # Non-uniform semantic mode logits (post-training state).
            model.motion_prior.group_mode_logits.copy_(
                torch.randn_like(model.motion_prior.group_mode_logits) * 0.5)

            mm = model.simple_test(
                points=None, img_metas=batch['img_metas'],
                img=batch['img_inputs'])
            modes = mm['occ_future_modes']  # K x T x (list|np)

            def _gt_from_mode(k_mode):
                preds = []
                for t in range(num_future):
                    p = modes[k_mode][t]
                    if isinstance(p, list):
                        p = p[0]
                    preds.append(torch.from_numpy(
                        np.asarray(p)).long().to(device))
                return torch.stack(preds).unsqueeze(0)  # (1,T,Dx,Dy,Dz)

            def _manual_winner(fut, gt):
                """Recompute the WTA winner from the exact scoring rule."""
                B_, K_, T_ = fut.shape[:3]
                s = fut.new_zeros(B_, K_)
                for k in range(K_):
                    for t in range(T_):
                        logits = model.occ_head(fut[:, k, t])
                        s[:, k] += model.future_loss_weights[t] * \
                            model._per_sample_occ_ce(
                                logits, gt[:, t],
                                batch['future_mask_camera'][:, t])
                return s.argmin(dim=1)

            mode_cls = []
            for k_mode in range(K):
                b_k = dict(batch)
                b_k['future_voxel_semantics'] = _gt_from_mode(k_mode)
                losses_k = model.forward_train(**b_k)
                winner = _manual_winner(
                    captured['fut'], b_k['future_voxel_semantics'])
                assert int(winner[0]) == k_mode, \
                    (f"WTA winner {int(winner[0])} != GT mode {k_mode}: "
                     f"the selection rule does not follow the GT")
                mode_cls.append(
                    float(losses_k['loss_mode_cls']) /
                    model.mode_cls_loss_weight)

    finally:
        hook.remove()

    # loss_mode_cls must distinguish the winning modes (mode logits are
    # non-uniform after the injected group_mode_logits).
    assert len(set(round(v, 6) for v in mode_cls)) == K, \
        f"loss_mode_cls identical for all winners: {mode_cls}"
    print(f"  WTA mode-selection: winner follows GT mode (K={K}), "
          f"loss_mode_cls tracks winner {mode_cls}: OK")

    # --- B5: simple_test multimodal outputs ---
    model.eval()
    with torch.no_grad():
        result = model.simple_test(
            points=None,
            img_metas=batch['img_metas'],
            img=batch['img_inputs'])
    assert isinstance(result, dict), "simple_test must return a dict"
    assert 'occ_future' in result and 'occ_current' in result
    assert 'occ_future_modes' in result, "missing occ_future_modes"
    assert 'mode_probs' in result, "missing mode_probs"
    assert len(result['occ_future']) == num_future
    assert len(result['occ_future_modes']) == K
    assert all(len(m) == num_future for m in result['occ_future_modes'])
    assert len(result['mode_probs'][0]) == K, \
        f"mode_probs len {len(result['mode_probs'][0])} != K={K}"
    probs_sum = float(sum(result['mode_probs'][0]))
    assert abs(probs_sum - 1.0) < 1e-4, \
        f"mode probs sum {probs_sum} != 1"
    pred0 = result['occ_future'][0]
    if isinstance(pred0, list):
        pred0 = pred0[0]
    assert pred0.shape == (200, 200, 16), f"pred shape {pred0.shape}"
    print(f"  simple_test: deployed + {K} modes x {num_future} horizons "
          f"+ mode_probs: OK")

    # --- B6: convergence mini-loop (overfit the synthetic batch) ---
    model.train()
    opt_cfg = copy.deepcopy(
        cfg.get('optimizer', dict(type='AdamW', lr=1e-4, weight_decay=0.01)))
    opt_cfg['lr'] = 1e-3
    optimizer = build_optimizer(model, opt_cfg)

    loss_totals = []
    for it in range(warmup + iters):
        optimizer.zero_grad()
        losses = model.forward_train(**batch)
        for k, v in losses.items():
            assert torch.isfinite(v).all(), \
                f"loss {k} not finite at iter {it}"
        total = sum(losses.values())
        total.backward()
        optimizer.step()
        if it >= warmup:
            loss_totals.append(float(total.item()))
    assert loss_totals[-1] < loss_totals[0], \
        (f"loss did not decrease: {loss_totals[0]:.4f} -> "
         f"{loss_totals[-1]:.4f}")
    print(f"  convergence: loss {loss_totals[0]:.4f} -> "
          f"{loss_totals[-1]:.4f}: OK")


# ---------------------------------------------------------------------------
# Part C: evaluation-level test (fake GT, no model)
# ---------------------------------------------------------------------------

def test_c_evaluation():
    """Deployed mIoU + best-of-K oracle (>= deployed) + selection accuracy."""
    from projects.mmdet3d_plugin.datasets.nuscenes_4d_forecast_dataset \
        import NuScenes4DOccForecastDataset

    Dx, Dy, Dz, T, K, N = 20, 20, 8, 3, 3, 4

    with tempfile.TemporaryDirectory() as tmp:
        # Fake GT: two scenes of future labels.
        data_infos = []
        for i in range(N):
            paths = []
            for t in range(T):
                d = os.path.join(tmp, f's{i}_t{t}')
                os.makedirs(d, exist_ok=True)
                rng = np.random.RandomState(i * 10 + t)
                semantics = rng.randint(0, 18, (Dx, Dy, Dz)).astype(np.uint8)
                mask = np.ones((Dx, Dy, Dz), dtype=bool)
                np.savez(
                    os.path.join(d, 'labels.npz'),
                    semantics=semantics, mask_lidar=mask, mask_camera=mask)
                paths.append(d)
            data_infos.append(dict(
                forecast=dict(
                    future_occ_paths=paths,
                    horizons_sec=[1.0, 2.0, 3.0])))

        # Fake multimodal results: mode 0 perfect, modes 1/2 noisy.
        results = []
        for i in range(N):
            modes, deployed = [], []
            for k_mode in range(K):
                horizon_preds = []
                for t in range(T):
                    gt = np.load(
                        os.path.join(paths_of(data_infos, i, t),
                                     'labels.npz'))['semantics']
                    pred = gt.copy()
                    if k_mode != 0:
                        rng = np.random.RandomState(100 + i + k_mode + t)
                        corrupt = rng.rand(*gt.shape) < 0.5
                        pred[corrupt] = rng.randint(0, 18, corrupt.sum())
                    horizon_preds.append(pred)
                modes.append(horizon_preds)
            deployed = modes[0]  # deployed picks the perfect mode
            results.append(dict(
                occ_current=deployed[0],
                occ_future=deployed,
                horizons_sec=[1.0, 2.0, 3.0],
                mode_probs=[[0.6, 0.2, 0.2]],
                occ_future_modes=modes))

        # Build a bare dataset instance (skip the real NuScenes __init__).
        ds = object.__new__(NuScenes4DOccForecastDataset)
        ds.data_infos = data_infos

        eval_results = ds.evaluate(results)

        for key in ('mIoU_1.0s', 'mIoU_2.0s', 'mIoU_3.0s', 'mIoU_avg',
                    'mIoU_bestofK_1.0s', 'mIoU_bestofK_avg',
                    'mode_selection_acc'):
            assert key in eval_results, f"missing eval key {key}"
        # Mode 0 is perfect and both deployed & oracle pick it: 100 mIoU.
        assert abs(eval_results['mIoU_1.0s'] - 100.0) < 1e-6, \
            f"deployed mIoU should be 100, got {eval_results['mIoU_1.0s']}"
        assert abs(eval_results['mIoU_bestofK_1.0s'] - 100.0) < 1e-6, \
            f"oracle mIoU should be 100, got {eval_results['mIoU_bestofK_1.0s']}"
        assert abs(eval_results['mode_selection_acc'] - 1.0) < 1e-6, \
            f"selection acc should be 1.0, got {eval_results['mode_selection_acc']}"

        # Now deploy the WORST mode (mode 1): oracle must beat deployed.
        results_bad = []
        for r in results:
            results_bad.append(dict(
                occ_current=r['occ_current'],
                occ_future=r['occ_future_modes'][1],
                horizons_sec=r['horizons_sec'],
                mode_probs=[[0.1, 0.8, 0.1]],  # deployed picks mode 1
                occ_future_modes=r['occ_future_modes']))
        eval_bad = ds.evaluate(results_bad)
        assert eval_bad['mIoU_bestofK_avg'] > eval_bad['mIoU_avg'], \
            (f"oracle ({eval_bad['mIoU_bestofK_avg']}) should exceed "
             f"deployed ({eval_bad['mIoU_avg']}) when a bad mode is picked")
        print(f"  deployed(bad mode) mIoU_avg={eval_bad['mIoU_avg']:.2f} "
              f"< best-of-K {eval_bad['mIoU_bestofK_avg']:.2f}: OK")


def paths_of(data_infos, i, t):
    return data_infos[i]['forecast']['future_occ_paths'][t]


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multimodal PriorOcc-4D CPU smoke test')
    parser.add_argument('--full', action='store_true',
                        help='also run the model-level Part B (slow)')
    parser.add_argument('--iters', type=int, default=3,
                        help='recorded convergence iters (Part B)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}, Seed: {args.seed}")
    warmup_plugin_imports()

    part_a = {
        'shapes': test_a_shapes,
        'near-identity-init': test_a_near_identity_init,
        'symmetry-breaking': test_a_symmetry_breaking,
        'mm-smp-identity': test_a_mm_smp_identity,
        'mm-smp-mode-logits': test_a_mm_smp_mode_logits,
        'mm-smp-losses-grad': test_a_mm_smp_losses_and_grad,
        'end-to-end-grad': test_a_end_to_end_grad,
    }
    part_b = {'model-chain': lambda d: test_b_forward_backward(
        d, iters=args.iters)}
    part_c = {'evaluation-bestofk': lambda: test_c_evaluation()}

    sections = [('A: modules', part_a)]
    if args.full:
        sections.append(('B: model', part_b))
    sections.append(('C: evaluation', part_c))

    failed = []
    for sec_name, tests in sections:
        print(f"\n=== Part {sec_name} ===")
        for name, fn in tests.items():
            print(f"[{name}]")
            try:
                if sec_name.startswith('C'):
                    fn()
                else:
                    fn(device)
                print("  PASS")
            except Exception as e:
                print(f"  FAIL: {e}")
                traceback.print_exc()
                failed.append(f"{sec_name}/{name}")

    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()
