#!/usr/bin/env python3
"""Multimodal occupancy-forecast uncertainty visualization.

Consumes the K-mode output of a multimodal PriorOcc4D model
(``occ_future_modes`` + ``mode_probs``) and renders WHERE the K future
hypotheses disagree — i.e. the model's predictive uncertainty. This is a
capability a deterministic forecaster fundamentally cannot offer.

Three uncertainty measures
---------------------------
A1 mode agreement   : per-voxel fraction of the K modes sharing the modal
                      semantic class. 1.0 = all agree (certain),
                      1/K = full disagreement (uncertain). Needs argmax only.
A2 occupied variance: variance across modes of the binary occupied/free
                      indicator (label != FREE). Safety-oriented: "is there
                      something here or not?". Needs argmax only.
B  predictive entropy: entropy of the pi-weighted fused class distribution
                      p(c|v) = sum_k pi_k * softmax_k(v). Smoothest, needs
                      per-mode voxel probabilities.

Data sources (pick one)
-----------------------
--self-test            : synthetic K-mode data, validates the whole pipeline
                         on CPU with no checkpoint / GPU / dataset needed.
--modes-npz PATH       : offline. NPZ with `mode_labels` (K,T,Dx,Dy,Dz) and
                         optional `mode_probs` (K,T,Dx,Dy,Dz,C) and
                         `mode_pi` (T,K) and `gt` (T,Dx,Dy,Dz).
--config C --checkpoint P : run a multimodal model's simple_test on one
                         sample (needs the real env + GPU + _forecast.pkl).

Headless: uses the Agg backend, so it runs on any CPU-only machine.

Examples
--------
    # 1) validate the pipeline with synthetic data (CPU, no deps beyond np/mpl)
    python tools/vis_multimodal_uncertainty.py --self-test \
        --output-dir work_dirs/mm_uncertainty_demo

    # 2) offline from an exported NPZ
    python tools/vis_multimodal_uncertainty.py \
        --modes-npz work_dirs/mm_out/sample0_modes.npz \
        --output-dir work_dirs/mm_uncertainty

    # 3) from a trained multimodal checkpoint (needs GPU + forecast data)
    python tools/vis_multimodal_uncertainty.py \
        --config projects/configs/priorocc/priorocc-4d-r50-mmodal.py \
        --checkpoint work_dirs/priorocc-4d-mmodal/latest.pth \
        --sample-index 0 --output-dir work_dirs/mm_uncertainty
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --------------------------------------------------------------------------
# Constants (self-contained; mirror tools/vis_occ_matplotlib.py)
# --------------------------------------------------------------------------
FREE_LABEL = 17
NUM_CLASSES = 18
VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

# RGB per class (0-255); index 17 (free) rendered as near-white background.
COLORMAP_255 = np.array([
    [0, 0, 0],           # 0 others
    [255, 120, 50],      # 1 barrier
    [255, 192, 203],     # 2 bicycle
    [255, 255, 0],       # 3 bus
    [0, 150, 245],       # 4 car
    [0, 255, 255],       # 5 construction_vehicle
    [255, 0, 255],       # 6 motorcycle
    [255, 0, 0],         # 7 pedestrian
    [128, 64, 128],      # 8 traffic_cone (placeholder)
    [0, 0, 255],         # 9 trailer
    [100, 100, 255],     # 10 truck
    [128, 128, 128],     # 11 driveable_surface
    [64, 64, 128],       # 12 other_flat
    [0, 128, 0],         # 13 sidewalk
    [128, 128, 0],       # 14 terrain
    [192, 192, 128],     # 15 manmade
    [0, 200, 0],         # 16 vegetation
    [255, 255, 255],     # 17 free
], dtype=np.uint8)
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free']


def _label_cmap():
    return ListedColormap(COLORMAP_255.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------
# Uncertainty measures
# --------------------------------------------------------------------------
def mode_agreement(labels):
    """Per-voxel modal-class agreement across modes.

    Args:
        labels: (K, ...) int array of argmax class per mode.

    Returns:
        agreement: (...) float in [1/K, 1]; higher = more certain.
    """
    K = labels.shape[0]
    spatial = labels.shape[1:]
    flat = np.asarray(labels).reshape(K, -1).astype(np.int64)  # (K, N)
    N = flat.shape[1]
    # matches[c, k, n] = (flat[k, n] == c)
    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    matches = (flat.T[:, None, :] == classes[None, :, None])  # (N, C, K)
    counts = matches.sum(axis=2)  # (N, C) how many modes vote class c
    modal_count = counts.max(axis=1)  # (N,)
    agreement = modal_count.astype(np.float32) / float(K)
    return agreement.reshape(spatial)


def occupied_variance(labels):
    """Variance across modes of the binary occupied/free indicator.

    Args:
        labels: (K, ...) int argmax class per mode.

    Returns:
        occ_var: (...) float in [0, 0.25]; higher = more uncertain geometry.
    """
    occupied = (np.asarray(labels) != FREE_LABEL).astype(np.float32)
    return occupied.var(axis=0)


def predictive_entropy(probs, pi):
    """Entropy of the pi-weighted fused class distribution.

    Args:
        probs: (K, ..., C) per-mode class probabilities.
        pi: (K,) mode probabilities (will be normalized).

    Returns:
        entropy: (...) float in [0, 1] (normalized by log C); higher = more
        uncertain.
    """
    probs = np.asarray(probs, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    pi = pi / (pi.sum() + 1e-12)
    # fused(c, v) = sum_k pi_k * probs_k(c, v)
    fused = np.tensordot(pi, probs, axes=(0, 0))  # (..., C)
    eps = 1e-12
    ent = -(fused * np.log(fused + eps)).sum(axis=-1)
    ent = ent / np.log(fused.shape[-1])  # normalize to [0, 1]
    return ent.astype(np.float32)


# --------------------------------------------------------------------------
# BEV projection helpers
# --------------------------------------------------------------------------
def label_to_bev(label):
    """Project a (Dx, Dy, Dz) label grid to a top-down (Dx, Dy) label map.

    For each (x, y) column take the top-most (largest z) occupied voxel;
    columns that are entirely free stay FREE_LABEL.
    """
    label = np.asarray(label)
    Dx, Dy, Dz = label.shape
    bev = np.full((Dx, Dy), FREE_LABEL, dtype=np.int64)
    occupied = label != FREE_LABEL
    for z in range(Dz):  # later z overwrites -> keeps the top-most
        m = occupied[:, :, z]
        bev[m] = label[m, z]
    return bev


def scalar_to_bev(scalar, reduce='max'):
    """Project a (Dx, Dy, Dz) scalar field to (Dx, Dy) by max/mean over z."""
    scalar = np.asarray(scalar, dtype=np.float32)
    if reduce == 'mean':
        return scalar.mean(axis=2)
    return scalar.max(axis=2)


def colorize_bev(bev_label):
    """Map a (Dx, Dy) label grid to an (Dx, Dy, 3) float RGB image."""
    bev_label = np.clip(bev_label, 0, NUM_CLASSES - 1).astype(np.int64)
    rgb = COLORMAP_255[bev_label].astype(np.float32) / 255.0
    return rgb


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _imshow_label(ax, bev_label, title):
    img = colorize_bev(bev_label)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_heat(ax, field, title, cmap, vmin=0.0, vmax=1.0):
    im = ax.imshow(np.transpose(field, (1, 0)), origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_horizon(mode_labels_t, agreement_t, occvar_t, entropy_t,
                 pi_t, gt_t, out_path, horizon_label):
    """Render one figure comparing K modes + uncertainty maps for one horizon.

    Args:
        mode_labels_t: (K, Dx, Dy, Dz) argmax labels per mode.
        agreement_t: (Dx, Dy, Dz) mode-agreement field.
        occvar_t: (Dx, Dy, Dz) occupied-variance field.
        entropy_t: (Dx, Dy, Dz) entropy field or None.
        pi_t: (K,) mode probabilities or None.
        gt_t: (Dx, Dy, Dz) ground-truth labels or None.
        out_path: where to save the PNG.
        horizon_label: e.g. 't+1s'.
    """
    K = mode_labels_t.shape[0]
    has_ent = entropy_t is not None
    has_gt = gt_t is not None

    # Row 1: K mode BEVs. Row 2: agreement / occ-var / (entropy) / (gt).
    n_col_row2 = 2 + int(has_ent) + int(has_gt)
    n_cols = max(K, n_col_row2)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 6.4))
    axes = np.atleast_2d(axes)

    pi_str = ''
    if pi_t is not None:
        pi_str = '  pi=[' + ', '.join(f'{p:.2f}' for p in pi_t) + ']'
    fig.suptitle(f'Multimodal occupancy forecast uncertainty — {horizon_label}'
                 f'{pi_str}', fontsize=11)

    # Row 1: each mode's top-down semantic BEV.
    for k in range(K):
        _imshow_label(axes[0, k], label_to_bev(mode_labels_t[k]),
                      f'mode {k}' + (f' (pi={pi_t[k]:.2f})'
                                     if pi_t is not None else ''))
    for j in range(K, n_cols):
        axes[0, j].axis('off')

    # Row 2: uncertainty maps (red = uncertain).
    col = 0
    _imshow_heat(axes[1, col], scalar_to_bev(agreement_t),
                 'mode agreement (low=uncertain)', 'RdYlGn', 0.0, 1.0)
    col += 1
    _imshow_heat(axes[1, col], scalar_to_bev(occvar_t),
                 'occupied variance (high=uncertain)', 'hot', 0.0, 0.25)
    col += 1
    if has_ent:
        _imshow_heat(axes[1, col], scalar_to_bev(entropy_t),
                     'predictive entropy (high=uncertain)', 'inferno', 0.0, 1.0)
        col += 1
    if has_gt:
        _imshow_label(axes[1, col], label_to_bev(gt_t), 'GT')
        col += 1
    for j in range(col, n_cols):
        axes[1, j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def summarize(agreement, occvar, entropy):
    """Return a small text summary of the uncertainty fields."""
    lines = [
        f'  mode agreement : mean={agreement.mean():.3f} '
        f'min={agreement.min():.3f} '
        f'frac<1.0={(agreement < 1.0).mean():.3f}',
        f'  occupied var   : mean={occvar.mean():.4f} '
        f'max={occvar.max():.4f} '
        f'frac>0={(occvar > 0).mean():.4f}',
    ]
    if entropy is not None:
        lines.append(
            f'  pred. entropy  : mean={entropy.mean():.3f} '
            f'max={entropy.max():.3f}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_modes_npz(path):
    """Load exported multimodal predictions from an NPZ.

    Expected keys:
        mode_labels: (K, T, Dx, Dy, Dz) int argmax labels per mode/horizon.
        mode_probs : (K, T, Dx, Dy, Dz, C) float, optional (enables entropy).
        mode_pi    : (T, K) float, optional per-horizon mode probabilities.
        gt         : (T, Dx, Dy, Dz) int, optional ground truth.
        horizons_sec: (T,) float, optional.

    Returns:
        dict with the loaded arrays (missing optional keys -> None).
    """
    data = np.load(path, allow_pickle=False)
    if 'mode_labels' not in data:
        raise KeyError(f"{path} missing 'mode_labels' (K,T,Dx,Dy,Dz)")
    out = dict(
        mode_labels=data['mode_labels'].astype(np.int64),
        mode_probs=(data['mode_probs'] if 'mode_probs' in data else None),
        mode_pi=(data['mode_pi'] if 'mode_pi' in data else None),
        gt=(data['gt'] if 'gt' in data else None),
        horizons_sec=(data['horizons_sec'] if 'horizons_sec' in data else None),
    )
    return out


def _export_modes_npz(result, out_npz):
    """Pack a multimodal simple_test result dict into an NPZ for offline use.

    result keys consumed: occ_future_modes (K x T x [np/list]), mode_probs
    (B x K, B assumed 1), horizons_sec.
    """
    modes = result['occ_future_modes']  # K x T, each (list|np) (Dx,Dy,Dz)
    K = len(modes)
    T = len(modes[0])

    def _arr(x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return np.asarray(x)

    mode_labels = np.stack(
        [[_arr(modes[k][t]) for t in range(T)] for k in range(K)]
    ).astype(np.int64)  # (K,T,Dx,Dy,Dz)

    save_kwargs = dict(mode_labels=mode_labels)
    if result.get('mode_probs'):
        pi = np.asarray(result['mode_probs'][0], dtype=np.float32)  # (K,)
        save_kwargs['mode_pi'] = np.tile(pi[None, :], (T, 1))  # (T,K)
    if result.get('horizons_sec'):
        save_kwargs['horizons_sec'] = np.asarray(
            result['horizons_sec'], dtype=np.float32)
    np.savez_compressed(out_npz, **save_kwargs)
    return out_npz


# --------------------------------------------------------------------------
# Inference from a checkpoint (needs real env + GPU + _forecast.pkl)
# --------------------------------------------------------------------------
def run_inference(config_path, checkpoint_path, sample_index=0):
    """Build the multimodal model, run simple_test on one sample.

    Returns:
        dict with mode_labels (K,T,Dx,Dy,Dz), mode_pi (T,K) or None,
        horizons_sec list or None.
    """
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset, build_dataloader

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    cfg = Config.fromfile(config_path)
    if getattr(cfg, 'plugin', False):
        import importlib
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device).eval()

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, shuffle=False,
        dist=False)

    for i, data in enumerate(loader):
        if i != sample_index:
            continue
        # Scatter to the model device (single-GPU / CPU simple path).
        from mmcv.parallel import collate, scatter
        img_metas = data['img_metas'][0].data[0]
        img = data['img_inputs'][0].data[0]
        with torch.no_grad():
            result = model.simple_test(
                points=None, img_metas=img_metas, img=img)
        if not isinstance(result, dict) or 'occ_future_modes' not in result:
            raise RuntimeError(
                'Model did not return multimodal outputs. Ensure '
                'enable_multimodal=True in the config.')
        mode_labels = np.stack([
            np.stack([np.asarray(m[0] if isinstance(m, list) else m)
                      for m in result['occ_future_modes'][k]])
            for k in range(len(result['occ_future_modes']))
        ]).astype(np.int64)
        mode_pi = None
        if result.get('mode_probs'):
            pi = np.asarray(result['mode_probs'][0], dtype=np.float32)
            mode_pi = np.tile(pi[None, :], (mode_labels.shape[1], 1))
        return dict(
            mode_labels=mode_labels, mode_pi=mode_pi, mode_probs=None,
            gt=None, horizons_sec=result.get('horizons_sec'))
    raise IndexError(
        f'sample_index {sample_index} out of range for the test set')


# --------------------------------------------------------------------------
# Self-test with synthetic multimodal data (CPU, no checkpoint)
# --------------------------------------------------------------------------
def make_synthetic_modes(K=3, T=3, Dx=200, Dy=200, Dz=16, seed=0):
    """Synthesize K-mode futures with a controllable disagreement region.

    Most voxels are identical across modes (static background -> agreement=1).
    A moving "object" block occupies different positions per mode, creating a
    localized high-uncertainty region — exactly what the tool should reveal.
    """
    rng = np.random.RandomState(seed)
    # Static background: shared across modes (drivable + some static classes).
    base = np.full((Dx, Dy, Dz), FREE_LABEL, dtype=np.int64)
    base[:, :, 0:2] = 11  # ground (driveable_surface)
    base[20:40, 20:40, 2:8] = 15  # a manmade block
    base[150:170, 150:170, 2:6] = 16  # vegetation

    labels = np.repeat(base[None], K, axis=0)  # (K,Dx,Dy,Dz)

    # A "vehicle" whose future position differs per mode (turn/straight/brake).
    for k in range(K):
        cx = 100 + k * 8            # spreads along x per mode
        cy = 100 + (k % 2) * 6
        labels[k, cx:cx + 10, cy:cy + 6, 2:6] = 4  # car
    # Per-horizon: shift the disagreement further at longer horizons.
    modes = np.repeat(labels[None], T, axis=0).transpose(1, 0, 2, 3, 4)
    # modes shape (K, T, Dx, Dy, Dz)
    for t in range(T):
        for k in range(K):
            shift = (t + 1) * (k - (K - 1) / 2.0) * 4
            s = int(round(shift))
            if s != 0:
                modes[k, t] = np.roll(modes[k, t], s, axis=0)

    # Per-mode probabilities (K,T,Dx,Dy,Dz,C): mostly confident, blended.
    probs = np.zeros((K, T, Dx, Dy, Dz, NUM_CLASSES), dtype=np.float32)
    for k in range(K):
        for t in range(T):
            oh = np.eye(NUM_CLASSES, dtype=np.float32)[modes[k, t]]
            probs[k, t] = 0.7 * oh + 0.3 / NUM_CLASSES
    pi = np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32)[:K], (T, 1))
    gt = modes[0].copy()  # pretend mode 0 is the truth
    horizons = np.array([1.0, 2.0, 3.0], dtype=np.float32)[:T]
    return dict(mode_labels=modes, mode_probs=probs, mode_pi=pi, gt=gt,
                horizons_sec=horizons)


# --------------------------------------------------------------------------
# Core render loop (shared by all data sources)
# --------------------------------------------------------------------------
def render(data, output_dir, metrics=('agreement', 'occupancy', 'entropy')):
    """Compute uncertainty and render one figure per horizon.

    Args:
        data: dict from load_modes_npz / make_synthetic_modes / run_inference.
        output_dir: where to write PNGs.
        metrics: which measures to render.

    Returns:
        list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = data['mode_labels']  # (K,T,Dx,Dy,Dz)
    K, T = labels.shape[0], labels.shape[1]
    probs = data.get('mode_probs')
    pi_all = data.get('mode_pi')  # (T,K) or None
    gt = data.get('gt')           # (T,Dx,Dy,Dz) or None
    horizons = data.get('horizons_sec')

    want_ent = 'entropy' in metrics and probs is not None
    if 'entropy' in metrics and probs is None:
        print('  [note] entropy requested but no mode_probs provided; '
              'skipping entropy map.')

    written = []
    for t in range(T):
        lab_t = labels[:, t]  # (K,Dx,Dy,Dz)
        agreement_t = mode_agreement(lab_t)
        occvar_t = occupied_variance(lab_t)
        entropy_t = None
        if want_ent:
            pi_t = (pi_all[t] if pi_all is not None
                    else np.ones(K, dtype=np.float32) / K)
            entropy_t = predictive_entropy(probs[:, t], pi_t)
        pi_t = pi_all[t] if pi_all is not None else None
        gt_t = gt[t] if gt is not None else None

        hz = (f't+{horizons[t]:.0f}s' if horizons is not None
              else f'horizon {t}')
        print(f'[{hz}] uncertainty summary:\n'
              + summarize(agreement_t, occvar_t, entropy_t))

        out_path = os.path.join(output_dir, f'uncertainty_{t:02d}.png')
        plot_horizon(lab_t, agreement_t, occvar_t, entropy_t, pi_t, gt_t,
                     out_path, hz)
        written.append(out_path)
        print(f'  -> saved {out_path}')
    return written


def run_self_test(output_dir):
    print('[self-test] synthesizing K=3 mode futures (CPU)...')
    data = make_synthetic_modes(K=3, T=3)
    print(f'  mode_labels: {data["mode_labels"].shape}, '
          f'mode_probs: {data["mode_probs"].shape}, '
          f'mode_pi: {data["mode_pi"].shape}, gt: {data["gt"].shape}')

    # Sanity assertions on the measures.
    lab = data['mode_labels'][:, 0]
    agr = mode_agreement(lab)
    assert agr.min() >= 1.0 / lab.shape[0] - 1e-6 and agr.max() <= 1.0 + 1e-6, \
        'agreement out of [1/K, 1]'
    # Background (all modes identical) must be fully certain.
    assert np.isclose(agr[0, 0, 0], 1.0), 'static background should agree'
    ov = occupied_variance(lab)
    assert ov.min() >= 0.0 and ov.max() <= 0.25 + 1e-6, 'occ-var out of range'
    ent = predictive_entropy(data['mode_probs'][:, 0], data['mode_pi'][0])
    assert ent.min() >= 0.0 and ent.max() <= 1.0 + 1e-6, 'entropy out of [0,1]'
    # There must EXIST a disagreement region (the moving block), else the
    # synthetic case is degenerate.
    assert (agr < 1.0).any(), 'no disagreement region found in synthetic data'
    print('  measure sanity: agreement/occ-var/entropy ranges OK, '
          'disagreement region present')

    written = render(data, output_dir)
    assert len(written) == data['mode_labels'].shape[1], 'not all horizons rendered'
    for p in written:
        assert os.path.exists(p) and os.path.getsize(p) > 1000, \
            f'output image missing/too small: {p}'
    print(f'[self-test] PASS — {len(written)} figures written to {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Multimodal occupancy-forecast uncertainty visualization')
    src = parser.add_argument_group('data source (pick one)')
    src.add_argument('--self-test', action='store_true',
                     help='run on synthetic data (CPU, no checkpoint)')
    src.add_argument('--modes-npz', type=str, default=None,
                     help='offline NPZ with mode_labels[/mode_probs/mode_pi/gt]')
    src.add_argument('--config', type=str, default=None,
                     help='multimodal model config (inference mode)')
    src.add_argument('--checkpoint', type=str, default=None,
                     help='model weights (inference mode)')
    src.add_argument('--sample-index', type=int, default=0,
                     help='test-set sample index (inference mode)')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/mm_uncertainty')
    parser.add_argument('--metric', type=str, default='all',
                        choices=['all', 'agreement', 'occupancy', 'entropy'],
                        help='which uncertainty measure(s) to render')
    parser.add_argument('--export-npz', type=str, default=None,
                        help='in inference mode, also dump modes to this NPZ')
    args = parser.parse_args()

    if args.metric == 'all':
        metrics = ('agreement', 'occupancy', 'entropy')
    else:
        metrics = (args.metric,)

    if args.self_test:
        run_self_test(args.output_dir)
        return

    if args.modes_npz:
        print(f'[npz] loading {args.modes_npz}')
        data = load_modes_npz(args.modes_npz)
        render(data, args.output_dir, metrics)
        return

    if args.config and args.checkpoint:
        print(f'[infer] config={args.config}')
        print(f'[infer] checkpoint={args.checkpoint} sample={args.sample_index}')
        data = run_inference(args.config, args.checkpoint, args.sample_index)
        if args.export_npz:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_npz)),
                        exist_ok=True)
            np.savez_compressed(
                args.export_npz,
                mode_labels=data['mode_labels'],
                **({'mode_pi': data['mode_pi']}
                   if data['mode_pi'] is not None else {}),
                **({'horizons_sec': np.asarray(data['horizons_sec'],
                                               dtype=np.float32)}
                   if data.get('horizons_sec') else {}))
            print(f'[infer] exported modes -> {args.export_npz}')
        render(data, args.output_dir, metrics)
        return

    parser.error(
        'provide one of: --self-test | --modes-npz PATH | '
        '--config C --checkpoint P')


if __name__ == '__main__':
    main()
