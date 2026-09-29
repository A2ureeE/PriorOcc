#!/usr/bin/env python
"""Create 4D forecast info pkl files from existing BEVDet info pkls.

Reads bevdetv2-nuscenes_infos_{train,val}.pkl and produces
bevdetv2-nuscenes_infos_{train,val}_forecast.pkl with a 'forecast' dict
attached to each info entry.

nuScenes keyframes are 2 Hz, so 1/2/3 s future horizons correspond to
+2/+4/+6 keyframe offsets (not +1/+2/+3).

The forecast dict contains:
  - history_indices: [idx_t2, idx_t1, idx_t] (indices into sorted infos)
  - history_tokens: [token_t2, token_t1, token_t]
  - future_indices: [idx_t2s, idx_t4s, idx_t6s] (+2/+4/+6 keyframes)
  - future_tokens: [token_t2s, token_t4s, token_t6s]
  - horizons_sec: [1.0, 2.0, 3.0]
  - future_occ_paths: [dir_t2s, dir_t4s, dir_t6s] (containing labels.npz)

Invalid anchors (missing history/future, cross-scene, bad timestamp,
missing GT files) get forecast=None. All entries are preserved so that
index-based lookups in get_adj_info() and evaluate() remain valid.

Usage:
    python tools/create_4d_forecast_infos.py --root-path data/nuscenes
    python tools/create_4d_forecast_infos.py --root-path data/nuscenes --max-samples 20
    python tools/create_4d_forecast_infos.py --root-path data/nuscenes --verify-only
"""

import os
import argparse
import pickle

FUTURE_OFFSETS = [2, 4, 6]
HISTORY_OFFSETS = [2, 1]
HORIZONS_SEC = [1.0, 2.0, 3.0]
TIMESTAMP_TOL_SEC = 0.3


def resolve_occ_path(occ_path, root_path):
    """Resolve occ_path to an existing directory containing labels.npz.

    Try the path as-is first. If labels.npz is not found, attempt to
    remap a leading 'data/nuscenes/' prefix to root_path.
    """
    labels_path = os.path.join(occ_path, 'labels.npz')
    if os.path.exists(labels_path):
        return occ_path

    for prefix in ['./data/nuscenes/', 'data/nuscenes/']:
        if occ_path.startswith(prefix):
            remapped = os.path.join(root_path, occ_path[len(prefix):])
            if os.path.exists(os.path.join(remapped, 'labels.npz')):
                return remapped
            break

    return None


def build_forecast(infos, anchor_idx, root_path, stats):
    """Build forecast dict for a single anchor frame.

    Args:
        infos: Sorted list of info dicts (by timestamp).
        anchor_idx: Index of the anchor (current) frame.
        root_path: Data root path for occ_path remapping.
        stats: Dict for accumulating rejection statistics.

    Returns:
        forecast dict if valid, None if invalid.
    """
    anchor_info = infos[anchor_idx]
    anchor_scene = anchor_info['scene_token']
    anchor_ts = anchor_info['timestamp']

    history_indices = []
    history_tokens = []
    for offset in HISTORY_OFFSETS:
        h_idx = anchor_idx - offset
        if h_idx < 0:
            stats['no_history'] += 1
            return None
        h_info = infos[h_idx]
        if h_info['scene_token'] != anchor_scene:
            stats['no_history'] += 1
            return None
        history_indices.append(h_idx)
        history_tokens.append(h_info['token'])

    history_indices.append(anchor_idx)
    history_tokens.append(anchor_info['token'])

    future_indices = []
    future_tokens = []
    future_occ_paths = []

    for i, offset in enumerate(FUTURE_OFFSETS):
        f_idx = anchor_idx + offset
        if f_idx >= len(infos):
            stats['cross_scene'] += 1
            return None

        f_info = infos[f_idx]

        if f_info['scene_token'] != anchor_scene:
            stats['cross_scene'] += 1
            return None

        dt_sec = (f_info['timestamp'] - anchor_ts) / 1e6
        if abs(dt_sec - HORIZONS_SEC[i]) > TIMESTAMP_TOL_SEC:
            stats['ts_mismatch'] += 1
            return None

        occ_path = resolve_occ_path(f_info['occ_path'], root_path)
        if occ_path is None:
            stats['missing_occ'] += 1
            return None

        future_indices.append(f_idx)
        future_tokens.append(f_info['token'])
        future_occ_paths.append(occ_path)

    return {
        'history_indices': history_indices,
        'history_tokens': history_tokens,
        'future_indices': future_indices,
        'future_tokens': future_tokens,
        'horizons_sec': list(HORIZONS_SEC),
        'future_occ_paths': future_occ_paths,
    }


def verify_forecast_pkl(forecast_pkl):
    """Verify a forecast pkl: check all forecast dicts are internally consistent."""
    with open(forecast_pkl, 'rb') as f:
        dataset = pickle.load(f)
    infos = dataset['infos']

    valid = 0
    for idx, info in enumerate(infos):
        fc = info.get('forecast', None)
        if fc is None:
            continue
        valid += 1

        assert len(fc['history_indices']) == 3, \
            f"history_indices len != 3 at idx {idx}"
        assert fc['history_indices'][2] == idx, \
            f"history_indices[2] != anchor idx at idx {idx}"
        for h_idx, h_token in zip(fc['history_indices'], fc['history_tokens']):
            assert 0 <= h_idx < len(infos), \
                f"history index out of bounds at idx {idx}"
            assert infos[h_idx]['scene_token'] == info['scene_token'], \
                f"history cross-scene at idx {idx}"
            assert infos[h_idx]['token'] == h_token, \
                f"history token mismatch at idx {idx}"

        assert len(fc['future_indices']) == 3, \
            f"future_indices len != 3 at idx {idx}"
        for i, (f_idx, f_token) in enumerate(
                zip(fc['future_indices'], fc['future_tokens'])):
            assert 0 <= f_idx < len(infos), \
                f"future index out of bounds at idx {idx}"
            assert infos[f_idx]['scene_token'] == info['scene_token'], \
                f"future cross-scene at idx {idx}"
            assert infos[f_idx]['token'] == f_token, \
                f"future token mismatch at idx {idx}"
            labels_path = os.path.join(fc['future_occ_paths'][i], 'labels.npz')
            assert os.path.exists(labels_path), \
                f"future occ missing: {labels_path} at idx {idx}"

        assert fc['horizons_sec'] == [1.0, 2.0, 3.0], \
            f"horizons_sec mismatch at idx {idx}"

    print(f'  {os.path.basename(forecast_pkl)}: '
          f'{valid} valid / {len(infos)} total anchors')


def create_forecast_infos(root_path, extra_tag, version,
                           max_samples=None, verify_only=False):
    """Create forecast info pkl files from existing BEVDet info pkls."""
    dataroot = root_path if root_path.endswith('/') else root_path + '/'

    if verify_only:
        for split in ['train', 'val']:
            forecast_pkl = os.path.join(
                dataroot, f'{extra_tag}_infos_{split}_forecast.pkl')
            if os.path.exists(forecast_pkl):
                verify_forecast_pkl(forecast_pkl)
            else:
                print(f'Skipping {split}: {forecast_pkl} not found')
        return

    for split in ['train', 'val']:
        input_pkl = os.path.join(dataroot, f'{extra_tag}_infos_{split}.pkl')
        output_pkl = os.path.join(
            dataroot, f'{extra_tag}_infos_{split}_forecast.pkl')

        if not os.path.exists(input_pkl):
            print(f'Skipping {split}: {input_pkl} not found')
            continue

        print(f'\nProcessing {split}: {input_pkl}')
        with open(input_pkl, 'rb') as f:
            dataset = pickle.load(f)

        infos = sorted(dataset['infos'], key=lambda e: e['timestamp'])
        dataset['infos'] = infos

        stats = {
            'total': len(infos), 'valid': 0, 'invalid': 0,
            'no_history': 0, 'cross_scene': 0,
            'ts_mismatch': 0, 'missing_occ': 0,
        }

        valid_count = 0
        for idx, info in enumerate(infos):
            forecast = build_forecast(infos, idx, root_path, stats)
            if forecast is not None:
                valid_count += 1
                if max_samples is not None and valid_count > max_samples:
                    info['forecast'] = None
                    stats['invalid'] += 1
                    continue
                info['forecast'] = forecast
                stats['valid'] += 1
            else:
                info['forecast'] = None
                stats['invalid'] += 1

        print(f'  total={stats["total"]}, valid={stats["valid"]}, '
              f'invalid={stats["invalid"]}')
        print(f'  reasons: no_history={stats["no_history"]}, '
              f'cross_scene={stats["cross_scene"]}, '
              f'ts_mismatch={stats["ts_mismatch"]}, '
              f'missing_occ={stats["missing_occ"]}')

        with open(output_pkl, 'wb') as f:
            pickle.dump(dataset, f)
        print(f'  Saved: {output_pkl}')

    print('\n--- Verification ---')
    for split in ['train', 'val']:
        forecast_pkl = os.path.join(
            dataroot, f'{extra_tag}_infos_{split}_forecast.pkl')
        if os.path.exists(forecast_pkl):
            verify_forecast_pkl(forecast_pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create 4D forecast info pkl files')
    parser.add_argument('--root-path', type=str, default='data/nuscenes',
                        help='Path to nuScenes dataset root')
    parser.add_argument('--extra-tag', type=str, default='bevdetv2-nuscenes',
                        help='Prefix for info filenames')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'],
                        help='nuScenes dataset version')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit valid forecast anchors for debugging')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing _forecast.pkl without writing')
    args = parser.parse_args()

    create_forecast_infos(
        root_path=args.root_path,
        extra_tag=args.extra_tag,
        version=args.version,
        max_samples=args.max_samples,
        verify_only=args.verify_only,
    )
