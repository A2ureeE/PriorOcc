"""Build stride-selected sub-pkls for quick hypothesis runs.

Both arms of an A/B must see exactly the same frames, so the same index list is
applied to the plain train pkl and the forecast train pkl.

Usage:
    python tools/stage1/make_subset.py --num 2000
"""
import argparse
import os

import mmcv

ROOT = 'data/nuscenes'
PAIRS = [
    ('bevdetv2-nuscenes_infos_train.pkl',
     'bevdetv2-nuscenes_infos_train_sub{n}.pkl'),
    ('bevdetv2-nuscenes_infos_train_forecast.pkl',
     'bevdetv2-nuscenes_infos_train_forecast_sub{n}.pkl'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=2000)
    args = ap.parse_args()
    n = args.num

    ref = mmcv.load(os.path.join(ROOT, PAIRS[0][0]))
    total = len(ref['infos'])
    stride = max(1, total // n)
    idxs = list(range(0, total, stride))[:n]
    print(f'total={total} stride={stride} selected={len(idxs)}')

    for src, dst in PAIRS:
        data = mmcv.load(os.path.join(ROOT, src))
        sub = dict(data)
        sub['infos'] = [data['infos'][i] for i in idxs]
        out = os.path.join(ROOT, dst.format(n=n))
        mmcv.dump(sub, out)
        print(f'{src} -> {out}  ({len(sub["infos"])} infos, '
              f'{os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
