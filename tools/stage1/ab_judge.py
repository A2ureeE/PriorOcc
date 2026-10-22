"""Judge the hole-filling A/B from two occupancy eval logs.

Rule (from the task spec):
    useful  <=>  A.mIoU >= B.mIoU  AND  A.smallclass >= B.smallclass
    "明显更差" -> useless.  A tiny tolerance (default 0.05 mIoU) absorbs the
    run-to-run noise of cudnn autotune so the verdict is not flipped by noise.

Small classes = pedestrian / bicycle / motorcycle (行人/自行车/摩托).

Primary verdict uses the non-EMA ep1 logs (same weight type as the 33.83
baseline); EMA logs, when given, are printed as corroboration only.

Usage:
    python tools/stage1/ab_judge.py --a a.log --b b.log [--a-ema ...] [--b-ema ...]
Exit code: 0 = useful, 2 = useless, 3 = parse error.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perclass_report import parse  # noqa: E402

SMALL = ['pedestrian', 'bicycle', 'motorcycle']


def small_avg(iou):
    vals = [iou[c] for c in SMALL if c in iou]
    return sum(vals) / len(vals) if vals else float('nan')


def summarize(path):
    iou, n, miou = parse(path, 0)
    if not iou:
        return None
    if miou is None:  # fall back to recomputing from per-class
        miou = sum(iou.values()) / len(iou)
    return dict(path=path, n=n, miou=miou, small=small_avg(iou), iou=iou)


def show(tag, s):
    per = ', '.join('{} {:.2f}'.format(c, s['iou'][c])
                    for c in SMALL if c in s['iou'])
    print('{}: mIoU {:.2f} | ped/bike/moto {:.2f} ({}) | {}'.format(
        tag, s['miou'], s['small'], per, s['path']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--a-ema', default=None)
    ap.add_argument('--b-ema', default=None)
    ap.add_argument('--tol', type=float, default=0.05)
    args = ap.parse_args()

    a = summarize(args.a)
    b = summarize(args.b)
    if a is None or b is None:
        print('PARSE_ERROR: missing per-class IoU in one of the logs')
        return 3

    print('---- hole-filling A/B (ep1, non-EMA, full 4818) ----')
    show('A(on) ', a)
    show('B(off)', b)
    dm = a['miou'] - b['miou']
    ds = a['small'] - b['small']
    print(f'delta(A-B): mIoU {dm:+.2f} | small-class {ds:+.2f} | tol {args.tol}')

    if args.a_ema and args.b_ema:
        ae, be = summarize(args.a_ema), summarize(args.b_ema)
        if ae and be:
            print('---- corroboration (EMA) ----')
            show('A(on) ', ae)
            show('B(off)', be)
            print(f'delta(A-B) EMA: mIoU {ae["miou"] - be["miou"]:+.2f} | '
                  f'small {ae["small"] - be["small"]:+.2f}')

    useful = (dm >= -args.tol) and (ds >= -args.tol)
    print(f'JUDGE {a["miou"]:.2f} {b["miou"]:.2f} {a["small"]:.2f} '
          f'{b["small"]:.2f}')
    print(f'VERDICT={"useful" if useful else "useless"}')
    print('REASON: ' + ('A 在总 mIoU 与小类均值上均不低于 B（容差内）'
                        if useful else
                        'A 在总 mIoU 或小类均值上明显低于 B'))
    return 0 if useful else 2


if __name__ == '__main__':
    sys.exit(main())
