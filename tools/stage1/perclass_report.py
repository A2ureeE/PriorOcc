"""Per-class IoU report / comparison for occupancy eval logs.

Parses the ``===> <class> - IoU = X`` lines printed by
``Metric_mIoU.count_miou()`` and compares two runs.

The 18 occ classes collapse into three groups:
  dynamic    : barrier bicycle bus car construction_vehicle motorcycle
               pedestrian traffic_cone trailer truck
  continuous : driveable_surface other_flat sidewalk terrain manmade
               vegetation            <- the "background hole-filling" targets
  static     : continuous + others

Examples:
    # one log, grouped summary
    python tools/stage1/perclass_report.py eval_ep0.log
    # compare stage-1 against the ep21 baseline (same 4818 pool)
    python tools/stage1/perclass_report.py stage1_ep12.log \
        --baseline experiment_reports/raw_eval/eval_ep21_same4818.log
    # single machine-readable line for curve files
    python tools/stage1/perclass_report.py log.txt --emit
"""
import argparse
import re
import sys

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
DYNAMIC = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck']
CONTINUOUS = ['driveable_surface', 'other_flat', 'sidewalk', 'terrain',
              'manmade', 'vegetation']
STATIC = ['others'] + CONTINUOUS

_IOU_RE = re.compile(r'===>\s+(\w+)\s+-\s+IoU\s+=\s+([0-9.]+)')
_MIOU_RE = re.compile(r'===>\s+mIoU of (\d+) samples:\s+([0-9.]+)')
# Each metric block starts with "===> per class IoU of N samples:"; a 4D log
# stacks four of them (current / 1s / 2s / 3s) in the same file.
_SECTION_RE = re.compile(r'===> per class IoU of \d+ samples:')


def parse(path, section=None):
    """Parse one metric block.

    Args:
        path: eval log.
        section: 0 = current frame, 1/2/3 = 1s/2s/3s; None = merge all
            blocks (only sane for single-frame logs).
    """
    text = open(path, 'r', errors='ignore').read().replace('\r', '\n')
    if section is not None:
        parts = _SECTION_RE.split(text)
        if section + 1 < len(parts):
            text = parts[section + 1]
    iou = {m.group(1): float(m.group(2)) for m in _IOU_RE.finditer(text)}
    m = _MIOU_RE.search(text)
    n, miou = (int(m.group(1)), float(m.group(2))) if m else (None, None)
    return iou, n, miou


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float('nan')


def group(iou):
    return dict(
        mIoU=mean([iou.get(c) for c in DYNAMIC + STATIC]),
        dynamic=mean([iou.get(c) for c in DYNAMIC]),
        static=mean([iou.get(c) for c in STATIC]),
        continuous=mean([iou.get(c) for c in CONTINUOUS]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--section', type=int, default=0,
                    help='metric block: 0=current(默认), 1/2/3=1s/2s/3s')
    ap.add_argument('--emit', action='store_true',
                    help='print "mIoU dynamic static continuous" only')
    args = ap.parse_args()

    iou, n, miou = parse(args.log, args.section)
    if not iou:
        print(f'ERROR: no per-class IoU found in {args.log}', file=sys.stderr)
        return 2
    g = group(iou)
    if miou is not None:
        g['mIoU'] = miou

    if args.emit:
        print(f'{g["mIoU"]:.2f} {g["dynamic"]:.2f} {g["static"]:.2f} '
              f'{g["continuous"]:.2f}')
        return 0

    base = None
    if args.baseline:
        base_iou, base_n, base_miou = parse(args.baseline, args.section)
        if not base_iou:
            print(f'ERROR: no per-class IoU in baseline {args.baseline}',
                  file=sys.stderr)
            return 2
        base = group(base_iou)
        if base_miou is not None:
            base['mIoU'] = base_miou

    title = f'{args.log}  ({n} samples, mIoU {g["mIoU"]:.2f})'
    print(title)
    if base is not None:
        print(f'baseline: {args.baseline}  ({base_n} samples, '
              f'mIoU {base["mIoU"]:.2f})')
        print(f'{"":22s}{"baseline":>10s}{"new":>10s}{"delta":>10s}')
        for key in ['mIoU', 'dynamic', 'static', 'continuous']:
            print(f'{"[group] " + key:22s}{base[key]:10.2f}{g[key]:10.2f}'
                  f'{g[key] - base[key]:+10.2f}')
        print()
        for c in CONTINUOUS + ['others'] + DYNAMIC:
            b, v = base_iou.get(c), iou.get(c)
            if b is None or v is None:
                continue
            tag = ' <- continuous' if c in CONTINUOUS else ''
            print(f'{c:22s}{b:10.2f}{v:10.2f}{v - b:+10.2f}{tag}')
    else:
        for key in ['mIoU', 'dynamic', 'static', 'continuous']:
            print(f'{"[group] " + key:22s}{g[key]:10.2f}')
        print()
        for c in CONTINUOUS + ['others'] + DYNAMIC:
            if c in iou:
                print(f'{c:22s}{iou[c]:10.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
