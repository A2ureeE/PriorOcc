"""CPU-side checkpoint compatibility audit (no GPU, no training).

Builds the model described by a config on CPU and compares its ``state_dict``
against a checkpoint key-by-key, reporting missing / unexpected / shape-mismatch
tensors grouped by top-level module. Used as the gate before hot-starting a run
from an external checkpoint (a shape mismatch would otherwise abort mmcv's
``load_checkpoint`` mid-run).

Examples:
    python tools/dryload_check.py projects/configs/priorocc/priorocc-r50-holefill-full.py \
        ckpts/priorocc-singleframe-ep21.pth
    python tools/dryload_check.py projects/configs/priorocc/priorocc-4d-r50-sep-frozen.py \
        /root/autodl-tmp/dataforpriorocc/work_dirs/stage1-holefill/best_ema.pth
"""
import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import projects.mmdet3d_plugin  # noqa: F401,E402  (registers custom modules)
from mmcv import Config  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402


def _group(keys):
    return collections.Counter(k.split('.')[0] for k in keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('ckpt')
    parser.add_argument(
        '--expect-missing',
        action='append',
        default=[],
        help='top-level module whose missing keys are expected (repeatable)')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # never hit the network / rely on torchvision weights for an audit
    if cfg.model.get('img_backbone') is not None:
        cfg.model.img_backbone.pretrained = None
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    ckpt = torch.load(args.ckpt, map_location='cpu')
    src = ckpt.get('state_dict', ckpt)

    tgt = model.state_dict()
    missing = [k for k in tgt if k not in src]
    unexpected = [k for k in src if k not in tgt]
    mismatch = [(k, tuple(src[k].shape), tuple(tgt[k].shape))
                for k in src if k in tgt and tuple(src[k].shape) != tuple(tgt[k].shape)]

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'== config: {args.config}')
    print(f'== ckpt  : {args.ckpt}')
    print(f'params total = {n_params / 1e6:.3f} M, '
          f'trainable = {n_train / 1e6:.3f} M, '
          f'frozen = {(n_params - n_train) / 1e6:.3f} M')
    print(f'matched = {len(tgt) - len(missing) - len(mismatch)} / {len(tgt)}')
    print(f'missing = {len(missing)} {dict(_group(missing))}')
    print(f'unexpected = {len(unexpected)} {dict(_group(unexpected))}')
    print(f'shape_mismatch = {len(mismatch)}')
    for key, got, want in mismatch:
        print(f'  MISMATCH {key}: ckpt {got} vs model {want}')

    # verdict: only explicitly expected modules may be missing
    bad_missing = [k for k in missing
                   if not any(k == p or k.startswith(p + '.')
                              for p in args.expect_missing)]
    if bad_missing or mismatch:
        print(f'VERDICT: FAIL  unexpected_missing={len(bad_missing)} '
              f'shape_mismatch={len(mismatch)}')
        for k in bad_missing[:40]:
            print(f'  MISSING {k}')
        return 1
    print('VERDICT: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
