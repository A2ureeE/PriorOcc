#!/usr/bin/env python3
import os
import re
import sys
import argparse
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIoU_PATTERNS = [
    re.compile(r"mIoU\s*[:=]\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"miou\s*[:=]\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"mean_iou\s*[:=]\s*(\d+\.?\d*)", re.IGNORECASE),
]


def parse_miou(text: str):
    """Try to parse mIoU from test output text.
    Returns (miou_value_float or None, matched_text)
    """
    last_match = None
    for pat in MIoU_PATTERNS:
        for m in pat.finditer(text):
            last_match = m
    if not last_match:
        return None, None
    try:
        val = float(last_match.group(1))
        # Normalize: some logs print percentage like 37.12, others in fraction 0.3712
        if val > 1.0:
            val = val / 100.0
        return val, last_match.group(0)
    except Exception:
        return None, None


def run_test(config, ckpt, gpus, eval_key, cwd):
    cmd = [
        os.path.join(cwd, 'tools', 'dist_test.sh'),
        config,
        ckpt,
        str(gpus),
        '--eval', eval_key,
    ]
    print(f"\n[Run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def save_log(log_dir, epoch, content):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'test_epoch_{epoch}_ema.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return log_path


def main():
    parser = argparse.ArgumentParser(description='Find best mIoU from tests in epoch range')
    parser.add_argument('--config', default='projects/configs/flashocc/flashocc-r50.py', help='Config path')
    parser.add_argument('--work-dir', default='work_dirs/flashocc-r50', help='Work dir for checkpoints')
    parser.add_argument('--start', type=int, default=10, help='Start epoch (inclusive)')
    parser.add_argument('--end', type=int, default=24, help='End epoch (inclusive)')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs used by dist_test.sh')
    parser.add_argument('--eval', default='mAP', help='Eval key passed to test script (default mAP)')
    parser.add_argument('--skip-missing', action='store_true', help='Skip epochs missing checkpoint instead of failing')
    args = parser.parse_args()

    best = {
        'epoch': None,
        'miou': -1.0,
        'match': None,
        'log': None,
    }

    for epoch in range(args.start, args.end + 1):
        ckpt = os.path.join(args.work_dir, f'epoch_{epoch}_ema.pth')
        if not os.path.isfile(ckpt):
            msg = f"[Skip] epoch {epoch}: checkpoint not found: {ckpt}"
            if args.skip_missing:
                print(msg)
                continue
            else:
                print(msg)
                sys.exit(1)

        ret, out = run_test(args.config, ckpt, args.gpus, args.eval, PROJECT_ROOT)
        log_path = save_log(PROJECT_ROOT, epoch, out)

        if ret != 0:
            print(f"[Warn] Test failed for epoch {epoch} (ret={ret}). See log: {log_path}")
            continue

        miou, matched = parse_miou(out)
        if miou is None:
            print(f"[Warn] mIoU not found in epoch {epoch} output. See log: {log_path}")
            continue

        # Rename log to include mIoU in file name
        try:
            new_log_path = os.path.join(PROJECT_ROOT, f'test_epoch_{epoch}_ema_miou_{miou:.4f}.log')
            os.rename(log_path, new_log_path)
            log_path = new_log_path
        except Exception:
            pass

        print(f"[OK] epoch {epoch}: mIoU={miou:.4f} (matched: '{matched}'), log: {log_path}")
        if miou > best['miou']:
            best.update({'epoch': epoch, 'miou': miou, 'match': matched, 'log': log_path})

    print("\n================ Summary ================")
    if best['epoch'] is None:
        print("No valid mIoU parsed in provided epoch range.")
        sys.exit(2)

    print(f"Best epoch: {best['epoch']}")
    print(f"Best mIoU:  {best['miou']:.4f}")
    print(f"Matched:    {best['match']}")
    print(f"Log file:   {best['log']}")
    print(f"Checkpoint: {os.path.join(args.work_dir, f'epoch_{best['epoch']}_ema.pth')}")


if __name__ == '__main__':
    main()