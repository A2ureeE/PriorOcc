#!/usr/bin/env python3

import os
import subprocess
import argparse
import re

def parse_miou(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
    match = re.search(r'mIoU of \d+ samples:\s*(\d+\.\d+)', content)
    if match:
        return float(match.group(1))
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the epoch with the best mIoU from test logs.')
    parser.add_argument('--config', required=True, help='Path to the config file')
    parser.add_argument('--work-dir', required=True, help='Work directory containing the checkpoints')
    parser.add_argument('--start', type=int, default=10, help='Starting epoch')
    parser.add_argument('--end', type=int, default=24, help='Ending epoch')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--eval', default='mAP', help='Evaluation key for dist_test.sh')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    best = {'miou': 0, 'epoch': None, 'log': None}

    for epoch in range(args.start, args.end + 1):
        ckpt = os.path.join(args.work_dir, f'epoch_{epoch}_ema.pth')
        log_file = f'test_epoch_{epoch}_ema.log'
        log_path = os.path.join(project_root, log_file)

        cmd = f'./tools/dist_test.sh {args.config} {ckpt} {args.gpus} --eval {args.eval} > {log_path} 2>&1'
        subprocess.call(cmd, shell=True)

        miou = parse_miou(log_path)
        if miou is not None:
            renamed_log = f'test_epoch_{epoch}_ema_miou_{miou:.2f}.log'
            renamed_path = os.path.join(project_root, renamed_log)
            os.rename(log_path, renamed_path)

            if miou > best['miou']:
                best['miou'] = miou
                best['epoch'] = epoch
                best['log'] = renamed_path
        else:
            print(f'[Warn] mIoU not found in epoch {epoch} output. See log: {log_path}')

    if best['epoch'] is not None:
        print('Best mIoU found:')
        print(f"Epoch:      {best['epoch']}")
        print(f"mIoU:       {best['miou']}")
        print(f"Log file:   {best['log']}")
        ckpt_path = os.path.join(args.work_dir, f"epoch_{best['epoch']}_ema.pth")
        print(f"Checkpoint: {ckpt_path}")
    else:
        print('No valid mIoU found in any tested epochs.')