#!/usr/bin/env python3
"""No-GPU slot checks for PriorOcc-4D future evaluator.

The real dataset evaluator depends on the project runtime. This lightweight
script validates the core forecasting invariant with synthetic arrays:
each horizon is evaluated against its own future GT, and changing only the
2-second prediction only changes the 2-second metric plus the average.
"""

import argparse

import numpy as np


HORIZONS = [1.0, 2.0, 3.0]
NUM_CLASSES = 18
SHAPE = (20, 20, 4)


def miou(pred, gt, mask, num_classes=NUM_CLASSES):
    values = []
    pred = pred[mask]
    gt = gt[mask]
    for cls_id in range(num_classes):
        pred_c = pred == cls_id
        gt_c = gt == cls_id
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_c, gt_c).sum()
        values.append(inter / union)
    if not values:
        return 0.0
    return float(np.mean(values))


def evaluate_future(preds, gts, masks):
    per_horizon = []
    for pred, gt, mask in zip(preds, gts, masks):
        per_horizon.append(miou(pred, gt, mask.astype(bool)))
    return {
        f"mIoU_{HORIZONS[idx]}s": value
        for idx, value in enumerate(per_horizon)
    } | {"mIoU_avg": float(np.mean(per_horizon))}


def make_future_gts():
    gts = []
    masks = []
    for idx in range(3):
        gt = np.zeros(SHAPE, dtype=np.uint8)
        gt[2 + idx:8 + idx, 4:10, 1:3] = idx + 1
        gt[10:15, 10:16, 0:2] = 11
        mask = np.ones(SHAPE, dtype=bool)
        gts.append(gt)
        masks.append(mask)
    return gts, masks


def verify_slot_isolation():
    gts, masks = make_future_gts()
    perfect_preds = [gt.copy() for gt in gts]
    perfect = evaluate_future(perfect_preds, gts, masks)
    for key, value in perfect.items():
        assert abs(value - 1.0) < 1e-8, f"{key} expected 1.0, got {value}"

    damaged_preds = [gt.copy() for gt in gts]
    damaged_preds[1] = np.zeros_like(damaged_preds[1])
    damaged = evaluate_future(damaged_preds, gts, masks)

    assert abs(damaged["mIoU_1.0s"] - 1.0) < 1e-8
    assert damaged["mIoU_2.0s"] < 1.0
    assert abs(damaged["mIoU_3.0s"] - 1.0) < 1e-8
    assert damaged["mIoU_avg"] < perfect["mIoU_avg"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=["all", "slots"], default="all")
    args = parser.parse_args()

    if args.test in ["all", "slots"]:
        verify_slot_isolation()
        print("evaluator slot isolation: PASS")


if __name__ == "__main__":
    main()
