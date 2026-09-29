#!/usr/bin/env python3
"""Diagnose PriorOcc-4D training stats without GPU dependencies.

Input can be either a JSON list of step dictionaries or a JSONL file. The
expected fields are intentionally simple so training smoke scripts can emit
them easily:

  step, loss_total, loss_occ_future_total, grad_norm, param_changed_ratio,
  logits_entropy, dominant_class_ratio, future_pairwise_diff,
  flow_mean_abs, flow_saturation_ratio, flow_grad_norm,
  gate_mean, gate_std, semantic_bev_std, mask_overlap
"""

import argparse
import json
import math
from pathlib import Path


DEFAULTS = {
    "min_future_loss_drop": 0.10,
    "min_param_changed_ratio": 1e-8,
    "max_dominant_class_ratio": 0.995,
    "min_logits_entropy": 1e-4,
    "min_future_pairwise_diff": 1e-8,
    "min_flow_after_train": 1e-6,
    "max_flow_saturation_ratio": 0.50,
    "min_gate_std": 1e-4,
    "min_semantic_bev_std": 1e-6,
    "max_mask_overlap": 0.05,
}


def load_stats(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"empty stats file: {path}")
    if text[0] == "[":
        stats = json.loads(text)
    else:
        stats = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(stats, list) or not stats:
        raise RuntimeError("stats must be a non-empty JSON list or JSONL")
    return stats


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def check_finite(stats, failures):
    for idx, item in enumerate(stats):
        for key, value in item.items():
            if isinstance(value, (int, float)) and not finite_number(value):
                failures.append(f"step {idx}: {key} is not finite: {value}")


def first_last(stats, key):
    values = [float(item[key]) for item in stats if key in item]
    if len(values) < 2:
        return None
    return values[0], values[-1]


def diagnose(stats, thresholds):
    failures = []
    check_finite(stats, failures)

    loss_pair = first_last(stats, "loss_occ_future_total")
    if loss_pair is None:
        loss_pair = first_last(stats, "loss_total")
    if loss_pair is not None:
        start, end = loss_pair
        required = start * (1.0 - thresholds["min_future_loss_drop"])
        if end > required:
            failures.append(
                f"loss did not drop enough: start={start:.6g}, "
                f"end={end:.6g}, required <= {required:.6g}")

    param_ratio = stats[-1].get("param_changed_ratio", None)
    if param_ratio is not None and \
            float(param_ratio) <= thresholds["min_param_changed_ratio"]:
        failures.append(
            f"param_changed_ratio too small: {float(param_ratio):.6g}")

    for item in stats:
        step = item.get("step", "?")
        entropy = item.get("logits_entropy", None)
        dominant = item.get("dominant_class_ratio", None)
        if entropy is not None and dominant is not None:
            if float(entropy) < thresholds["min_logits_entropy"] and \
                    float(dominant) > thresholds["max_dominant_class_ratio"]:
                failures.append(
                    f"step {step}: logits collapsed, entropy={entropy}, "
                    f"dominant_class_ratio={dominant}")

        diversity = item.get("future_pairwise_diff", None)
        if diversity is not None and \
                float(diversity) <= thresholds["min_future_pairwise_diff"]:
            failures.append(
                f"step {step}: future horizons are identical, "
                f"future_pairwise_diff={diversity}")

        flow_abs = item.get("flow_mean_abs", None)
        flow_grad = item.get("flow_grad_norm", None)
        if flow_abs is not None and flow_grad is not None:
            if float(flow_grad) > 0 and \
                    float(flow_abs) <= thresholds["min_flow_after_train"]:
                failures.append(
                    f"step {step}: flow may be identity-collapsed, "
                    f"flow_mean_abs={flow_abs}, flow_grad_norm={flow_grad}")

        flow_sat = item.get("flow_saturation_ratio", None)
        if flow_sat is not None and \
                float(flow_sat) > thresholds["max_flow_saturation_ratio"]:
            failures.append(
                f"step {step}: flow saturation too high: {flow_sat}")

        gate_mean = item.get("gate_mean", None)
        gate_std = item.get("gate_std", None)
        if gate_mean is not None and gate_std is not None:
            if (float(gate_mean) < 0.01 or float(gate_mean) > 0.99) and \
                    float(gate_std) < thresholds["min_gate_std"]:
                failures.append(
                    f"step {step}: gate collapsed, mean={gate_mean}, "
                    f"std={gate_std}")

        sem_std = item.get("semantic_bev_std", None)
        if sem_std is not None and \
                float(sem_std) < thresholds["min_semantic_bev_std"]:
            failures.append(
                f"step {step}: semantic BEV is nearly constant, std={sem_std}")

        overlap = item.get("mask_overlap", None)
        if overlap is not None and \
                float(overlap) > thresholds["max_mask_overlap"]:
            failures.append(
                f"step {step}: dynamic/static mask overlap too high: {overlap}")

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-json", required=True)
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float,
                            default=value)
    args = parser.parse_args()

    thresholds = {
        key: getattr(args, key)
        for key in DEFAULTS
    }
    stats = load_stats(args.stats_json)
    failures = diagnose(stats, thresholds)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("training diagnostics: PASS")


if __name__ == "__main__":
    main()
