#!/usr/bin/env python3
"""Static audit for PriorOcc-4D implementation.

This script intentionally avoids torch/mmcv imports. It checks whether the
new files are present and whether several design-critical code paths are
wired in the expected way. It is not a substitute for training smoke tests.
"""

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "projects/mmdet3d_plugin/models/detectors/priorocc_4d.py",
    "projects/mmdet3d_plugin/models/model_utils/dyn_sta_decoder.py",
    "projects/mmdet3d_plugin/models/model_utils/scmf.py",
    "projects/mmdet3d_plugin/models/model_utils/future_semantic.py",
    "projects/mmdet3d_plugin/models/model_utils/sem_consistency.py",
    "projects/mmdet3d_plugin/datasets/nuscenes_4d_forecast_dataset.py",
    "projects/mmdet3d_plugin/datasets/pipelines/loading_future_occ.py",
    "projects/mmdet3d_plugin/datasets/pipelines/loading_temporal_seg2d.py",
    "projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py",
    "tools/create_4d_forecast_infos.py",
    "tools/verify_priorocc_4d.py",
]


def read(rel_path):
    path = ROOT / rel_path
    return path.read_text(encoding="utf-8")


def line_of(text, pattern):
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def add_issue(issues, severity, rel_path, line, message):
    issues.append((severity, rel_path, line, message))


def check_presence(issues):
    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            add_issue(issues, "HIGH", rel_path, None, "required file missing")


def check_detector(issues):
    rel_path = "projects/mmdet3d_plugin/models/detectors/priorocc_4d.py"
    text = read(rel_path)

    required = [
        "class PriorOcc4D",
        "SemanticInjector",
        "sem_logits=seg_logits",
        "torch.stack(",
        "future_voxel_semantics",
        "loss_occ_future_",
        "future_gt_semantic_bev",
        "loss_sem_consistency",
    ]
    for marker in required:
        if marker not in text:
            add_issue(issues, "HIGH", rel_path, None,
                      f"detector missing expected marker: {marker}")

    if "with torch.no_grad():" in text and \
            "freeze_history_frames" not in text:
        add_issue(
            issues, "HIGH", rel_path, line_of(text, "with torch.no_grad():"),
            "historical frame SGDM/SemanticInjector is under no_grad; "
            "history seg losses and multi-frame semantic prior will not train")

    if "'gt_semantic_2d' in kwargs" in text and "gt_semantic_2d_history" not in text:
        add_issue(
            issues, "HIGH", rel_path, line_of(text, "'gt_semantic_2d' in kwargs"),
            "3-frame seg logits are supervised by a single-frame gt_semantic_2d; "
            "expected gt_semantic_2d_history with per-frame routing")

    if "self.sem_consistency(seg_logits_list)" in text and \
            "sem_cons_masks" not in text:
        add_issue(
            issues, "MED", rel_path,
            line_of(text, "self.sem_consistency(seg_logits_list)"),
            "semantic consistency receives only logits; design requires "
            "visible/high-confidence/static/common-view masking")

    cfg_text = read("projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py")
    if "future_sem_logits, _ =" in text and \
            "enable_future_semantic=False" not in cfg_text:
        add_issue(
            issues, "MED", rel_path,
            line_of(text, "future_sem_logits, _ ="),
            "future semantic branch is BEV-supervised only; avoid calling it "
            "future 2D semantic unless projection/rendering labels are added")


def check_config(issues):
    rel_path = "projects/configs/priorocc/priorocc-4d-r50-stgdm-scmf.py"
    text = read(rel_path)

    if "type='LoadSemanticSeg2D'" in text and "LoadTemporalSemanticSeg2D" not in text:
        add_issue(
            issues, "HIGH", rel_path, line_of(text, "type='LoadSemanticSeg2D'"),
            "4D train pipeline uses single-frame 2D semantic loader; "
            "expected temporal 2D semantic labels for t,t-1,t-2")

    if "gt_semantic_2d_history" not in text:
        add_issue(
            issues, "HIGH", rel_path, line_of(text, "'gt_semantic_2d'"),
            "Collect3D does not collect gt_semantic_2d_history")

    if "future_offsets=[2, 4, 6]" not in text:
        add_issue(issues, "MED", rel_path, None,
                  "future offsets are not explicitly +2/+4/+6")

    if "_forecast.pkl" not in text:
        add_issue(
            issues, "HIGH", rel_path, None,
            "4D config does not point train/val/test ann_file to forecast pkl")

    if "enable_future_semantic=True" in text:
        add_issue(
            issues, "MED", rel_path, line_of(text, "enable_future_semantic=True"),
            "future semantic is enabled by default although real future BEV/2D "
            "semantic labels are not wired in train_pipeline")


def check_future_loader(issues):
    rel_path = "projects/mmdet3d_plugin/datasets/pipelines/loading_future_occ.py"
    text = read(rel_path)

    if "continue" in text and "not os.path.exists" in text:
        add_issue(
            issues, "HIGH", rel_path, line_of(text, "not os.path.exists"),
            "missing future labels are silently zero-filled; design requires "
            "clear RuntimeError with path/token to avoid invalid supervision")

    for key in ["semantics", "mask_lidar", "mask_camera"]:
        pattern = f"occ_labels['{key}']"
        if pattern not in text:
            add_issue(issues, "HIGH", rel_path, None,
                      f"future loader does not read {key}")

    if "shape" not in text:
        add_issue(issues, "MED", rel_path, None,
                  "future loader lacks explicit shape validation")


def check_dataset(issues):
    rel_path = "projects/mmdet3d_plugin/datasets/nuscenes_4d_forecast_dataset.py"
    text = read(rel_path)

    if "from core.evaluation.occ_metrics import Metric_mIoU" in text:
        add_issue(
            issues, "HIGH", rel_path,
            line_of(text, "from core.evaluation.occ_metrics import Metric_mIoU"),
            "absolute import likely fails under plugin package; use project "
            "relative import path")

    if re.search(r"for i, result in enumerate\(occ_results\)", text) and \
            "len(occ_results) != len(self.data_infos)" not in text:
        add_issue(
            issues, "MED", rel_path,
            line_of(text, "for i, result in enumerate(occ_results)"),
            "evaluation indexes data_infos by output order; skipped invalid "
            "forecast anchors can misalign predictions and GT")

    if "continue" in text and "count_miou()" in text:
        add_issue(
            issues, "MED", rel_path, line_of(text, "continue"),
            "evaluation silently skips missing future GT; should report invalid "
            "samples or fail in verification mode")


def check_verification_coverage(issues):
    expected = {
        "tools/verify_priorocc_4d_pipeline.py": "pipeline synthetic loader test",
        "tools/diagnose_priorocc_4d_training.py": "training collapse diagnostics",
        "tools/verify_priorocc_4d_evaluator.py": "future evaluator slot test",
    }
    for rel_path, desc in expected.items():
        if not (ROOT / rel_path).exists():
            add_issue(issues, "MED", rel_path, None, f"missing {desc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-on", choices=["HIGH", "MED", "LOW"],
                        default="HIGH")
    args = parser.parse_args()

    severity_rank = {"HIGH": 3, "MED": 2, "LOW": 1}
    issues = []

    check_presence(issues)
    check_detector(issues)
    check_config(issues)
    check_future_loader(issues)
    check_dataset(issues)
    check_verification_coverage(issues)

    issues.sort(key=lambda x: (-severity_rank[x[0]], x[1], x[2] or 0))

    if not issues:
        print("PASS: no static audit issues found")
        return 0

    for severity, rel_path, line, message in issues:
        loc = f"{rel_path}:{line}" if line else rel_path
        print(f"[{severity}] {loc} - {message}")

    should_fail = any(
        severity_rank[s] >= severity_rank[args.fail_on]
        for s, _, _, _ in issues)
    return 1 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
