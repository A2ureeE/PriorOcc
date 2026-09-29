#!/usr/bin/env python3
"""No-GPU contract checks for PriorOcc-4D forecast pipeline.

This tool does not import torch/mmcv. It validates the file-level contract used
by LoadFutureOccGTFromFile with synthetic labels.npz files, so it can run on a
plain CPU workstation before moving the project to a training server.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np


EXPECTED_SHAPE = (200, 200, 16)
NUM_CLASSES = 18


def make_label(path, value):
    path.mkdir(parents=True, exist_ok=True)
    semantics = np.full(EXPECTED_SHAPE, value, dtype=np.uint8)
    semantics[0, 1, 2] = min(value + 1, NUM_CLASSES - 1)
    mask_lidar = np.ones(EXPECTED_SHAPE, dtype=np.uint8)
    mask_camera = np.ones(EXPECTED_SHAPE, dtype=np.uint8)
    np.savez_compressed(
        path / "labels.npz",
        semantics=semantics,
        mask_lidar=mask_lidar,
        mask_camera=mask_camera)
    return semantics, mask_lidar, mask_camera


def load_npz(label_file):
    if not label_file.exists():
        raise RuntimeError(f"missing labels file: {label_file}")
    labels = np.load(label_file)
    for key in ["semantics", "mask_lidar", "mask_camera"]:
        if key not in labels:
            raise RuntimeError(f"{label_file} missing key: {key}")
        if labels[key].shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"{label_file}:{key} shape {labels[key].shape} "
                f"!= {EXPECTED_SHAPE}")
    semantics = labels["semantics"]
    if semantics.min() < 0 or semantics.max() >= NUM_CLASSES:
        raise RuntimeError(
            f"{label_file}: class range [{semantics.min()}, "
            f"{semantics.max()}] outside [0, {NUM_CLASSES - 1}]")
    return labels


def verify_loader_contract():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        future_dirs = []
        originals = []
        for idx, value in enumerate([1, 5, 9]):
            future_dir = tmp_dir / f"future_{idx}"
            originals.append(make_label(future_dir, value))
            future_dirs.append(future_dir)

        loaded_semantics = []
        loaded_masks = []
        for future_dir in future_dirs:
            labels = load_npz(future_dir / "labels.npz")
            loaded_semantics.append(labels["semantics"])
            loaded_masks.append(labels["mask_camera"])

        semantics = np.stack(loaded_semantics, axis=0)
        masks = np.stack(loaded_masks, axis=0)
        assert semantics.shape == (3,) + EXPECTED_SHAPE
        assert masks.shape == (3,) + EXPECTED_SHAPE
        assert [int(semantics[i, 0, 0, 0]) for i in range(3)] == [1, 5, 9]

        flipped_x = np.flip(semantics, axis=1)
        flipped_y = np.flip(semantics, axis=2)
        assert flipped_x[0, -1, 1, 2] == semantics[0, 0, 1, 2]
        assert flipped_y[0, 0, -2, 2] == semantics[0, 0, 1, 2]

        try:
            load_npz(tmp_dir / "missing" / "labels.npz")
        except RuntimeError as exc:
            assert "missing labels file" in str(exc)
        else:
            raise AssertionError("missing labels file did not fail")

        bad_dir = tmp_dir / "bad_key"
        bad_dir.mkdir()
        np.savez_compressed(
            bad_dir / "labels.npz",
            semantics=np.zeros(EXPECTED_SHAPE, dtype=np.uint8),
            mask_camera=np.ones(EXPECTED_SHAPE, dtype=np.uint8))
        try:
            load_npz(bad_dir / "labels.npz")
        except RuntimeError as exc:
            assert "missing key" in str(exc)
        else:
            raise AssertionError("missing key did not fail")

        bad_shape = tmp_dir / "bad_shape"
        bad_shape.mkdir()
        np.savez_compressed(
            bad_shape / "labels.npz",
            semantics=np.zeros((4, 4, 2), dtype=np.uint8),
            mask_lidar=np.ones((4, 4, 2), dtype=np.uint8),
            mask_camera=np.ones((4, 4, 2), dtype=np.uint8))
        try:
            load_npz(bad_shape / "labels.npz")
        except RuntimeError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("bad shape did not fail")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=["all", "loader"], default="all")
    args = parser.parse_args()

    if args.test in ["all", "loader"]:
        verify_loader_contract()
        print("pipeline contract: PASS")


if __name__ == "__main__":
    main()
