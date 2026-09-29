"""Pipeline transform for loading future occupancy GT from multiple frames."""

import os
import numpy as np
import torch

from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadFutureOccGTFromFile(object):
    """Load future-frame occupancy GT labels for forecasting.

    Reads T future ``labels.npz`` files from the forecast dict's
    ``future_occ_paths`` and applies the same BDA flip as
    ``LoadOccGTFromFile``.

    Input results keys:
        - forecast: dict with 'future_occ_paths' (list of T dir paths)

    Output results keys:
        - future_voxel_semantics: (T, Dx, Dy, Dz) long
        - future_mask_lidar: (T, Dx, Dy, Dz) float
        - future_mask_camera: (T, Dx, Dy, Dz) float
    """

    def __init__(self, expected_shape=(200, 200, 16), num_classes=18):
        self.expected_shape = tuple(expected_shape) \
            if expected_shape is not None else None
        self.num_classes = num_classes

    def _context(self, results):
        sample_token = results.get('sample_idx', None)
        if sample_token is None and 'curr' in results:
            sample_token = results['curr'].get('token', None)
        return f"sample={sample_token}" if sample_token is not None else "sample=<unknown>"

    def _load_one(self, occ_path, results):
        if not os.path.exists(occ_path):
            raise RuntimeError(
                f"Missing future occupancy label: {occ_path} "
                f"({self._context(results)})")

        occ_labels = np.load(occ_path)
        required_keys = ['semantics', 'mask_lidar', 'mask_camera']
        missing_keys = [key for key in required_keys if key not in occ_labels]
        if missing_keys:
            raise RuntimeError(
                f"Future occupancy label {occ_path} missing keys "
                f"{missing_keys} ({self._context(results)})")

        semantics = torch.from_numpy(occ_labels['semantics'])
        mask_lidar = torch.from_numpy(occ_labels['mask_lidar'])
        mask_camera = torch.from_numpy(occ_labels['mask_camera'])

        if self.expected_shape is not None:
            for name, tensor in [
                    ('semantics', semantics),
                    ('mask_lidar', mask_lidar),
                    ('mask_camera', mask_camera)]:
                if tuple(tensor.shape) != self.expected_shape:
                    raise RuntimeError(
                        f"Future occupancy {name} shape {tuple(tensor.shape)} "
                        f"!= {self.expected_shape} in {occ_path} "
                        f"({self._context(results)})")

        if semantics.numel() > 0:
            min_label = int(semantics.min())
            max_label = int(semantics.max())
            if min_label < 0 or max_label >= self.num_classes:
                raise RuntimeError(
                    f"Future occupancy semantics label range "
                    f"[{min_label}, {max_label}] outside [0, "
                    f"{self.num_classes - 1}] in {occ_path} "
                    f"({self._context(results)})")

        return semantics.long(), mask_lidar.float(), mask_camera.float()

    def __call__(self, results):
        forecast = results.get('forecast', None)
        if forecast is None:
            return results

        future_paths = forecast.get('future_occ_paths', [])
        if not future_paths:
            return results

        future_semantics = []
        future_mask_lidar = []
        future_mask_camera = []

        for horizon_idx, occ_dir in enumerate(future_paths):
            occ_path = os.path.join(occ_dir, 'labels.npz')
            try:
                semantics, mask_lidar, mask_camera = \
                    self._load_one(occ_path, results)
            except RuntimeError as exc:
                horizons = forecast.get('horizons_sec', [])
                horizon = horizons[horizon_idx] \
                    if horizon_idx < len(horizons) else horizon_idx
                raise RuntimeError(
                    f"{exc}; future_index={horizon_idx}, "
                    f"horizon={horizon}") from exc

            if results.get('flip_dx', False):
                semantics = torch.flip(semantics, [0])
                mask_lidar = torch.flip(mask_lidar, [0])
                mask_camera = torch.flip(mask_camera, [0])
            if results.get('flip_dy', False):
                semantics = torch.flip(semantics, [1])
                mask_lidar = torch.flip(mask_lidar, [1])
                mask_camera = torch.flip(mask_camera, [1])

            future_semantics.append(semantics)
            future_mask_lidar.append(mask_lidar)
            future_mask_camera.append(mask_camera)

        results['future_voxel_semantics'] = torch.stack(future_semantics)
        results['future_mask_lidar'] = torch.stack(future_mask_lidar)
        results['future_mask_camera'] = torch.stack(future_mask_camera)

        return results
