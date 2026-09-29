"""nuScenes 4D forecast dataset with per-horizon evaluation."""

import numpy as np
import torch

from mmdet3d.datasets.builder import DATASETS
from .nuscenes_dataset_occ import NuScenesDatasetOccpancy
from ..core.evaluation.occ_metrics import Metric_mIoU


@DATASETS.register_module()
class NuScenes4DOccForecastDataset(NuScenesDatasetOccpancy):
    """nuScenes OCC dataset with 4D forecasting support.

    Extends NuScenesDatasetOccpancy to:
    - Expose the ``forecast`` dict (future_occ_paths, tokens, horizons)
      via ``get_data_info``.
    - Evaluate per-horizon mIoU (1s/2s/3s) and average.
    """

    def __init__(self, *args, filter_invalid_forecast=True, **kwargs):
        self.filter_invalid_forecast = filter_invalid_forecast
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        data_infos = super().load_annotations(ann_file)
        if self.filter_invalid_forecast:
            data_infos = [
                info for info in data_infos
                if info.get('forecast', None) is not None
            ]
        return data_infos

    def get_data_info(self, index):
        info = super().get_data_info(index)
        forecast = self.data_infos[index].get('forecast', None)
        if forecast is not None:
            info['forecast'] = forecast
        return info

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        """Evaluate per-horizon occupancy mIoU.

        Args:
            occ_results: list of dicts from simple_test, each containing
                'occ_current' and optionally 'occ_future' (list of T preds).

        Returns:
            dict: per-horizon mIoU and average.
        """
        num_future = None
        for r in occ_results:
            if isinstance(r, dict) and 'occ_future' in r:
                num_future = len(r['occ_future'])
                break

        if num_future is None:
            return super().evaluate(
                occ_results, runner=runner, show_dir=show_dir, **eval_kwargs)

        if len(occ_results) != len(self.data_infos):
            raise RuntimeError(
                f"Forecast evaluation got {len(occ_results)} results for "
                f"{len(self.data_infos)} dataset samples; prediction/GT "
                f"alignment would be ambiguous.")

        metrics = [Metric_mIoU(
            num_classes=18, use_lidar_mask=False, use_image_mask=True)
            for _ in range(num_future)]

        for i, result in enumerate(occ_results):
            if not isinstance(result, dict) or 'occ_future' not in result:
                raise RuntimeError(
                    f"Forecast evaluation expects result dict with "
                    f"'occ_future' at index {i}, got {type(result)}")

            forecast = self.data_infos[i].get('forecast', None)
            if forecast is None:
                raise RuntimeError(
                    f"Missing forecast metadata for eval sample index {i}")

            future_paths = forecast.get('future_occ_paths', [])
            if len(future_paths) < num_future:
                raise RuntimeError(
                    f"Forecast sample index {i} has {len(future_paths)} "
                    f"future paths, expected {num_future}")

            for k in range(num_future):
                occ_path = future_paths[k]
                import os
                label_file = os.path.join(occ_path, 'labels.npz')
                if not os.path.exists(label_file):
                    raise RuntimeError(
                        f"Missing future occupancy label for evaluation: "
                        f"{label_file}")

                occ_labels = np.load(label_file)
                gt_semantics = occ_labels['semantics']
                mask_lidar = occ_labels['mask_lidar']
                mask_camera = occ_labels['mask_camera']

                pred_occ = result['occ_future'][k]
                if isinstance(pred_occ, list):
                    pred_occ = pred_occ[0]
                if isinstance(pred_occ, torch.Tensor):
                    pred_occ = pred_occ.cpu().numpy()

                metrics[k].add_batch(
                    pred_occ, gt_semantics, mask_lidar, mask_camera)

        eval_results = {}
        miou_values = []
        if len(self.data_infos) > 0:
            horizons = self.data_infos[0].get(
                'forecast', {}).get('horizons_sec', [1, 2, 3])
        else:
            horizons = [1, 2, 3]

        for k in range(num_future):
            miou = metrics[k].count_miou()
            key = f'mIoU_{horizons[k]}s'
            eval_results[key] = miou
            miou_values.append(miou)

        if miou_values:
            eval_results['mIoU_avg'] = float(np.mean(miou_values))

        return eval_results
