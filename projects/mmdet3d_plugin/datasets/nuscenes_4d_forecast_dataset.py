"""nuScenes 4D forecast dataset with per-horizon evaluation."""

import os

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
    - Evaluate multimodal forecasts: deployed (semantic mode selection)
      mIoU plus best-of-K oracle mIoU and mode-selection accuracy.
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

    @staticmethod
    def _unwrap_pred(pred):
        """Extract a single-sample (Dx, Dy, Dz) numpy prediction."""
        if isinstance(pred, list):
            pred = pred[0]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        return pred

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        """Evaluate per-horizon occupancy mIoU.

        Args:
            occ_results: list of dicts from simple_test, each containing
                'occ_current' and optionally 'occ_future' (list of T preds).
                Multimodal models additionally provide 'occ_future_modes'
                (K x T preds) and 'mode_probs' (B x K probabilities): the
                deployed prediction 'occ_future' then uses semantic mode
                selection, and a best-of-K oracle mIoU is also reported.

        Returns:
            dict: per-horizon mIoU and average. Multimodal results add
            mIoU_bestofK_{h}s / mIoU_bestofK_avg and mode_selection_acc.
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

        # Multimodal detection: first result exposing all-mode predictions.
        has_modes = bool(occ_results) and all(
            isinstance(r, dict) and 'occ_future_modes' in r
            for r in occ_results[:1])
        oracle_metrics = None
        mode_hits, mode_total = 0, 0
        if has_modes:
            oracle_metrics = [
                Metric_mIoU(num_classes=18, use_lidar_mask=False,
                            use_image_mask=True)
                for _ in range(num_future)]

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

            # Load every horizon's GT once; reused by deployed + oracle.
            future_gts = []
            for k in range(num_future):
                label_file = os.path.join(future_paths[k], 'labels.npz')
                if not os.path.exists(label_file):
                    raise RuntimeError(
                        f"Missing future occupancy label for evaluation: "
                        f"{label_file}")
                occ_labels = np.load(label_file)
                future_gts.append((
                    occ_labels['semantics'],
                    occ_labels['mask_lidar'],
                    occ_labels['mask_camera']))

            # ---- deployed (semantic mode selection) predictions ----
            for k in range(num_future):
                gt_semantics, mask_lidar, mask_camera = future_gts[k]
                pred_occ = self._unwrap_pred(result['occ_future'][k])
                metrics[k].add_batch(
                    pred_occ, gt_semantics, mask_lidar, mask_camera)

            # ---- best-of-K oracle scoring (multimodal only) ----
            if oracle_metrics is not None and 'occ_future_modes' in result:
                modes = result['occ_future_modes']  # K x T preds
                num_modes = len(modes)
                best_k, best_hists, best_score = None, None, -np.inf
                for k_mode in range(num_modes):
                    hists, score = [], 0.0
                    for t in range(num_future):
                        gt_semantics, _, mask_camera = future_gts[t]
                        pred_m = self._unwrap_pred(modes[k_mode][t])
                        m = mask_camera.astype(bool)
                        hist, _, _ = metrics[t].hist_info(
                            18, pred_m[m], gt_semantics[m])
                        hists.append(hist)
                        score += float(np.nanmean(
                            metrics[t].per_class_iu(hist)))
                    if score > best_score:
                        best_k, best_hists, best_score = k_mode, hists, score
                for t in range(num_future):
                    oracle_metrics[t].hist += best_hists[t]
                    oracle_metrics[t].cnt += 1

                # Deployed mode vs oracle mode agreement.
                if result.get('mode_probs'):
                    probs = result['mode_probs'][0]
                    sel_k = int(np.argmax(probs))
                    mode_total += 1
                    if sel_k == best_k:
                        mode_hits += 1

        eval_results = {}
        miou_values = []
        if len(self.data_infos) > 0:
            horizons = self.data_infos[0].get(
                'forecast', {}).get('horizons_sec', [1, 2, 3])
        else:
            horizons = [1, 2, 3]

        for k in range(num_future):
            miou = metrics[k].count_miou()['mIoU']
            key = f'mIoU_{horizons[k]}s'
            eval_results[key] = miou
            miou_values.append(miou)

        if miou_values:
            eval_results['mIoU_avg'] = float(np.mean(miou_values))

        # Multimodal: best-of-K oracle upper bound + selection accuracy.
        if oracle_metrics is not None:
            oracle_values = []
            for k in range(num_future):
                om = oracle_metrics[k]
                iou = om.per_class_iu(om.hist)
                miou = float(
                    np.nanmean(iou[:om.num_classes - 1]) * 100)
                eval_results[f'mIoU_bestofK_{horizons[k]}s'] = miou
                oracle_values.append(miou)
            if oracle_values:
                eval_results['mIoU_bestofK_avg'] = float(
                    np.mean(oracle_values))
            if mode_total > 0:
                eval_results['mode_selection_acc'] = mode_hits / mode_total

        return eval_results
