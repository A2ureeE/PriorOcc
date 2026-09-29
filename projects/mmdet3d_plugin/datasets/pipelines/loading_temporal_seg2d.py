"""Pipeline transform for loading temporal 2D semantic labels across frames."""

import os.path as osp
import numpy as np
import cv2

from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadTemporalSemanticSeg2D:
    """Load 2D semantic pseudo-labels for current + historical frames.

    Produces ``gt_semantic_2d_history`` of shape (T_frames, N_cams, H, W)
    where T_frames = 3 (current t, t-1, t-2), matching the ordering of
    ``seg_logits_list`` in the model.

    Args:
        seg_prefix: root dir for pseudo-label PNGs.
        num_classes: number of semantic classes.
        ignore_index: label for missing/ignored pixels.
        target_size: (H, W) matching post-augmentation image size.
    """

    def __init__(self,
                 seg_prefix='data/nuscenes/seg_2d_labels',
                 num_classes=17,
                 ignore_index=255,
                 target_size=(256, 704)):
        self.seg_prefix = seg_prefix
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.target_size = target_size

        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

    def _get_seg_path(self, img_path):
        if 'samples/' in img_path:
            rel_path = 'samples/' + img_path.split('samples/')[-1]
        else:
            rel_path = osp.basename(img_path)
        seg_path = osp.join(self.seg_prefix, rel_path)
        seg_path = seg_path.rsplit('.', 1)[0] + '.png'
        return seg_path

    def _load_and_resize(self, seg_path):
        if osp.exists(seg_path):
            seg_label = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if seg_label is None:
                seg_label = np.full(
                    self.target_size, self.ignore_index, dtype=np.uint8)
        else:
            seg_label = np.full(
                self.target_size, self.ignore_index, dtype=np.uint8)
        if seg_label.shape != self.target_size:
            seg_label = cv2.resize(
                seg_label,
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_NEAREST)
        return seg_label

    def _load_frame_cams(self, frame_info, cam_names=None):
        """Load seg labels for all cameras of one frame."""
        labels = []
        if cam_names is None:
            cam_names = frame_info.get('cam_names', self.cam_names) \
                if isinstance(frame_info, dict) else self.cam_names

        cams = None
        if isinstance(frame_info, dict) and 'cams' in frame_info:
            cams = frame_info['cams']

        for idx, cam_name in enumerate(cam_names):
            if cams is not None and cam_name in cams:
                img_path = cams[cam_name]['data_path']
                seg_path = self._get_seg_path(img_path)
                labels.append(self._load_and_resize(seg_path))
            else:
                labels.append(
                    np.full(self.target_size, self.ignore_index,
                            dtype=np.uint8))
        return np.stack(labels, axis=0)

    def __call__(self, results):
        # Authoritative camera order is set by PrepareImageInputs (pipeline step
        # 1) and follows data_config['cams']; seg_logits_list uses the same
        # order. Reading it here keeps gt_semantic_2d_history camera-aligned
        # with the model predictions (the hardcoded self.cam_names differs).
        cam_names = results.get('cam_names', self.cam_names)
        frames = []

        if 'curr' in results:
            frames.append(self._load_frame_cams(results['curr'], cam_names))
        else:
            frames.append(
                np.full((len(cam_names),) + self.target_size,
                        self.ignore_index, dtype=np.uint8))

        adjacent = results.get('adjacent', [])
        for adj in adjacent:
            frames.append(self._load_frame_cams(adj, cam_names))

        while len(frames) < 3:
            frames.append(frames[-1])

        results['gt_semantic_2d_history'] = np.stack(frames[:3], axis=0)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'seg_prefix={self.seg_prefix}, '
                f'num_classes={self.num_classes})')
