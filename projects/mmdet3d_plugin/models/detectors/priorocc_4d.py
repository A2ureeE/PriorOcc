import inspect

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet3d.models import DETECTORS, builder

from .bevdet_occ import BEVDepth4DOCC


@DETECTORS.register_module()
class PriorOcc4D(BEVDepth4DOCC):
    """PriorOcc-4D: 4D occupancy forecasting with semantic-conditioned motion.

    Phase B: integrates SemanticInjector into the 4D multi-frame pipeline.
    Every historical frame runs through image_encoder -> SemanticInjector ->
    view_transformer(sem_logits=...). Per-frame seg_logits and depth are
    collected, historical BEV is aligned via shift_feature(), and current-frame
    losses (loss_depth, loss_occ, 3-frame loss_2d_seg_history) are computed.

    Args:
        num_future: number of future timesteps to predict.
        num_semantic_classes: number of 2D semantic classes.
        dynamic_class_ids: list of dynamic class indices.
        static_class_ids: list of static class indices.
        enable_dyn_sta_decoder: enable dynamic/static separation.
        enable_motion_encoder: enable semantic motion feature encoder.
        enable_semantic_attention: enable semantic motion attention.
        enable_scmf: enable semantic-conditioned motion field.
        enable_future_prediction: enable future occupancy prediction.
        max_flow_cells: max flow magnitude in BEV cells.
        future_loss_weights: per-horizon loss weights.
        horizons_sec: forecast horizons in seconds.
        dyn_sta_decoder: config dict for dynamic/static separator.
        motion_encoder: config dict for motion feature encoder.
        semantic_attention: config dict for semantic attention.
        delta_combiner: config dict for per-class delta combiner.
        scmf: config dict for SCMF module.
        future_predictor: config dict for future predictor.
    """

    def __init__(self,
                 num_future=3,
                 num_semantic_classes=17,
                 dynamic_class_ids=None,
                 static_class_ids=None,
                 enable_dyn_sta_decoder=True,
                 enable_motion_encoder=True,
                 enable_semantic_attention=True,
                 enable_scmf=True,
                 enable_future_prediction=True,
                 enable_future_semantic=False,
                 enable_semantic_consistency=False,
                 freeze_history_frames=False,
                 max_flow_cells=5.0,
                 future_loss_weights=(1.0, 0.7, 0.5),
                 horizons_sec=(1.0, 2.0, 3.0),
                 dyn_sta_decoder=None,
                 motion_encoder=None,
                 semantic_attention=None,
                 delta_combiner=None,
                 scmf=None,
                 future_predictor=None,
                 bev_projector=None,
                 future_semantic=None,
                 sem_consistency=None,
                 enable_motion_prior=False,
                 motion_prior=None,
                 motion_prior_loss_weights=None,
                 enable_sem_continuity=False,
                 sem_continuity=None,
                 sem_continuity_loss_weights=None,
                 continuity_apply_occ=False,
                 **kwargs):
        kwargs.pop('use_language_self_gating', None)
        kwargs.pop('language_self_gating', None)
        super(PriorOcc4D, self).__init__(**kwargs)

        self.num_future = num_future
        self.num_semantic_classes = num_semantic_classes
        self.max_flow_cells = max_flow_cells
        self.future_loss_weights = list(future_loss_weights)
        self.horizons_sec = list(horizons_sec)
        self.freeze_history_frames = freeze_history_frames

        if dynamic_class_ids is None:
            dynamic_class_ids = list(range(11))
        if static_class_ids is None:
            static_class_ids = list(range(11, num_semantic_classes))
        self.dynamic_class_ids = list(dynamic_class_ids)
        self.static_class_ids = list(static_class_ids)

        self.enable_flags = dict(
            dyn_sta_decoder=enable_dyn_sta_decoder,
            motion_encoder=enable_motion_encoder,
            semantic_attention=enable_semantic_attention,
            scmf=enable_scmf,
            future_prediction=enable_future_prediction,
            future_semantic=enable_future_semantic,
            sem_consistency=enable_semantic_consistency,
            motion_prior=enable_motion_prior,
            sem_continuity=enable_sem_continuity)

        self._dyn_sta_decoder_cfg = dyn_sta_decoder
        self._motion_encoder_cfg = motion_encoder
        self._semantic_attention_cfg = semantic_attention
        self._delta_combiner_cfg = delta_combiner
        self._scmf_cfg = scmf
        self._future_predictor_cfg = future_predictor
        self._bev_projector_cfg = bev_projector
        self._future_semantic_cfg = future_semantic
        self._sem_consistency_cfg = sem_consistency
        self._motion_prior_cfg = motion_prior
        self._sem_continuity_cfg = sem_continuity

        self.continuity_apply_occ = continuity_apply_occ
        self.motion_prior_loss_weights = dict(
            static_flow=0.05, rigid_smooth=0.02, nonrigid_bound=0.02)
        if motion_prior_loss_weights:
            self.motion_prior_loss_weights.update(motion_prior_loss_weights)
        self.sem_continuity_loss_weights = dict(
            cont_2d=0.05, cont_bev=0.05, cont_occ=0.0)
        if sem_continuity_loss_weights:
            self.sem_continuity_loss_weights.update(sem_continuity_loss_weights)

        self.dyn_sta_decoder = None
        self.motion_encoder = None
        self.semantic_attention = None
        self.delta_combiner = None
        self.scmf = None
        self.future_predictor = None
        self.bev_projector = None
        self.future_semantic = None
        self.sem_consistency = None
        self.motion_prior = None
        self.sem_continuity = None

        self._build_4d_modules()

    def _build_4d_modules(self):
        """Build 4D motion modules from config."""
        if self.enable_flags['dyn_sta_decoder'] and self._dyn_sta_decoder_cfg:
            self.dyn_sta_decoder = builder.build_neck(
                self._dyn_sta_decoder_cfg)
        if self.enable_flags['motion_encoder'] and self._motion_encoder_cfg:
            self.motion_encoder = builder.build_neck(
                self._motion_encoder_cfg)
        if self.enable_flags['semantic_attention'] and \
                self._semantic_attention_cfg:
            self.semantic_attention = builder.build_neck(
                self._semantic_attention_cfg)
        if self._delta_combiner_cfg:
            self.delta_combiner = builder.build_neck(
                self._delta_combiner_cfg)
        if self.enable_flags['scmf'] and self._scmf_cfg:
            self.scmf = builder.build_neck(self._scmf_cfg)
        if self.enable_flags['future_prediction'] and \
                self._future_predictor_cfg:
            self.future_predictor = builder.build_neck(
                self._future_predictor_cfg)
        if self._bev_projector_cfg:
            self.bev_projector = builder.build_neck(
                self._bev_projector_cfg)
        if self.enable_flags['future_semantic'] and \
                self._future_semantic_cfg:
            self.future_semantic = builder.build_neck(
                self._future_semantic_cfg)
        if self.enable_flags['sem_consistency'] and \
                self._sem_consistency_cfg:
            self.sem_consistency = builder.build_neck(
                self._sem_consistency_cfg)
        if self.enable_flags['motion_prior'] and self._motion_prior_cfg:
            self.motion_prior = builder.build_neck(self._motion_prior_cfg)
        if self.enable_flags['sem_continuity'] and self._sem_continuity_cfg:
            self.sem_continuity = builder.build_neck(self._sem_continuity_cfg)

    def loss_2d_seg(self, seg_logits, gt_semantic_2d):
        """Compute 2D semantic segmentation loss.

        Copied from BEVDetOCC since BEVDepth4DOCC does not inherit from it.

        Args:
            seg_logits: (B*N, num_classes, H, W)
            gt_semantic_2d: (B, N, H, W) or (B*N, H, W) or None
        Returns:
            dict: loss_2d_seg
        """
        if gt_semantic_2d is None:
            return dict(loss_2d_seg=seg_logits.sum() * 0.0)

        loss_weight = 1.0
        ignore_index = 255

        if getattr(self, 'semantic_injector', None) is not None:
            cfg = getattr(self.semantic_injector, 'loss_2d_seg', None)
            if isinstance(cfg, dict):
                loss_weight = float(cfg.get('loss_weight', 1.0))
                ignore_index = int(cfg.get('ignore_index', 255))

        if gt_semantic_2d.dim() == 4:
            gt_semantic_2d = gt_semantic_2d.view(
                -1, gt_semantic_2d.shape[-2], gt_semantic_2d.shape[-1])

        if seg_logits.shape[-2:] != gt_semantic_2d.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=gt_semantic_2d.shape[-2:],
                mode='bilinear', align_corners=True)

        loss_seg = F.cross_entropy(
            seg_logits, gt_semantic_2d.long(), ignore_index=ignore_index)
        return dict(loss_2d_seg=loss_seg * loss_weight)

    def prepare_bev_feat(self, img, sensor2ego, ego2global, intrin, post_rot,
                         post_tran, bda, mlp_input):
        """Extract BEV features for a single frame, with SemanticInjector.

        Args:
            img: (B, N_views, 3, H, W)
            sensor2ego: (B, N_views, 4, 4)
            ego2global: (B, N_views, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
            mlp_input: (B, N_views, 27)

        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
            seg_logits: (B*N, C_sem, fH, fW) or None
        """
        x, _ = self.image_encoder(img)  # (B, N, C, fH, fW)

        seg_logits = None
        if self.semantic_injector is not None:
            B, N, C, fH, fW = x.shape
            x = x.view(B * N, C, fH, fW)
            x, seg_logits = self.semantic_injector(x)
            x = x.view(B, N, -1, fH, fW)

        view_input = [x, sensor2ego, ego2global, intrin, post_rot,
                      post_tran, bda, mlp_input]

        sig = inspect.signature(self.img_view_transformer.forward)
        if 'sem_logits' in sig.parameters:
            view_result = self.img_view_transformer(
                view_input, sem_logits=seg_logits)
            if len(view_result) == 3:
                bev_feat, depth, refined_sem_logits = view_result
                if refined_sem_logits is not None:
                    seg_logits = refined_sem_logits
            else:
                bev_feat, depth = view_result
        else:
            bev_feat, depth = self.img_view_transformer(view_input)

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, seg_logits

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """Extract image features across all frames.

        Args:
            img_inputs: tuple of (imgs, sensor2egos, ego2globals, intrins,
                post_rots, post_trans, bda)
            img_metas:
            **kwargs:
        Returns:
            x: [(B, C', H', W')]
            depth: (B*N_views, D, fH, fW) — key frame depth
            seg_logits_list: list of (B*N, C_sem, fH, fW) or None per frame
        """
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, _ = self.prepare_inputs(img_inputs)

        bev_feat_list = []
        depth_list = []
        seg_logits_list = []
        key_frame = True

        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots,
                post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = \
                        sensor2keyegos[0], ego2globals[0]

                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot,
                    post_tran, bda)

                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input)
                if key_frame or not self.freeze_history_frames:
                    bev_feat, depth, seg_logits = \
                        self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth, seg_logits = \
                            self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
                seg_logits = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            seg_logits_list.append(seg_logits)
            key_frame = False

        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],
                    [sensor2keyegos[0], sensor2keyegos[adj_id]], bda)

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        key_frame_metas = (
            sensor2keyegos[0], ego2globals[0], intrins[0],
            post_rots[0], post_trans[0], bda)
        return [x], depth_list[0], seg_logits_list, bev_feat_list, \
            key_frame_metas

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points.

        Returns 6 values: img_feats, pts_feats, depth, seg_logits_list,
        bev_feat_list, key_frame_metas.
        """
        img_feats, depth, seg_logits_list, bev_feat_list, key_frame_metas = \
            self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth, seg_logits_list, \
            bev_feat_list, key_frame_metas

    def _compute_motion_flow(self, fused_bev, bev_feat_list, key_frame_metas,
                             seg_logits_list, depth,
                             return_semantic_masks=False):
        """Run the SCMF motion pipeline and return per-horizon flow.

        Args:
            fused_bev: (B, C, H, W) encoded BEV feature.
            bev_feat_list: list of (B, C_raw, H, W) raw per-frame BEV
                [t, t-1, t-2] (index 0 = key frame).
            key_frame_metas: 6-tuple for SemanticBEVProjector.
            seg_logits_list: list of (B*N, C_sem, fH, fW) or None per frame.
            depth: (B*N, D, fH, fW) post-softmax depth for key frame.

        Returns:
            flow: (B, T, 2, H, W) or None if SCMF is disabled.
            refined_bev: (B, C, H, W) — possibly delta-combiner enriched.
            semantic_masks: optional list of masks for semantic consistency.
        """
        if self.scmf is None or self.bev_projector is None:
            if return_semantic_masks:
                return None, fused_bev, None, None
            return None, fused_bev

        seg_logits_key = seg_logits_list[0]
        if seg_logits_key is None:
            if return_semantic_masks:
                return None, fused_bev, None, None
            return None, fused_bev

        semantic_bev, visibility = self.bev_projector(
            seg_logits_key, depth, self.img_view_transformer, key_frame_metas)

        dyn_mask, sta_mask, per_cls_masks = self.dyn_sta_decoder(
            semantic_bev, visibility)
        semantic_consistency_masks = None
        if return_semantic_masks and seg_logits_list is not None:
            valid_mask = ((sta_mask > 0.5) & (visibility > 0)).float()
            semantic_consistency_masks = []
            for seg_logits in seg_logits_list:
                if seg_logits is None:
                    semantic_consistency_masks.append(None)
                    continue
                mask_i = F.interpolate(
                    valid_mask, size=seg_logits.shape[-2:],
                    mode='nearest')
                if mask_i.shape[0] != seg_logits.shape[0]:
                    repeat = seg_logits.shape[0] // mask_i.shape[0]
                    assert repeat * mask_i.shape[0] == seg_logits.shape[0], \
                        "Cannot broadcast semantic consistency mask from " \
                        f"{mask_i.shape[0]} to {seg_logits.shape[0]}"
                    mask_i = mask_i.repeat_interleave(repeat, dim=0)
                semantic_consistency_masks.append(mask_i)

        aligned_bev_history = torch.stack(
            bev_feat_list[::-1], dim=1)
        motion_feat = self.motion_encoder(aligned_bev_history, semantic_bev)

        attn_feat = self.semantic_attention(
            fused_bev, motion_feat, per_cls_masks)

        if self.delta_combiner is not None:
            combined_delta = self.delta_combiner(
                fused_bev, motion_feat, per_cls_masks)
            fused_bev = fused_bev + combined_delta

        flow = self.scmf(fused_bev, motion_feat, semantic_bev, attn_feat)
        if self.motion_prior is not None and flow is not None:
            flow = self.motion_prior(flow, per_cls_masks)
        if return_semantic_masks:
            motion_aux = dict(
                per_cls_masks=per_cls_masks,
                sta_mask=sta_mask,
                dyn_mask=dyn_mask,
                semantic_bev=semantic_bev,
                visibility=visibility)
            return flow, fused_bev, semantic_consistency_masks, motion_aux
        return flow, fused_bev

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training for 4D occupancy with motion pipeline.

        Returns:
            dict: losses including loss_depth, loss_occ,
                loss_2d_seg_history_{i}, and loss_occ_future_{k}s.
        """
        img_feats, pts_feats, depth, seg_logits_list, bev_feat_list, \
            key_frame_metas = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        gt_depth = kwargs['gt_depth']
        losses['loss_depth'] = \
            self.img_view_transformer.get_depth_loss(gt_depth, depth)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature, scale_factor=2, mode='bilinear',
                align_corners=True)
        occ_logits = self.occ_head(occ_bev_feature)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        losses.update(self.occ_head.loss(
            occ_logits, voxel_semantics, mask_camera))

        gt_sem_2d_history = kwargs.get('gt_semantic_2d_history', None)
        if gt_sem_2d_history is None and 'gt_semantic_2d' in kwargs:
            gt_sem_2d_history = kwargs['gt_semantic_2d']

        if seg_logits_list is not None and gt_sem_2d_history is not None:
            for i, seg_logits in enumerate(seg_logits_list):
                if seg_logits is not None:
                    if gt_sem_2d_history.dim() == 5:
                        gt_sem_2d = gt_sem_2d_history[:, i]
                    else:
                        gt_sem_2d = gt_sem_2d_history
                    loss_dict = self.loss_2d_seg(seg_logits, gt_sem_2d)
                    losses[f'loss_2d_seg_history_{i}'] = \
                        loss_dict['loss_2d_seg']

        flow, refined_bev, sem_cons_masks, motion_aux = \
            self._compute_motion_flow(
                occ_bev_feature, bev_feat_list, key_frame_metas,
                seg_logits_list, depth, return_semantic_masks=True)

        if self.sem_consistency is not None and \
                seg_logits_list is not None:
            cons_loss = self.sem_consistency(seg_logits_list, sem_cons_masks)
            if cons_loss is not None:
                losses['loss_sem_consistency'] = cons_loss * 0.05

        # Semantic Motion Prior regularization (fills the missing L_motion_reg)
        if self.motion_prior is not None and flow is not None and \
                motion_aux is not None:
            mp = self.motion_prior.regularization_losses(
                flow, motion_aux['per_cls_masks'])
            w = self.motion_prior_loss_weights
            losses['loss_static_flow'] = mp['static_flow'] * w['static_flow']
            losses['loss_rigid_smooth'] = mp['rigid_smooth'] * w['rigid_smooth']
            losses['loss_nonrigid_bound'] = \
                mp['nonrigid_bound'] * w['nonrigid_bound']

        # Semantic continuity (background hole-filling)
        if self.sem_continuity is not None:
            seg_key = seg_logits_list[0] if seg_logits_list else None
            sem_bev = motion_aux['semantic_bev'] if motion_aux else None
            vis = motion_aux['visibility'] if motion_aux else None
            cont = self.sem_continuity(
                seg_logits_key=seg_key,
                semantic_bev=sem_bev,
                visibility=vis,
                occ_logits=(occ_logits if self.continuity_apply_occ else None),
                mask_camera=mask_camera)
            w = self.sem_continuity_loss_weights
            if cont.get('loss_sem_continuity_2d') is not None:
                losses['loss_sem_continuity_2d'] = \
                    cont['loss_sem_continuity_2d'] * w['cont_2d']
            if cont.get('loss_sem_continuity_bev') is not None:
                losses['loss_sem_continuity_bev'] = \
                    cont['loss_sem_continuity_bev'] * w['cont_bev']
            if self.continuity_apply_occ and \
                    cont.get('loss_sem_continuity_occ') is not None:
                losses['loss_sem_continuity_occ'] = \
                    cont['loss_sem_continuity_occ'] * w['cont_occ']

        if self.future_predictor is not None and \
                'future_voxel_semantics' in kwargs:
            future_voxel_semantics = kwargs['future_voxel_semantics']
            future_mask_camera = kwargs.get('future_mask_camera')

            if flow is None and hasattr(self.future_predictor, 'conv_gru'):
                B, C, H, W = refined_bev.shape
                flow = torch.zeros(
                    B, self.num_future, 2, H, W,
                    device=refined_bev.device, dtype=refined_bev.dtype)

            future_bevs = self.future_predictor(refined_bev, flow)
            for k in range(self.num_future):
                future_feat_k = future_bevs[:, k]
                future_gt_k = future_voxel_semantics[:, k]
                if future_mask_camera is not None:
                    future_mask_k = future_mask_camera[:, k]
                else:
                    future_mask_k = mask_camera
                loss_future = self.forward_occ_train(
                    future_feat_k, future_gt_k, future_mask_k)
                losses[f'loss_occ_future_{k+1}s'] = \
                    loss_future['loss_occ'] * self.future_loss_weights[k]

            if self.future_semantic is not None and \
                    'future_gt_semantic_bev' in kwargs:
                future_sem_logits, _ = \
                    self.future_semantic(future_bevs)
                future_gt_sem_bev = kwargs['future_gt_semantic_bev']
                for k in range(self.num_future):
                    sem_log_k = future_sem_logits[k]
                    gt_k = future_gt_sem_bev[:, k]
                    if sem_log_k.shape[-2:] != gt_k.shape[-2:]:
                        sem_log_k = F.interpolate(
                            sem_log_k, size=gt_k.shape[-2:],
                            mode='bilinear', align_corners=True)
                    loss_sem = F.cross_entropy(
                        sem_log_k, gt_k, ignore_index=255)
                    losses[f'loss_future_semantic_{k+1}s'] = \
                        loss_sem * 0.1

        return losses

    def simple_test_occ(self, img_feats, img_metas=None):
        """Override parent to use get_occ instead of get_occ_gpu.

        BEVDepth4DOCC.simple_test_occ calls get_occ_gpu which only exists
        on BEVOCCHead2D_V2, not BEVOCCHead2D.
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)
        return occ_preds

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function — returns current and future occupancy predictions.

        Returns:
            When future_predictor is available: dict with occ_current,
            occ_future (list of T predictions), and horizons_sec.
            Otherwise: list of current occupancy predictions.
        """
        img_feats, _, depth, seg_logits_list, bev_feat_list, \
            key_frame_metas = self.extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature, scale_factor=2, mode='bilinear',
                align_corners=True)

        occ_current = self.simple_test_occ(occ_bev_feature, img_metas)

        if self.future_predictor is not None:
            flow, refined_bev = self._compute_motion_flow(
                occ_bev_feature, bev_feat_list, key_frame_metas,
                seg_logits_list, depth)

            if flow is None and hasattr(self.future_predictor, 'conv_gru'):
                B, C, H, W = refined_bev.shape
                flow = torch.zeros(
                    B, self.num_future, 2, H, W,
                    device=refined_bev.device, dtype=refined_bev.dtype)

            future_bevs = self.future_predictor(refined_bev, flow)
            occ_future = []
            for k in range(self.num_future):
                future_pred = self.simple_test_occ(
                    future_bevs[:, k], img_metas)
                occ_future.append(future_pred)
            return dict(
                occ_current=occ_current,
                occ_future=occ_future,
                horizons_sec=self.horizons_sec)

        return occ_current

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None,
                      **kwargs):
        """Override parent to handle 6-value extract_feat return."""
        img_feats, _, _, _, _, _ = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature, scale_factor=2, mode='bilinear',
                align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs
