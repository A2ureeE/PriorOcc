_base_ = ['./priorocc-r50.py']

numC_Trans = 64
multi_adj_frame_id_cfg = (1, 3, 1)

num_future = 3
max_flow_cells = 5.0

dynamic_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
static_class_ids = [11, 12, 13, 14, 15, 16]

horizons_sec = [1.0, 2.0, 3.0]
future_loss_weights = [1.0, 0.7, 0.5]

forecast_cfg = dict(
    num_future=num_future,
    horizons_sec=horizons_sec,
    future_loss_weights=future_loss_weights,
    history_offsets=[2, 1],
    future_offsets=[2, 4, 6],
)

model = dict(
    type='PriorOcc4D',
    num_adj=2,
    align_after_view_transfromation=False,
    with_prev=True,
    num_future=num_future,
    num_semantic_classes=17,
    dynamic_class_ids=dynamic_class_ids,
    static_class_ids=static_class_ids,
    max_flow_cells=max_flow_cells,
    future_loss_weights=future_loss_weights,
    horizons_sec=horizons_sec,
    enable_dyn_sta_decoder=True,
    enable_motion_encoder=True,
    enable_semantic_attention=True,
    enable_scmf=True,
    enable_future_prediction=True,
    enable_future_semantic=False,
    enable_semantic_consistency=True,
    freeze_history_frames=False,
    bev_projector=dict(
        type='SemanticBEVProjector',
        num_semantic_classes=17,
    ),
    dyn_sta_decoder=dict(
        type='SemanticDynStaSeparator',
        num_semantic_classes=17,
        dynamic_class_ids=dynamic_class_ids,
        static_class_ids=static_class_ids,
    ),
    motion_encoder=dict(
        type='SemanticMotionFeatureEncoder',
        raw_bev_channels=64,
        bev_channels=256,
        num_semantic_classes=17,
    ),
    semantic_attention=dict(
        type='SemanticMotionAttention',
        bev_channels=256,
        num_semantic_classes=17,
        num_heads=4,
        hidden_dim=256,
    ),
    delta_combiner=dict(
        type='PerClassDeltaCombiner',
        bev_channels=256,
        num_semantic_classes=17,
        binary_mask=False,
    ),
    scmf=dict(
        type='SemanticConditionedMotionField',
        bev_channels=256,
        num_semantic_classes=17,
        num_future=num_future,
        max_flow_cells=max_flow_cells,
        hidden_dim=256,
    ),
    future_predictor=dict(
        type='SCMFEnhancedPredictor',
        bev_channels=256,
        num_future=num_future,
        hidden_dim=256,
    ),
    future_semantic=dict(
        type='FutureSemanticPredictor',
        bev_channels=256,
        num_classes=17,
        num_future=num_future,
        hidden_dim=256,
    ),
    sem_consistency=dict(
        type='SemConsistencyLoss',
        num_classes=17,
        conf_threshold=0.5,
    ),
    # Fallback: DirectFutureOccupancyHead (no flow needed)
    # future_predictor=dict(
    #     type='DirectFutureOccupancyHead',
    #     bev_channels=256,
    #     num_future=num_future,
    #     hidden_dim=256,
    # ),
    img_bev_encoder_backbone=dict(
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
    ),
)

load_from = "ckpts/bevdet-r50-cbgs.pth"

# Local copies needed for pipeline construction — mmcv config inheritance
# merges config dicts but does NOT expose base Python variables.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# Multi-frame pipelines (sequential=True for 4D)
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadFutureOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='LoadTemporalSemanticSeg2D',
        seg_prefix=data_root + 'seg_2d_labels',
        num_classes=17,
        ignore_index=255,
        target_size=(256, 704)),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_depth', 'voxel_semantics', 'mask_lidar',
            'mask_camera', 'gt_semantic_2d_history',
            'future_voxel_semantics', 'future_mask_camera',
            'future_mask_lidar'
        ])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

share_data_config = dict(
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

data = dict(
    train=dict(type='NuScenes4DOccForecastDataset',
               ann_file=data_root + 'bevdetv2-nuscenes_infos_train_forecast.pkl',
               pipeline=train_pipeline, **share_data_config),
    val=dict(type='NuScenes4DOccForecastDataset',
             ann_file=data_root + 'bevdetv2-nuscenes_infos_val_forecast.pkl',
             pipeline=test_pipeline, **share_data_config),
    test=dict(type='NuScenes4DOccForecastDataset',
              ann_file=data_root + 'bevdetv2-nuscenes_infos_val_forecast.pkl',
              pipeline=test_pipeline, **share_data_config),
)

evaluation = dict(interval=2, start=2, pipeline=test_pipeline)
