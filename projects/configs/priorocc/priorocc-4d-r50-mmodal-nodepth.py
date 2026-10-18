# PriorOcc-4D Multimodal —— 关闭 depth 监督的纯相机版本
# 继承 priorocc-4d-r50-mmodal.py（含 K-mode 多模态 + SMP + SDP + hole-filling 全套创新），
# 仅覆盖：
#   1) img_view_transformer.loss_depth_weight = 0（模型侧自动跳过 depth loss）
#   2) train/test pipeline 去掉 LoadPointsFromFile / PointToMultiViewDepth / gt_depth / points
#   3) load_from = None（仓库内暂无预训练权重；如需从单帧权重初始化改此处）
_base_ = ['./priorocc-4d-r50-mmodal.py']

# ---- 关闭 depth 监督 ----
model = dict(
    img_view_transformer=dict(
        loss_depth_weight=0.0,
    ),
)

load_from = None

# 混合精度训练：约 1.5-2x 提速 + 显存减半（dynamic loss scale 自适应）
fp16 = dict(loss_scale='dynamic')

# ---- mmcv 继承不暴露基类 Python 变量，需本地重定义 ----
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

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# ---- 纯相机 train pipeline（无 lidar / 无 gt_depth）----
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
        type='LoadTemporalSemanticSeg2D',
        seg_prefix=data_root + 'seg_2d_labels',
        num_classes=17,
        ignore_index=255,
        target_size=(256, 704)),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'voxel_semantics', 'mask_lidar', 'mask_camera',
            'gt_semantic_2d_history', 'future_voxel_semantics',
            'future_mask_camera', 'future_mask_lidar'
        ])
]

# ---- 纯相机 test pipeline（无 lidar / 无 points）----
test_pipeline = [
    dict(
        type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
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
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

evaluation = dict(interval=2, start=2, pipeline=test_pipeline)
