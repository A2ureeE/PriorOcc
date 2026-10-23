# 最终模型 Phase 1 + Depth 监督 — 恢复 BEVDepth 标准做法
#
# 与 final-p1 的唯一区别: 训练期恢复 LiDAR 投影深度监督（推理仍纯相机）
#   - pipeline 加回 LoadPointsFromFile + PointToMultiViewDepth
#   - Collect3D 补 gt_depth
#   - loss_depth_weight = 3.0 (FlashOcc 单帧 baseline 默认值; FlashOcc 4D 不做未来预测故 0.05 不适用)
#
# 依据: ep21(有depth) 当前帧 33.83 vs 全部无depth 4D run 31~32
#       depth 失锚是第三个未识别的退化成因
_base_ = ['./priorocc-4d-r50-final-p1.py']

# ---- 恢复 depth 监督 ----
model = dict(
    img_view_transformer=dict(
        loss_depth_weight=3.0,
    ),
    # ① 历史帧 no_grad + ② 历史帧拼批（共用一套 backbone，当前帧仍训练）
    #    历史帧前向不变、只是不回传梯度 → 无分布漂移，可无缝续训
    freeze_history_frames=True,
    batch_history_frames=True,
)

# ---- mmcv 不继承基类 Python 变量, 本地重定义 pipeline ----
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

# ---- 带 depth 的 train pipeline（test/val 不变, 推理纯相机）----
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
        file_client_args=dict(backend='disk')),
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

data = dict(
    train=dict(pipeline=train_pipeline),
)
