# PriorOcc-4D 阶段 1：补洞版单帧感知（perception-only）
#
# 目的：在不背未来预测多任务税、全量帧、补洞开启的干净条件下，把当前帧
# 占用感知练到高于单帧 ep21（4818 同池 33.83），产出可被阶段 2 冻结的骨干。
#
# 与基座 priorocc-4d-r50-smp-sdp-holefill-nodepth.py 的差异：
#   1. 退化为纯单帧：num_adj=0 => num_frame=1（BEV 编码器入口 64 通道，与
#      单帧 ep21 权重 1:1 对齐）。注意代码里 num_frame = num_adj + 1，所以
#      "只用当前帧"写的是 num_adj=0，不是 1。
#   2. 关闭全部与未来预测耦合的模块（SCMF / ConvGRU 预测器 / 语义运动注意力
#      / 动静态分离 / 跨帧语义一致性 / 运动先验）：单帧路径下它们只产生被丢弃
#      的中间量，白吃显存与算力。
#   3. 补洞（SemanticContinuityLoss）三层全开，cont_occ 打开（默认是关的），
#      这是本阶段"填洞"的主力监督。
#   4. 训练集换成全量 28130 帧（bevdetv2-nuscenes_infos_train.pkl），无 future
#      字段的帧只贡献当前帧 loss。
#   5. 用单帧 ep21 权重热启。
#
# 验收判据：当前帧 val mIoU（4818 同池）> 33.83；连续类
# （driveable_surface / sidewalk / terrain / vegetation）IoU 较 ep21 上升。
_base_ = ['./priorocc-4d-r50-smp-sdp-holefill-nodepth.py']

numC_Trans = 64
# 单帧路径：0 个相邻帧（range(1,1,1) 为空）=> 无 adjacent，BEV 入口 64 通道
multi_adj_frame_id_cfg = (1, 1, 1)

model = dict(
    # ---- 单帧退化 ----
    num_adj=0,
    with_prev=False,
    align_after_view_transfromation=False,
    img_bev_encoder_backbone=dict(numC_input=numC_Trans),
    # SDP（语义深度先验查表）在 ep21 里是关闭的，若此处打开会多出一张
    # 随机初始化的 prior table（dry-load 实测唯一感知失配点），既破坏
    # "只差补洞"的干净对照，也会在阶段 2 被冻结成随机常量。故与 ep21 对齐关闭。
    img_view_transformer=dict(
        depthnet_cfg=dict(use_semantic_depth_prior=False)),
    # ---- 只训当前帧 ----
    enable_future_prediction=False,
    enable_multimodal=False,
    enable_motion_prior=False,
    enable_future_semantic=False,
    enable_semantic_consistency=False,
    # ---- 关掉运动链（未来头不需要，感知也不需要）----
    enable_scmf=False,
    enable_motion_encoder=False,
    enable_semantic_attention=False,
    enable_dyn_sta_decoder=False,
    scmf=None,
    future_predictor=None,
    motion_prior=None,
    dyn_sta_decoder=None,
    motion_encoder=None,
    semantic_attention=None,
    delta_combiner=None,
    future_semantic=None,
    sem_consistency=None,
    # ---- 补洞开启（occ 层是本阶段主角）----
    enable_sem_continuity=True,
    sem_continuity_loss_weights=dict(
        cont_2d=0.05, cont_bev=0.05, cont_occ=0.05),
    continuity_apply_occ=True,
)

load_from = 'ckpts/priorocc-singleframe-ep21.pth'

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

data_root = 'data/nuscenes/'

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# ---- 纯相机单帧 train pipeline（无 lidar / 无 gt_depth / 无 future）----
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadSemanticSeg2D',
        seg_prefix=data_root + 'seg_2d_labels',
        num_classes=17,
        ignore_index=255,
        target_size=(256, 704)),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'voxel_semantics', 'mask_lidar', 'mask_camera',
            'gt_semantic_2d'
        ])
]

# ---- 纯相机单帧 test pipeline ----
test_pipeline = [
    dict(
        type='PrepareImageInputs', data_config=data_config, sequential=False),
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

share_data_config = dict(
    img_info_prototype='bevdet',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

data = dict(
    # 全量训练帧：28130 帧，只训当前帧
    train=dict(
        type='NuScenesDatasetOccpancy',
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        **share_data_config),
    # 验证/测试沿用 4D 的 4818 帧池（forecast pkl 过滤后），保证与
    # 单帧 ep21@4818=33.83、4D ep15_ema@4818=32.10 严格同池可比。
    # 本配置无未来头，simple_test 只返回当前帧，evaluate 自动回落到
    # NuScenesDatasetOccpancy.evaluate（当前帧 mIoU）。
    val=dict(
        type='NuScenes4DOccForecastDataset',
        ann_file=data_root + 'bevdetv2-nuscenes_infos_val_forecast.pkl',
        pipeline=test_pipeline,
        filter_invalid_forecast=True,
        **share_data_config),
    test=dict(
        type='NuScenes4DOccForecastDataset',
        ann_file=data_root + 'bevdetv2-nuscenes_infos_val_forecast.pkl',
        pipeline=test_pipeline,
        filter_invalid_forecast=True,
        **share_data_config),
)

# 12 轮够用：ep21 已是 21 轮收敛权重，本阶段只是"补洞 + 全量帧"精修。
# lr 在 ep8 衰减一次，末段 4 轮低 lr 收敛。
runner = dict(type='EpochBasedRunner', max_epochs=12)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8])

# 全池评测开销大（4818 帧 ~20 min），每 4 轮评一次非 EMA；
# EMA 趋势由外部守护脚本按偶数轮快测（--limit 300）补齐。
evaluation = dict(interval=4, start=4, pipeline=test_pipeline)
