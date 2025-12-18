"""
FlashOCC-R50 配置文件 (nuScenes mini 版本)

用于在 nuScenes mini 数据集上快速验证训练流程。
相比完整版本的修改：
- 使用更小的 batch size
- 减少训练 epoch
- 禁用预训练权重加载（可选）
- 调整 evaluation 间隔

使用方式：
    python tools/train.py projects/configs/flashocc/flashocc-r50-mini.py
"""

_base_ = ['./flashocc-r50.py']

# 数据配置 - 使用 mini 数据集
data_root = 'data/nuscenes/'

data = dict(
    samples_per_gpu=1,  # mini 数据集较小，使用小 batch
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
    ),
    val=dict(
        ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl',
    ),
    test=dict(
        ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl',
    ),
)

# 优化器 - 小 batch 可以用更小的 lr
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=1e-2)

# 训练配置 - 减少 epoch 用于快速验证
runner = dict(type='EpochBasedRunner', max_epochs=2)  # 仅 2 个 epoch 验证流程

# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,  # mini 数据集样本少，减少 warmup
    warmup_ratio=0.001,
    step=[1, ])  # 在第 1 个 epoch 后降低 lr

# 评估配置
evaluation = dict(interval=1, start=1, pipeline={{_base_.test_pipeline}})

# 检查点配置
checkpoint_config = dict(interval=1, max_keep_ckpts=2)

# 禁用 EMA（mini 数据集不需要）
custom_hooks = []

# 禁用预训练权重（如果没有下载）
load_from = None

# 日志配置
log_config = dict(
    interval=10,  # 每 10 个 iter 打印一次日志
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
