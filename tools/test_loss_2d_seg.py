import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from projects.mmdet3d_plugin.models.model_utils.semantic_injector import SemanticInjector

def test_loss_2d_seg():
    # 定义输入参数
    in_channels = 256
    out_channels = 256
    num_classes = 17
    norm_cfg = dict(type='BN')
    loss_2d_seg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        ignore_index=255,
        loss_weight=0.2
    )

    # 初始化 SemanticInjector
    semantic_injector = SemanticInjector(
        in_channels=in_channels,
        out_channels=out_channels,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_2d_seg=loss_2d_seg
    )

    # 验证 loss_2d_seg 是否正确传递
    assert semantic_injector.loss_2d_seg == loss_2d_seg, "loss_2d_seg 参数未正确传递！"
    print("loss_2d_seg 参数已正确传递！")

if __name__ == "__main__":
    test_loss_2d_seg()