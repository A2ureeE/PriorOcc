import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS

@NECKS.register_module()
class SemanticInjector(nn.Module):
    """
    SemanticInjector for FlashOcc Scheme C.
    Injects 2D semantic priors into the feature map before View Transformer.
    
    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels (input to LSS).
        num_classes (int): Number of semantic classes for 2D segmentation.
        norm_cfg (dict): Config for normalization layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes,
                 norm_cfg=dict(type='BN')):
        super(SemanticInjector, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        # SegHead: Conv3x3 -> BN -> ReLU -> Conv1x1
        self.seg_head = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels,
                num_classes,
                kernel_size=1,
                padding=0,
                bias=True)
        )

        # FusionLayer: Concat(Feat, Logits) -> Conv1x1 -> BN -> ReLU
        self.fusion_layer = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels + num_classes,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B*N, C, H, W) Input features from backbone/neck.
        Returns:
            feat_fused: (B*N, out_channels, H, W) Fused features for LSS.
            seg_logits: (B*N, num_classes, H, W) 2D semantic logits for loss.
        """
        # 1. Generate 2D Semantic Logits
        seg_logits = self.seg_head(x)
        
        # 2. Feature Fusion
        # Concatenate original features with semantic logits along channel dimension
        feat_cat = torch.cat([x, seg_logits], dim=1)
        
        # Fuse and reduce channels
        feat_fused = self.fusion_layer(feat_cat)
        
        return feat_fused, seg_logits
