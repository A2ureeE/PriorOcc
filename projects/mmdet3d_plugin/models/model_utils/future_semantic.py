import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class FutureSemanticPredictor(nn.Module):
    """Predict per-horizon BEV semantic states from future BEV features.

    Each future BEV feature (from SCMFEnhancedPredictor) is decoded into
    per-class semantic logits via a 1x1 conv head. Softmax'd probabilities
    are projected back to BEV channel space as conditioning features that
    can be fed into future occupancy refinement.

    Loss is computed in the detector via cross_entropy against future 2D
    semantic pseudo-labels (or BEV semantic GT if available).

    Args:
        bev_channels: BEV feature channels.
        num_classes: number of semantic classes.
        num_future: number of future timesteps.
        hidden_dim: hidden dimension in per-horizon decoder.
    """

    def __init__(self,
                 bev_channels=256,
                 num_classes=17,
                 num_future=3,
                 hidden_dim=256):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_classes = num_classes
        self.num_future = num_future

        self.future_sem_heads = nn.ModuleList([
            nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d'),
                    bev_channels, hidden_dim, 3, padding=1, bias=False),
                build_norm_layer(dict(type='BN'), hidden_dim)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, num_classes, 1, bias=True))
            for _ in range(num_future)
        ])

        self.sem_to_bev_proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'),
                num_classes, bev_channels, 1, bias=False),
            build_norm_layer(dict(type='BN'), bev_channels)[1],
            nn.ReLU(inplace=True))

        for head in self.future_sem_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, future_bevs):
        """Args:
            future_bevs: (B, T, C, H, W) future BEV features.

        Returns:
            future_sem_logits: list of (B, num_classes, H, W) per-horizon.
            future_sem_bevs: list of (B, C, H, W) conditioning features.
        """
        B, T, C, H, W = future_bevs.shape
        assert T == self.num_future, \
            f"Expected {self.num_future} future steps, got {T}"

        future_sem_logits = []
        future_sem_bevs = []
        for k in range(T):
            bev_k = future_bevs[:, k]
            logits_k = self.future_sem_heads[k](bev_k)
            future_sem_logits.append(logits_k)
            sem_prob = F.softmax(logits_k, dim=1)
            sem_bev_k = self.sem_to_bev_proj(sem_prob)
            future_sem_bevs.append(sem_bev_k)

        return future_sem_logits, future_sem_bevs
