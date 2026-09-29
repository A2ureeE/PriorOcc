import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS


@NECKS.register_module()
class SemConsistencyLoss(nn.Module):
    """Temporal semantic consistency loss in 2D image space.

    Computes symmetric KL divergence between the key frame's semantic
    predictions and each historical frame's predictions (stop-gradient on
    historical). The loss is masked to high-confidence regions where both
    frames have confident predictions, naturally excluding dynamic,
    occluded, and newly-appeared regions.

    Historical frames are computed under torch.no_grad() in extract_img_feat,
    so only the key frame's SemanticInjector receives gradients. This acts
    as a teacher-forcing regularizer: the key frame is pushed toward
    consistency with historical predictions.

    Args:
        num_classes: number of semantic classes.
        conf_threshold: minimum max softmax probability to count as confident.
        eps: epsilon for numerical stability in log.
    """

    def __init__(self, num_classes=17, conf_threshold=0.5, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.eps = eps

    def forward(self, seg_logits_list, masks=None):
        """Args:
            seg_logits_list: list of (B*N, C_sem, fH, fW) per frame.
                Index 0 = key frame (t), 1 = t-1, 2 = t-2.
            masks: optional list of (B*N, 1, fH, fW) masks. The key-frame
                high-confidence mask is intersected with each history mask.

        Returns:
            loss: scalar symmetric KL divergence.
        """
        key_logits = seg_logits_list[0]
        if key_logits is None:
            return None

        p_key = F.softmax(key_logits, dim=1)
        log_p_key = torch.log(p_key + self.eps)
        conf_key = p_key.max(dim=1)[0]

        losses = []
        for i in range(1, len(seg_logits_list)):
            hist_logits = seg_logits_list[i]
            if hist_logits is None:
                continue
            if hist_logits.shape[-2:] != key_logits.shape[-2:]:
                hist_logits = F.interpolate(
                    hist_logits, size=key_logits.shape[-2:],
                    mode='bilinear', align_corners=True)

            p_hist = F.softmax(hist_logits.detach(), dim=1)
            log_p_hist = torch.log(p_hist + self.eps)
            conf_hist = p_hist.max(dim=1)[0]

            mask = (conf_key > self.conf_threshold) & \
                   (conf_hist > self.conf_threshold)
            if masks is not None and masks[0] is not None and \
                    masks[i] is not None:
                key_mask = masks[0]
                hist_mask = masks[i]
                if key_mask.shape[-2:] != key_logits.shape[-2:]:
                    key_mask = F.interpolate(
                        key_mask.float(), size=key_logits.shape[-2:],
                        mode='nearest')
                if hist_mask.shape[-2:] != key_logits.shape[-2:]:
                    hist_mask = F.interpolate(
                        hist_mask.float(), size=key_logits.shape[-2:],
                        mode='nearest')
                key_mask = key_mask[:, 0] > 0.5
                hist_mask = hist_mask[:, 0] > 0.5
                mask = mask & key_mask & hist_mask
            mask_f = mask.float()

            kl_kh = (p_key * (log_p_key - log_p_hist)).sum(dim=1)
            kl_hk = (p_hist * (log_p_hist - log_p_key)).sum(dim=1)
            sym_kl = 0.5 * (kl_kh + kl_hk)

            masked = sym_kl * mask_f
            count = mask_f.sum().clamp(min=1.0)
            losses.append(masked.sum() / count)

        if not losses:
            return key_logits.sum() * 0.0

        return torch.stack(losses).mean()
