"""Region-of-importance (ROI) weighting for SEC-VCM training.

The ROI map is a per-pixel importance map in [0, 1], produced offline by an
instance segmentation model (see ``scripts/precompute_roi_masks.py``).  It is
used to spatially re-weight the objectives that act on the semantic branch, so
that the semantic decoder spends its capacity on the regions a machine vision
system actually consumes.

Everything in this module is training-only: no ROI map is read at inference
time and no extra module is added to the codec.

Two invariants are worth keeping in mind, because the whole comparison rests
on them:

1. ``bg_weight == 1.0`` reproduces the unweighted baseline *exactly* (every
   weight becomes 1.0 and every reduction degenerates to ``tensor.mean()``).
   That makes "baseline vs ROI" a single-flag difference.
2. Weight maps are normalised to unit mean per sample, so switching the ROI on
   does not change the effective magnitude of any loss term.  Without this,
   an apparent ROI gain could just be a re-tuned loss weight.
"""

import torch
import torch.nn.functional as F


# Loss terms that can be spatially re-weighted.  See DMC.forward_one_frame.
ROI_TARGETS = ("biec", "swin", "cnn", "dino", "mse")

# Default: the conditional-entropy (BiEC) term plus the two convolutional
# teacher feature losses.  These are the terms that operate on spatial feature
# maps at the scales BiEC aligns (1/4, 1/8, 1/16), so a spatial prior is
# meaningful for them.
ROI_DEFAULT_TARGETS = ("biec", "swin", "cnn")


class RoiConfig:
    """Configuration for ROI weighting.

    Attributes:
        enabled:    master switch.  When False every helper here is a no-op.
        bg_weight:  weight given to background pixels.  Foreground is always
                    1.0, so ``bg_weight=1.0`` means "no ROI" and
                    ``bg_weight=0.2`` means "background counts 5x less".
        threshold:  binarisation threshold applied to the stored soft map.
        soft:       use the stored confidence directly instead of thresholding.
        normalize:  rescale each weight map to unit mean.
        targets:    which loss terms get weighted.
    """

    def __init__(self, enabled=False, bg_weight=0.5, threshold=0.5, soft=False,
                 normalize=True, targets=ROI_DEFAULT_TARGETS):
        unknown = set(targets) - set(ROI_TARGETS)
        if unknown:
            raise ValueError(f"unknown roi targets {sorted(unknown)}, valid: {ROI_TARGETS}")
        self.enabled = bool(enabled)
        self.bg_weight = float(bg_weight)
        self.threshold = float(threshold)
        self.soft = bool(soft)
        self.normalize = bool(normalize)
        self.targets = frozenset(targets)

    def wants(self, target):
        return self.enabled and target in self.targets

    def __repr__(self):
        return (f"RoiConfig(enabled={self.enabled}, bg_weight={self.bg_weight}, "
                f"threshold={self.threshold}, soft={self.soft}, "
                f"normalize={self.normalize}, targets={sorted(self.targets)})")

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--roi', action='store_true',
                            help="Enable ROI-weighted semantic alignment (requires precomputed masks).")
        parser.add_argument('--roi_bg_weight', type=float, default=0.5,
                            help="Weight of background pixels; 1.0 reproduces the unweighted baseline.")
        parser.add_argument('--roi_threshold', type=float, default=0.5,
                            help="Threshold applied to the stored soft ROI map.")
        parser.add_argument('--roi_soft', action='store_true',
                            help="Use the stored confidence map directly instead of thresholding.")
        parser.add_argument('--roi_no_normalize', action='store_true',
                            help="Do not rescale weight maps to unit mean (not recommended).")
        parser.add_argument('--roi_targets', nargs='+', default=list(ROI_DEFAULT_TARGETS),
                            help=f"Loss terms to re-weight, any of {ROI_TARGETS}.")
        return parser

    @classmethod
    def from_args(cls, args):
        return cls(enabled=getattr(args, 'roi', False),
                   bg_weight=getattr(args, 'roi_bg_weight', 0.5),
                   threshold=getattr(args, 'roi_threshold', 0.5),
                   soft=getattr(args, 'roi_soft', False),
                   normalize=not getattr(args, 'roi_no_normalize', False),
                   targets=tuple(getattr(args, 'roi_targets', ROI_DEFAULT_TARGETS)))


def build_weight_map(roi, cfg):
    """Turn a stored ROI map into a normalised weight map.

    Args:
        roi: (B, 1, H, W) importance in [0, 1].
        cfg: RoiConfig.

    Returns:
        (B, 1, H, W) weights with unit mean per sample, or None if disabled.
    """
    if roi is None or cfg is None or not cfg.enabled:
        return None
    roi = roi.clamp(0.0, 1.0)
    mask = roi if cfg.soft else (roi >= cfg.threshold).to(roi.dtype)
    weight = cfg.bg_weight + (1.0 - cfg.bg_weight) * mask
    if cfg.normalize:
        denom = weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        weight = weight / denom
    return weight


def resize_weight(weight, size):
    """Resample a weight map to a feature-map size.

    Downsampling uses area averaging, which is the right reduction for an
    importance map: the weight of a feature cell is the mean importance of the
    pixels in its footprint.
    """
    if weight is None:
        return None
    size = (int(size[0]), int(size[1]))
    if tuple(weight.shape[-2:]) == size:
        return weight
    if weight.shape[-2] >= size[0] and weight.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(weight, size)
    return F.interpolate(weight, size=size, mode='bilinear', align_corners=False)


def weighted_mean(value, weight):
    """Weighted mean of a (B, C, H, W) tensor under a (B, 1, h, w) weight map.

    Equals ``value.mean()`` when ``weight`` is None or uniform.
    """
    if weight is None:
        return value.mean()
    weight = resize_weight(weight, value.shape[-2:])
    total = weight.sum() * value.shape[1]
    return (value * weight).sum() / total.clamp_min(1e-6)


def weighted_mean_tokens(value, weight, grid_hw):
    """Weighted mean of a (B, N, C) token tensor laid out on a ``grid_hw`` grid.

    Used for DINOv2, whose features are patch tokens rather than a feature map.
    Falls back to a plain mean if the token count does not match the grid (e.g.
    if the backbone starts returning prefix tokens).
    """
    if weight is None:
        return value.mean()
    grid = resize_weight(weight, grid_hw)
    tokens = grid.flatten(2).transpose(1, 2)          # (B, N, 1)
    if tokens.shape[1] != value.shape[1]:
        return value.mean()
    total = tokens.sum() * value.shape[2]
    return (value * tokens).sum() / total.clamp_min(1e-6)


def weighted_pixel_sum(value, weight):
    """Re-weight a (B, C, H, W) per-pixel error before the caller sums it.

    The caller keeps its own ``sum(dim=(1,2,3)) / (H*W)`` convention, so this
    only applies the (unit-mean) weights and leaves the reduction alone.
    """
    if weight is None:
        return value
    return value * resize_weight(weight, value.shape[-2:])


def region_stats(value, roi, threshold=0.5):
    """Mean of ``value`` inside and outside the ROI, for diagnostics.

    These are the numbers that show *why* an ROI run differs from baseline:
    foreground error should drop while background error rises.  Returns
    ``(fg, bg)`` as detached scalars; an empty region reports 0.
    """
    if roi is None:
        zero = value.detach().new_zeros(())
        return zero, zero
    with torch.no_grad():
        mask = (roi >= threshold).to(value.dtype)
        mask = resize_weight(mask, value.shape[-2:])
        mask = (mask >= 0.5).to(value.dtype)
        value = value.detach()
        channels = value.shape[1]
        fg_n = mask.sum() * channels
        bg_n = (1.0 - mask).sum() * channels
        fg = (value * mask).sum() / fg_n if fg_n > 0 else value.new_zeros(())
        bg = (value * (1.0 - mask)).sum() / bg_n if bg_n > 0 else value.new_zeros(())
    return fg, bg


def dino_grid_size(height, width, patch=14):
    """Token grid produced by DinoV2 for an input of this size.

    Mirrors the resize inside ``DinoV2.forward`` so the ROI map can be pooled
    onto the same grid.
    """
    new_h = max(round(height / patch), 1) * patch
    new_w = max(round(width / patch), 1) * patch
    return new_h // patch, new_w // patch
