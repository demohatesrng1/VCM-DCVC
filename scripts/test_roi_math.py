"""Self-checks for the ROI weighting math. Needs torch only, runs on CPU in seconds.

    python scripts/test_roi_math.py

The first check is the one that matters for the experiment: with bg_weight=1.0 the
ROI path must reduce to the plain mean *exactly*, so "baseline vs ROI" is a single
flag and not a second, accidental change.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from secvcm.roi import (RoiConfig, build_weight_map, resize_weight, weighted_mean,
                        weighted_mean_tokens, weighted_pixel_sum, region_stats,
                        dino_grid_size)

failures = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def make_roi(b=2, h=32, w=32):
    roi = torch.zeros(b, 1, h, w)
    roi[:, :, 8:24, 8:24] = 1.0          # a centred 16x16 "object" = 25% of the frame
    return roi


def main():
    torch.manual_seed(0)
    roi = make_roi()
    value = torch.rand(2, 5, 32, 32)

    # 1. bg_weight = 1.0 must be a no-op
    cfg_off = RoiConfig(enabled=True, bg_weight=1.0)
    w_off = build_weight_map(roi, cfg_off)
    check("bg_weight=1.0 gives all-ones weights", torch.allclose(w_off, torch.ones_like(w_off)))
    check("bg_weight=1.0 reduces to the plain mean",
          torch.allclose(weighted_mean(value, w_off), value.mean(), atol=1e-6),
          f"{weighted_mean(value, w_off).item():.8f} vs {value.mean().item():.8f}")
    check("weight=None reduces to the plain mean",
          torch.allclose(weighted_mean(value, None), value.mean()))

    # 2. disabled config yields no weights at all
    check("disabled config returns None", build_weight_map(roi, RoiConfig(enabled=False)) is None)

    # 3. unit-mean normalisation, so enabling ROI does not rescale the loss
    cfg = RoiConfig(enabled=True, bg_weight=0.2)
    w = build_weight_map(roi, cfg)
    check("weights have unit mean", torch.allclose(w.mean(), torch.tensor(1.0), atol=1e-6),
          f"mean={w.mean().item():.6f}")
    fg_w = w[0, 0, 16, 16].item()
    bg_w = w[0, 0, 0, 0].item()
    check("foreground weighted above background", fg_w > bg_w, f"fg={fg_w:.3f} bg={bg_w:.3f}")
    check("weight ratio matches bg_weight", abs(bg_w / fg_w - 0.2) < 1e-5,
          f"ratio={bg_w / fg_w:.5f}")

    # 4. the weighted mean actually follows the foreground
    fg_heavy = torch.zeros(2, 5, 32, 32)
    fg_heavy[:, :, 8:24, 8:24] = 1.0
    check("weighted mean tracks the foreground",
          weighted_mean(fg_heavy, w) > fg_heavy.mean(),
          f"weighted={weighted_mean(fg_heavy, w).item():.4f} plain={fg_heavy.mean().item():.4f}")

    # 5. resizing to feature scales preserves the unit mean (1/4, 1/8, 1/16)
    for scale in (4, 8, 16):
        size = (32 // scale, 32 // scale)
        w_s = resize_weight(w, size)
        check(f"resize to 1/{scale} keeps unit mean",
              torch.allclose(w_s.mean(), torch.tensor(1.0), atol=1e-5),
              f"mean={w_s.mean().item():.6f} shape={tuple(w_s.shape)}")

    # 6. weighted mean at a coarser scale still equals the plain mean when uniform
    coarse = torch.rand(2, 7, 8, 8)
    check("uniform weights at a coarse scale reduce to the mean",
          torch.allclose(weighted_mean(coarse, w_off), coarse.mean(), atol=1e-6))

    # 7. region diagnostics split the frame the way the ROI does
    fg, bg = region_stats(fg_heavy, roi, 0.5)
    check("region_stats separates fg/bg", abs(fg.item() - 1.0) < 1e-6 and abs(bg.item()) < 1e-6,
          f"fg={fg.item():.4f} bg={bg.item():.4f}")
    fg2, bg2 = region_stats(coarse, None, 0.5)
    check("region_stats without an ROI reports zeros", fg2.item() == 0.0 and bg2.item() == 0.0)

    # 8. pixel-domain re-weighting keeps the total error scale comparable
    err = torch.rand(2, 3, 32, 32)
    check("weighted_pixel_sum is identity without weights",
          torch.equal(weighted_pixel_sum(err, None), err))
    check("weighted_pixel_sum preserves scale within 2x",
          0.5 < (weighted_pixel_sum(err, w).sum() / err.sum()).item() < 2.0,
          f"ratio={(weighted_pixel_sum(err, w).sum() / err.sum()).item():.3f}")

    # 9. DINOv2 token path
    grid = dino_grid_size(256, 256)
    check("dino grid for 256x256 is 18x18", grid == (18, 18), f"grid={grid}")
    tokens = torch.rand(2, grid[0] * grid[1], 9)
    check("token weighting reduces to the mean when uniform",
          torch.allclose(weighted_mean_tokens(tokens, w_off, grid), tokens.mean(), atol=1e-6))
    check("token weighting falls back safely on a token-count mismatch",
          torch.allclose(weighted_mean_tokens(torch.rand(2, 7, 9), w, grid),
                         torch.tensor(0.0), atol=1.0))

    # 10. threshold ablation actually changes the partition on a soft map
    soft = torch.rand(2, 1, 32, 32)
    lo = build_weight_map(soft, RoiConfig(enabled=True, bg_weight=0.2, threshold=0.2))
    hi = build_weight_map(soft, RoiConfig(enabled=True, bg_weight=0.2, threshold=0.8))
    check("threshold changes the foreground area", not torch.allclose(lo, hi))

    # 11. bad target names are rejected loudly
    try:
        RoiConfig(enabled=True, targets=('biec', 'nonsense'))
        rejected = False
    except ValueError:
        rejected = True
    check("unknown roi target is rejected", rejected)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
