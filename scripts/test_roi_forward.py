"""End-to-end smoke test for the ROI plumbing. CPU-only, no checkpoints needed.

    python scripts/test_roi_forward.py          # run from the repository root

Builds the real DMC with stand-in teachers (correct shapes, random weights) and
pushes one frame through it. This exercises the parts that are easy to get wrong
and expensive to discover on a GPU box six hours into a run: tensor shapes at
every BiEC scale, the DINOv2 token path, the Stage 1 (--skip_semantic) path, and
the invariant that bg_weight=1.0 is bit-identical to no ROI at all.

It does not check that ROI weighting *helps* -- that is what scripts/pilot_ab.sh
is for.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn

from secvcm.models.video_model import DMC
from secvcm.roi import RoiConfig

# Swin-Tiny feature widths, which fix the DistributionGeneration channel counts.
SWIN_CHANNELS = {'res2': 96, 'res3': 192, 'res4': 384, 'res5': 768}

failures = []


def check(name, condition, detail=""):
    print(f"[{'ok  ' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


class StubSwinBackbone(nn.Module):
    """Emits feature maps with Swin-Tiny's shapes at strides 4/8/16/32."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleDict({
            key: nn.Conv2d(3, ch, 3, stride=stride, padding=1)
            for (key, ch), stride in zip(SWIN_CHANNELS.items(), (4, 8, 16, 32))
        })

    def forward(self, x):
        return {key: conv(x) for key, conv in self.convs.items()}


class StubMask2Former(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = StubSwinBackbone()
        self.register_buffer('pixel_mean', torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1))


class StubDino(nn.Module):
    """Mimics dinov2's get_intermediate_layers: a list of (B, N, C) token tensors."""

    def __init__(self, dim=384, patch=14):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)

    def get_intermediate_layers(self, x, n):
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return [tokens for _ in range(n)]


def build(roi_cfg=None, skip_semantic=False):
    torch.manual_seed(0)
    if skip_semantic:
        return DMC(swin_model=None, cnn_model=None, dino_model=None,
                   skip_semantic=True, roi_cfg=None).eval()
    import torchvision
    return DMC(swin_model=StubMask2Former(),
               cnn_model=torchvision.models.resnet18(weights=None),
               dino_model=StubDino(),
               roi_cfg=roi_cfg).eval()


def empty_dpb(x):
    return {"ref_frame": x, "ref_frame_semantic": None, "ref_feature": None,
            "ref_feature_semantic": None, "ref_y": None, "ref_mv_y": None}


def make_inputs(b=1, size=256):
    torch.manual_seed(1)
    x = torch.rand(b, 3, size, size)
    ref = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    roi = torch.zeros(b, 1, size, size)
    roi[:, :, size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 1.0
    return x, ref, roi


def scalars(result):
    keys = ('mse', 'mse_semantic', 'bpp', 'lpips_swin', 'lpips_cnn', 'lpips_dinov2',
            'conditional_entropy_4', 'conditional_entropy_8', 'conditional_entropy_16',
            'entropy_fg', 'entropy_bg')
    return {k: float(result[k].mean()) for k in keys}


def main():
    if not os.path.isdir('./pretrain/flow_pretrain_np'):
        print("run this from the repository root (./pretrain/flow_pretrain_np must exist)")
        return 1

    x, ref, roi = make_inputs()

    # --- Stage 1 path: no semantic branch, no teachers ------------------------
    with torch.no_grad():
        out = build(skip_semantic=True)(x, empty_dpb(ref), lmd_index=0)
    check("skip_semantic forward runs", torch.isfinite(out['bpp']).all())
    check("skip_semantic reports zero semantic distortion", float(out['mse_semantic'].mean()) == 0.0)
    check("skip_semantic reports zero entropy", float(out['conditional_entropy']) == 0.0)
    check("skip_semantic still produces a real rate", float(out['bpp'].mean()) > 0.0,
          f"bpp={float(out['bpp'].mean()):.4f}")
    check("skip_semantic pixel branch is finite", torch.isfinite(out['mse']).all())

    # --- Full semantic branch, no ROI ----------------------------------------
    with torch.no_grad():
        base = scalars(build(roi_cfg=None)(x, empty_dpb(ref), lmd_index=0))
    check("baseline forward runs", all(v == v for v in base.values()))
    check("baseline BiEC terms are non-zero",
          base['conditional_entropy_4'] > 0 and base['conditional_entropy_16'] > 0,
          f"e4={base['conditional_entropy_4']:.3f} e16={base['conditional_entropy_16']:.3f}")
    check("baseline reports no ROI diagnostics", base['entropy_fg'] == 0.0 and base['entropy_bg'] == 0.0)

    # --- ROI enabled but neutral (bg_weight = 1.0) ---------------------------
    with torch.no_grad():
        neutral = scalars(build(roi_cfg=RoiConfig(enabled=True, bg_weight=1.0))(
            x, empty_dpb(ref), lmd_index=0, roi=roi))
    same = all(abs(neutral[k] - base[k]) < 1e-6
               for k in base if k not in ('entropy_fg', 'entropy_bg'))
    check("bg_weight=1.0 matches the no-ROI baseline exactly", same,
          f"max delta={max(abs(neutral[k] - base[k]) for k in base if k not in ('entropy_fg', 'entropy_bg')):.2e}")
    check("neutral ROI still reports fg/bg diagnostics",
          neutral['entropy_fg'] > 0 and neutral['entropy_bg'] > 0,
          f"fg={neutral['entropy_fg']:.3f} bg={neutral['entropy_bg']:.3f}")

    # --- ROI actually weighting ----------------------------------------------
    cfg = RoiConfig(enabled=True, bg_weight=0.2, targets=('biec', 'swin', 'cnn'))
    with torch.no_grad():
        weighted = scalars(build(roi_cfg=cfg)(x, empty_dpb(ref), lmd_index=0, roi=roi))
    check("ROI changes the BiEC term",
          abs(weighted['conditional_entropy_4'] - base['conditional_entropy_4']) > 1e-6,
          f"{base['conditional_entropy_4']:.4f} -> {weighted['conditional_entropy_4']:.4f}")
    check("ROI changes the swin teacher term",
          abs(weighted['lpips_swin'] - base['lpips_swin']) > 1e-9,
          f"{base['lpips_swin']:.6f} -> {weighted['lpips_swin']:.6f}")
    check("ROI leaves untargeted terms alone (dino not in targets)",
          abs(weighted['lpips_dinov2'] - base['lpips_dinov2']) < 1e-9)
    check("ROI leaves the rate untouched (base codec is frozen in warm-up)",
          abs(weighted['bpp'] - base['bpp']) < 1e-9,
          f"bpp={weighted['bpp']:.6f}")
    check("ROI leaves mse_semantic alone when 'mse' is not targeted",
          abs(weighted['mse_semantic'] - base['mse_semantic']) < 1e-9)

    # --- targeting the pixel term too ----------------------------------------
    cfg_mse = RoiConfig(enabled=True, bg_weight=0.2, targets=('biec', 'swin', 'cnn', 'mse'))
    with torch.no_grad():
        with_mse = scalars(build(roi_cfg=cfg_mse)(x, empty_dpb(ref), lmd_index=0, roi=roi))
    check("adding 'mse' to targets weights mse_semantic",
          abs(with_mse['mse_semantic'] - base['mse_semantic']) > 1e-9,
          f"{base['mse_semantic']:.6f} -> {with_mse['mse_semantic']:.6f}")

    # --- gradients reach the semantic modules --------------------------------
    model = build(roi_cfg=cfg).train()
    out = model(x, empty_dpb(ref), lmd_index=0, roi=roi)
    loss = (out['conditional_entropy_4'] + out['conditional_entropy_8'] + out['conditional_entropy_16']) / 3.0
    loss.backward()
    grads = [p.grad for p in model.semantic_decoder.parameters() if p.grad is not None]
    check("BiEC gradients reach the semantic decoder",
          len(grads) > 0 and any(g.abs().sum() > 0 for g in grads),
          f"{len(grads)} tensors with grad")

    # --- non-square input, to catch hardcoded shape assumptions --------------
    # (MS-SSIM needs >160px per side, so 256x192 is the smallest useful shape here)
    x2, ref2, roi2 = make_inputs(size=256)
    x2, ref2, roi2 = x2[..., :192], ref2[..., :192], roi2[..., :192]     # 256x192
    with torch.no_grad():
        out2 = build(roi_cfg=cfg)(x2, empty_dpb(ref2), lmd_index=0, roi=roi2)
    check("non-square input works", torch.isfinite(out2['conditional_entropy_4']).all(),
          f"shape={tuple(x2.shape)}")

    # --- batch of 2, to catch per-sample normalisation mistakes --------------
    x3, ref3, roi3 = make_inputs(b=2)
    roi3[1] = 0.0                                    # second sample has no foreground
    with torch.no_grad():
        out3 = build(roi_cfg=cfg)(x3, empty_dpb(ref3), lmd_index=0, roi=roi3)
    check("batch with an empty ROI map works",
          torch.isfinite(out3['conditional_entropy_4']).all() and torch.isfinite(out3['bpp']).all(),
          "sample 1 has no foreground")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
