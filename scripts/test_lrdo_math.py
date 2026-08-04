"""Self-checks for the ROI-LRDO math. Needs torch only, runs on CPU in seconds.

    python scripts/test_lrdo_math.py

Two checks carry the experiment:

  * SGA must pass a non-zero gradient into the latent. ``quant`` is a plain
    ``torch.round`` in eval mode, so without the swap every gradient is zero and
    the optimisation silently does nothing -- which looks exactly like "LRDO
    didn't help".
  * ``bg_weight=1.0`` must reduce to the unweighted objective *exactly*, so
    "LRDO vs ROI-LRDO" is a single flag and not a second, accidental change.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from secvcm.lrdo import (LrdoConfig, sga_quantize, sga_quantization, tau_at,
                         distortion_terms)
from secvcm.roi import build_weight_map, weighted_mean

failures = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


class StubModel:
    """Minimal stand-in for CompressionModel.quant."""

    def quant(self, x, force_detach=False):
        return torch.round(x)


def make_roi(b=1, h=32, w=32):
    roi = torch.zeros(b, 1, h, w)
    roi[:, :, 8:24, 8:24] = 1.0          # centred 16x16 "object" = 25% of the frame
    return roi


def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------------ SGA --
    x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.0, 1.5, 7.0])
    q = sga_quantize(x, tau=0.5)
    check("sga leaves exact integers untouched",
          torch.allclose(q[[0, 2, 4, 6]], x[[0, 2, 4, 6]], atol=1e-5),
          f"{q[[0, 2, 4, 6]].tolist()}")
    check("sga output stays within [floor, ceil]",
          bool((q >= torch.floor(x) - 1e-5).all() and (q <= torch.ceil(x) + 1e-5).all()))

    # SGA is a *sampler*, so a value sitting on a bin boundary may legitimately
    # land on either side however small tau is. Check the two things that are
    # actually required: values away from the midpoint round deterministically,
    # and disagreement overall is rare.
    vals = torch.rand(4096) * 10 - 5
    hard = sga_quantize(vals, tau=0.01)
    frac = vals - torch.floor(vals)
    decided = (frac - 0.5).abs() > 0.1
    check("small tau rounds unambiguous values exactly",
          torch.equal(hard[decided], torch.round(vals)[decided]),
          f"{int(decided.sum())} unambiguous values")
    disagree = (hard - torch.round(vals)).abs().gt(1e-3).float().mean().item()
    check("small tau disagrees with rounding only rarely",
          disagree < 0.02, f"disagreement rate {disagree:.4%}")

    soft = sga_quantize(vals, tau=0.5)
    check("larger tau is not hard rounding",
          not torch.allclose(soft, torch.round(vals), atol=1e-3))

    # The one that matters: gradient has to reach the latent.
    z = torch.rand(64, requires_grad=True)
    sga_quantize(z, tau=0.5).sum().backward()
    check("sga passes a gradient into the latent",
          z.grad is not None and bool((z.grad.abs() > 0).any()),
          f"nonzero grads: {int((z.grad.abs() > 0).sum())}/64")

    z2 = torch.rand(64, requires_grad=True)
    torch.round(z2).sum().backward()
    check("plain rounding passes zero gradient (why the swap exists)",
          bool((z2.grad.abs() == 0).all()))

    # --------------------------------------------------------------- anneal --
    taus = [tau_at(i, 20, 0.5, 0.05) for i in range(20)]
    check("tau schedule starts at tau_init", abs(taus[0] - 0.5) < 1e-6)
    check("tau schedule ends at tau_min", abs(taus[-1] - 0.05) < 1e-6, f"{taus[-1]:.4f}")
    check("tau schedule is monotone decreasing",
          all(a >= b for a, b in zip(taus, taus[1:])))
    check("tau schedule is independent of iters at the endpoints",
          abs(tau_at(199, 200, 0.5, 0.05) - 0.05) < 1e-6)

    # ------------------------------------------------------ context manager --
    model = StubModel()
    w = torch.rand(16, requires_grad=True)
    with sga_quantization(model, tau=0.5):
        check("context manager swaps quant", 'quant' in model.__dict__)
        out = model.quant(w)
        check("swapped quant is differentiable", out.requires_grad)
    check("context manager leaves no instance attribute behind",
          'quant' not in model.__dict__)
    check("restored quant is hard rounding",
          torch.allclose(model.quant(w).detach(), torch.round(w.detach())))
    w2 = torch.rand(16, requires_grad=True)
    model.quant(w2).sum().backward()
    check("restored quant passes zero gradient again",
          bool((w2.grad.abs() == 0).all()))

    # ------------------------------------------------------- ROI invariants --
    roi = make_roi()
    cfg_off = LrdoConfig(enabled=True, bg_weight=1.0, w_lpips=0.0)
    w_off = build_weight_map(roi, cfg_off.roi_config())
    check("bg_weight=1.0 gives all-ones weights",
          torch.allclose(w_off, torch.ones_like(w_off)))

    x_ref = torch.rand(1, 3, 32, 32)
    x_hat = torch.rand(1, 3, 32, 32)
    mse_plain, _ = distortion_terms(x_ref, x_hat, None, cfg_off)
    mse_off, _ = distortion_terms(x_ref, x_hat, w_off, cfg_off)
    unweighted = ((x_ref - x_hat) ** 2).mean()
    check("bg_weight=1.0 gives the same distortion as no ROI at all",
          torch.allclose(mse_off, mse_plain, atol=1e-8),
          f"{mse_off.item():.8f} vs {mse_plain.item():.8f}")

    # The LRDO distortion must use the codec's own MSE convention, or lambda is
    # silently scaled and LRDO trades quality for bits. forward_one_frame uses
    # sum(dim=(1,2,3)) / (H*W), i.e. 3x the plain mean for RGB.
    codec_mse = (((x_ref - x_hat) ** 2).sum(dim=(1, 2, 3)) / (32 * 32)).mean()
    check("LRDO MSE matches the codec's sum/(H*W) convention",
          torch.allclose(mse_plain, codec_mse, atol=1e-6),
          f"{mse_plain.item():.8f} vs {codec_mse.item():.8f}")
    check("LRDO MSE is 3x the plain mean for RGB",
          torch.allclose(mse_plain, unweighted * 3, atol=1e-6))

    # A foreground-heavy error should be penalised more once the ROI is on.
    err = torch.zeros(1, 3, 32, 32)
    err[:, :, 8:24, 8:24] = 1.0                      # error only inside the ROI
    cfg_roi = LrdoConfig(enabled=True, bg_weight=0.2, w_lpips=0.0)
    w_roi = build_weight_map(roi, cfg_roi.roi_config())
    fg_off = weighted_mean(err, w_off)
    fg_on = weighted_mean(err, w_roi)
    check("ROI weighting up-weights foreground error",
          fg_on > fg_off, f"{fg_on.item():.4f} > {fg_off.item():.4f}")

    err_bg = 1.0 - err
    bg_off = weighted_mean(err_bg, w_off)
    bg_on = weighted_mean(err_bg, w_roi)
    check("ROI weighting down-weights background error",
          bg_on < bg_off, f"{bg_on.item():.4f} < {bg_off.item():.4f}")

    check("unit-mean normalisation keeps the overall scale",
          abs(w_roi.mean().item() - 1.0) < 1e-5, f"mean {w_roi.mean().item():.6f}")

    # ------------------------------------------------------------- config ---
    check("uses_roi is False at bg_weight=1.0",
          not LrdoConfig(enabled=True, bg_weight=1.0).uses_roi)
    check("uses_roi is True at bg_weight<1.0",
          LrdoConfig(enabled=True, bg_weight=0.5).uses_roi)
    check("uses_roi is False when disabled",
          not LrdoConfig(enabled=False, bg_weight=0.5).uses_roi)

    try:
        LrdoConfig(target='nonsense')
        rejected = False
    except ValueError:
        rejected = True
    check("unknown lrdo target is rejected", rejected)

    try:
        LrdoConfig(iters=0)
        rejected = False
    except ValueError:
        rejected = True
    check("iters=0 is rejected", rejected)

    class Args:
        lrdo = True
        lrdo_lambdas = [40.0, 85.0, 170.0, 380.0]
        lrdo_bg_weight = 0.5

    check("from_args picks lambda by rate index",
          LrdoConfig.from_args(Args(), rate_idx=2).lmbda == 170.0)
    check("from_args clamps an out-of-range rate index",
          LrdoConfig.from_args(Args(), rate_idx=9).lmbda == 380.0)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
