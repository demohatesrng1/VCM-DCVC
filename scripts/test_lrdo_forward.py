"""End-to-end smoke test for ROI-LRDO. CPU-only, no checkpoints needed.

    python scripts/test_lrdo_forward.py         # run from the repository root

Builds the real DMC (random weights, inference_mode so no teachers are needed)
and runs Algorithm 1 on one small frame. It checks the things that are easy to
get wrong and expensive to discover on a GPU box an hour into an encode:

  * feeding the encoder's own latent back through ``y_in`` reproduces the
    unmodified forward pass *exactly* -- if this drifts, every LRDO number is
    measured against the wrong baseline;
  * gradient actually reaches the latent through the whole codec, rate term
    included;
  * the objective goes down, and the bitstream genuinely changes;
  * ``bg_weight=1.0`` gives the same trajectory as plain LRDO.

It does not check that LRDO *helps* a downstream task. That needs the rate-task
evaluation loop.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from secvcm.lrdo import LrdoConfig, optimize_frame
from secvcm.models.video_model import DMC

failures = []


def check(name, condition, detail=""):
    print(f"[{'ok  ' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def build_model():
    torch.manual_seed(0)
    model = DMC(inference_mode=True)
    model.eval()                      # quant must be hard rounding, not the STE
    for p in model.parameters():
        p.requires_grad_(False)       # only the latent is a free variable
    return model


def make_inputs(size=192):
    # >160 px per side: forward_one_frame computes MS-SSIM, which needs that much
    # for its 4 downsamplings. 192 is the smallest multiple of 64 that clears it.
    torch.manual_seed(1)
    x = torch.rand(1, 3, size, size)
    ref = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    dpb = {
        "ref_frame": ref,
        "ref_frame_semantic": ref,
        "ref_feature": None,
        "ref_feature_semantic": None,
        "ref_y": None,
        "ref_mv_y": None,
    }
    roi = torch.zeros(1, 1, size, size)
    roi[:, :, size // 4:3 * size // 4, size // 4:3 * size // 4] = 1.0
    return x, dpb, roi


def main():
    model = build_model()
    x, dpb, roi = make_inputs()
    q = torch.tensor([1.0])

    # ------------------------------------------------- the latent hook is exact
    with torch.no_grad():
        base = model.forward_one_frame(x, dpb, mv_y_q_scale=q, y_q_scale=q,
                                       return_latents=True)
        fed = model.forward_one_frame(x, dpb, mv_y_q_scale=q, y_q_scale=q,
                                      y_in=base['y_latent'],
                                      mv_y_in=base['mv_y_latent'])

    check("return_latents yields a 4-D main latent",
          base['y_latent'].dim() == 4, tuple(base['y_latent'].shape))
    check("feeding y back through y_in reproduces bpp exactly",
          torch.equal(base['bpp'], fed['bpp']),
          f"{base['bpp'].item():.8f} vs {fed['bpp'].item():.8f}")
    check("feeding y back through y_in reproduces the reconstruction exactly",
          torch.equal(base['dpb']['ref_frame'], fed['dpb']['ref_frame']))
    check("feeding y back through y_in reproduces the semantic output exactly",
          torch.equal(base['dpb']['ref_frame_semantic'],
                      fed['dpb']['ref_frame_semantic']))

    # ------------------------------------------- gradient reaches the latent
    from secvcm.lrdo import sga_quantization
    y = base['y_latent'].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        with sga_quantization(model, tau=0.5):
            out = model.forward_one_frame(x, dpb, mv_y_q_scale=q, y_q_scale=q,
                                          y_in=y, mv_y_in=base['mv_y_latent'])
        out['bpp'].mean().backward()
    check("rate term passes a gradient into the latent",
          y.grad is not None and bool((y.grad.abs() > 0).any()),
          f"grad norm {y.grad.norm().item():.4e}")

    y2 = base['y_latent'].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        with sga_quantization(model, tau=0.5):
            out2 = model.forward_one_frame(x, dpb, mv_y_q_scale=q, y_q_scale=q,
                                           y_in=y2, mv_y_in=base['mv_y_latent'])
        ((x - out2['dpb']['ref_frame_semantic']) ** 2).mean().backward()
    check("distortion term passes a gradient into the latent",
          y2.grad is not None and bool((y2.grad.abs() > 0).any()),
          f"grad norm {y2.grad.norm().item():.4e}")

    # -------------------------------------------------------- Algorithm 1 runs
    # LPIPS off: it needs a weight download, and the loop is what is under test.
    cfg = LrdoConfig(enabled=True, iters=6, lr=1e-2, lmbda=170.0,
                     bg_weight=1.0, w_lpips=0.0, target='semantic')
    torch.manual_seed(7)
    result, stats = optimize_frame(model, x, dpb, cfg,
                                   mv_y_q_scale=q, y_q_scale=q)

    check("optimize_frame returns a full result dict",
          all(k in result for k in ('bpp', 'bit', 'dpb')))
    check("the objective decreases", stats['loss'][-1] < stats['loss'][0],
          f"{stats['loss'][0]:.4f} -> {stats['loss'][-1]:.4f}")
    check("the bitstream actually changes",
          abs(stats['bpp_after'] - stats['bpp_before']) > 1e-6,
          f"bpp {stats['bpp_before']:.5f} -> {stats['bpp_after']:.5f}")
    check("the final pass is hard-quantised (quant restored)",
          'quant' not in model.__dict__)
    check("stats record one entry per iteration", len(stats['loss']) == cfg.iters)

    # ------------------------------------------------------- step budget ----
    # The whole diagnosis of a flat BD-rate rests on this: with a tiny travel
    # budget (iters*lr) the latent cannot cross a quantisation boundary, so no
    # symbol is re-coded and the result is flat for a reason that has nothing to
    # do with whether LRDO helps.
    check("stats report symbol churn and displacement",
          'frac_symbols_changed' in stats and 'mean_abs_dy' in stats)

    tiny = LrdoConfig(enabled=True, iters=3, lr=1e-5, lmbda=170.0,
                      bg_weight=1.0, w_lpips=0.0, target='semantic')
    torch.manual_seed(7)
    _, stats_tiny = optimize_frame(model, x, dpb, tiny, mv_y_q_scale=q, y_q_scale=q)
    big = LrdoConfig(enabled=True, iters=3, lr=5.0, lmbda=170.0,
                     bg_weight=1.0, w_lpips=0.0, target='semantic')
    torch.manual_seed(7)
    _, stats_big = optimize_frame(model, x, dpb, big, mv_y_q_scale=q, y_q_scale=q)
    check("a tiny travel budget re-codes (almost) nothing",
          stats_tiny['frac_symbols_changed'] < 0.01,
          f"{stats_tiny['frac_symbols_changed']:.2%} of symbols changed")
    check("a large travel budget re-codes a lot",
          stats_big['frac_symbols_changed'] > stats_tiny['frac_symbols_changed'],
          f"{stats_big['frac_symbols_changed']:.2%} vs {stats_tiny['frac_symbols_changed']:.2%}")
    check("displacement scales with the step budget",
          stats_big['mean_abs_dy'] > stats_tiny['mean_abs_dy'],
          f"|dy| {stats_big['mean_abs_dy']:.4f} vs {stats_tiny['mean_abs_dy']:.6f}")

    # ------------------------------------------- survives an outer no_grad()
    # test_video.run_test wraps its whole frame loop in torch.no_grad(), so the
    # optimisation has to re-enable grad for itself or it silently does nothing.
    torch.manual_seed(7)
    with torch.no_grad():
        _, stats_ng = optimize_frame(model, x, dpb, cfg, mv_y_q_scale=q, y_q_scale=q)
    check("optimize_frame works inside torch.no_grad()",
          stats_ng['loss'][-1] < stats_ng['loss'][0],
          f"{stats_ng['loss'][0]:.4f} -> {stats_ng['loss'][-1]:.4f}")
    check("outer no_grad does not change the result",
          stats_ng['loss'] == stats['loss'])

    # ---------------------------------------------------- picklable for spawn
    import pickle
    check("LrdoConfig survives pickling (ProcessPoolExecutor uses spawn)",
          pickle.loads(pickle.dumps(cfg)).lmbda == cfg.lmbda)

    # ------------------------------------------------------------- invariant 1
    torch.manual_seed(7)
    _, stats_none = optimize_frame(model, x, dpb, cfg, roi=None,
                                   mv_y_q_scale=q, y_q_scale=q)
    torch.manual_seed(7)
    _, stats_unit = optimize_frame(model, x, dpb, cfg, roi=roi,
                                   mv_y_q_scale=q, y_q_scale=q)
    drift = max(abs(a - b) for a, b in zip(stats_none['loss'], stats_unit['loss']))
    check("bg_weight=1.0 matches plain LRDO exactly",
          drift == 0.0, f"max |loss difference| {drift:.2e}")

    # An actual ROI run must diverge from it, or the flag is doing nothing.
    cfg_roi = LrdoConfig(enabled=True, iters=6, lr=1e-2, lmbda=170.0,
                         bg_weight=0.2, w_lpips=0.0, target='semantic')
    torch.manual_seed(7)
    _, stats_roi = optimize_frame(model, x, dpb, cfg_roi, roi=roi,
                                  mv_y_q_scale=q, y_q_scale=q)
    check("bg_weight<1.0 changes the trajectory",
          stats_roi['loss'] != stats_unit['loss'],
          f"roi {stats_roi['loss'][-1]:.4f} vs plain {stats_unit['loss'][-1]:.4f}")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
