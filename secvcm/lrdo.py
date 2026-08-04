"""ROI-based latent rate-distortion optimisation (ROI-LRDO) for SEC-VCM.

A port of Algorithm 1 / Eq. 5 of PO-RTIntra [1] onto this video codec.

The idea in one line: freeze every weight, and optimise the *latent* of each
frame at encode time under an ROI-weighted rate-distortion objective

    L = R + lambda * (w1 * L_mse^roi + w2 * L_lpips^roi)

using Stochastic Gumbel Annealing [2] as a differentiable stand-in for rounding.

Why this is worth having, given ``secvcm/roi.py`` already does ROI weighting:

  * It needs no training.  It runs against the released checkpoint, so an
    experiment costs minutes on one GPU instead of days on four.
  * It actually moves bits.  The Stage 2 schedule (``semantic_v3``) freezes the
    encoder and entropy model for 8 of its 10 epochs and carries no rate term,
    so training-time ROI weighting cannot reallocate a single bit -- it can only
    change what the semantic decoder reconstructs.  LRDO edits ``y`` directly,
    so the bitstream genuinely changes.
  * It is decoder-compatible.  Nothing about the decoder or the bitstream format
    changes; only the encoder does more work.

Two caveats that belong in any write-up:

  * PO-RTIntra is an *intra* codec: one image, one latent, no history.  Here
    frame t's latent propagates into ``ref_frame`` / ``ref_feature`` / ``ref_y``
    for every later frame, so optimising frames greedily one at a time is not
    the same as optimising the sequence.  ``optimize_frame`` is greedy.  That is
    a real limitation and also the interesting open problem.
  * PO-RTIntra reports its rate-distortion curves *without* LRDO and shows it
    only qualitatively, so that paper does not establish the size of the gain.

Two invariants, mirroring the ones in ``secvcm/roi.py``, because the comparison
rests on them:

  1. ``bg_weight == 1.0`` reproduces unweighted LRDO *exactly* -- every weight
     becomes 1.0 and every reduction degenerates to ``tensor.mean()``.  That
     makes "LRDO vs ROI-LRDO" a single-flag difference.
  2. Weight maps are normalised to unit mean, so switching the ROI on does not
     change the effective magnitude of the distortion term relative to the rate
     term.  Without this, an apparent ROI gain could just be a different
     effective lambda.  (PO-RTIntra does not normalise; it does not need to,
     because it is not running a controlled A/B.)

[1] Ma, Zhang, Fu & Chen, "PO-RTIntra: Perception-Weighted Rate Allocation and
    ROI Latent Optimization on DCVC-RT Intra".
[2] Yang, Bamler & Mandt, "Improving Inference for Neural Image Compression",
    NeurIPS 2020 -- the source of Stochastic Gumbel Annealing.
"""

import math

import torch

from .roi import RoiConfig, build_weight_map, weighted_mean


# --------------------------------------------------------------------------- #
# Stochastic Gumbel Annealing
# --------------------------------------------------------------------------- #

def sga_quantize(x, tau, eps=1e-6):
    """Differentiable stand-in for ``torch.round``.

    Samples between ``floor(x)`` and ``ceil(x)`` with a relaxed categorical whose
    logits push mass toward whichever bound is nearer.  As ``tau -> 0`` this
    converges to hard rounding; at larger ``tau`` it stays smooth enough to carry
    a useful gradient into ``x``.

    Exact integers are a fixed point: both bounds equal ``x``, so the output is
    ``x`` regardless of the sample.
    """
    tau = max(float(tau), 1e-3)
    x_floor = torch.floor(x)
    x_ceil = torch.ceil(x)

    # Distance to each bound, clamped away from +-1 so atanh stays finite.
    d_floor = torch.clamp(x - x_floor, -1.0 + eps, 1.0 - eps)
    d_ceil = torch.clamp(x_ceil - x, -1.0 + eps, 1.0 - eps)

    # Near the floor, d_floor is small -> logit_floor ~ 0 while logit_ceil -> -inf.
    logits = torch.stack((-torch.atanh(d_floor) / tau,
                          -torch.atanh(d_ceil) / tau), dim=-1)

    dist = torch.distributions.RelaxedOneHotCategorical(tau, logits=logits)
    sample = dist.rsample()
    bounds = torch.stack((x_floor, x_ceil), dim=-1)
    return (bounds * sample).sum(dim=-1)


class sga_quantization:
    """Context manager that swaps ``model.quant`` for the SGA sampler.

    ``CompressionModel.quant`` returns ``torch.round(x)`` in eval mode, which has
    zero gradient everywhere -- so latent optimisation is impossible without
    replacing it.  Every quantisation site in the codec (``y``, ``z``, ``mv_y``,
    ``mv_z``) goes through this one method, so a single swap covers all of them.

    ``tau`` is read from ``self.tau`` at call time, so the caller can anneal it
    across iterations without rebuilding the context.
    """

    def __init__(self, model, tau):
        self.model = model
        self.tau = tau
        self._had_own = False
        self._saved = None

    def __enter__(self):
        # ``quant`` is normally a class method reached through the instance, so
        # there is nothing in __dict__ to restore -- the swap has to be *removed*
        # on exit rather than reassigned, or the instance keeps shadowing the
        # class for the rest of the run.
        self._had_own = 'quant' in self.model.__dict__
        self._saved = self.model.__dict__.get('quant')

        def quant(x, force_detach=False):
            return sga_quantize(x, self.tau)

        self.model.quant = quant
        return self

    def __exit__(self, *exc):
        if self._had_own:
            self.model.quant = self._saved
        else:
            del self.model.quant
        self._saved = None
        return False


def tau_at(step, iters, tau_init, tau_min):
    """Exponential anneal from ``tau_init`` down to ``tau_min`` over ``iters``.

    Yang et al. anneal on a fixed exponential schedule; tying the rate to the
    iteration budget means changing ``--lrdo_iters`` does not silently change how
    hard the final sample is rounded.
    """
    if iters <= 1:
        return tau_min
    rate = math.log(max(tau_init, 1e-8) / max(tau_min, 1e-8)) / (iters - 1)
    return max(tau_min, tau_init * math.exp(-rate * step))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LRDO_TARGETS = ('semantic', 'recon')


class LrdoConfig:
    """Configuration for ROI-LRDO.

    Attributes:
        enabled:      master switch.
        iters:        N in Algorithm 1.  Cost is linear in this.
        lr:           step size for the Adam update on the latent.
        lmbda:        lambda in ``L = R + lambda * D``.  PO-RTIntra aligns this
                      with the training-phase lambda of the matching rate point,
                      which for this codebase is ``--lambdas`` in main_ddp.py.
        bg_weight:    background weight; 1.0 means "no ROI" (plain LRDO).
        threshold:    binarisation threshold for the stored soft map.
        soft:         use the stored confidence directly instead of thresholding.
        normalize:    rescale weight maps to unit mean.  See invariant 2 above.
        w_mse:        w1' in Eq. 5.
        w_lpips:      w2' in Eq. 5.  Set 0 to drop the LPIPS term entirely.
        lpips_match:  rescale the LPIPS term so its magnitude matches the MSE
                      term, as PO-RTIntra specifies.  The factor is fixed at the
                      first iteration so the objective stays stationary.
        target:       which reconstruction to optimise.  'semantic' is the
                      machine-facing output (``ref_frame_semantic``), which is
                      what test_video.py writes to disk; 'recon' is the pixel
                      branch.
        optimize_mv:  also optimise the motion latent.  Off by default: PO-RTIntra
                      has no motion stream, and it doubles the free variables.
    """

    def __init__(self, enabled=False, iters=30, lr=5e-3, lmbda=170.0,
                 bg_weight=1.0, threshold=0.5, soft=False, normalize=True,
                 w_mse=0.5, w_lpips=0.5, lpips_match=True, target='semantic',
                 optimize_mv=False, tau_init=0.5, tau_min=0.05):
        if target not in LRDO_TARGETS:
            raise ValueError(f"unknown lrdo target {target!r}, valid: {LRDO_TARGETS}")
        if iters < 1:
            raise ValueError(f"lrdo iters must be >= 1, got {iters}")
        self.enabled = bool(enabled)
        self.iters = int(iters)
        self.lr = float(lr)
        self.lmbda = float(lmbda)
        self.bg_weight = float(bg_weight)
        self.threshold = float(threshold)
        self.soft = bool(soft)
        self.normalize = bool(normalize)
        self.w_mse = float(w_mse)
        self.w_lpips = float(w_lpips)
        self.lpips_match = bool(lpips_match)
        self.target = target
        self.optimize_mv = bool(optimize_mv)
        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)

    @property
    def uses_roi(self):
        """True when the weight map is not identically 1.0."""
        return self.enabled and self.bg_weight != 1.0

    def roi_config(self):
        """The equivalent RoiConfig, so the weighting is identical to training."""
        return RoiConfig(enabled=True, bg_weight=self.bg_weight,
                         threshold=self.threshold, soft=self.soft,
                         normalize=self.normalize, targets=('mse',))

    def __repr__(self):
        return (f"LrdoConfig(enabled={self.enabled}, iters={self.iters}, lr={self.lr}, "
                f"lmbda={self.lmbda}, bg_weight={self.bg_weight}, w_mse={self.w_mse}, "
                f"w_lpips={self.w_lpips}, target={self.target!r}, "
                f"optimize_mv={self.optimize_mv})")

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--lrdo', action='store_true',
                            help="Enable latent rate-distortion optimisation at encode time.")
        parser.add_argument('--lrdo_iters', type=int, default=30,
                            help="N in Algorithm 1. Encode time is roughly linear in this.")
        parser.add_argument('--lrdo_lr', type=float, default=5e-3,
                            help="Adam step size on the latent.")
        parser.add_argument('--lrdo_lambdas', type=float, nargs='+',
                            default=[40.0, 85.0, 170.0, 380.0, 640.0],
                            help="Per-rate-point lambda, indexed by rate_idx. Matches "
                                 "--lambdas in train/main_ddp.py.")
        parser.add_argument('--lrdo_bg_weight', type=float, default=1.0,
                            help="Background weight. 1.0 = plain LRDO (no ROI); "
                                 "<1.0 = ROI-LRDO.")
        parser.add_argument('--lrdo_threshold', type=float, default=0.5,
                            help="Threshold applied to the stored soft ROI map.")
        parser.add_argument('--lrdo_soft', action='store_true',
                            help="Use the stored confidence map directly instead of thresholding.")
        parser.add_argument('--lrdo_no_normalize', action='store_true',
                            help="Do not rescale weight maps to unit mean. This is "
                                 "PO-RTIntra's literal formulation, but it couples "
                                 "bg_weight to the effective lambda.")
        parser.add_argument('--lrdo_w_mse', type=float, default=0.5,
                            help="w1' in Eq. 5.")
        parser.add_argument('--lrdo_w_lpips', type=float, default=0.5,
                            help="w2' in Eq. 5. Set 0 to drop the LPIPS term.")
        parser.add_argument('--lrdo_no_lpips_match', action='store_true',
                            help="Do not rescale the LPIPS term to the MSE magnitude.")
        parser.add_argument('--lrdo_target', type=str, default='semantic',
                            choices=list(LRDO_TARGETS),
                            help="Which reconstruction the distortion is measured on.")
        parser.add_argument('--lrdo_optimize_mv', action='store_true',
                            help="Also optimise the motion latent.")
        parser.add_argument('--lrdo_roi_dir', type=str, default=None,
                            help="Directory of per-frame ROI PNGs mirroring the frame "
                                 "names (im00001.png ...). Omit for plain LRDO.")
        return parser

    @classmethod
    def from_args(cls, args, rate_idx=0):
        lambdas = getattr(args, 'lrdo_lambdas', [170.0])
        lmbda = lambdas[min(rate_idx, len(lambdas) - 1)]
        return cls(enabled=getattr(args, 'lrdo', False),
                   iters=getattr(args, 'lrdo_iters', 30),
                   lr=getattr(args, 'lrdo_lr', 5e-3),
                   lmbda=lmbda,
                   bg_weight=getattr(args, 'lrdo_bg_weight', 1.0),
                   threshold=getattr(args, 'lrdo_threshold', 0.5),
                   soft=getattr(args, 'lrdo_soft', False),
                   normalize=not getattr(args, 'lrdo_no_normalize', False),
                   w_mse=getattr(args, 'lrdo_w_mse', 0.5),
                   w_lpips=getattr(args, 'lrdo_w_lpips', 0.5),
                   lpips_match=not getattr(args, 'lrdo_no_lpips_match', False),
                   target=getattr(args, 'lrdo_target', 'semantic'),
                   optimize_mv=getattr(args, 'lrdo_optimize_mv', False))


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #

_SPATIAL_LPIPS = {}


def spatial_lpips(device):
    """A spatially-resolved AlexNet LPIPS, built once per device.

    The model already carries ``self.alexnet_model``, but that one reduces to a
    scalar per sample.  Eq. 5 weights LPIPS *per pixel*, which needs the spatial
    variant.
    """
    key = str(device)
    if key not in _SPATIAL_LPIPS:
        import lpips
        net = lpips.LPIPS(net='alex', spatial=True).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _SPATIAL_LPIPS[key] = net
    return _SPATIAL_LPIPS[key]


def _to_lpips_range(x):
    return torch.clamp(x * 2.0 - 1.0, min=-1.0, max=1.0)


def distortion_terms(x, x_hat, weight, cfg):
    """ROI-weighted MSE and LPIPS, i.e. ``L_mse^roi`` and ``L_lpips^roi``.

    ``weight`` is a (B, 1, H, W) unit-mean map or None.  With None -- or with an
    all-ones map -- both terms reduce to the plain spatial mean, which is
    invariant 1.
    """
    mse_map = (x - x_hat) ** 2
    mse = weighted_mean(mse_map, weight)

    if cfg.w_lpips == 0.0:
        return mse, torch.zeros((), device=x.device, dtype=mse.dtype)

    net = spatial_lpips(x.device)
    lpips_map = net(_to_lpips_range(x), _to_lpips_range(x_hat))
    lpips_val = weighted_mean(lpips_map, weight)
    return mse, lpips_val


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #

def optimize_frame(model, x, dpb, cfg, roi=None,
                   mv_y_q_scale=None, y_q_scale=None, verbose=False):
    """Run ROI-LRDO for one P-frame and return the final hard-quantised result.

    Mirrors Algorithm 1 of PO-RTIntra:

        m, y0   <- S(x), E(x)                       (lines 3)
        repeat N times:
            y~  <- SGA(y)                           (line 5)
            x^  <- D(y~)                            (line 6)
            R   <- R(y~)                            (line 7)
            d   <- alpha_r * m * d + alpha_n * (1-m) * d   (lines 8-11)
            L   <- R + lambda (w1 d_mse + w2 d_lpips)      (line 12)
            y   <- y - eta * grad_y L               (line 13)
        x^, bits <- D(round(yN)), AE(round(yN))     (line 15)

    Args:
        model:   a DMC in eval mode.  Weights are never touched.
        x:       (1, 3, H, W) padded input frame in [0, 1].
        dpb:     decoded picture buffer from the previous frame.
        cfg:     LrdoConfig.
        roi:     (1, 1, H, W) importance map in [0, 1], or None for plain LRDO.

    Returns:
        ``(result, stats)`` where ``result`` is exactly what
        ``forward_one_frame`` returns for the optimised latent under *hard*
        rounding -- so bits and reconstructions are the real ones, not the
        relaxed ones -- and ``stats`` records the optimisation trajectory.
    """
    weight = build_weight_map(roi, cfg.roi_config()) if roi is not None else None

    # Line 3: y0 <- E(x).  One clean encoder pass, no gradient.
    with torch.no_grad():
        init = model.forward_one_frame(x, dpb, mv_y_q_scale=mv_y_q_scale,
                                       y_q_scale=y_q_scale, return_latents=True)

    y = init['y_latent'].detach().clone().requires_grad_(True)
    variables = [y]
    if cfg.optimize_mv:
        mv_y = init['mv_y_latent'].detach().clone().requires_grad_(True)
        variables.append(mv_y)
    else:
        mv_y = init['mv_y_latent'].detach()

    optimizer = torch.optim.Adam(variables, lr=cfg.lr)
    lpips_scale = None
    stats = {'loss': [], 'bpp': [], 'mse': [], 'lpips': []}

    # Lines 4-14.
    with torch.enable_grad():
        for step in range(cfg.iters):
            tau = tau_at(step, cfg.iters, cfg.tau_init, cfg.tau_min)
            with sga_quantization(model, tau):
                out = model.forward_one_frame(x, dpb, mv_y_q_scale=mv_y_q_scale,
                                              y_q_scale=y_q_scale,
                                              y_in=y, mv_y_in=mv_y)

            x_hat = out['dpb']['ref_frame_semantic'] if cfg.target == 'semantic' \
                else out['dpb']['ref_frame']
            rate = out['bpp'].mean()
            mse, lpips_val = distortion_terms(x, x_hat, weight, cfg)

            # PO-RTIntra normalises L_lpips to the magnitude of L_mse.  Fixing the
            # factor at step 0 keeps the objective stationary across iterations.
            if cfg.w_lpips != 0.0 and cfg.lpips_match and lpips_scale is None:
                lpips_scale = (mse.detach() / lpips_val.detach().clamp_min(1e-12)).item()
            scale = lpips_scale if lpips_scale is not None else 1.0

            loss = rate + cfg.lmbda * (cfg.w_mse * mse + cfg.w_lpips * scale * lpips_val)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            stats['loss'].append(loss.item())
            stats['bpp'].append(rate.item())
            stats['mse'].append(mse.item())
            stats['lpips'].append(float(lpips_val))
            if verbose:
                print(f"    [lrdo] it {step:3d}  tau {tau:.3f}  loss {loss.item():.4f}  "
                      f"bpp {rate.item():.4f}  mse {mse.item():.6f}")

    # Line 15: the real encode, with hard rounding restored.
    with torch.no_grad():
        result = model.forward_one_frame(x, dpb, mv_y_q_scale=mv_y_q_scale,
                                         y_q_scale=y_q_scale,
                                         y_in=y.detach(),
                                         mv_y_in=mv_y.detach())
    stats['bpp_before'] = float(init['bpp'].mean())
    stats['bpp_after'] = float(result['bpp'].mean())
    return result, stats
