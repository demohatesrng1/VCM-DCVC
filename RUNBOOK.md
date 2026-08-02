# RUNBOOK — reproducing SEC-VCM and adding ROI-weighted semantic alignment

Everything here runs from the repository root. Paths live in `scripts/env.sh`; edit
that file once and every other script picks it up.

---

## 0. What was added, and why

| File | Purpose |
|---|---|
| `secvcm/roi.py` | ROI weight-map construction and weighted reductions. |
| `scripts/precompute_roi_masks.py` | Offline instance-mask extraction → one 8-bit PNG per frame. |
| `scripts/test_roi_math.py` | CPU unit checks of the weighting math. |
| `scripts/test_roi_forward.py` | CPU end-to-end forward test with stand-in teachers, no checkpoints. |
| `scripts/env.sh` | All dataset/checkpoint paths in one place. |
| `scripts/precompute_roi.sh`, `train_stage1.sh`, `train_stage2.sh`, `pilot_ab.sh` | Runnable pipeline. |

Changes to existing files:

- **`train/main_ddp.py`** — two new schedules (`stage1_lpips`, `semantic_pilot`), the
  `--roi*` / `--skip_semantic` / `--m2f_norm` flags, ROI maps threaded into the train
  and valid loops, per-region diagnostics logged. `set_requires_grad` now tolerates
  attributes that are `None` (it raised `TypeError` otherwise once teachers became
  optional). Imports made robust to being launched as a script.
- **`train/video_train.py`** — detectron2/Mask2Former imports moved out of module scope
  so Stage 1 can run without them; `M2F_WEIGHTS` / `DINOV2_WEIGHTS` env overrides; ROI
  maps threaded through the cascaded rollout.
- **`secvcm/models/video_model.py`** — `forward_one_frame(..., roi=None)`; weighted
  reductions on the BiEC and teacher terms; `skip_semantic` mode; `m2f_norm` switch;
  fg/bg diagnostics.
- **`scripts/test_video.py`** — teachers built lazily; `--inference_mode` (default on)
  runs the codec with no teacher checkpoints at all.

**Nothing changes the codec architecture.** With `--roi_bg_weight 1.0` the loss is
bit-identical to the released one — `scripts/test_roi_forward.py` verifies this to
`0.00e+00`. That is what makes "baseline vs ROI" a one-flag comparison.

---

## 1. Before anything else (no GPU needed)

```bash
python scripts/test_roi_math.py       # 21 checks
python scripts/test_roi_forward.py    # 19 checks, builds the real DMC on CPU
```

Both should print `all checks passed`. If they do, the ROI plumbing is sound and every
later failure is an environment, data or checkpoint problem — which is a much easier
thing to debug at 3am.

## 2. Downloads

```bash
python DCVC/DCVC-family/DCVC-HEM/checkpoints/download.py   # from the DCVC repo
```

Put `acmmm2022_image_psnr.pth.tar` and `acmmm2022_video_psnr.pth.tar` in `./checkpoint/`.
Those OneDrive links are from 2022 and are the single most likely thing to be dead —
check them on day one, not day five.

Teachers (Stage 2 only): Mask2Former ytvis swin-tiny → `$M2F_WEIGHTS`, and
`dinov2_vits14_reg4_pretrain.pth` → `pretrain/`. ResNet-18 downloads itself.

> The shipped `video_maskformer2_swin_tiny_bs16_8ep.yaml` hardcodes
> `/opt/data/private/syx/.../ytvis-swin-T.pkl`, a path from the authors' machine.
> Set `M2F_WEIGHTS` instead of editing the yaml; the loader now fails with a clear
> message rather than a confusing detectron2 error.

## 3. The gate — one honest number

```bash
source scripts/env.sh
python scripts/test_video.py \
    --i_frame_model_path $HEM_IMAGE_CKPT \
    --model_path $HEM_VIDEO_CKPT \
    --rate_num 4 --test_config ./dataset_config_example.json \
    --cuda True -w 1 --write_stream 0 --output_path gate.json
```

Two things to be clear about:

- This is a **DCVC-HEM** number, not a SEC-VCM number. The semantic branch is randomly
  initialised until Stage 2, so `semantic_psnrs` in the output will be garbage while
  `psnrs`/`bpp` are meaningful. That is expected.
- **Do not build the C++ entropy extension.** With `--write_stream 0` the rANS coder is
  never constructed, and SEC-VCM's video model has no `compress`/`decompress` methods at
  all (DCVC-HEM has them; this repo dropped them), so `--write_stream 1` cannot work.
  All reported bits are entropy estimates. This removes about a week from the plan.

The real week-1 risk is the **task evaluation loop**, which is not in this repo:
`scripts/extract_mask2former_log.py` only greps AP out of a detectron2 log you have to
produce yourself. Stand that up on *uncompressed* frames first, then swap in decoded
ones. `scripts/Read_and_Plot/bjontegaard_metric.py` gives you `BD_RATE` — it takes
(rate, quality) pairs, so it works for (bpp, mAP) as well as (bpp, PSNR).

> Dataset trap: YouTube-VIS 2019 validation annotations are **not public** (scoring runs
> on a CodaLab server). For a one-month project, pick a task you can score locally.

## 4. Stage 1 — LPIPS fine-tune

Microsoft released DCVC-HEM weights but no DCVC-HEM training code. The loss Stage 1
needs (`codec_rd_lpips` = `λ·(mse + 0.05·lpips_alexnet) + bpp`) was already implemented
in `Change_loss`; only a schedule that reached it was missing. Now:

```bash
bash scripts/train_stage1.sh
```

`--skip_semantic` keeps the semantic branch and all three teachers out of the graph, so
**Stage 1 needs neither detectron2 nor any teacher checkpoint** — HEM weights plus Vimeo
are enough to start. It is also meaningfully faster for the same reason.

Fallback if the clock gets tight: skip Stage 1 and start Stage 2 from the PSNR weights
directly. Absolute numbers will be lower, but your result is the baseline-vs-ROI delta
under identical conditions, and that survives.

## 5. ROI maps

```bash
bash scripts/precompute_roi.sh check     # 32 frames + preview JPEGs — LOOK AT THESE
bash scripts/precompute_roi.sh train     # full pass, sharded over $NGPU GPUs
bash scripts/precompute_roi.sh valid
```

The `check` pass writes `roi_preview/*.jpg` as `original | red-tinted ROI`. Five minutes
here protects every number downstream. Default backend is torchvision Mask R-CNN
(weights download automatically, soft masks, zero setup); `--backend mask2former` matches
the paper's teacher family but needs detectron2 and a COCO instance checkpoint.

Storage: ~450k small PNGs for the full Vimeo train list. The run is resumable (existing
files are skipped) and shardable.

## 6. The pilot — do this before the long runs

```bash
bash scripts/pilot_ab.sh ./checkpoint/stage1_lpips/<ckpt>.model 2000 0.5
```

Three warm-up epochs, capped at 2000 iterations each, two arms differing only in
`--roi_bg_weight`. Then:

```bash
grep '\[roi\]' pilot_baseline.log | tail -20
grep '\[roi\]' pilot_roi.log      | tail -20
```

`entropy_fg` should fall faster in the ROI arm while `entropy_bg` is allowed to rise. If
neither budges, the weighting is not reaching the objective and a full 10-epoch run will
not fix that — better to learn it in an afternoon than after four days.

## 7. Full runs

```bash
bash scripts/train_stage2.sh baseline ./checkpoint/stage1_lpips/<ckpt>.model
bash scripts/train_stage2.sh roi      ./checkpoint/stage1_lpips/<ckpt>.model 0.5
```

`--stage_extend 1` is documented in-repo as ≈5 days on one RTX 4090, so budget
accordingly for two runs on 4×3090. Watch memory at the epoch 2→3 boundary, where
`train_seq` goes to 6 and batch drops to 2/GPU while three teachers run twice per frame.

---

## What the schedule actually does (and what it means for the claim)

The README's training table is wrong; the code in `get_training_strategy` is right:

| Epochs | Trainable | Loss | Rate term? |
|---|---|---|---|
| 0–7 | semantic modules only | `semantic_d_dinov2` | **no** |
| 8–9 | everything | `semantic_rd` | yes |

For 8 of 10 epochs the encoder and entropy model are frozen and there is no bpp term, so
**the bitstream is bit-identical to the Stage 1 model** and ROI weighting cannot move a
single bit. It changes what the semantic decoder reconstructs. Bits can only be
reallocated in epochs 8–9.

So frame the contribution as **ROI-weighted semantic alignment** — higher task accuracy
at unchanged rate, which is still a genuine BD-rate gain — not as ROI bit allocation. If
you want bits to move, lengthen the final `semantic_rd` phase; that is a deliberate
design change, not a bug fix.

Related: the "bi-directional" entropy constraint is only bi-directional in epochs 8–9.
The `*_reverse` heads are absent from `attributes_semantic`, so they stay at random init
through epoch 7, and `semantic_d_dinov2` uses only the forward terms. They enter
abruptly at epoch 8 via `semantic_rd`. That is a free ablation and a landmine.

## Ablations the flags already support

| Question | Flags |
|---|---|
| How much background suppression? | `--roi_bg_weight 0.2 / 0.5 / 1.0` |
| Which terms should be weighted? | `--roi_targets biec` / `biec swin cnn` / `biec swin cnn mse` |
| Hard or soft ROI? | `--roi_soft`, `--roi_threshold 0.3 / 0.5 / 0.7` |
| Does the map need object context? | `--dilate 4` at precompute time |
| Is the released Mask2Former scaling a bug? | `--m2f_norm legacy` vs `fixed` |

On that last one: `format_convert_lpips_mask2former` multiplies the input by 255 **twice**,
putting it ~255× outside the range implied by the config's `PIXEL_MEAN`/`PIXEL_STD`. Since
the BiEC targets come from this path, it matters. It is left as `legacy` by default so the
released behaviour is reproducible — measure both before deciding which is "correct".

## Known upstream issues left alone

- `get_conditional_entropy` calls `exit()` on NaN/Inf. Under DDP that kills one rank and
  hangs the rest. Left as-is (changing it changes behaviour), but if a run goes silent,
  this is the first place to look.
- `mse_cnn_res4/res5` and `mse_swin_res4/res5` are computed and never used.
- MS-SSIM requires >160 px per side, so the 256×256 training crop is near the floor.
