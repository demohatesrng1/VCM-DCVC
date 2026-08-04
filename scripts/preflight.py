"""Preflight check for the CUDA machine. Run this before anything else.

    python scripts/preflight.py                       # environment + code only
    python scripts/preflight.py --full                # also checkpoints, data, GPU

Every check is independent and failures do not abort the run, so one pass tells
you everything that is wrong rather than only the first thing. Nothing here
trains, downloads or writes: it is safe to run repeatedly.

The point is to fail in 30 seconds instead of six hours into a job.
"""

import argparse
import importlib
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

failures = []
warnings = []


def check(name, condition, detail=""):
    print(f"[{'ok  ' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)
    return condition


def warn(name, condition, detail=""):
    if condition:
        print(f"[ok  ] {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"[warn] {name}" + (f"  {detail}" if detail else ""))
        warnings.append(name)
    return condition


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def check_environment():
    section("environment")
    print(f"       python {sys.version.split()[0]}")
    try:
        import torch
    except Exception as exc:                                     # noqa: BLE001
        check("torch imports", False, str(exc))
        return None
    check("torch imports", True, f"version {torch.__version__}")

    cuda = torch.cuda.is_available()
    check("CUDA is available", cuda,
          "torch was built without CUDA, or no visible GPU" if not cuda else "")
    if cuda:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024 ** 3)
            print(f"       gpu {i}: {props.name}  {total:.0f} GB  sm_{props.major}{props.minor}")
        # The GOP=6 phase with three teachers is the memory high-water mark.
        smallest = min(torch.cuda.get_device_properties(i).total_memory
                       for i in range(torch.cuda.device_count())) / (1024 ** 3)
        warn("smallest GPU has >= 20 GB", smallest >= 20,
             f"{smallest:.0f} GB -- Stage 2 at GOP=6 with three teachers needs ~24 GB")
    return torch


def check_packages():
    section("packages")
    for mod, why in (('torchvision', 'ResNet-18 teacher, Mask R-CNN ROI backend'),
                     ('lpips', 'LPIPS terms, built into DMC.__init__'),
                     ('pytorch_msssim', 'MS-SSIM in forward_one_frame'),
                     ('numpy', 'everywhere'),
                     ('PIL', 'frame and ROI map I/O')):
        try:
            importlib.import_module(mod)
            check(f"{mod} imports", True, why)
        except Exception as exc:                                 # noqa: BLE001
            check(f"{mod} imports", False, f"{why}: {exc}")

    # detectron2 is only needed to TRAIN stage 2. Inference and LRDO do not use it.
    try:
        importlib.import_module('detectron2')
        print("[ok  ] detectron2 imports  (only needed for Stage 2 training)")
    except Exception:                                            # noqa: BLE001
        print("[info] detectron2 not installed -- fine for the codec, LRDO and "
              "evaluation; only Stage 2 training needs it")


def check_code():
    section("code")
    try:
        from secvcm.models.video_model import DMC                # noqa: F401
        from secvcm.models.image_model import IntraNoAR          # noqa: F401
        check("secvcm imports", True)
    except Exception as exc:                                     # noqa: BLE001
        check("secvcm imports", False,
              f"{exc}  -- run from the repository root, or set PYTHONPATH=$PWD")
        return False
    try:
        from secvcm.roi import RoiConfig                         # noqa: F401
        from secvcm.lrdo import LrdoConfig, optimize_frame       # noqa: F401
        check("roi and lrdo modules import", True)
    except Exception as exc:                                     # noqa: BLE001
        check("roi and lrdo modules import", False, str(exc))
        return False
    return True


def check_checkpoints(args):
    section("checkpoints")
    from secvcm.models.video_model import DMC
    from secvcm.models.image_model import IntraNoAR

    for label, path, cls in (("i-frame", args.i_frame_model_path, IntraNoAR),
                             ("video", args.model_path, DMC)):
        if not path:
            warn(f"{label} checkpoint given", False, "not passed, skipping")
            continue
        if not check(f"{label} checkpoint exists", os.path.isfile(path), path):
            continue
        size = os.path.getsize(path) / (1024 ** 2)
        if not warn(f"{label} checkpoint is a plausible size", size > 1.0,
                    f"{size:.1f} MB -- a tiny file is usually a saved HTML error page"):
            continue
        try:
            y_q, mv_q = cls.get_q_scales_from_ckpt(path)
            check(f"{label} checkpoint loads and has q_scales", True,
                  f"{len(y_q)} rate points")
        except Exception as exc:                                 # noqa: BLE001
            try:
                q = cls.get_q_scales_from_ckpt(path)
                check(f"{label} checkpoint loads and has q_scales", True,
                      f"{len(q)} rate points")
            except Exception as exc2:                            # noqa: BLE001
                check(f"{label} checkpoint loads and has q_scales", False,
                      f"{exc} / {exc2}")


def check_dataset(args):
    section("dataset")
    if not args.test_config:
        warn("test config given", False, "not passed, skipping")
        return
    if not check("test config exists", os.path.isfile(args.test_config), args.test_config):
        return
    try:
        with open(args.test_config) as fp:
            config = json.load(fp)
    except Exception as exc:                                     # noqa: BLE001
        check("test config parses as JSON", False, str(exc))
        return
    check("test config parses as JSON", True)

    root = config.get('root_path', '')
    check("root_path exists", os.path.isdir(root), root or "(empty)")

    for ds_name, ds in config.get('test_classes', {}).items():
        if ds.get('test', 0) == 0:
            print(f"[info] {ds_name}: test=0, skipped by test_video.py")
            continue
        base = os.path.join(root, ds.get('base_path', ''))
        for seq_name, seq in ds.get('sequences', {}).items():
            folder = os.path.join(base, seq_name)
            if not check(f"{ds_name}/{seq_name} folder exists", os.path.isdir(folder), folder):
                continue
            # PNGReader accepts im1.png or im00001.png and nothing else.
            names = set(os.listdir(folder))
            padded = 'im00001.png' in names
            plain = 'im1.png' in names
            check(f"{ds_name}/{seq_name} uses a supported frame naming",
                  padded or plain,
                  "expected im1.png or im00001.png; PNGReader rejects anything else")
            n_png = sum(1 for f in names if f.endswith('.png'))
            want = seq.get('frames', 0)
            warn(f"{ds_name}/{seq_name} has >= 'frames' PNGs", n_png >= want,
                 f"{n_png} on disk, config says {want}")


def check_gpu_forward(args, torch):
    section("gpu smoke test")
    if torch is None or not torch.cuda.is_available():
        warn("gpu forward", False, "no CUDA, skipped")
        return
    from secvcm.models.video_model import DMC
    from secvcm.lrdo import LrdoConfig, optimize_frame
    from secvcm.utils.stream_helper import get_state_dict

    device = 'cuda:0'
    try:
        net = DMC(inference_mode=True)
        if args.model_path and os.path.isfile(args.model_path):
            missing = net.load_state_dict(get_state_dict(args.model_path), strict=False)
            unexpected = len(getattr(missing, 'unexpected_keys', []))
            warn("video checkpoint has no unexpected keys", unexpected == 0,
                 f"{unexpected} unexpected -- expected if the ckpt was saved with teachers")
        net = net.to(device).eval()
        check("DMC builds and moves to GPU", True)
    except Exception as exc:                                     # noqa: BLE001
        check("DMC builds and moves to GPU", False, str(exc))
        return

    # 192 px is the floor: forward_one_frame computes MS-SSIM, which needs >160.
    x = torch.rand(1, 3, 192, 192, device=device)
    dpb = {"ref_frame": x.clone(), "ref_frame_semantic": x.clone(),
           "ref_feature": None, "ref_feature_semantic": None,
           "ref_y": None, "ref_mv_y": None}
    q = torch.tensor([1.0], device=device)
    try:
        with torch.no_grad():
            out = net.forward_one_frame(x, dpb, mv_y_q_scale=q, y_q_scale=q)
        check("one P-frame forward pass runs on GPU", True,
              f"bpp {out['bpp'].mean().item():.4f}")
    except Exception as exc:                                     # noqa: BLE001
        check("one P-frame forward pass runs on GPU", False, str(exc))
        return

    try:
        cfg = LrdoConfig(enabled=True, iters=3, w_lpips=0.0)
        with torch.no_grad():
            _, stats = optimize_frame(net, x, dpb, cfg, mv_y_q_scale=q, y_q_scale=q)
        moved = abs(stats['bpp_after'] - stats['bpp_before'])
        check("LRDO runs on GPU and changes the bitstream", moved > 0,
              f"bpp {stats['bpp_before']:.5f} -> {stats['bpp_after']:.5f}")
    except Exception as exc:                                     # noqa: BLE001
        check("LRDO runs on GPU", False, str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                        help="Also check checkpoints, dataset and run a GPU smoke test.")
    parser.add_argument('--i_frame_model_path', type=str,
                        default=os.environ.get('HEM_IMAGE_CKPT'))
    parser.add_argument('--model_path', type=str,
                        default=os.environ.get('HEM_VIDEO_CKPT'))
    parser.add_argument('--test_config', type=str, default='./dataset_config_example.json')
    args = parser.parse_args()

    torch = check_environment()
    check_packages()
    ok = check_code()

    if args.full and ok:
        check_checkpoints(args)
        check_dataset(args)
        check_gpu_forward(args, torch)
    elif not args.full:
        print("\n[info] run with --full to also check checkpoints, data and the GPU")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        if warnings:
            print(f"{len(warnings)} warning(s): {warnings}")
        return 1
    if warnings:
        print(f"all checks passed, {len(warnings)} warning(s): {warnings}")
        return 0
    print("all checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
