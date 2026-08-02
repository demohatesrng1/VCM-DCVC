"""Precompute per-frame ROI (region-of-importance) maps for SEC-VCM training.

The maps are written as 8-bit grayscale PNGs in a tree that mirrors the frame
tree, so ``train/dataset.py`` can find them by path substitution alone.  Doing
this offline keeps the segmentation model out of the training loop entirely:
training reads one small PNG per frame and nothing else changes.

Two backends:

  maskrcnn      torchvision Mask R-CNN R50-FPN.  Weights download automatically,
                so this needs no setup and is the fastest way to get unblocked.
                Produces a *soft* map (mask probability), which makes the
                --roi_threshold ablation meaningful.
  mask2former   Mask2Former COCO instance segmentation, i.e. the same family as
                the BiEC teacher.  Produces a binary map.  Needs detectron2 and
                a checkpoint.

Examples
--------
Vimeo-90k septuplet, sharded over 2 GPUs:

    python scripts/precompute_roi_masks.py \\
        --src_root /data/vimeo_septuplet/sequences \\
        --dst_root /data/vimeo_septuplet/roi \\
        --list /data/vimeo_septuplet/sep_trainlist.txt \\
        --backend maskrcnn --device cuda:0 --shard 0 --num_shards 2

    python scripts/precompute_roi_masks.py ... --device cuda:1 --shard 1 --num_shards 2

Check a handful of maps before launching the full run:

    python scripts/precompute_roi_masks.py ... --limit 32 --preview_dir /tmp/roi_preview
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def parse_args():
    p = argparse.ArgumentParser(description="Precompute ROI maps for SEC-VCM training.")
    p.add_argument('--src_root', required=True, help="Root of the frame tree.")
    p.add_argument('--dst_root', required=True, help="Root of the ROI tree to write.")
    p.add_argument('--list', default='',
                   help="Optional list file (e.g. Vimeo sep_trainlist.txt); one relative "
                        "sequence directory per line. If omitted, src_root is walked.")
    p.add_argument('--backend', default='maskrcnn', choices=['maskrcnn', 'mask2former'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch', type=int, default=8, help="Frames per forward (maskrcnn only).")
    p.add_argument('--score_thresh', type=float, default=0.5,
                   help="Instances below this detection score are ignored.")
    p.add_argument('--dilate', type=int, default=0,
                   help="Grow the ROI by this many pixels (max-pool), to keep object context.")
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0, help="Stop after this many frames (0 = all).")
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--preview_dir', default='', help="Write overlay previews here for inspection.")
    p.add_argument('--preview_num', type=int, default=16)
    # mask2former backend
    p.add_argument('--m2f_config', default='third_party/mask2former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml')
    p.add_argument('--m2f_weights', default='')
    return p.parse_args()


def iter_frames(args):
    """Yield absolute frame paths in a deterministic order."""
    if args.list:
        with open(args.list) as f:
            rels = [line.strip() for line in f if line.strip()]
        for rel in rels:
            seq_dir = os.path.join(args.src_root, rel)
            if not os.path.isdir(seq_dir):
                continue
            for name in sorted(os.listdir(seq_dir)):
                if name.lower().endswith(IMAGE_EXTS):
                    yield os.path.join(seq_dir, name)
    else:
        for root, _, files in os.walk(args.src_root):
            for name in sorted(files):
                if name.lower().endswith(IMAGE_EXTS):
                    yield os.path.join(root, name)


def dst_path_for(src_path, args):
    rel = os.path.relpath(src_path, args.src_root)
    rel = os.path.splitext(rel)[0] + '.png'
    return os.path.join(args.dst_root, rel)


class MaskRCNNRoi:
    """torchvision Mask R-CNN. Soft ROI map: max instance mask probability."""

    supports_batching = True

    def __init__(self, device, score_thresh):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        except ImportError:  # torchvision < 0.13
            model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model = model.eval().to(device)
        self.device = device
        self.score_thresh = score_thresh

    @torch.no_grad()
    def __call__(self, images):
        outputs = self.model(images)
        maps = []
        for out, img in zip(outputs, images):
            h, w = img.shape[-2:]
            keep = out['scores'] >= self.score_thresh
            if keep.any():
                masks = out['masks'][keep]              # (N, 1, H, W) probabilities
                roi = masks.amax(dim=0)                 # (1, H, W)
            else:
                roi = img.new_zeros(1, h, w)
            maps.append(roi)
        return maps


class Mask2FormerRoi:
    """Mask2Former COCO instance segmentation. Binary ROI map."""

    supports_batching = False

    def __init__(self, device, score_thresh, config, weights):
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config

        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'mask2former'))
        from mask2former import add_maskformer2_config

        if not weights:
            raise SystemExit("--m2f_weights is required for the mask2former backend "
                             "(COCO instance segmentation checkpoint).")
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = str(device)
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)
        self.device = device
        self.score_thresh = score_thresh

    @torch.no_grad()
    def __call__(self, images):
        maps = []
        for img in images:
            bgr = (img.flip(0) * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            out = self.predictor(bgr)
            inst = out["instances"]
            h, w = img.shape[-2:]
            roi = img.new_zeros(1, h, w)
            if len(inst) > 0:
                keep = inst.scores >= self.score_thresh
                if keep.any():
                    masks = inst.pred_masks[keep].to(img.dtype)   # (N, H, W)
                    roi = masks.amax(dim=0, keepdim=True).to(img.device)
            maps.append(roi)
        return maps


def build_backend(args, device):
    if args.backend == 'maskrcnn':
        return MaskRCNNRoi(device, args.score_thresh)
    return Mask2FormerRoi(device, args.score_thresh, args.m2f_config, args.m2f_weights)


def dilate(roi, pixels):
    if pixels <= 0:
        return roi
    k = 2 * pixels + 1
    return F.max_pool2d(roi.unsqueeze(0), kernel_size=k, stride=1, padding=pixels).squeeze(0)


def save_preview(src_path, image, roi, out_dir, index):
    os.makedirs(out_dir, exist_ok=True)
    rgb = (image * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    heat = roi.squeeze(0).cpu().numpy()
    tint = np.zeros_like(rgb)
    tint[..., 0] = 255
    overlay = (rgb * (1 - 0.5 * heat[..., None]) + tint * (0.5 * heat[..., None])).astype(np.uint8)
    strip = np.concatenate([rgb, overlay], axis=1)
    name = f"{index:04d}_" + os.path.basename(os.path.dirname(src_path)) + "_" + os.path.basename(src_path)
    Image.fromarray(strip).save(os.path.join(out_dir, os.path.splitext(name)[0] + '.jpg'), quality=90)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    backend = build_backend(args, device)
    batch_size = args.batch if backend.supports_batching else 1

    todo = []
    total_seen = 0
    for i, src in enumerate(iter_frames(args)):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard:
            continue
        total_seen += 1
        dst = dst_path_for(src, args)
        if not args.overwrite and os.path.exists(dst):
            continue
        todo.append((src, dst))
        if args.limit and len(todo) >= args.limit:
            break

    print(f"[shard {args.shard}/{args.num_shards}] {len(todo)} frames to process "
          f"(of {total_seen} in this shard)")
    if not todo:
        return

    written = 0
    previewed = 0
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        images = []
        for src, _ in chunk:
            arr = np.asarray(Image.open(src).convert('RGB'), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr).permute(2, 0, 1).to(device))

        rois = backend(images)

        for (src, dst), image, roi in zip(chunk, images, rois):
            roi = dilate(roi.clamp(0, 1), args.dilate)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            arr = (roi.squeeze(0) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(arr, mode='L').save(dst, optimize=True)
            written += 1
            if args.preview_dir and previewed < args.preview_num:
                save_preview(src, image, roi, args.preview_dir, previewed)
                previewed += 1

        if (start // max(batch_size, 1)) % 50 == 0:
            done = min(start + batch_size, len(todo))
            print(f"  {done}/{len(todo)}", flush=True)

    print(f"[shard {args.shard}/{args.num_shards}] wrote {written} ROI maps to {args.dst_root}")
    if args.preview_dir:
        print(f"previews: {args.preview_dir}")


if __name__ == '__main__':
    main()
