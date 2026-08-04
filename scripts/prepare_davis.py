"""Turn DAVIS 2017 into something test_video.py can read, and write its config.

    python scripts/prepare_davis.py --davis_root /data/DAVIS --dst_root /data/davis_png

DAVIS ships frames as ``JPEGImages/480p/<seq>/00000.jpg``.  PNGReader accepts
only ``im1.png`` or ``im00001.png`` and asserts that every frame matches the
width/height in the config, so pointing test_video.py at DAVIS directly fails --
usually in a confusing way, halfway through a sequence.  This script converts the
frames once and generates the matching dataset config, with the real per-sequence
resolution and frame count read off disk rather than assumed.

Nothing is resized.  DAVIS 480p is 854x480 for most sequences, which
``get_padding_size`` pads to 896x512 at encode time and crops back afterwards.

The run is resumable: existing PNGs are skipped, so re-running after an
interruption is cheap.  Use ``--limit 3`` to do a few sequences first and check
the whole pipeline end to end before converting all of them.
"""

import argparse
import json
import os
import sys

from PIL import Image

# DAVIS layout.
JPEG_SUBDIR = os.path.join('JPEGImages', '480p')
SPLIT_SUBDIR = os.path.join('ImageSets', '2017')


def read_split(davis_root, split):
    """Sequence names for a DAVIS split, or None if the split file is absent."""
    path = os.path.join(davis_root, SPLIT_SUBDIR, f'{split}.txt')
    if not os.path.isfile(path):
        return None
    with open(path) as fp:
        return [line.strip() for line in fp if line.strip()]


def discover_sequences(davis_root, split):
    jpeg_root = os.path.join(davis_root, JPEG_SUBDIR)
    if not os.path.isdir(jpeg_root):
        raise SystemExit(f"not found: {jpeg_root}\n"
                         f"--davis_root should be the folder containing "
                         f"JPEGImages/, Annotations/ and ImageSets/")

    names = read_split(davis_root, split)
    if names is None:
        print(f"warning: no ImageSets/2017/{split}.txt, using every sequence on disk")
        names = sorted(os.listdir(jpeg_root))

    sequences = []
    for name in names:
        folder = os.path.join(jpeg_root, name)
        if os.path.isdir(folder):
            sequences.append(name)
        else:
            print(f"warning: {name} is in the split file but not on disk, skipping")
    return sequences


def convert_sequence(src_folder, dst_folder, padding, dry_run=False):
    """Convert one sequence. Returns (width, height, frame_count).

    Raises if frames disagree on size, because PNGReader asserts on that and the
    failure is much easier to read here than mid-encode.
    """
    frames = sorted(f for f in os.listdir(src_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if not frames:
        raise ValueError(f"no frames in {src_folder}")

    if not dry_run:
        os.makedirs(dst_folder, exist_ok=True)

    size = None
    for index, name in enumerate(frames):
        # PNGReader starts at index 1, so frame 0 becomes im00001.png.
        dst = os.path.join(dst_folder, f"im{str(index + 1).zfill(padding)}.png")
        if os.path.exists(dst) and not dry_run:
            with Image.open(dst) as img:
                cur = img.size
        else:
            with Image.open(os.path.join(src_folder, name)) as img:
                img = img.convert('RGB')
                cur = img.size
                if not dry_run:
                    img.save(dst)
        if size is None:
            size = cur
        elif cur != size:
            raise ValueError(f"{src_folder}: frame {name} is {cur}, expected {size}. "
                             f"PNGReader requires one size per sequence.")

    width, height = size
    return width, height, len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis_root', required=True,
                        help="Folder containing JPEGImages/, Annotations/, ImageSets/")
    parser.add_argument('--dst_root', required=True,
                        help="Where to write the converted frame tree.")
    parser.add_argument('--config_out', default='./dataset_config_davis.json',
                        help="Dataset config for test_video.py --test_config.")
    parser.add_argument('--split', default='val',
                        help="DAVIS split file to use (val, train, ...).")
    parser.add_argument('--gop', type=int, default=32,
                        help="Intra period. 32 matches the SEC-VCM evaluation protocol.")
    parser.add_argument('--padding', type=int, default=5, choices=[1, 5],
                        help="Frame index width: 5 gives im00001.png, 1 gives im1.png.")
    parser.add_argument('--limit', type=int, default=0,
                        help="Convert only the first N sequences (0 = all).")
    parser.add_argument('--ds_name', default='DAVIS2017',
                        help="Dataset key written into the config.")
    parser.add_argument('--dry_run', action='store_true',
                        help="Scan and write the config, but do not write PNGs.")
    args = parser.parse_args()

    sequences = discover_sequences(args.davis_root, args.split)
    if args.limit > 0:
        sequences = sequences[:args.limit]
    if not sequences:
        raise SystemExit("no sequences found")

    print(f"{len(sequences)} sequence(s) from split '{args.split}'"
          + (" (dry run)" if args.dry_run else ""))

    entries = {}
    total_frames = 0
    failed = []
    for i, name in enumerate(sequences, 1):
        src = os.path.join(args.davis_root, JPEG_SUBDIR, name)
        dst = os.path.join(args.dst_root, name)
        try:
            width, height, count = convert_sequence(src, dst, args.padding, args.dry_run)
        except Exception as exc:                                 # noqa: BLE001
            print(f"  [{i}/{len(sequences)}] {name}: FAILED  {exc}")
            failed.append(name)
            continue
        entries[name] = {"width": width, "height": height,
                         "frames": count, "gop": args.gop}
        total_frames += count
        print(f"  [{i}/{len(sequences)}] {name}: {width}x{height}, {count} frames")

    if not entries:
        raise SystemExit("nothing converted successfully")

    config = {
        "root_path": os.path.join(os.path.abspath(args.dst_root), ""),
        "test_classes": {
            args.ds_name: {
                "test": 1,
                "src_type": "png",
                "base_path": "",
                "sequences": entries,
            }
        }
    }
    with open(args.config_out, 'w') as fp:
        json.dump(config, fp, indent=2)

    print(f"\nwrote {args.config_out}: {len(entries)} sequences, {total_frames} frames, "
          f"gop={args.gop}")
    if failed:
        print(f"{len(failed)} sequence(s) failed: {failed}")
    print("\nnext:")
    print(f"  python scripts/preflight.py --full --test_config {args.config_out}")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
