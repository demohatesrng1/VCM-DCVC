#!/bin/bash
# One unattended run that answers "is LRDO worth carrying, and at what budget?"
#
#   bash scripts/lrdo_sweep.sh
#
# Runs the no-LRDO baseline once, then several (iters, lr) configurations, then
# prints a single summary table: BD-rate per sequence plus two diagnostics that
# say WHY a configuration did or did not work.
#
# The diagnostics matter more than the BD-rate here. Adam moves each latent by
# roughly `lr` per step, so it travels about iters*lr in total. But the codec
# divides y by quant_step (>= 0.5, typically 1-4) and rounds, so changing one
# coded symbol needs a move of 0.5*quant_step. With iters*lr = 0.05 -- the old
# default of 10 steps at 5e-3 -- essentially nothing can cross a boundary, and a
# flat BD-rate says nothing about the method. `symbols_changed` measures this
# directly.
#
# Expect roughly 1.5-2 hours. Run it in tmux and walk away.

set -e

: "${HEM_IMAGE_CKPT:?set HEM_IMAGE_CKPT first}"
: "${HEM_VIDEO_CKPT:?set HEM_VIDEO_CKPT first}"
: "${REPO:?set REPO first}"

CONFIG=${CONFIG:-$REPO/dataset_config_davis.json}
FRAMES=${FRAMES:-6}
RATES=${RATES:-4}
OUT=${OUT:-$REPO/sweep}

# (iters lr) pairs. iters*lr is the latent's travel budget in y units.
CONFIGS=(
    "10 0.005"    # the original default: travel 0.05, expected to change nothing
    "50 0.02"     # travel 1.0
    "100 0.02"    # travel 2.0
    "100 0.05"    # travel 5.0
)

mkdir -p "$OUT"
echo "output -> $OUT"
echo "config -> $CONFIG   frames=$FRAMES  rates=$RATES"
echo

COMMON="--i_frame_model_path $HEM_IMAGE_CKPT --model_path $HEM_VIDEO_CKPT \
        --rate_num $RATES --test_config $CONFIG --cuda True -w 1 \
        --write_stream 0 --force_frame_num $FRAMES"

if [ ! -f "$OUT/base.json" ]; then
    echo "=== baseline (no LRDO) ==="
    python scripts/test_video.py $COMMON --output_path "$OUT/base.json" \
        > "$OUT/base.log" 2>&1
    echo "    done"
else
    echo "=== baseline already present, skipping ==="
fi

for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    iters=$1; lr=$2
    tag="i${iters}_lr${lr}"
    if [ -f "$OUT/$tag.json" ]; then
        echo "=== $tag already present, skipping ==="
        continue
    fi
    echo "=== iters=$iters lr=$lr  (travel $(python -c "print(f'{$iters*$lr:.2f}')")) ==="
    start=$(date +%s)
    python scripts/test_video.py $COMMON \
        --lrdo --lrdo_iters "$iters" --lrdo_lr "$lr" \
        --lrdo_bg_weight 1.0 --lrdo_target recon \
        --lrdo_w_mse 1.0 --lrdo_w_lpips 0.0 \
        --output_path "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
    echo "    done in $(( $(date +%s) - start ))s"
done

echo
python scripts/lrdo_report.py --sweep_dir "$OUT"
