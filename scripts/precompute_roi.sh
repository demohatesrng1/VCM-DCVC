#!/bin/bash
# Generate the ROI maps for Vimeo-90k, sharded across the available GPUs.
#
# Do the --limit run first and look at the previews. A bad ROI map makes every
# downstream number meaningless, and it is a five-minute check.
#
# Usage:
#   bash scripts/precompute_roi.sh check     # 32 frames + preview images
#   bash scripts/precompute_roi.sh train     # full training list
#   bash scripts/precompute_roi.sh valid     # validation list
set -e
source scripts/env.sh

mode=${1:-check}
backend=${2:-maskrcnn}
ngpu=${NGPU:-1}

common="--src_root ${VIMEO_ROOT} --dst_root ${ROI_ROOT} --backend ${backend} --score_thresh 0.5"

case "${mode}" in
    check)
        python scripts/precompute_roi_masks.py ${common} \
            --list ${VIMEO_TRAIN_LIST} \
            --device cuda:0 --limit 32 --preview_dir ./roi_preview --overwrite
        echo
        echo "Look at ./roi_preview/*.jpg (original | red-tinted ROI) before running the full pass."
        ;;
    train|valid)
        if [ "${mode}" = "train" ]; then list=${VIMEO_TRAIN_LIST}; else list=${VIMEO_VALID_LIST}; fi
        pids=()
        for ((i=0; i<ngpu; i++)); do
            python scripts/precompute_roi_masks.py ${common} \
                --list ${list} \
                --device cuda:${i} --shard ${i} --num_shards ${ngpu} &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait ${pid}; done
        echo "ROI maps written to ${ROI_ROOT}"
        ;;
    *)
        echo "unknown mode '${mode}' (expected check, train or valid)"; exit 1 ;;
esac
