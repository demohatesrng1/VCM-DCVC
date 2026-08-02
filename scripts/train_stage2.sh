#!/bin/bash
# Stage 2: SEC-VCM semantic training.
#
# Runs both arms of the experiment from one script so the two are guaranteed to
# differ only in the ROI flags:
#
#   bash scripts/train_stage2.sh baseline  <stage1_ckpt>
#   bash scripts/train_stage2.sh roi       <stage1_ckpt> [bg_weight]
#
# Schedule (train/main_ddp.py, semantic_v3): 8 epochs training the semantic
# modules only with the base codec frozen and no rate term, then 2 epochs with
# everything unfrozen and bpp active. Bits can only move in those last 2 epochs.
set -e
source scripts/env.sh

arm=${1:-baseline}
pretrain=${2:-}
bg_weight=${3:-0.5}

if [ -z "${pretrain}" ]; then
    echo "usage: bash scripts/train_stage2.sh {baseline|roi} <stage1_checkpoint> [bg_weight]"
    exit 1
fi

# The baseline arm runs with --roi --roi_bg_weight 1.0 rather than with the ROI
# switched off. Two reasons: bg_weight=1.0 makes every weight exactly 1.0, so the
# loss is bit-identical to the unweighted one (verified by scripts/test_roi_forward.py);
# and loading the ROI maps consumes the augmentation RNG the same way in both arms,
# so the two runs see the same random crops. The only difference between the arms is
# then the number in --roi_bg_weight.
case "${arm}" in
    baseline) roi_args="--roi --roi_bg_weight 1.0" ; name="secvcm_baseline" ;;
    roi)      roi_args="--roi --roi_bg_weight ${bg_weight} --roi_targets biec swin cnn"
              name="secvcm_roi_bg${bg_weight}" ;;
    *) echo "unknown arm '${arm}' (expected baseline or roi)"; exit 1 ;;
esac

mkdir -p ./checkpoint/${name}

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
    --nproc_per_node=${NGPU} --use_env train/main_ddp.py \
    --save_path ./checkpoint/${name} \
    --pretrain ${pretrain} \
    --used_data all_vimeo --model hem \
    --train_schedule semantic_v3 \
    --stage_extend 1 \
    ${roi_args} \
    --save_epoch 2 -b 4 \
    -l ./checkpoint/${name}/log.txt | tee ${name}.log
