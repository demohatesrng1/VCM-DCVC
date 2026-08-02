#!/bin/bash
# Cheap A/B pilot: does ROI weighting move the semantic branch at all?
#
# Runs the warm-up phase only (train_schedule semantic_pilot), twice, changing
# nothing but the ROI flags. During warm-up the base codec is frozen and there
# is no rate term, so the bitstream is identical in both arms by construction --
# any difference you see is the semantic branch reacting to the weighting, which
# is exactly the question. Answer this before spending days on two full runs.
#
# What to look at afterwards, in the [roi] line of the logs and in tensorboard
# under roi/*:
#   entropy_fg should fall faster in the roi arm than in the baseline arm,
#   entropy_bg is allowed to rise. If neither budges, the weighting is not
#   reaching the objective and more training will not fix that.
#
# Usage: bash scripts/pilot_ab.sh <stage1_checkpoint> [iters_per_epoch] [bg_weight]
set -e
source scripts/env.sh

pretrain=${1:-}
iters=${2:-2000}
bg_weight=${3:-0.5}

if [ -z "${pretrain}" ]; then
    echo "usage: bash scripts/pilot_ab.sh <stage1_checkpoint> [iters_per_epoch] [bg_weight]"
    exit 1
fi

run_arm () {
    local name=$1; shift
    mkdir -p ./checkpoint/${name}
    echo "=============== ${name} ==============="
    TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
        --nproc_per_node=${NGPU} --use_env train/main_ddp.py \
        --save_path ./checkpoint/${name} \
        --pretrain ${pretrain} \
        --used_data all_vimeo --model hem \
        --train_schedule semantic_pilot \
        --data_num ${iters} \
        --keep_index 2 \
        "$@" \
        --save_epoch 1 -b 4 \
        -l ./checkpoint/${name}/log.txt | tee ${name}.log
}

# --keep_index 2 pins the rate point so both arms see the same lambda instead of a
# random one, which would otherwise dominate a run this short.
# The baseline arm uses --roi --roi_bg_weight 1.0, not "no --roi": bg_weight=1.0 makes
# every weight exactly 1.0 (bit-identical loss) while keeping the data path, the ROI
# PNG loads and hence the augmentation RNG identical across the two arms. Validation
# runs unweighted in both arms, so the valid losses are directly comparable.
run_arm pilot_baseline --roi --roi_bg_weight 1.0
run_arm pilot_roi      --roi --roi_bg_weight ${bg_weight} --roi_targets biec swin cnn

echo
echo "Both arms done. Compare the [roi] lines:"
echo "  grep '\[roi\]' pilot_baseline.log | tail -20"
echo "  grep '\[roi\]' pilot_roi.log      | tail -20"
