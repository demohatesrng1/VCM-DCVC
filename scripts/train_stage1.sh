#!/bin/bash
# Stage 1: LPIPS fine-tuning of the DCVC-HEM PSNR weights.
#
# Produces the checkpoint Stage 2 requires. Microsoft released DCVC-HEM weights
# but no DCVC-HEM training code, so this runs on SEC-VCM's own trainer using the
# codec_rd_lpips loss (lambda*(mse + 0.05*lpips_alexnet) + bpp) that was already
# implemented in Change_loss.
#
# --skip_semantic keeps the semantic branch and all three teachers out of the
# graph, so this stage needs neither detectron2 nor any teacher checkpoint:
# HEM weights + Vimeo are enough to start.
#
# Usage: bash scripts/train_stage1.sh
set -e
source scripts/env.sh

name="stage1_lpips"
mkdir -p ./checkpoint/${name}

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
    --nproc_per_node=${NGPU} --use_env train/main_ddp.py \
    --save_path ./checkpoint/${name} \
    --pretrain ${HEM_VIDEO_CKPT} \
    --used_data all_vimeo --model hem \
    --train_schedule stage1_lpips \
    --skip_semantic \
    --stage_extend 1 \
    --save_epoch 1 -b 4 \
    -l ./checkpoint/${name}/log.txt | tee ${name}.log

echo
echo "Stage 1 done. Feed the last checkpoint in ./checkpoint/${name}/ to Stage 2 via --pretrain."
