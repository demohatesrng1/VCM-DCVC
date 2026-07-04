#!/bin/bash
# SEC-VCM: Example training script
# Usage: bash scripts/train.sh

export PYTHONPATH=$PWD:$PWD/third_party/mask2former:$PWD/third_party/dino:$PWD/third_party/detectron2

name="secvcm"

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env train/main_ddp.py \
    --save_path ./checkpoint/${name} \
    --used_data all_vimeo --stage_extend 1 --model hem \
    --train_schedule semantic_v3 \
    --save_epoch 4 -b 4 \
    -l ./checkpoint/${name}/log.txt | tee train.log
