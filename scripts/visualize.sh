#!/bin/bash
# SEC-VCM: Example visualization script
# Usage: bash scripts/visualize.sh

export PYTHONPATH=$PWD:$PWD/third_party/mask2former:$PWD/third_party/dino:$PWD/third_party/detectron2

name="secvcm"
model_name="ckpt-hem10-re2"
rm -f "test.log"

output_path="./checkpoint/${name}/${model_name}-vis.json"
CUDA_VISIBLE_DEVICES=0 python scripts/test_video.py \
    --i_frame_model_path ./checkpoint/acmmm2022_image_psnr.pth.tar \
    --cuda True -w 1 --write_stream 0 --rate_num 4 \
    --model_path ./checkpoint/${name}/${model_name}.model \
    --output_path $output_path \
    --test_config ./dataset_config_example.json \
    --save_decoded_frame True \
    --decoded_frame_path "results/temp_folder" | tee -a test.log
