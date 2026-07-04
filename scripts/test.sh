#!/bin/bash
# SEC-VCM: Example test script
# Usage: bash scripts/test.sh

export PYTHONPATH=$PWD:$PWD/third_party/mask2former:$PWD/third_party/dino:$PWD/third_party/detectron2

name="secvcm"
model_name="ckpt-hem10-re2"
rm -f "test.log"
rate_num=4

output_path="./checkpoint/${name}/${model_name}-test.json"
CUDA_VISIBLE_DEVICES=0 python scripts/test_video.py \
    --i_frame_model_path ./checkpoint/acmmm2022_image_psnr.pth.tar \
    --cuda True -w 1 --write_stream 0 --rate_num ${rate_num} \
    --model_path ./checkpoint/${name}/${model_name}.model \
    --output_path $output_path \
    --test_config ./dataset_config_example.json \
    --save_decoded_frame False | tee -a test.log
