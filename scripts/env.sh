#!/bin/bash
# Shared paths for SEC-VCM training. Edit this file once; every other script
# sources it. Run scripts from the repository root.
#
#   source scripts/env.sh

export PYTHONPATH=$PWD:$PWD/third_party/mask2former:$PWD/third_party/dino:$PWD/third_party/detectron2

# ---------------------------------------------------------------- datasets ---
# Vimeo-90k septuplet (~82 GB). sequences/ holds NNNNN/NNNN/im1.png .. im7.png
export VIMEO_ROOT=/data/vimeo_septuplet/sequences
export VIMEO_TRAIN_LIST=/data/vimeo_septuplet/sep_trainlist.txt
export VIMEO_VALID_LIST=/data/vimeo_septuplet/sep_testlist.txt
# A short valid list keeps per-epoch validation cheap; make one with e.g.
#   head -n 256 sep_testlist.txt > sep_testlist_short.txt
export VIMEO_VALID_LIST_SHORT=/data/vimeo_septuplet/sep_testlist_short.txt

# Precomputed ROI maps (scripts/precompute_roi_masks.py). Mirrors VIMEO_ROOT.
export ROI_ROOT=/data/vimeo_septuplet/roi

# ------------------------------------------------------------- checkpoints ---
# DCVC-HEM PSNR weights from microsoft/DCVC (DCVC-family/DCVC-HEM/checkpoints/download.py)
export HEM_IMAGE_CKPT=$PWD/checkpoint/acmmm2022_image_psnr.pth.tar
export HEM_VIDEO_CKPT=$PWD/checkpoint/acmmm2022_video_psnr.pth.tar

# Teacher checkpoints. The shipped mask2former yaml hardcodes an absolute path
# from the authors' machine, so this override is required, not optional.
export M2F_WEIGHTS=$PWD/checkpoint/ytvis-swin-T.pkl
export DINOV2_WEIGHTS=$PWD/pretrain/dinov2_vits14_reg4_pretrain.pth

# ------------------------------------------------------------------- rigs ----
export NGPU=4
export GPUS=0,1,2,3
