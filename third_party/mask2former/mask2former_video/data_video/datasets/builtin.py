# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    # pre-defined by mask2former
    "ytvis_2019_train": ("/opt/data/private/syx/dataset/ytvis2019/train/JPEGImages",
                         "/opt/data/private/syx/dataset/ytvis2019/train.json"),
    "ytvis_2019_val": ("/opt/data/private/syx/dataset/ytvis2019/valid/JPEGImages",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT.json"),
    "ytvis_2019_test": ("ytvis2019/test/JPEGImages",
                        "ytvis2019/test.json"),
    
    # defined by myself
    # (1) training dataset
    "ytvis_2019_train_merge_x265": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-x265-png.json"),
    "ytvis_2019_train_merge_x264": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-x264-png.json"),
    "ytvis_2019_train_merge_v1": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v1-png.json"),
    "ytvis_2019_train_merge_v2": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v2-png.json"),
    "ytvis_2019_train_merge_v3": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v3-png.json"),
    "ytvis_2019_train_merge_v4": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v4-png.json"),
    "ytvis_2019_train_merge_v6": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v6-png.json"),
    "ytvis_2019_train_merge_v7": ("/opt/data/private/syx/dataset/ytvis2019/train/",
                         "/opt/data/private/syx/dataset/ytvis2019/train-merge-v7-png.json"),
                         
    # (2) evaluation dataset
    "ytvis_2019_val_x265_20": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-20",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_23": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-23",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_26": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-26",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_29": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-29",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_32": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-32",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_35": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-35",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    
    "ytvis_2019_val_x264_20": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-20",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_23": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-23",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_26": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-26",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_29": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-29",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_32": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-32",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_35": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-35",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    
    "ytvis_2019_val_dcvc_psnr_0": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-0",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_psnr_1": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-1",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_psnr_2": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-2",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_psnr_3": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-3",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),

    "ytvis_2019_val_dcvc_dc_psnr_0": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-0",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_1": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-1",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_2": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-2",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_3": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-3",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),

    "ytvis_2019_val_smc_24": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-SMC-24",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_smc_28": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-SMC-28",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_smc_32": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-SMC-32",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    "ytvis_2019_val_smc_36": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-SMC-36",
                       "/opt/data/private/syx/dataset/ytvis2019/valid-pair-png.json"),
    
    # compared method: PromptIR
    "ytvis_2019_val_x265_26_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-26-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_29_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-29-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_32_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-32-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x265_35_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x265-35-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    
    "ytvis_2019_val_x264_26_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-26-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_29_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-29-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_32_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-32-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_x264_35_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-x264-35-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    
    "ytvis_2019_val_dcvc_psnr_0_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-0-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_psnr_1_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-1-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_psnr_2_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-2-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_psnr_3_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-PSNR-3-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),

    "ytvis_2019_val_dcvc_dc_psnr_0_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-0-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_1_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-1-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_2_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-2-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    "ytvis_2019_val_dcvc_dc_psnr_3_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-3-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    
    "ytvis_2019_val_dcvc_dc_psnr_3_promptIR": ("/opt/data/private/syx/dataset/ytvis2019/valid/PNGImages-DCVC-DC-PSNR-3-denoise-PromptIR",
                       "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT_png.json"),
    
    # test for DCVC-HEM official
    "ytvis2019val_hem_psnr_0": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_psnr/DMC_0", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_0.json"),
    "ytvis2019val_hem_psnr_1": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_psnr/DMC_1", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_1.json"),
    "ytvis2019val_hem_psnr_2": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_psnr/DMC_2", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_2.json"),
    "ytvis2019val_hem_psnr_3": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_psnr/DMC_3", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_3.json"),
    
    "ytvis2019val_hem_ssim_0": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_ssim/DMC_0", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_0.json"),
    "ytvis2019val_hem_ssim_1": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_ssim/DMC_1", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_1.json"),
    "ytvis2019val_hem_ssim_2": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_ssim/DMC_2", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_2.json"),
    "ytvis2019val_hem_ssim_3": ("/opt/data/private/syx/DCVC-HEM/results/decoded_frames_ytvis2019valid_official_ssim/DMC_3", "/opt/data/private/syx/DCVC-HEM/results/test_config_ytvis2019valid/ytvis2019valid_3.json"),
    
    # temp folder and temp dataset for test
    "ytvis_2019_val_dcvc_hem_temp_0": ("/opt/data/private/syx/DCVC-HEM/DCVC-HEM-v1/results/decoded_frames_ytvis2019valid_temp/DMC_0",
                        "/opt/data/private/syx/DCVC-HEM/dataset/downstream_test_config_ytvis2019valid/ytvis2019valid_0.json"),
    "ytvis_2019_val_dcvc_hem_temp_1": ("/opt/data/private/syx/DCVC-HEM/DCVC-HEM-v1/results/decoded_frames_ytvis2019valid_temp/DMC_1",
                        "/opt/data/private/syx/DCVC-HEM/dataset/downstream_test_config_ytvis2019valid/ytvis2019valid_1.json"),
    "ytvis_2019_val_dcvc_hem_temp_2": ("/opt/data/private/syx/DCVC-HEM/DCVC-HEM-v1/results/decoded_frames_ytvis2019valid_temp/DMC_2",
                        "/opt/data/private/syx/DCVC-HEM/dataset/downstream_test_config_ytvis2019valid/ytvis2019valid_2.json"),
    "ytvis_2019_val_dcvc_hem_temp_3": ("/opt/data/private/syx/DCVC-HEM/DCVC-HEM-v1/results/decoded_frames_ytvis2019valid_temp/DMC_3",
                        "/opt/data/private/syx/DCVC-HEM/dataset/downstream_test_config_ytvis2019valid/ytvis2019valid_3.json"),

}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
