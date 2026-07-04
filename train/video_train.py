from typing import Callable, Union
import torch
from torch import nn
from torch.nn.modules.module import Module, _grad_t
from torch.utils.hooks import RemovableHandle

from secvcm.models.video_model import DMC as HEM
import os
import random, torchvision
import copy
import itertools
import logging
import os, json

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'mask2former'))
from mask2former import add_maskformer2_config
from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_maskformer2_video_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)

def save_model(model, epoch, relative_epoch, iter, path):
    path = os.path.join(path, "epoch{}-relative{}-iter{}.model".format(epoch, relative_epoch, iter))
    torch.save(model.state_dict(), path)
    print("save model: ", path)

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f, map_location='cpu')
        
    # Handle multi-GPU model key names
    if next(iter(pretrained_dict)).startswith('module.'):
        pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
        
    model.load_state_dict(pretrained_dict, strict=False)
    del pretrained_dict
    torch.cuda.empty_cache()
    
    f = str(f) # name format follows "epoch0-relative0-iter0.model"
    if 'iter' in f and 'epoch' in f and 'relative' in f and '.model' in f:
        st = f.find('epoch') + 5
        ed = f.find('-relative', st)
        epoch = int(f[st:ed])
        
        st = f.find('relative') + 8
        ed = f.find('-iter', st)
        relative_epoch = int(f[st:ed])
        
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        iters = int(f[st:ed])
        return epoch, relative_epoch, iters 
    else:
        return -1, None, 0

class DCVC_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.hem = None
        self.validing = False
        self.frame_weight = None

    def valid(self):
        self.validing = True
    
    def end_valid(self):
        self.validing = False
    
    def cal_avg_result(self, result_lst, isWeighted=False, isOnlyLast=False):
        avg_results = {}
        
        if isOnlyLast:
            last_result = result_lst[-1]
            i = len(result_lst) - 1
            if isWeighted:
                last_result["mse"] = last_result["mse"] * self.frame_weight[len(result_lst)%len(self.frame_weight)]
            return last_result

        for _, key in enumerate(result_lst[0].keys()):
            total = 0
            if key == 'mse' and isWeighted:
                for i,result in enumerate(result_lst):
                    total += result[key] * self.frame_weight[(i+1)%len(self.frame_weight)]
            else:
                for result in result_lst:
                    total += result[key]
            avg_results[key] = total / len(result_lst) 
        return avg_results

    def get_detached_dpb(self, dpb, frame_index, isDetach, max_frame=32):
        if isDetach == "feature_fix_6":
            detach_nodes = ["ref_feature", "ref_mv_feature"]
            detach_idx = [True if (i+1) % 6 == 0 else False for i in range(max_frame)]
        elif isDetach == "feature_fix_4":
            detach_nodes = ["ref_feature", "ref_mv_feature"]
            detach_idx = [True if (i+1) % 4 == 0 else False for i in range(max_frame)]
        elif isDetach == "all_fix_6":
            detach_nodes = ["ref_feature", "ref_mv_feature", "ref_y", "ref_mv_y"]
            detach_idx = [True if (i+1) % 6 == 0 else False for i in range(max_frame)]
        else:
            raise TypeError(isDetach)
        
        if detach_idx[frame_index] == False:
            return dpb
        for k,v in dpb.items():
            if k in detach_nodes:
                dpb[k] = v.clone().detach()
        return dpb

    def forward_multi(self, images, dpb, lmd_index, isCascaded, isDetach="no"):
        result_lst = []
        for i, img in enumerate(images):
            if i == 0:
                if dpb["ref_frame"] is None: # If isRepeat, then not None; use the repeated first P-frame as ref_frame
                    dpb["ref_frame"] = images[0]
                continue
            result = self.hem(img, dpb, lmd_index=lmd_index, frame_idx=i)
            dpb = result.pop('dpb')
            if not isCascaded:
                dpb["ref_frame"] = img

            if isDetach != "no":
                dpb = self.get_detached_dpb(dpb, i, isDetach)

            result_lst.append(result)
        return result_lst
    
    def repeat_first_Pframe(self, images, dpb, lmd_index, strategy=None):
        with torch.no_grad():
            nums = strategy["repeat_num"]
            degrade = strategy["repeat_degrade"]
            if degrade == "normal":
                pass
            elif degrade == "min_index":
                lmd_index = 0
            else:
                raise TypeError(degrade)

            img0 = images[0]
            img1 = images[1]
            rand_num = random.choice(nums)
            if rand_num == 0:
                return images, dpb
            for i in range(rand_num+1):
                if i == 0:
                    dpb["ref_frame"] = img0
                    continue
                result = self.hem(img1, dpb, lmd_index=lmd_index, frame_idx=i)
                dpb = result.pop('dpb')
            for k in dpb.keys():
                if k not in ['ref_frame', 'ref_feature']:
                    dpb[k] = None
            return images[1:], dpb

    def forward(self, images, dpb, lmd_index, strategy):
        isCascaded = strategy["isCascaded"]
        isWeighted = strategy["isWeighted"]
        isDetach = strategy["isDetach"]
        isOnlyLast = strategy["isOnlyLast"]
        self.frame_weight = strategy["weights"]
        isRepeat = strategy["isRepeat"]
        
        images = torch.split(images, 3, dim=1)
        if self.training and isRepeat:
            images, dpb = self.repeat_first_Pframe(images, dpb, lmd_index, strategy)
        result_lst = self.forward_multi(images, dpb, lmd_index=lmd_index, isCascaded=isCascaded, isDetach=isDetach)
        avg_results = self.cal_avg_result(result_lst, isWeighted=isWeighted, isOnlyLast=isOnlyLast)
        return avg_results
    
    def set_noise_level(self, noise_level=0.5):
        def add_noise(x):
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -noise_level, noise_level)
            noise = noise.clone().detach()
            return x + noise
        self.hem.add_noise = add_noise

class HEM_train(DCVC_base):
    def __init__(self):
        super().__init__()
        
        # pretrained mask2former model
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_maskformer2_video_config(cfg)
        cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "..", "third_party", "mask2former", "configs/youtubevis_2019/swin/video_maskformer2_swin_tiny_bs16_8ep.yaml")) 
        cfg.freeze()
        mask2former_model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(mask2former_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        print("Mask2Former model loaded successfully.")
        # Pretrained ResNet-18 model
        resnet18_model = torchvision.models.resnet18(pretrained=True)
        print("ResNet-18 model loaded successfully.")
        # pretrained dinov2 model
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'dino')) 
        from dinov2.hub.backbones import dinov2_vits14_reg 
        dino_model = dinov2_vits14_reg()
        dino_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'pretrain', 'dinov2_vits14_reg4_pretrain.pth')))
        print("DINOv2 model loaded successfully.")

        self.hem = HEM(swin_model=mask2former_model, cnn_model=resnet18_model, dino_model=dino_model, inference_mode=False)
    
    def forward(self, images, lmd_index, strategy):
        dpb = {
            "ref_frame": None,
            "ref_frame_semantic": None,
            "ref_feature": None,
            "ref_feature_semantic": None, 
            "ref_y": None,
            "ref_mv_y": None,
        }
        return super().forward(images, dpb, lmd_index, strategy)


