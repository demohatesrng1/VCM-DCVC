import os
import argparse
import torch
import cv2
import logging
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
import datetime
from pytorch_msssim import ms_ssim
from tensorboardX import SummaryWriter
from .drawuvg import uvgdrawplt
from .dataset import *
from .video_train import *
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
gpu_num = torch.cuda.device_count()


print_step = 100
cal_step = 10
warmup_step = 0
test_step = 10000
tot_epoch = 10000
tot_step = 3000000
decay_interval = 2300000
lr_decay = 0.1
tb_logger = None
global_step = 0
cur_lr = base_lr = 1e-4
num_workers = None
gpu_batch = None


parser = argparse.ArgumentParser(description='DCVC reimplement')

parser.add_argument('--ref_i_dir', type=str, default='')
parser.add_argument('-l', '--log', default='',help='output training details')
parser.add_argument('-p', '--pretrain', default='',help='load pretrain model')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testhevcb', action='store_true')
parser.add_argument("--gop", type=int, default=None)
parser.add_argument("--data_num", type=int, default=None, help="The data number of each epoch.")
parser.add_argument("--epoch", type=int, default=None, help="Specify the number of epochs.")
parser.add_argument('--save_path', type=str,)
parser.add_argument('--stage_extend', type=float, default=1, help="Setting 1 means 5 days training using RTX4090, setting 2 leads to 10 days, etc.")
parser.add_argument('--used_data', type=str, default="all_vimeo", help="all_vimeo, all_youhq, vimeo_youhq")
parser.add_argument('-b', "--batchsize_pergpu", type=int, default=4)
parser.add_argument("--data_scale", type=str, default="original", help="Downsample youhq, default: original, other: mid, small, mixed")
parser.add_argument('--model', type=str, default="hem")
parser.add_argument('--last_lr', type=float, default=None, help="Keep stage, keep the lr")
parser.add_argument('--freeze_q', action='store_true')
parser.add_argument('--keep_index', type=int, default=None)
parser.add_argument('--train_schedule', type=str, default="semantic_v3")
parser.add_argument('--isDetach', type=str, default="no") # default min_frame=10
parser.add_argument('--isWeighted', action='store_true')
parser.add_argument('--weights', nargs='+', type=float, default=[0.5, 1.2, 0.5, 0.9])
parser.add_argument("--lambdas", nargs='+', type=float, default=[40, 85, 170, 380, 640])
parser.add_argument("--repeat_num", nargs='+', type=int, default=[0, 4, 12])
parser.add_argument("--repeat_degrade", type=str, default="normal")
parser.add_argument("--noise_level", type=float, default=0.5)
parser.add_argument("--save_epoch", type=int, default=1)

import time
last_time_called = None
def time_interval(info="No info", isprint=False):
    if not isprint:
        return
    global last_time_called
    current_time = time.time()
    if last_time_called is None:
        print("This is the first call.")
    else:
        interval = current_time - last_time_called
        print(info, f" time: {interval:.6f} seconds")
    last_time_called = current_time

import torchvision.transforms.functional as TF
def save_frames(input_tensor, save_dir, isExit=True):
    """
    Save each frame of input_tensor to the specified directory.

    Args:
    - input_tensor (Tensor): Input tensor of shape B, 3*F, H, W.
    - save_dir (str): Directory path where frames will be saved.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get batch size and number of frames
    batch_size = input_tensor.size(0)
    num_frames = input_tensor.size(1) // 3

    for batch_idx in range(batch_size):
        # Get the sample for this batch index
        input_sample = input_tensor[batch_idx]  # shape: 3*F, H, W
        # Split the tensor into F frames
        images = torch.split(input_sample, 3, dim=0)

        # Save each frame
        for i, image in enumerate(images):
            # Convert to PIL image
            pil_image = TF.to_pil_image(image)

            # Save image to file
            filename = os.path.join(save_dir, f'batch{batch_idx}_frame{i+1}.png')
            pil_image.save(filename)

    print("save_frames")
    if isExit:
        exit()

def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:  # // gpu_num:
        lr = base_lr
    else:
        lr = base_lr * (lr_decay ** (global_step // decay_interval))
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def Var(x):
    return Variable(x.cuda())

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

#     # find_unused_parameters = False if training_strategy.cur_strategy["param"] == "both" else True
#     # if args.model == "dc" and training_strategy.cur_strategy["param"] == "both" and training_strategy.cur_strategy["train_seq"] >= 6:
#     #     find_unused_parameters = False


def Change_dataset(train_strategy, mode="train"):
    num_frames = train_strategy.cur_strategy["train_seq"]
    if args.used_data == "all_vimeo" or mode=="valid":
        return DataSet_vimeo(num_frames=num_frames, mode=mode)
    elif args.used_data in ["all_youhq", "vimeo_youhq"]:
        print(args.used_data)
        return DataSet_youhq(num_frames=num_frames, scale=args.data_scale)
    elif args.used_data == "all_bvidvc":
        return DataSet_bvidvc(num_frames=num_frames)


class training_strategy_class:
    def __init__(self, iters, data_batch = (64608, 1)):
        self.training_strategy = self.get_training_strategy(args.train_schedule)    # a training_strategy list
        self.check_assert(self.training_strategy) # check input string is valid
        self.data_batch = data_batch
        self.cur_strategy = None # current starategy
        self.stage_extend = args.stage_extend    # Extend the training stage by a certain multiple.
        self.relative_epoch = None
        self.get_cur_strategy(iters)
    
    def get_training_strategy(self, strategy_type):
        # VCM stage training strategy as described in Table I of the SEC-VCM paper.
        # "M" = motion-related modules, "C" = contextual-coding-related modules,
        # "S" = proposed semantic modules for machine-oriented reconstruction.
        # Each epoch contains about 64K iterations.
        #
        # Stage | Modules | Loss          | lr              | GOP | Epochs
        # VCM   | S       | L_e, L_cons   | 1e-4            | 2   | 1
        # VCM   | S       | L_e, L_cons   | 1e-4            | 3   | 1
        # VCM   | S       | L_e, L_cons   | 1e-4            | 6   | 1
        # VCM   | S       | L_e, L_cons   | 1e-4 -> 1e-5    | 6   | 5
        # VCM   | M,C,S   | L_total       | 1e-5            | 6   | 2
        if strategy_type == 'semantic_v3':
            training_strategy = \
            [{"epo": 0, "lr": 1e-4, "param": "semantic", "loss": "semantic_d_dinov2", "train_seq": 2, "comment": "not_cascaded"}] * 1 + \
            [{"epo": 1, "lr": 1e-4, "param": "semantic", "loss": "semantic_d_dinov2", "train_seq": 3, "comment": "cascaded"}] * 1 + \
            [{"epo": 2, "lr": 1e-4, "param": "semantic", "loss": "semantic_d_dinov2", "train_seq": 6, "comment": "cascaded"}] * 1 + \
            [{"epo": 3, "lr": 1e-4, "param": "semantic", "loss": "semantic_d_dinov2", "train_seq": 6, "comment": "cascaded"}] * 1 + \
            [{"epo": 4, "lr": 1e-5, "param": "semantic", "loss": "semantic_d_dinov2", "train_seq": 6, "comment": "cascaded"}] * 4 + \
            [{"epo": 8, "lr": 1e-5, "param": "semantic_all", "loss": "semantic_rd", "train_seq": 6, "comment": "cascaded"}] * 2
        else:
            raise TypeError(strategy_type)
        return training_strategy

    def check_assert(self, training_strategy):
        for i, cur in enumerate(training_strategy):
            assert cur["comment"] in ["cascaded", "not_cascaded", "not_cascaded_last_frame", "cascaded_repeat"]

    def get_cur_strategy(self, iters):
        def parse_comment(cur):
            cmd = cur.pop("comment")
            cur["isCascaded"] = True if cmd in ["cascaded", "cascaded_repeat"] else False
            cur["isOnlyLast"] = True if cmd == "not_cascaded_last_frame" else False
            cur["isRepeat"] = True if cmd == "cascaded_repeat" else False
            return cur
        
        # Assumption batch size 1. If we set 4 following official paper, the total iters is about 0.5M, which is too small
        data = self.data_batch[0]
        batch = self.data_batch[1]
        
        # Determine the relative epoch of paper. 
        if iters % (data / batch) == 0:
            self.relative_epoch = iters // (data / batch)
            self.status = "first actual epoch"
        else:
            self.relative_epoch = iters // (data / batch)
            self.status = "normal actual epoch"
        self.relative_epoch = min(self.relative_epoch, len(self.training_strategy)-1)
        # The epoch index starts from 0. 
        self.relative_epoch = int(self.relative_epoch)
        cur = self.training_strategy[self.relative_epoch].copy()
        cur = parse_comment(cur)

        cur["isDetach"] = args.isDetach
        cur["isWeighted"] = args.isWeighted
        cur["weights"] = args.weights
        cur["lambdas"] = args.lambdas
        if cur["isRepeat"]:
            cur["repeat_num"] = args.repeat_num
            cur["repeat_degrade"] = args.repeat_degrade
        cur["noise_level"] = args.noise_level
        self.cur_strategy = cur


not_found = []
def set_requires_grad(module, attribute_names, flag, other_flag=None):
    
    # set other_flag
    global not_found
    not_found = []
    if other_flag is not None:
        for name, parameter in module.named_parameters():
            parameter.requires_grad = other_flag
    # set flag
    for name in attribute_names:
        if hasattr(module, name):
            m = getattr(module, name)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = flag
            elif isinstance(m, nn.Parameter):
                m.requires_grad = flag
            else:
                raise TypeError(f"The attribute {name} is neither an nn.Module nor an nn.Parameter.")
        else:
            not_found.append(name)
            
            
attributes_q = [
    'y_q_basic',
    'y_q_scale',
    'y_q_basic_enc',
    'y_q_basic_dec',
    'y_q_scale_enc',
    'y_q_scale_dec',
]
attributes_q_mv = [
    'mv_y_q_basic',
    'mv_y_q_scale',
    'mv_y_q_basic_enc',
    'mv_y_q_basic_dec',
    'mv_y_q_scale_enc',
    'mv_y_q_scale_dec',
]
attributes_mv = [
    'optic_flow',
    'mv_encoder',
    'mv_decoder',
    'mv_hyper_prior_encoder',
    'mv_hyper_prior_decoder',
    'mv_y_prior_fusion',
    'mv_y_spatial_prior',
    'mv_y_prior_fusion_adaptor_0',
    'mv_y_prior_fusion_adaptor_1',
    'mv_y_spatial_prior_adaptor_1',
    'mv_y_spatial_prior_adaptor_2',
    'mv_y_spatial_prior_adaptor_3',
    'bit_estimator_z_mv',
]
attributes_semantic = [
    'semantic_decoder',
    'semantic_generation_net',
    'distribution_generation4',
    'distribution_generation8',
    'distribution_generation16',
]
attributes_vfm = [
    'resnet18_model', 
    'semantic_model',
    'dinov2_model',
    'alexnet_model', 
]


def Change_optim(train_strategy, net):
    module = net.module

    if dist.get_rank() == 0 and not_found:
        print(f"Attributes not found in module.hem: {', '.join(not_found)}")

    param = train_strategy.cur_strategy["param"]
    if param == "inter":
        set_requires_grad(module.hem, attributes_q_mv + attributes_mv, True, other_flag=False)
        set_requires_grad(module.hem, attributes_semantic, False)
    elif param == "recon":
        set_requires_grad(module.hem, attributes_q + attributes_q_mv + attributes_mv, False, other_flag=True)
        set_requires_grad(module.hem, attributes_semantic, False)
    elif param == "both":
        set_requires_grad(module.hem, attributes_q + attributes_q_mv + attributes_mv, True, other_flag=True)
        set_requires_grad(module.hem, attributes_semantic, False)
    elif param == "semantic":
        set_requires_grad(module.hem, attributes_semantic, True, other_flag=False)
    elif param == "semantic_all":
        set_requires_grad(module.hem, [], True, other_flag=True)
    else:
        print(f"Error: param \"{param}\" is not supported. ")
        exit()

    # check if it's RD-loss or D-loss
    loss = train_strategy.cur_strategy["loss"]
    if loss in ["me_mse", "recon_mse", "semantic_d", "semantic_d_dinov2", "semantic_d_without_entropy"]:
        set_requires_grad(module.hem, ["bit_estimator_z_mv", "bit_estimator_z"], False)
    elif loss in ["me_rdc_mse"]:
        set_requires_grad(module.hem, ["bit_estimator_z_mv"], True)
    elif loss in ["recon_rdc_mse"]:
        set_requires_grad(module.hem, ["bit_estimator_z"], True)
    elif loss in ["total_rdc_mse", "codec_rd_lpips"]:
        set_requires_grad(module.hem, ["bit_estimator_z_mv", "bit_estimator_z"], True)
    elif loss in ["semantic_rd"]:
        assert param == "semantic_all", "Error: when loss is 'semantic_rd', param must be 'semantic_rd'. "
    else:
        print(f"Error: loss \"{loss}\" is not supported. ")
        exit()
        
    if args.freeze_q:
        set_requires_grad(module.hem, attributes_q + attributes_q_mv, False)
        print("freeze q")
        
    # always set these semantic-related modules' grad to false
    set_requires_grad(module.hem, attributes_vfm, False)
    
    global cur_lr
    cur_lr = train_strategy.cur_strategy["lr"] if args.last_lr is None else args.last_lr

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, module.hem.parameters()), lr = cur_lr)
    
    # show all optimized parameters
    if dist.get_rank() == 0:
        logger.info("all optimized parameters:")
        for name, param in net.named_parameters():
            if param.requires_grad:
                logger.info(f"optimize: {name}")
    return optimizer
    

def Change_batchsize(train_strategy):
    num_frames = train_strategy.cur_strategy["train_seq"]
    if args.model == "hem": 
        if num_frames <= 3: 
            workers, batch = num_workers, gpu_batch # 1
        elif num_frames <= 5:   
            workers, batch = num_workers, gpu_batch # 1
        elif num_frames <= 7:
            workers, batch = num_workers//2, gpu_batch//2 # 1
        else:                   
            workers, batch = 2, 1
    elif "dc" in args.model:
        if num_frames <= 3: 
            workers, batch = num_workers, gpu_batch
        elif num_frames <= 6:   
            workers, batch = num_workers//2, gpu_batch//2
        elif num_frames <= 7:
            workers, batch = num_workers//2, gpu_batch//2
        else:                   
            workers, batch = 2, 1
    train_strategy.cur_strategy["batchsize_pergpu"] = max(batch, 1)
    return max(workers, 1), max(batch, 1)


def Change_loss(train_strategy, result, lmd):
    # pixel-domain loss
    mse = torch.mean(result['mse'])
    mse_me = torch.mean(result['me_mse'])
    mse_semantic = torch.mean(result['mse_semantic'])
    msssim = torch.mean(result['ssim'])
    msssim_semantic = torch.mean(result['ssim_semantic'])
    # bitrate loss
    bpp_y = torch.mean(result['bpp_y'])
    bpp_z = torch.mean(result['bpp_z'])
    bpp_mv_y = torch.mean(result['bpp_mv_y'])
    bpp_mv_z = torch.mean(result['bpp_mv_z'])
    bpp = torch.mean(result['bpp'])
    # perception loss
    lpips_alexnet = torch.mean(result['lpips_alexnet'])
    lpips_swin  = torch.mean(result['lpips_swin'])
    lpips_cnn = torch.mean(result['lpips_cnn'])
    lpips_dinov2 = torch.mean(result['lpips_dinov2'])
    # entropy loss
    entropy = result['conditional_entropy']
    entropy4 = result['conditional_entropy_4']
    entropy8 = result['conditional_entropy_8']
    entropy16 = result['conditional_entropy_16'] 
    entropy4_reverse = result['conditional_entropy_4_reverse']
    entropy8_reverse = result['conditional_entropy_8_reverse']
    entropy16_reverse = result['conditional_entropy_16_reverse'] 

    loss_type = train_strategy.cur_strategy["loss"]
    if loss_type == "me_mse":
        rd_loss = lmd * mse_me
    elif loss_type == "me_rdc_mse":
        rd_loss = lmd * mse_me + bpp_mv_y + bpp_mv_z
    elif loss_type == "recon_mse":
        rd_loss = lmd * mse
    elif loss_type == "recon_rdc_mse":
        rd_loss = lmd * mse + bpp_y + bpp_z
    elif loss_type == "total_rdc_mse":
        rd_loss = lmd * mse + bpp
    elif loss_type == "semantic_d":
        rd_loss = lmd * (0.1 * mse_semantic + lpips_swin + lpips_cnn) + entropy
    elif loss_type == "semantic_d_dinov2":
        rd_loss = lmd * (0.1 * mse_semantic + 0.1 * (1.0 - msssim_semantic) + lpips_swin + lpips_cnn + lpips_dinov2) + (entropy4 + entropy8 + entropy16) / 3.0
    elif loss_type == "semantic_d_without_entropy":
        rd_loss = lmd * (0.1 * mse_semantic + lpips_swin + lpips_cnn)
    elif loss_type == "semantic_rd":
        rd_loss = lmd * mse + bpp
        rd_loss = rd_loss + lmd * (0.1 * mse_semantic + lpips_swin + lpips_cnn) + entropy 
    elif loss_type == "codec_rd_lpips":
        rd_loss = lmd * (mse + 0.05 * lpips_alexnet) + bpp 
    else:
        print("Error, loss type not found: ", loss_type)
        raise TypeError
    return rd_loss, mse_me, mse, mse_semantic, msssim, msssim_semantic, lpips_alexnet, lpips_swin, lpips_cnn, lpips_dinov2, entropy, entropy4, entropy8, entropy16, entropy4_reverse, entropy8_reverse, entropy16_reverse, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp 


def mse_psnr(mse):
    if mse > 0:
        psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10)).cpu().detach().numpy()
    else:
        psnr = 100
    return psnr


def valid(epoch, num_workers, batch_size, lmd_index=3, train_strategy=None):
    with torch.no_grad():
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=valid_sampler)
        net.eval()
        net.module.valid()

        # record valid loss
        local_sumbpp = local_sumbpp_y = local_sumbpp_z = local_sumbpp_mv_y = local_sumbpp_mv_z = 0
        local_sumloss = local_sumpsnr = local_sumpsnr_me = local_sumpsnr_semantic = 0
        local_summsssim = local_summsssim_semantic = 0
        local_sumlpips_alexnet = local_sumlpips_swin = local_sumlpips_cnn = local_sumlpips_dinov2 = 0
        local_sumentropy = local_sumentropy4 = local_sumentropy8 = local_sumentropy16 = local_sumentropy4_reverse = local_sumentropy8_reverse = local_sumentropy16_reverse = 0
        
        t0 = datetime.datetime.now()
        lmd_list = train_strategy.cur_strategy["lambdas"]
        index = lmd_index

        test_num = 0
        for batch_idx, input in enumerate(valid_loader):
            lmd = lmd_list[index]
            images = to_variable(input)
            result = net(images, index, strategy=train_strategy.cur_strategy)
            rd_loss, mse_me, mse, mse_semantic, msssim, msssim_semantic, lpips_alexnet, lpips_swin, lpips_cnn, lpips_dinov2, entropy, entropy4, entropy8, entropy16, entropy4_reverse, entropy8_reverse, entropy16_reverse, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp = Change_loss(train_strategy, result, lmd)
        
            psnr = mse_psnr(mse)
            psnr_me = mse_psnr(mse_me)
            psnr_semantic = mse_psnr(mse_semantic)
            loss_ = rd_loss.cpu().detach().numpy()

            local_sumloss += loss_
            local_sumpsnr += psnr
            local_sumpsnr_me += psnr_me
            local_sumpsnr_semantic += psnr_semantic
            local_summsssim += msssim.cpu().detach().numpy()
            local_summsssim_semantic += msssim_semantic.cpu().detach().numpy()
            local_sumlpips_alexnet += lpips_alexnet.cpu().detach().numpy()
            local_sumlpips_swin += lpips_swin.cpu().detach().numpy()
            local_sumlpips_cnn += lpips_cnn.cpu().detach().numpy()
            local_sumlpips_dinov2 += lpips_dinov2.cpu().detach().numpy()
            local_sumentropy += entropy.cpu().detach().numpy()
            local_sumentropy4 += entropy4.cpu().detach().numpy()
            local_sumentropy8 += entropy8.cpu().detach().numpy()
            local_sumentropy16 += entropy16.cpu().detach().numpy()
            local_sumentropy4_reverse += entropy4_reverse.cpu().detach().numpy()
            local_sumentropy8_reverse += entropy8_reverse.cpu().detach().numpy()
            local_sumentropy16_reverse += entropy16_reverse.cpu().detach().numpy()
            local_sumbpp += bpp.cpu().detach().numpy()
            local_sumbpp_y += bpp_y.cpu().detach().numpy()
            local_sumbpp_z += bpp_z.cpu().detach().numpy()
            local_sumbpp_mv_y += bpp_mv_y.cpu().detach().numpy()
            local_sumbpp_mv_z += bpp_mv_z.cpu().detach().numpy()
            test_num += 1
        
        # Reduce across all processes
        global_sumloss = torch.tensor(local_sumloss / test_num).cuda().clone().detach()
        global_sumpsnr = torch.tensor(local_sumpsnr / test_num).cuda().clone().detach()
        global_sumpsnr_me = torch.tensor(local_sumpsnr_me / test_num).cuda().clone().detach()
        global_sumpsnr_semantic = torch.tensor(local_sumpsnr_semantic / test_num).cuda().clone().detach()
        global_summsssim = torch.tensor(local_summsssim / test_num).cuda().clone().detach()
        global_summsssim_semantic = torch.tensor(local_summsssim_semantic / test_num).cuda().clone().detach()
        global_sumlpips_alexnet = torch.tensor(local_sumlpips_alexnet / test_num).cuda().clone().detach()
        global_sumlpips_swin = torch.tensor(local_sumlpips_swin / test_num).cuda().clone().detach()
        global_sumlpips_cnn = torch.tensor(local_sumlpips_cnn / test_num).cuda().clone().detach()
        global_sumlpips_dinov2 = torch.tensor(local_sumlpips_dinov2 / test_num).cuda().clone().detach()
        global_sumentropy = torch.tensor(local_sumentropy / test_num).cuda().clone().detach()
        global_sumentropy4 = torch.tensor(local_sumentropy4 / test_num).cuda().clone().detach()
        global_sumentropy8 = torch.tensor(local_sumentropy8 / test_num).cuda().clone().detach()
        global_sumentropy16 = torch.tensor(local_sumentropy16 / test_num).cuda().clone().detach()
        global_sumentropy4_reverse = torch.tensor(local_sumentropy4_reverse / test_num).cuda().clone().detach()
        global_sumentropy8_reverse = torch.tensor(local_sumentropy8_reverse / test_num).cuda().clone().detach()
        global_sumentropy16_reverse = torch.tensor(local_sumentropy16_reverse / test_num).cuda().clone().detach()
        global_sumbpp = torch.tensor(local_sumbpp / test_num).cuda().clone().detach()
        global_sumbpp_y = torch.tensor(local_sumbpp_y / test_num).cuda().clone().detach()
        global_sumbpp_z = torch.tensor(local_sumbpp_z / test_num).cuda().clone().detach()
        global_sumbpp_mv_y = torch.tensor(local_sumbpp_mv_y / test_num).cuda().clone().detach()
        global_sumbpp_mv_z = torch.tensor(local_sumbpp_mv_z / test_num).cuda().clone().detach()
        
        dist.reduce(global_sumloss, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumpsnr, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumpsnr_me, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumpsnr_semantic, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_summsssim, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_summsssim_semantic, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumlpips_alexnet, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumlpips_swin, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumlpips_cnn, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumlpips_dinov2, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy4, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy8, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy16, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy4_reverse, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy8_reverse, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumentropy16_reverse, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumbpp, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumbpp_y, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumbpp_z, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumbpp_mv_y, 0, op=dist.ReduceOp.SUM)
        dist.reduce(global_sumbpp_mv_z, 0, op=dist.ReduceOp.SUM)
        
        # Only main process does the logging and final averaging
        if dist.get_rank() == 0:
            global_sumloss /= dist.get_world_size()
            global_sumpsnr /= dist.get_world_size()
            global_sumpsnr_me /= dist.get_world_size()
            global_sumpsnr_semantic /= dist.get_world_size()
            global_summsssim /= dist.get_world_size()
            global_summsssim_semantic /= dist.get_world_size()
            global_sumlpips_alexnet /= dist.get_world_size()
            global_sumlpips_swin /= dist.get_world_size()
            global_sumlpips_cnn /= dist.get_world_size()
            global_sumlpips_dinov2 /= dist.get_world_size()
            global_sumentropy /= dist.get_world_size()
            global_sumentropy4 /= dist.get_world_size()
            global_sumentropy8 /= dist.get_world_size()
            global_sumentropy16 /= dist.get_world_size()
            global_sumentropy4_reverse /= dist.get_world_size()
            global_sumentropy8_reverse /= dist.get_world_size()
            global_sumentropy16_reverse /= dist.get_world_size()
            global_sumbpp /= dist.get_world_size()
            global_sumbpp_y /= dist.get_world_size()
            global_sumbpp_z /= dist.get_world_size()
            global_sumbpp_mv_y /= dist.get_world_size()
            global_sumbpp_mv_z /= dist.get_world_size()
            
            t1 = datetime.datetime.now()
            deltatime = t1 - t0; deltatime = deltatime.seconds + 1e-6 * deltatime.microseconds 
            
            log = 'Valid Epoch: {:03} Loss: {:.6f}, Bpp:{:.3f}, Bpp_mv:{:.3f}, time:{:.3f}, index:{}'.format(
                epoch, global_sumloss, global_sumbpp, global_sumbpp_mv_y+global_sumbpp_mv_z, deltatime, lmd_index
            )
            log += '\n\t PSNR: {}\t PSNR_me: {:.3f}\t PSNR_semantic: {:.3f}\t MSSSIM: {:.3f}\t MSSSIM_semantic: {:.3f}\t lpips_alexnet: {:.6f}\t lpips_swin: {:.6f}\t lpips_cnn: {:.6f}\t lpips_dinov2: {:.6f}\n'.format(
                global_sumpsnr, global_sumpsnr_me, global_sumpsnr_semantic, global_summsssim, global_summsssim_semantic, global_sumlpips_alexnet, global_sumlpips_swin, global_sumlpips_cnn, global_sumlpips_dinov2
            )
            log += '\n\t entropy: {:.6f}\t entropy4: {:.6f}\t entropy8: {:.3f}\t entropy16: {:.3f}'.format(
                global_sumentropy, global_sumentropy4, global_sumentropy8, global_sumentropy16
            )
            log += '\n\t entropy4_reverse: {:.6f}\t entropy8_reverse: {:.3f}\t entropy16_reverse: {:.3f}'.format(
                global_sumentropy4_reverse, global_sumentropy8_reverse, global_sumentropy16_reverse
            )
            if dist.get_rank() == 0:
                logger.info(log)
        net.module.end_valid()
        return global_sumloss.item()


def train(epoch, global_step, num_workers, batch_size, train_strategy):
    if args.data_num:
        if 0 >= args.data_num:
            return global_step
    if dist.get_rank() == 0:
        print("epoch", epoch)
    global gpu_batch
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size,
                              pin_memory=True, sampler=train_sampler)
    train_sampler.set_epoch(epoch)
    net.train()

    global optimizer, cur_lr 
    
    bat_cnt = 0
    cal_cnt = 0
    sumbpp = sumbpp_y = sumbpp_z = sumbpp_mv_y = sumbpp_mv_z = sumloss = 0
    sumpsnr = sumpsnr_me = sumpsnr_semantic = summsssim = summsssim_semantic = 0
    sumlpips_alexnet = sumlpips_swin = sumlpips_cnn = sumlpips_dinov2 = sumentropy = sumentropy4 = sumentropy8 = sumentropy16 = sumentropy4_reverse = sumentropy8_reverse = sumentropy16_reverse = 0

    t0 = datetime.datetime.now()
    lmd_list = train_strategy.cur_strategy["lambdas"]
    for batch_idx, input in enumerate(train_loader):
        
        if args.data_num:
            if batch_idx >= args.data_num:
                break
        global_step += 1
        bat_cnt += 1
        index = random.randint(0,3) if args.keep_index is None else args.keep_index
        lmd = lmd_list[index]
        images = to_variable(input)


        result = net(images, index, strategy=train_strategy.cur_strategy)
        rd_loss, mse_me, mse, mse_semantic, msssim, msssim_semantic, lpips_alexnet, lpips_swin, lpips_cnn, lpips_dinov2, entropy, entropy4, entropy8, entropy16, entropy4_reverse, entropy8_reverse, entropy16_reverse, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp = Change_loss(train_strategy, result, lmd)
        optimizer.zero_grad()
        rd_loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()

        if global_step % cal_step == 0:
            cal_cnt += 1
            psnr_me = mse_psnr(mse_me)
            psnr = mse_psnr(mse)
            psnr_semantic = mse_psnr(mse_semantic)
            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            sumpsnr_me += psnr_me
            sumpsnr_semantic += psnr_semantic
            summsssim += msssim.cpu().detach()
            summsssim_semantic += msssim_semantic.cpu().detach()
            sumlpips_alexnet += lpips_alexnet.cpu().detach()
            sumlpips_swin += lpips_swin.cpu().detach()
            sumlpips_cnn += lpips_cnn.cpu().detach()
            sumlpips_dinov2 += lpips_dinov2.cpu().detach()
            sumentropy += entropy.cpu().detach()
            sumentropy4 += entropy4.cpu().detach()
            sumentropy8 += entropy8.cpu().detach()
            sumentropy16 += entropy16.cpu().detach()
            sumentropy4_reverse += entropy4_reverse.cpu().detach()
            sumentropy8_reverse += entropy8_reverse.cpu().detach()
            sumentropy16_reverse += entropy16_reverse.cpu().detach()
            sumbpp += bpp.cpu().detach()
            sumbpp_y += bpp_y.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()
            sumbpp_mv_y += bpp_mv_y.cpu().detach()
            sumbpp_mv_z += bpp_mv_z.cpu().detach() 

        if (batch_idx % print_step) == 0 and bat_cnt>1:
            if dist.get_rank() == 0:
                tb_logger.add_scalar('lr', cur_lr, global_step)
                tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
                tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
                tb_logger.add_scalar('psnr_me', sumpsnr_me / cal_cnt, global_step)
                tb_logger.add_scalar('psnr_semantic', sumpsnr_semantic / cal_cnt, global_step)
                tb_logger.add_scalar('msssim', summsssim / cal_cnt, global_step)
                tb_logger.add_scalar('msssim_semantic', summsssim_semantic / cal_cnt, global_step)
                tb_logger.add_scalar('lpips_alexnet', sumlpips_alexnet / cal_cnt, global_step)
                tb_logger.add_scalar('lpips_swin', sumlpips_swin / cal_cnt, global_step)
                tb_logger.add_scalar('lpips_cnn', sumlpips_cnn / cal_cnt, global_step)
                tb_logger.add_scalar('lpips_dinov2', sumlpips_dinov2 / cal_cnt, global_step)
                tb_logger.add_scalar('entropy', sumentropy / cal_cnt, global_step)
                tb_logger.add_scalar('entropy4', sumentropy4 / cal_cnt, global_step)
                tb_logger.add_scalar('entropy8', sumentropy8 / cal_cnt, global_step)
                tb_logger.add_scalar('entropy16', sumentropy16 / cal_cnt, global_step)
                tb_logger.add_scalar('entropy4_reverse', sumentropy4_reverse / cal_cnt, global_step)
                tb_logger.add_scalar('entropy8_reverse', sumentropy8_reverse / cal_cnt, global_step)
                tb_logger.add_scalar('entropy16_reverse', sumentropy16_reverse / cal_cnt, global_step)
                tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
                tb_logger.add_scalar('bpp_y', sumbpp_y / cal_cnt, global_step)
                tb_logger.add_scalar('bpp_z', sumbpp_z / cal_cnt, global_step)
                tb_logger.add_scalar('bpp_mv', sumbpp_mv_y / cal_cnt, global_step)
                tb_logger.add_scalar('bpp_mv_z', sumbpp_mv_z / cal_cnt, global_step)
                
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{} index:{}'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), sumloss / cal_cnt, cur_lr, deltatime.seconds + 1e-6 * deltatime.microseconds, index
            )
            log += '\n bpp: {:.4f}, bpp_y: {:.4f}, bpp_z: {:.4f}, bpp_mv_y: {:.4f}, bpp_mv_z: {:.6f}'.format(
                sumbpp / cal_cnt, sumbpp_y / cal_cnt, sumbpp_z / cal_cnt, sumbpp_mv_y / cal_cnt, sumbpp_mv_z / cal_cnt,
            )
            log += '\n [video branch] psnr_me: {:.2f}, psnr: {:.2f}, lpips_alexnet: {:.6f}, msssim: {:.3f}'.format(
                sumpsnr_me / cal_cnt, sumpsnr / cal_cnt, sumlpips_alexnet / cal_cnt, summsssim / cal_cnt
            )
            log += '\n [semantic branch] psnr_semantic: {:.2f}, msssim_semantic: {:.3f}, lpips_swin: {:.6f}, lpips_cnn: {:.6f}, lpips_dinov2: {:.6f}'.format(
                sumpsnr_semantic / cal_cnt, summsssim_semantic / cal_cnt, sumlpips_swin / cal_cnt, sumlpips_cnn / cal_cnt, sumlpips_dinov2 / cal_cnt
            )
            log += '\n [semantic branch] entropy: {:.4f}, entropy4: {:.4f}, entropy8: {:.4f}, entropy16: {:.4f}'.format(
                sumentropy / cal_cnt, sumentropy4 / cal_cnt, sumentropy8 / cal_cnt, sumentropy16 / cal_cnt
            )
            log += '\n [semantic branch] entropy4_reverse: {:.4f}, entropy8_reverse: {:.4f}, entropy16_reverse: {:.4f}'.format(
                sumentropy4_reverse / cal_cnt, sumentropy8_reverse / cal_cnt, sumentropy16_reverse / cal_cnt
            )
            if dist.get_rank() == 0:
                logger.info(log) 

            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_y = sumbpp_z = sumbpp_mv_y = sumbpp_mv_z = sumloss = sumpsnr = sumpsnr_me = summsssim = summsssim_semantic = 0
            sumpsnr_semantic = sumlpips_alexnet = sumlpips_swin = sumlpips_cnn = sumlpips_dinov2 = sumentropy = sumentropy4 = sumentropy8 = sumentropy16 = sumentropy4_reverse = sumentropy8_reverse = sumentropy16_reverse = 0
            
            t0 = t1
            
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    if dist.get_rank() == 0:
        logger.info(log)
    return global_step


def to_variable(x, is_Training=True):
    if torch.cuda.is_available():
        x = x.cuda(local_rank, non_blocking=True)
    return Variable(x, requires_grad=is_Training)


if __name__ == "__main__":
    args = parser.parse_args()
    gpu_batch = args.batchsize_pergpu
    num_workers = int(gpu_batch*2)

    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    if dist.get_rank() == 0: # Only print on rank 0
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        logger = logging.getLogger("VideoCompression")

    if dist.get_rank() == 0:
    # Code executed on rank 0 process
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path) 

        formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
        stdhandler = logging.StreamHandler()
        stdhandler.setLevel(logging.INFO)
        stdhandler.setFormatter(formatter)
        logger.addHandler(stdhandler)
        if args.log != '':
            filehandler = logging.FileHandler(args.log)
            filehandler.setLevel(logging.INFO)
            filehandler.setFormatter(formatter)
            logger.addHandler(filehandler)
        logger.setLevel(logging.INFO)
        if args.model == "hem":
            logger.info("DCVC-HEM training")
        elif args.model == "dc":
            logger.info("DCVC-DC training")
            exit()
        else:
            logger.info(f"{args.model} training")
        logger.info(args)
    
    # create model
    assert args.model == "hem", "args.model must be \"hem\""
    model = HEM_train()
    # display number of parameters
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Total number of parameters: {total_params:.2f}M")
    # load pretrained weight
    if args.pretrain != '':
        if dist.get_rank() == 0:
            logger.info(f"loading pretrain : {args.pretrain}")
        finished_epoch, _, global_step = load_model(model.hem, args.pretrain)
    else:
        finished_epoch = -1
    # move model to GPU and transfer it into parallel mode
    net = model.cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if dist.get_rank() == 0:
        tb_logger = SummaryWriter(args.save_path)

    # training and validation
    for epoch in range(finished_epoch+1, tot_epoch):
        # determine current relative epoch and trianing details
        train_strategy = training_strategy_class(global_step)
        num_worker_stg, batchsize_stg = Change_batchsize(train_strategy)
        # determine dataset
        train_dataset = Change_dataset(train_strategy, mode="train")
        valid_dataset = Change_dataset(train_strategy, mode="valid")
        # determine optimizer 
        optimizer = Change_optim(train_strategy, net)
        if dist.get_rank() == 0:
            logger.info(f"epoch: {epoch}, relative epoch: {train_strategy.relative_epoch}, global_steps: {global_step}")
            logger.info(train_strategy.cur_strategy)
            logger.info(f'lr: {cur_lr}, num_frames: {train_dataset.num_frames}, batch_per_gpu: {batchsize_stg}')
        
        # training
        global_step = train(epoch, global_step, num_worker_stg, batchsize_stg, train_strategy)

        # save model ckpt
        if dist.get_rank() == 0:
            if (epoch + 1) % args.save_epoch == 0:
                save_model(model.hem, epoch, train_strategy.relative_epoch, global_step, path=args.save_path)

        # validation
        valid(epoch, num_worker_stg, batchsize_stg, lmd_index=0, train_strategy=train_strategy)
        valid(epoch, num_worker_stg, batchsize_stg, lmd_index=1, train_strategy=train_strategy)
        valid(epoch, num_worker_stg, batchsize_stg, lmd_index=2, train_strategy=train_strategy)
        valid(epoch, num_worker_stg, batchsize_stg, lmd_index=3, train_strategy=train_strategy)
