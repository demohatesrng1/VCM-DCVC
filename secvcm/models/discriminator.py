
import time, os

import torchvision
import torch, math
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from .video_net import ME_Spynet, flow_warp, ResBlock, bilineardownsacling, LowerBound, UNet, \
get_enc_dec_models, get_hyper_enc_dec_models
from ..layers.layers import conv3x3, subpel_conv1x1, subpel_conv3x3
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, get_rounded_q, get_state_dict

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(0.1)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),  # BatchNorm1d for stability
            nn.ReLU(),             # ReLU activation
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)  # Force feature maps to 4x4
        x = self.fc_layers(x)
        return x
    

def get_gan_loss(d_real, d_fake):
    real_loss = torch.mean((d_real - 1) ** 2)  # Minimize the difference with 1 for real images
    fake_loss = torch.mean(d_fake ** 2)  # Minimize the difference with 0 for fake images
    return (real_loss + fake_loss) * 0.5