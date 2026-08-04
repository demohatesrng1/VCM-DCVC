# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time, os

import torchvision
import torch, math
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from .video_net import ME_Spynet, flow_warp, ResBlock, bilineardownsacling, LowerBound, UNet, get_enc_dec_models, get_hyper_enc_dec_models
from ..layers.layers import conv3x3, subpel_conv1x1, subpel_conv3x3
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
get_rounded_q, get_state_dict
from ..roi import build_weight_map, weighted_mean, weighted_mean_tokens, \
weighted_pixel_sum, region_stats, dino_grid_size

import lpips

class ResNet18(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self.backbone.eval()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad = False) 
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad = False)

    def forward(self, x):
        # standardrize
        x = (x - self.mean) / self.std
        # forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        out1 = self.backbone.maxpool(x)
        out2 = self.backbone.layer1(out1)
        out3 = self.backbone.layer2(out2)
        out4 = self.backbone.layer3(out3)
        out5 = self.backbone.layer4(out4)
        return {'stem': out1, 'res2': out2, 'res3': out3, 'res4': out4, 'res5': out5}


class DinoV2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self.backbone.eval()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad = False) 
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad = False)

    def forward(self, x):
        # standardrize
        x = (x - self.mean) / self.std
        # resize 
        _, _, h, w = x.shape
        new_h = round(h / 14) * 14
        new_w = round(w / 14) * 14
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # forward
        features = self.backbone.get_intermediate_layers(x, 12)    
        return {'coarse': features[3], 'fine': features[7]}


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ContextualEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                                start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                                start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                                start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                                start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class SemanticDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.first_conv = torch.nn.Conv2d(channel_M, channel_N, kernel_size=3, padding=1, stride=1)
        self.res1 = ResBlock(channel_N, bottleneck=True, slope=0.1, start_from_relu=True, end_with_relu=True)
        self.up1 = subpel_conv3x3(channel_N, channel_N, 2) 
        self.res2 = ResBlock(channel_N, bottleneck=True, slope=0.1, start_from_relu=True, end_with_relu=True)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res3 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1, start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res4 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1, start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, latent, context2, context3):
        feature = self.first_conv(latent)
        out16 = self.res1(feature)
        out8 = self.res2(self.up1(out16))  
        out4 = self.res3(torch.cat([self.up2(out8), context3], dim=1))
        out2 = self.res4(torch.cat([self.up3(out4), context2], dim=1))
        f = self.up4(out2)
        return f, out2, out4, out8, out16 


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class GatedGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        
        self.factor_generator = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1),
            ResBlock(channel=channel, bottleneck=True), 
            nn.Sigmoid()
        )
            
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res, recon_feature):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        
        alpha = self.factor_generator(torch.cat((feature, recon_feature), dim=1))
        feature = alpha * feature + (1 - alpha) * recon_feature
        
        recon = self.recon_conv(feature)
        return feature, recon


class DistributionGeneration(nn.Module):
    def __init__(self, in_channles, out_channels):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channles, out_channels, kernel_size=3, stride=1, padding=1)
        self.resblock = ResBlock(out_channels, bottleneck=True)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.resblock(x)
        means, scales = x.chunk(2, 1)
        return means, scales


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, swin_model=None, cnn_model=None, dino_model=None, inference_mode=False,
                 roi_cfg=None, m2f_norm='legacy', skip_semantic=False):
        """
        roi_cfg:       secvcm.roi.RoiConfig, or None for the unweighted baseline.
        m2f_norm:      'legacy' keeps the released (x*255*255) scaling of the Mask2Former
                       input; 'fixed' uses the single x*255 that matches the config's
                       PIXEL_MEAN/PIXEL_STD.  Kept as a switch so the two can be compared
                       rather than silently changed.
        skip_semantic: run the pixel branch only.  Used by Stage 1 (LPIPS fine-tuning of
                       the base codec), where the semantic branch and all three teachers
                       are dead weight.
        """
        super().__init__(y_distribution='laplace', z_channel=64, mv_z_channel=64)
        self.DMC_version = '1.19'

        channel_mv = 64
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        self.optic_flow = ME_Spynet()

        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1)
        )

        self.mv_y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_mv * 4, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, padding=1)
        )

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)

        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.contextual_hyper_prior_decoder = nn.Sequential(
            conv3x3(channel_N, channel_M),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_N, channel_M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1),
        )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_M * 5, channel_M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, padding=1)
        )

        self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        self.recon_generation_net = ReconGeneration()
        
        # decoder for semantic frame
        self.semantic_decoder = SemanticDecoder(channel_N=channel_N, channel_M=channel_M)
        self.semantic_generation_net = GatedGeneration() 

        self.mv_y_q_basic = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_basic = nn.Parameter(torch.ones((1, channel_M, 1, 1)))
        self.y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.anchor_num = int(anchor_num)

        self._initialize_weights()
        
        # -------------- pretrained model must not be initialized -------
        
        self.alexnet_model = lpips.LPIPS(net='alex')

        self.inference_mode = inference_mode
        self.skip_semantic = skip_semantic
        self.roi_cfg = roi_cfg
        assert m2f_norm in ('legacy', 'fixed'), f"unknown m2f_norm: {m2f_norm}"
        self.m2f_norm = m2f_norm
        # The teachers and the BiEC distribution heads only exist while the semantic
        # branch is being trained.  Inference drops them, and so does Stage 1.
        self.use_semantic = not (self.inference_mode or self.skip_semantic)
        # moudules only for training
        if self.use_semantic:
            # semantic teachers
            self.resnet18_model = ResNet18(cnn_model)
            self.semantic_model = swin_model
            self.dinov2_model = DinoV2(dino_model)
            assert self.resnet18_model is not None, "Error, you must assign a ResNet18 model!"
            assert self.semantic_model is not None, "Error, you must assign a swin model!"
            assert self.dinov2_model is not None, "Error, you must assign a DinoV2 model!"
            # distribution estimation
            self.distribution_generation16 = DistributionGeneration(384, 128)
            self.distribution_generation8 = DistributionGeneration(192, 128)
            self.distribution_generation4 = DistributionGeneration(96, 256)

            self.distribution_generation16_reverse = DistributionGeneration(64, 768)
            self.distribution_generation8_reverse = DistributionGeneration(64, 384)
            self.distribution_generation4_reverse = DistributionGeneration(128, 192)
        else:
            # semantic teachers
            self.resnet18_model = None
            self.semantic_model = None
            self.dinov2_model = None
            # distribution estimation
            self.distribution_generation16 = None
            self.distribution_generation8 = None
            self.distribution_generation4 = None

        
    def get_conditional_entropy(self, feature, mean, sigma, weight=None, roi=None):
        """Conditional entropy of ``feature`` under a Laplace prior (the BiEC term).

        ``weight`` is an optional (B, 1, h, w) ROI weight map with unit mean; when it
        is None the reduction is the plain mean of the original implementation.
        ``roi`` is only used to report foreground/background entropy separately.

        Returns ``(entropy, fg, bg)`` where fg/bg are detached diagnostics.
        """
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print("Found NaN or Inf in mean"); exit()
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            print("Found NaN or Inf in sigma"); exit()
        if torch.isnan(feature).any() or torch.isinf(feature).any():
            print("Found NaN or Inf in feature"); exit()

        outputs = feature
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)
        mean_entropy = weighted_mean(bits, weight)
        fg, bg = region_stats(bits, roi, self.roi_threshold)
        return mean_entropy, fg, bg

    @property
    def roi_threshold(self):
        return self.roi_cfg.threshold if self.roi_cfg is not None else 0.5

    def multi_scale_feature_extractor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor_P(dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scales = ckpt["y_q_scale"]
        mv_y_q_scales = ckpt["mv_y_q_scale"]
        return y_q_scales.reshape(-1), mv_y_q_scales.reshape(-1)

    def get_curr_mv_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.mv_y_q_basic, 0.5)
        return q_basic * q_scale

    def get_curr_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.y_q_basic, 0.5)
        return q_basic * q_scale 

    def encode_decode(self, x, dpb, output_path=None,
                        pic_width=None, pic_height=None,
                        mv_y_q_scale=None, y_q_scale=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            mv_y_q_scale, mv_y_q_index = get_rounded_q(mv_y_q_scale)
            y_q_scale, y_q_index = get_rounded_q(y_q_scale)

            encoded = self.compress(x, dpb,
                                    mv_y_q_scale, y_q_scale)
            encode_p(encoded['bit_stream'], mv_y_q_index, y_q_index, output_path)
            bits = filesize(output_path) * 8
            mv_y_q_index, y_q_index, string = decode_p(output_path)

            start = time.time()
            decoded = self.decompress(dpb, string,
                                        pic_height, pic_width,
                                        mv_y_q_index / 100, y_q_index / 100)
            decoding_time = time.time() - start
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "decoding_time": decoding_time,
            }
            return result

        encoded = self.forward_one_frame(x, dpb,
                                            mv_y_q_scale=mv_y_q_scale, y_q_scale=y_q_scale)
        result = {
            "dpb": encoded['dpb'],
            "bit_y": encoded['bit_y'].item(),
            "bit_z": encoded['bit_z'].item(),
            "bit_mv_y": encoded['bit_mv_y'].item(),
            "bit_mv_z": encoded['bit_mv_z'].item(),
            "bit": encoded['bit'].item(),
            "decoding_time": 0,
            "lpips_alexnet": encoded["lpips_alexnet"].item(),
            "lpips_swin": encoded["lpips_swin"].item(),
            "lpips_cnn": encoded["lpips_cnn"].item(),
            "lpips_dinov2": encoded["lpips_dinov2"].item(),
            "conditional_entropy": encoded["conditional_entropy"].item(), 
        }
        return result
        
    def format_convert_lpips_mask2former(self, x):
        """Map x in [0, 1] to the Mask2Former input convention.

        'legacy' is the scaling in the released code: x is multiplied by 255 twice,
        which puts the input ~255x outside the range implied by the config's
        PIXEL_MEAN/PIXEL_STD ([123.675, ...] / [58.395, ...]).  'fixed' applies the
        single scaling those statistics expect.  Both are kept because the BiEC
        targets come from this path, so the choice changes what the codec is aligned
        to and has to be measured rather than assumed.
        """
        x = x * 255.0
        if self.m2f_norm == 'fixed':
            return (x - self.semantic_model.pixel_mean) / self.semantic_model.pixel_std
        return (x * 255.0 - self.semantic_model.pixel_mean) / self.semantic_model.pixel_std

    def forward_one_frame(self, x, dpb, mv_y_q_scale=None, y_q_scale=None, lmd_index=None, roi=None,
                          y_in=None, mv_y_in=None, return_latents=False):
        """One P-frame.

        ``y_in`` / ``mv_y_in`` replace the corresponding *encoder outputs* (after
        the q-scale division, before quantisation).  When given, that encoder is
        skipped entirely.  This is what lets secvcm/lrdo.py treat the latent as a
        free variable at encode time; leave them None and nothing changes.

        ``return_latents`` adds ``y_latent`` / ``mv_y_latent`` to the result so a
        caller can seed such an optimisation.  It is off by default because
        train/video_train.py averages every key of this dict across frames.
        """
        ref_frame = dpb["ref_frame"]
        # add lmd_index
        if lmd_index is None:
            curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
            curr_y_q = self.get_curr_y_q(y_q_scale)
        else:
            curr_mv_y_q = self.get_curr_mv_y_q(self.mv_y_q_scale[lmd_index])
            curr_y_q = self.get_curr_y_q(self.y_q_scale[lmd_index])

        if mv_y_in is None:
            est_mv = self.optic_flow(x, ref_frame)
            mv_y = self.mv_encoder(est_mv)
            mv_y = mv_y / curr_mv_y_q
        else:
            mv_y = mv_y_in
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat)

        if y_in is None:
            y = self.contextual_encoder(x, context1, context2, context3)
            y = y / curr_y_q
        else:
            y = y_in
        z = self.contextual_hyper_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)

        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        if self.skip_semantic:
            # Stage 1: the semantic branch is not trained and its teachers are absent.
            semantic_image_feature = feature
            semantic_image = recon_image
            out2 = out4 = out8 = out16 = None
        else:
            semantic_image_feature, out2, out4, out8, out16  = self.semantic_decoder(y_hat, context2, context3)
            _, semantic_image = self.semantic_generation_net(semantic_image_feature, context1, feature)

        # distortion loss
        B, _, H, W = x.size()
        pixel_num = H * W

        # ROI weight map: unit mean, so enabling it changes *where* the semantic
        # objectives look, not how strongly they pull overall.
        roi_w = build_weight_map(roi, self.roi_cfg) if self.roi_cfg is not None else None

        mse = self.mse(x, recon_image)
        ssim = self.ssim(x, recon_image)
        me_mse = self.mse(x, warp_frame)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
        me_mse = torch.sum(me_mse, dim=(1, 2, 3)) / pixel_num

        if self.skip_semantic:
            mse_semantic = torch.zeros_like(mse)
            ssim_semantic = torch.zeros_like(ssim)
            mse_semantic_fg = mse_semantic_bg = mse.detach().new_zeros(())
        else:
            mse_semantic_map = self.mse(x, semantic_image)
            ssim_semantic = self.ssim(x, semantic_image)
            mse_semantic_fg, mse_semantic_bg = region_stats(mse_semantic_map, roi, self.roi_threshold)
            if self.roi_cfg is not None and self.roi_cfg.wants('mse'):
                mse_semantic_map = weighted_pixel_sum(mse_semantic_map, roi_w)
            mse_semantic = torch.sum(mse_semantic_map, dim=(1, 2, 3)) / pixel_num

        def renorm_for_alexnet_lpips(x):
            return torch.clip(x * 2 - 1, min=-1.0, max=1.0)
        self.alexnet_model.eval()
        lpips_alexnet = self.alexnet_model(renorm_for_alexnet_lpips(x), renorm_for_alexnet_lpips(recon_image))

        # Which terms the ROI map re-weights.  None => plain mean, i.e. the baseline.
        w_swin = roi_w if (self.roi_cfg is not None and self.roi_cfg.wants('swin')) else None
        w_cnn = roi_w if (self.roi_cfg is not None and self.roi_cfg.wants('cnn')) else None
        w_dino = roi_w if (self.roi_cfg is not None and self.roi_cfg.wants('dino')) else None
        w_biec = roi_w if (self.roi_cfg is not None and self.roi_cfg.wants('biec')) else None

        if self.use_semantic:
            self.dinov2_model.eval()
            self.dinov2_model.backbone.eval()
            perception_features_dinov2 = self.dinov2_model(x)
            perception_features_dinov2_hat = self.dinov2_model(semantic_image)
            dino_grid = dino_grid_size(H, W)
            mse_dinov2_coarse = weighted_mean_tokens(self.mse(perception_features_dinov2['coarse'], perception_features_dinov2_hat['coarse']), w_dino, dino_grid)
            mse_dinov2_fine = weighted_mean_tokens(self.mse(perception_features_dinov2['fine'], perception_features_dinov2_hat['fine']), w_dino, dino_grid)
            lpips_dinov2 = (mse_dinov2_coarse + mse_dinov2_fine) / 2.0
        else:
            lpips_dinov2 = torch.tensor(0).to(mse.device)

        if self.use_semantic:
            self.resnet18_model.eval()
            self.resnet18_model.backbone.eval()
            perception_features_cnn = self.resnet18_model(x)
            perception_features_cnn_hat = self.resnet18_model(semantic_image)
            mse_cnn_res2 = weighted_mean(self.mse(perception_features_cnn['res2'], perception_features_cnn_hat['res2']), w_cnn)
            mse_cnn_res3 = weighted_mean(self.mse(perception_features_cnn['res3'], perception_features_cnn_hat['res3']), w_cnn)
            mse_cnn_res4 = weighted_mean(self.mse(perception_features_cnn['res4'], perception_features_cnn_hat['res4']), w_cnn)
            mse_cnn_res5 = weighted_mean(self.mse(perception_features_cnn['res5'], perception_features_cnn_hat['res5']), w_cnn)
            lpips_cnn = (mse_cnn_res2 + mse_cnn_res3) / 2.0
        else:
            lpips_cnn = torch.tensor(0).to(mse.device)


        if self.use_semantic:
            self.semantic_model.eval()
            self.semantic_model.backbone.eval()
            perception_features_swin = self.semantic_model.backbone(self.format_convert_lpips_mask2former(x))
            perception_features_swin_hat = self.semantic_model.backbone(self.format_convert_lpips_mask2former(semantic_image))
            swin_res2_err = self.mse(perception_features_swin['res2'], perception_features_swin_hat['res2'])
            mse_swin_res2 = weighted_mean(swin_res2_err, w_swin)
            mse_swin_res3 = weighted_mean(self.mse(perception_features_swin['res3'], perception_features_swin_hat['res3']), w_swin)
            mse_swin_res4 = weighted_mean(self.mse(perception_features_swin['res4'], perception_features_swin_hat['res4']), w_swin)
            mse_swin_res5 = weighted_mean(self.mse(perception_features_swin['res5'], perception_features_swin_hat['res5']), w_swin)
            lpips_swin = (mse_swin_res2 + mse_swin_res3) / 2.0
            lpips_swin_fg, lpips_swin_bg = region_stats(swin_res2_err, roi, self.roi_threshold)
        else:
            lpips_swin = torch.tensor(0).to(mse.device)
            lpips_swin_fg = lpips_swin_bg = mse.detach().new_zeros(())

        # conditional entropy loss
        if self.use_semantic:
            means4, scales4 = self.distribution_generation4(perception_features_swin["res2"])
            means8, scales8 = self.distribution_generation8(perception_features_swin["res3"])
            means16, scales16 = self.distribution_generation16(perception_features_swin["res4"])
            entropy4, fg4, bg4 = self.get_conditional_entropy(out4, means4, scales4, w_biec, roi)
            entropy8, fg8, bg8 = self.get_conditional_entropy(out8, means8, scales8, w_biec, roi)
            entropy16, fg16, bg16 = self.get_conditional_entropy(out16, means16, scales16, w_biec, roi)

            means4_reverse, scales4_reverse = self.distribution_generation4_reverse(out4)
            means8_reverse, scales8_reverse = self.distribution_generation8_reverse(out8)
            means16_reverse, scales16_reverse = self.distribution_generation16_reverse(out16)
            entropy4_reverse, _, _ = self.get_conditional_entropy(perception_features_swin["res2"], means4_reverse, scales4_reverse, w_biec)
            entropy8_reverse, _, _ = self.get_conditional_entropy(perception_features_swin["res3"], means8_reverse, scales8_reverse, w_biec)
            entropy16_reverse, _, _ = self.get_conditional_entropy(perception_features_swin["res4"], means16_reverse, scales16_reverse, w_biec)

            entropy = (entropy4 + entropy8 + entropy16 + entropy4_reverse + entropy8_reverse + entropy16_reverse) / 6.0
            # Diagnostics: the mechanism claim is "foreground entropy drops, background
            # entropy is allowed to rise".  Averaged over the three forward scales.
            entropy_fg = (fg4 + fg8 + fg16) / 3.0
            entropy_bg = (bg4 + bg8 + bg16) / 3.0
        else:
            entropy4 = torch.tensor(0).to(mse.device)
            entropy8 = torch.tensor(0).to(mse.device)
            entropy16 = torch.tensor(0).to(mse.device)
            entropy4_reverse = torch.tensor(0).to(mse.device)
            entropy8_reverse = torch.tensor(0).to(mse.device)
            entropy16_reverse = torch.tensor(0).to(mse.device)
            entropy = torch.tensor(0).to(mse.device)
            entropy_fg = entropy_bg = mse.detach().new_zeros(())
        
        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        result = {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "me_mse": me_mse,
                "mse": mse,
                "mse_semantic": mse_semantic,
                "ssim": ssim,
                "ssim_semantic": ssim_semantic, 
                "lpips_alexnet": lpips_alexnet, 
                "lpips_swin": lpips_swin,
                "lpips_cnn": lpips_cnn,
                "lpips_dinov2": lpips_dinov2, 
                "conditional_entropy": entropy, 
                "conditional_entropy_4": entropy4,
                "conditional_entropy_8": entropy8,
                "conditional_entropy_16": entropy16,
                "conditional_entropy_4_reverse": entropy4_reverse,
                "conditional_entropy_8_reverse": entropy8_reverse,
                "conditional_entropy_16_reverse": entropy16_reverse,
                # ROI diagnostics (detached, zero when no ROI map is supplied)
                "entropy_fg": entropy_fg,
                "entropy_bg": entropy_bg,
                "lpips_swin_fg": lpips_swin_fg,
                "lpips_swin_bg": lpips_swin_bg,
                "mse_semantic_fg": mse_semantic_fg,
                "mse_semantic_bg": mse_semantic_bg,
                "dpb": {
                    "ref_frame": recon_image,
                    "ref_frame_semantic": semantic_image,
                    "ref_feature": feature,
                    "ref_feature_semantic": semantic_image_feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }
        if return_latents:
            # Pre-quantisation encoder outputs, i.e. the variables secvcm/lrdo.py
            # optimises. Opt-in: video_train.cal_avg_result averages every key.
            result["y_latent"] = y
            result["mv_y_latent"] = mv_y
            # The quantised symbols themselves. Comparing these before and after
            # latent optimisation says how much was actually re-coded, which is the
            # difference between "LRDO does not help" and "LRDO never moved".
            result["y_q"] = y_q
        return result

    def forward(self, x, dpb, mv_y_q_scale=None, y_q_scale=None, lmd_index=None, frame_idx=None, roi=None):
        return self.forward_one_frame(x, dpb, mv_y_q_scale=mv_y_q_scale, y_q_scale=y_q_scale,
                                      lmd_index=lmd_index, roi=roi)
