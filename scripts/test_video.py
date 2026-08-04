# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json, sys
import multiprocessing
import time

import torch, torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image
from secvcm.models.video_model import DMC
from secvcm.models.image_model import IntraNoAR
from secvcm.lrdo import LrdoConfig, optimize_frame
from secvcm.utils.common import str2bool, interpolate_log, create_folder, generate_log_json, dump_json
from secvcm.utils.stream_helper import get_padding_size, get_state_dict
from secvcm.utils.png_reader import PNGReader
from tqdm import tqdm
from pytorch_msssim import ms_ssim

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.video_train import build_mask2former, build_dinov2


_TEACHERS = None


def load_teachers():
    """Build the three teachers, once per process.

    They are only needed to report the lpips_*/entropy diagnostics: the decoded
    frames are identical without them.  Loading them lazily (instead of at import
    time, as in the original script) means --inference_mode can run the codec with
    no detectron2, no Mask2Former checkpoint and no DINOv2 checkpoint.
    """
    global _TEACHERS
    if _TEACHERS is None:
        mask2former_model = build_mask2former()
        print("Mask2Former model loaded successfully.")
        resnet18_model = torchvision.models.resnet18(pretrained=True)
        print("ResNet-18 model loaded successfully.")
        dino_model = build_dinov2()
        print("DINOv2 model loaded successfully.")
        _TEACHERS = (mask2former_model, resnet18_model, dino_model)
    return _TEACHERS


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--i_frame_q_scales', type=float, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--model_path',  type=str)
    parser.add_argument('--p_frame_y_q_scales', type=float, nargs="+")
    parser.add_argument('--p_frame_mv_y_q_scales', type=float, nargs="+")
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--inference_mode', type=str2bool, nargs='?', const=True, default=True,
                        help="Run the codec without the teacher models. Decoded frames are "
                             "identical; the lpips_*/entropy diagnostics are reported as 0. "
                             "Set False to reproduce the released script's behaviour.")
    parser.add_argument('--m2f_norm', type=str, default='legacy', choices=['legacy', 'fixed'],
                        help="Mask2Former input scaling used for the diagnostics; must match "
                             "the setting the checkpoint was trained with.")
    LrdoConfig.add_argparse_args(parser)

    args = parser.parse_args()
    return args


class RoiReader:
    """Reads per-frame ROI maps written by scripts/precompute_roi_masks.py.

    Mirrors PNGReader's naming (im1.png / im00001.png) so an ROI tree produced
    for a frame tree lines up without any extra bookkeeping.  A missing map is
    not an error: it yields None, which makes that frame plain LRDO.
    """

    def __init__(self, roi_folder, padding):
        self.roi_folder = roi_folder
        self.padding = padding
        self.missing = 0

    def read(self, frame_index, device):
        path = os.path.join(self.roi_folder,
                            f"im{str(frame_index + 1).zfill(self.padding)}.png")
        if not os.path.exists(path):
            self.missing += 1
            return None
        roi = np.asarray(Image.open(path).convert('L')).astype('float32') / 255.0
        return torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).to(device)


def read_image_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    print("save to path:", save_path)
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def run_test(video_net, i_frame_net, args, device):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0

    if args['src_type'] == 'png':
        print(args['img_path'])
        src_reader = PNGReader(args['img_path'], args['src_width'], args['src_height'])

    # Encode-time latent RDO. Off unless --lrdo was passed, in which case P-frames
    # go through secvcm/lrdo.py instead of a single encoder pass.
    lrdo_cfg = args.get('lrdo_cfg')
    lrdo_on = lrdo_cfg is not None and lrdo_cfg.enabled
    roi_reader = None
    if lrdo_on:
        assert not write_stream, (
            "--lrdo cannot be combined with --write_stream: this repo's video model "
            "has no compress/decompress, so all bits are entropy estimates.")
        if args.get('lrdo_roi_dir'):
            roi_reader = RoiReader(os.path.join(args['lrdo_roi_dir'], args['video_path']),
                                   src_reader.padding)
    lrdo_bpp_before = []
    lrdo_bpp_after = []
    lrdo_frac_changed = []
    lrdo_mean_abs_dy = []

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    frame_pixel_num = 0
    recon_lpips_alexnet = []
    semantic_entropy = []
    semantic_lpips_cnn = []
    semantic_lpips_swin = []
    semantic_lpips_dino = []
    semantic_psnrs = []
    semantic_msssims = []

    start_time = time.time()
    p_frame_number = 0
    overall_p_decoding_time = 0
    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            rgb = src_reader.read_one_frame(src_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            bin_path = os.path.join(args['bin_folder'], f"{frame_idx}.bin") \
                if write_stream else None

            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'], bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                recon_semantic_frame = recon_frame
                frame_types.append(0)
                bits.append(result["bit"])
            else:
                if lrdo_on:
                    roi = roi_reader.read(frame_idx, device) if roi_reader is not None else None
                    if roi is not None:
                        # Pad in lockstep with the frame. The padded strip carries no
                        # content, so it is background as far as the ROI is concerned.
                        roi = torch.nn.functional.pad(
                            roi, (padding_l, padding_r, padding_t, padding_b),
                            mode="constant", value=0.0)
                    raw, lrdo_stats = optimize_frame(
                        video_net, x_padded, dpb, lrdo_cfg, roi=roi,
                        mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                        y_q_scale=args['p_frame_y_q_scale'],
                        verbose=verbose >= 2)
                    result = {
                        "dpb": raw["dpb"],
                        "bit": raw["bit"].item(),
                        "bit_y": raw["bit_y"].item(),
                        "bit_z": raw["bit_z"].item(),
                        "bit_mv_y": raw["bit_mv_y"].item(),
                        "bit_mv_z": raw["bit_mv_z"].item(),
                        "decoding_time": 0,
                        "lpips_alexnet": raw["lpips_alexnet"].item(),
                        "lpips_swin": raw["lpips_swin"].item(),
                        "lpips_cnn": raw["lpips_cnn"].item(),
                        "lpips_dinov2": raw["lpips_dinov2"].item(),
                        "conditional_entropy": raw["conditional_entropy"].item(),
                    }
                    lrdo_bpp_before.append(lrdo_stats['bpp_before'])
                    lrdo_bpp_after.append(lrdo_stats['bpp_after'])
                    lrdo_frac_changed.append(lrdo_stats['frac_symbols_changed'])
                    lrdo_mean_abs_dy.append(lrdo_stats['mean_abs_dy'])
                else:
                    result = video_net.encode_decode(x_padded, dpb, bin_path,
                                                     pic_height=pic_height, pic_width=pic_width,
                                                     mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                                                     y_q_scale=args['p_frame_y_q_scale'])
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                recon_semantic_frame = dpb["ref_frame_semantic"]
                
                frame_types.append(1)
                bits.append(result['bit'])
                p_frame_number += 1
                overall_p_decoding_time += result['decoding_time']

                # Feature-based metrics
                recon_lpips_alexnet.append(result['lpips_alexnet'])
                semantic_entropy.append(result['conditional_entropy'])
                semantic_lpips_cnn.append(result['lpips_cnn'])
                semantic_lpips_swin.append(result['lpips_swin'])
                semantic_lpips_dino.append(result['lpips_dinov2'])

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            recon_semantic_frame = recon_semantic_frame.clamp_(0, 1)
            x_semantic_hat = F.pad(recon_semantic_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            
            # psnr and ms-ssim of reconstructed frame
            psnr = PSNR(x_hat, x)
            msssim = ms_ssim(x_hat, x, data_range=1).item()
            psnrs.append(psnr)
            msssims.append(msssim)

            # PSNR and MS-SSIM of semantic frame
            semantic_psnr = PSNR(x_semantic_hat, x)
            semantic_msssim = ms_ssim(x_semantic_hat, x, data_range=1).item()
            semantic_psnrs.append(semantic_psnr)
            semantic_msssims.append(semantic_msssim)

            frame_end_time = time.time()


            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds,",
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1]:.4f}, MS-SSIM: {msssims[-1]:.4f} ")

            if save_decoded_frame:
                save_path = os.path.join(args['decoded_frame_folder'], f'{frame_idx}.png')
                save_torch_image(x_semantic_hat, save_path)

    test_time = time.time() - start_time
    if verbose >= 1 and p_frame_number > 0:
        print(f"decoding {p_frame_number} P frames, "
              f"average {overall_p_decoding_time/p_frame_number * 1000:.0f} ms.")

    other_info = {
        'recon_lpips_alexnet': recon_lpips_alexnet, 
        'semantic_psnrs': semantic_psnrs,
        'semantic_msssims': semantic_msssims,
        'semantic_entropy': semantic_entropy,
        'semantic_lpips_cnn': semantic_lpips_cnn, 
        'semantic_lpips_swin': semantic_lpips_swin,
        'semantic_lpips_dino': semantic_lpips_dino,
    }
    if lrdo_on:
        # Per-frame bpp before and after the latent optimisation: the direct
        # evidence that LRDO moved bits at all.
        other_info['lrdo_bpp_before'] = lrdo_bpp_before
        other_info['lrdo_bpp_after'] = lrdo_bpp_after
        # Fraction of coded symbols the optimisation actually changed, and how far
        # the latent travelled. Near-zero means the step budget never crossed a
        # quantisation boundary, not that the method fails.
        other_info['lrdo_frac_symbols_changed'] = lrdo_frac_changed
        other_info['lrdo_mean_abs_dy'] = lrdo_mean_abs_dy
        other_info['lrdo_iters'] = lrdo_cfg.iters
        other_info['lrdo_lr'] = lrdo_cfg.lr
        if roi_reader is not None and roi_reader.missing:
            print(f"warning: {roi_reader.missing} ROI maps missing under "
                  f"{roi_reader.roi_folder}; those frames ran as plain LRDO")

    log_result = generate_log_json(frame_num, frame_types, bits, psnrs, msssims,
                                   frame_pixel_num, test_time, other_info)
    return log_result


def encode_one(args, device):
    i_state_dict = get_state_dict(args['i_frame_model_path'])
    i_frame_net = IntraNoAR()
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    if args['force_intra']:
        video_net = None
    else:
        p_state_dict = get_state_dict(args['model_path'])

        if args.get('inference_mode', True):
            video_net = DMC(inference_mode=True, m2f_norm=args.get('m2f_norm', 'legacy'))
        else:
            mask2former_model, resnet18_model, dino_model = load_teachers()
            video_net = DMC(inference_mode=False, cnn_model=resnet18_model, swin_model=mask2former_model,
                            dino_model=dino_model, m2f_norm=args.get('m2f_norm', 'legacy'))
        video_net.load_state_dict(p_state_dict, strict=False)
        video_net = video_net.to(device)
        video_net.eval()

    if args['write_stream']:
        if video_net is not None:
            video_net.update(force=True)
        i_frame_net.update(force=True)

    sub_dir_name = args['video_path']
    gop_size = args['gop']
    frame_num = args['frame_num']

    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args['decoded_frame_path'], sub_dir_name, str(args['rate_idx']))
        create_folder(decoded_frame_folder)
    else:
        decoded_frame_folder = None

    
    args['img_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['gop_size'] = gop_size
    args['frame_num'] = frame_num
    args['bin_folder'] = bin_folder
    args['decoded_frame_folder'] = decoded_frame_folder

    result = run_test(video_net, i_frame_net, args, device=device)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(use_cuda, args):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    result = encode_one(args, device)
    return result


def main():
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args()

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)
    print("q_scales in intra ckpt: ", end='')
    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()
    if args.i_frame_q_scales is not None:
        assert len(args.i_frame_q_scales) == rate_num
        i_frame_q_scales = args.i_frame_q_scales
        print(f"testing {rate_num} rate points with pre-defined intra y q_scales: ", end='')
    elif len(i_frame_q_scales) == rate_num:
        print(f"testing {rate_num} rate points with intra y q_scales in ckpt: ", end='')
    else:
        max_q_scale = i_frame_q_scales[0]
        min_q_scale = i_frame_q_scales[-1]
        i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_num)
        print(f"testing {rate_num} rates, using intra y q_scales: ", end='')

    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()

    if not args.force_intra:
        p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(args.model_path)
        print("y_q_scales in inter ckpt: ", end='')
        for q in p_frame_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scales in inter ckpt: ", end='')
        for q in p_frame_mv_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        if args.p_frame_y_q_scales is not None:
            assert len(args.p_frame_y_q_scales) == rate_num
            assert len(args.p_frame_mv_y_q_scales) == rate_num
            p_frame_y_q_scales = args.p_frame_y_q_scales
            p_frame_mv_y_q_scales = args.p_frame_mv_y_q_scales
            print(f"testing {rate_num} rate points with pre-defined inter q_scales")
        elif len(p_frame_y_q_scales) == rate_num:
            print(f"testing {rate_num} rate points with inter q_scales in ckpt")
        else:
            max_y_q_scale = p_frame_y_q_scales[0]
            min_y_q_scale = p_frame_y_q_scales[-1]
            p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_num)

            max_mv_y_q_scale = p_frame_mv_y_q_scales[0]
            min_mv_y_q_scale = p_frame_mv_y_q_scales[-1]
            p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_num)
        print("y_q_scales for testing: ", end='')
        for q in p_frame_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scales for testing: ", end='')
        for q in p_frame_mv_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['i_frame_model_path'] = args.i_frame_model_path
                cur_args['i_frame_q_scale'] = i_frame_q_scales[rate_idx]
                if not args.force_intra:
                    cur_args['model_path'] = args.model_path
                    cur_args['p_frame_y_q_scale'] = p_frame_y_q_scales[rate_idx]
                    cur_args['p_frame_mv_y_q_scale'] = p_frame_mv_y_q_scales[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                if args.force_intra:
                    cur_args['gop'] = 1
                if args.force_intra_period > 0:
                    cur_args['gop'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}_DMC_{rate_idx}'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['inference_mode'] = args.inference_mode
                cur_args['m2f_norm'] = args.m2f_norm
                # Built here rather than in the worker so lambda is picked from
                # --lrdo_lambdas by this rate point, mirroring training.
                cur_args['lrdo_cfg'] = LrdoConfig.from_args(args, rate_idx=rate_idx)
                cur_args['lrdo_roi_dir'] = args.lrdo_roi_dir

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(
                    worker,
                    args.cuda,
                    cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][f"{rate:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
