import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

out_channel_N = 64
out_channel_M = 96
out_channel_mv = 128

vimeo_root = os.environ.get("VIMEO_ROOT", "/path/to/vimeo/sequences/")
train_lst = os.environ.get("VIMEO_TRAIN_LIST", "/path/to/vimeo/sep_trainlist.txt")
valid_lst = os.environ.get("VIMEO_VALID_LIST", "/path/to/vimeo/sep_testlist.txt")
valid_lst_short = os.environ.get("VIMEO_VALID_LIST_SHORT", "/path/to/vimeo/sep_testlist.txt")

youhq_root = os.environ.get("YOUHQ_ROOT", "/path/to/youhq_png3/")
youhq_train_lst = os.environ.get("YOUHQ_TRAIN_LIST", "/path/to/youhq_png3/youhq_trainlist.txt")
youhq_root_small = os.environ.get("YOUHQ_ROOT_SMALL", "/path/to/youhq_png3_resized/")
youhq_root_mid = os.environ.get("YOUHQ_ROOT_MID", "/path/to/youhq_png3_resized/")

bvidvc_train_lst = "/e2edataset/BVI-DVC/train_list.txt"
bvidvc_root = "/e2edataset/BVI-DVC/png"

# Root of the precomputed ROI maps (see scripts/precompute_roi_masks.py).  The
# tree mirrors the frame tree exactly, one 8-bit grayscale PNG per frame.
roi_root = os.environ.get("ROI_ROOT", "")

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

def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def get_numbers(pick_num, total_num=7, isSkip=True, isReverse=True):
    arr = range(total_num)

    # reverse data augmentation
    if isReverse and random.randint(0, 1) > 0:
        arr = arr[::-1]
    
    if pick_num == total_num:
        return arr
    current_index = random.randint(0, total_num - pick_num)
    if not isSkip:
        return arr[current_index : current_index + pick_num]
    
    # The following algorithm makes it more likely that later frames will be more compact. Therefore, there is a reverse flag that could solve it.
    isInverse = True if random.randint(0, 1) > 0 else False 
    if isInverse:
        arr = arr[::-1]
    assert pick_num <= total_num
    
    result = []
    result.append(arr[current_index])    
    for i in range(1, pick_num):
        # Choose the next index as either the current_index + 1 or current_index + 2
        if total_num - current_index - 1  <= pick_num - i:
            step = 1
        else:
            step = random.choice([1, 2])
        next_index = current_index + step
        
        result.append(arr[next_index])
        current_index = next_index

    if isInverse:
        result = result[::-1]
    return result        


    
#     # augmentation setting....





    


            
            

    

    



  
            
            

    
    



  
            
            

    

    





    

#                 image_lists[i].append(y + "/im" + str(numbers[i]+1) + '.png')






class DataSet_Base_mp4(data.Dataset):
    def __init__(self, video_dir=None, im_height=256, im_width=256, num_frames=2, isStep=True):
        self.video_dir = video_dir
        self.video_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(video_dir)
            for f in files if f.endswith('.mp4')
        ]
        self.transform = transforms.Compose([
            transforms.RandomCrop((im_height, im_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.total_frames = 30
        self.get_numbers = get_numbers
        self.num_frames = min(num_frames, self.total_frames)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        
        # Use OpenCV to read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        numbers = self.get_numbers(self.num_frames, self.total_frames)
        for i in range(self.total_frames):
            if i not in numbers:
                cap.grab()
                continue
            ret, frame = cap.read()
            if not ret:
                break                
            frames.append(frame[:, :, ::-1])  # BGR to RGB
        assert len(frames) == self.num_frames
        
        cap.release()
        
        frames = np.stack(frames)
        frames = torch.from_numpy(frames)  # [T, H, W, C=3]
        frames = frames.to(torch.float32) / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous() # [T, C=3, H, W]
        T,C,H,W = frames.shape
        frames = frames.view(-1, H, W)
        frames = self.transform(frames)
        return frames

class DataSet_Base(data.Dataset):
    def __init__(self, im_height, im_width, transform_scale=None):
        self.transform = transforms.Compose([
            transforms.RandomCrop((im_height, im_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.num_threads = 8
        # ROI support (off unless enable_roi is called)
        self.roi_root = ""
        self.roi_missing = "error"
        self.data_roots = []

    def enable_roi(self, root, data_roots, missing="error"):
        """Return a per-frame ROI map alongside every clip.

        ``data_roots`` are the frame-tree roots this dataset draws from; the ROI
        tree mirrors them below ``root``.  ``missing`` selects what happens when a
        map is absent: 'error' (default, so a partial precompute run cannot quietly
        turn into a half-weighted experiment) or 'ones' (uniform, i.e. no ROI for
        that frame).
        """
        assert missing in ("error", "ones"), missing
        if not root:
            raise ValueError("ROI enabled but no ROI root given (set ROI_ROOT or --roi_root).")
        self.roi_root = root
        self.roi_missing = missing
        self.data_roots = [r for r in data_roots if r]
        return self

    @property
    def roi_enabled(self):
        return bool(self.roi_root)

    def roi_path_of(self, image_path):
        p = os.path.normpath(image_path)
        for root in self.data_roots:
            root_n = os.path.normpath(root)
            if p.startswith(root_n + os.sep):
                return os.path.join(self.roi_root, os.path.relpath(p, root_n))
        raise ValueError(f"cannot map '{image_path}' to an ROI path; data roots are {self.data_roots}")

    def __len__(self):
        return len(self.image_lists[0])


    def __getitem__(self, index):
        def load_image(frame_index):
            image = Image.open(self.image_lists[frame_index][index]).convert('RGB')
            return to_tensor(image)

        def load_roi(frame_index):
            path = self.roi_path_of(self.image_lists[frame_index][index])
            if not os.path.exists(path):
                if self.roi_missing == "error":
                    raise FileNotFoundError(
                        f"missing ROI map '{path}'. Run scripts/precompute_roi_masks.py over the "
                        f"whole split, or pass --roi_missing ones to fall back to uniform weights.")
                return torch.ones_like(images[frame_index][:1])
            return to_tensor(Image.open(path).convert('L'))

        to_tensor = transforms.ToTensor()
        images = [None] * self.num_frames  # Pre-allocate a fixed-size list

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_frame_index = {executor.submit(load_image, frame_index): frame_index for frame_index in range(self.num_frames)}

            for future in as_completed(future_to_frame_index):
                frame_index = future_to_frame_index[future]
                images[frame_index] = future.result()

        images_tensor = torch.cat(images, dim=0)

        if not self.roi_enabled:
            if self.transform:
                images_tensor = self.transform(images_tensor)
            return images_tensor

        rois = [None] * self.num_frames
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(load_roi, frame_index): frame_index for frame_index in range(self.num_frames)}
            for future in as_completed(futures):
                rois[futures[future]] = future.result()
        roi_tensor = torch.cat(rois, dim=0)

        # Crop and flip the frames and their ROI maps as one tensor, so the two can
        # never drift apart under the random augmentation.
        stacked = torch.cat([images_tensor, roi_tensor], dim=0)
        if self.transform:
            stacked = self.transform(stacked)
        split = 3 * self.num_frames
        return stacked[:split], stacked[split:]

class DataSet_youhq(DataSet_Base):
    def __init__(self, mode="train", scale="original", im_height=256, im_width=256, num_frames=2, isStep=True):
        super().__init__(im_height, im_width, transform_scale=(1., 1.)) # changed
        self.mode = mode
        self.path = youhq_train_lst
        self.num_frames = num_frames
        self.total_frames = 30
        self.get_numbers = get_numbers
        self.isStep = isStep

        if scale == "original":   data_root = [youhq_root]
        elif scale == "mid":    data_root = [youhq_root_mid]
        elif scale == "small":  data_root = [youhq_root_small]
        elif scale == "mixed":  data_root = [youhq_root, youhq_root_mid, youhq_root_small]
        else: raise TypeError
        self.image_lists = self.get_video(rootdir=data_root, filefolderlist=youhq_train_lst) # changed
        print("dataset find image: ", len(self.image_lists[0]))

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((im_height, im_width), scale=(1., 1./16), ratio=(1, 1)),
            transforms.RandomCrop((im_height, im_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    
    def get_video(self, rootdir, filefolderlist):
        with open(filefolderlist) as f:
            data = f.readlines()
    
        image_lists = [[] for _ in range(self.num_frames)]

        for line in data:
            line = line.split()[0]
            rootdir_sample = random.choice(rootdir)
            y = os.path.join(rootdir_sample, line.rstrip())
            numbers = get_numbers(self.num_frames, self.total_frames, self.isStep)
            for i in range(self.num_frames):
                image_lists[i].append(y + "/frame_" + str(numbers[i]+1).zfill(4) + '.png')

        return image_lists

class DataSet_vimeo(DataSet_Base):
    def __init__(self, mode="train", im_height=256, im_width=256, num_frames=2, isStep=True, transform_scale=(1., 1.)):
        super().__init__(im_height, im_width, transform_scale=transform_scale)
        if mode == "train":
            self.path = train_lst
        elif mode == "valid":
            self.path = valid_lst_short
        else:
            raise TypeError("Should be train or valid!")
        self.mode = mode
        self.total_frames = 7
        self.num_frames = num_frames
        assert num_frames <= 7, f"Error, num_frames is {num_frames}, which is larger than 7. "

        self.get_numbers = get_numbers
        self.isStep = isStep
        
        self.image_lists = self.get_video(vimeo_root, filefolderlist=self.path)

        if mode == "valid":
            self.transform = transforms.Compose([
                transforms.CenterCrop((im_height, im_width)),
            ])
    
    def get_video(self, rootdir, filefolderlist):
        with open(filefolderlist) as f:
            import random; random.seed(233)
            data = f.readlines()
            random.shuffle(data)
            data = data[:64608]
            print("Randomly select 64608 videos from vimeo-plus dataset")
    
        image_lists = [[] for _ in range(self.num_frames)]

        for line in data:
            y = os.path.join(rootdir, line.rstrip())
            if self.mode == "train":
                numbers = self.get_numbers(self.num_frames, self.total_frames, self.isStep)
            else:
                numbers = range(self.num_frames)
            for i in range(self.num_frames):
                image_lists[i].append(y + "/im" + str(numbers[i]+1) + '.png')

        return image_lists

class DataSet_bvidvc(DataSet_Base):
    def __init__(self, mode="train", scale="original", num_scale=20, im_height=256, im_width=256, num_frames=2, isStep=True):
        super().__init__(im_height, im_width, transform_scale=(1., 1.)) # changed
        self.mode = mode
        self.num_frames = num_frames
        self.total_frames = 64
        self.get_numbers = get_numbers
        self.isStep = isStep


        self.image_lists = self.get_video(rootdir=bvidvc_root, filefolderlist=bvidvc_train_lst) # changed
        self.image_lists = [x * num_scale for x in self.image_lists]
        print("dataset find image: ", len(self.image_lists[0]))

        if scale == "original":
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop((im_height, im_width)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            if scale == "small":
                size = 256
            elif scale == "mid":
                size = 512
            else:
                raise TypeError
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop((im_height, im_width)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
    
    def get_video(self, rootdir, filefolderlist):
        with open(filefolderlist) as f:
            data = f.readlines()
    
        image_lists = [[] for _ in range(self.num_frames)]

        for line in data:
            line = line.split()[0]
            rootdir_sample = rootdir
            y = os.path.join(rootdir_sample, line.rstrip())
            numbers = get_numbers(self.num_frames, self.total_frames, self.isStep)
            for i in range(self.num_frames):
                image_lists[i].append(y + "/frame_" + str(numbers[i]+1).zfill(4) + '.png')

        return image_lists


def read_image_one_HQ(path, freeSizeX, freeSizeY, turn_1, turn_2, perm):
    image_pil = Image.open(path)
    new_width = int(image_pil.width * 0.25)
    new_height = int(image_pil.height * 0.25)
    image_pil_resize = image_pil.resize((new_width, new_height), Image.LANCZOS)
    image = np.array(image_pil_resize)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float()

    maxFreeSizeX = image.shape[2] - 256
    maxFreeSizeY = image.shape[1] - 256
    freeSizeX = min(freeSizeX, maxFreeSizeX)
    freeSizeY = min(freeSizeY, maxFreeSizeY)


    image = random_crop(image, freeSizeX, freeSizeY)
    image = random_flip(image, turn_1, turn_2)

    return image

def random_crop(img, freeSizeX, freeSizeY):
    img_crop = img[:, freeSizeY:freeSizeY + 256, freeSizeX:freeSizeX + 256]
    return img_crop

def random_flip(images, turn_1, turn_2):
    if turn_1 == 1:
        images = torch.flip(images, [1])
    if turn_2 == 1:
        images = torch.flip(images, [2])

    return images





#                 image_list[i].append(y + "/frame_" + str(numbers).zfill(4) + '.png')





