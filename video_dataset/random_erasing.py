# --------------------------------------------------------
# This file originates from: https://github.com/facebookresearch/SlowFast/blob/fee19d699c49a81f33b890c5ff592bbb11aa5c54/slowfast/datasets/random_erasing.py
# -------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
pulished under an Apache License 2.0.

COMMENT FROM ORIGINAL:
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng
Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
import torch
import os
import sys

import configs

gpu_id = configs.gpu_id
device = torch.device("cuda:" + str(gpu_id))

def _get_pixels(
    per_pixel, rand_color, patch_size, dtype=torch.float32, device=device
):
    """
    生成像素值用于擦除区域
    """
    try:
        # 尝试在 GPU 上生成
        if per_pixel:
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif rand_color:
            return torch.empty(
                (patch_size[0], 1, 1), dtype=dtype, device=device
            ).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
    except RuntimeError as e:
        # GPU 内存不足时回退到 CPU
        print(f"====> random_erasing.py: GPU 内存不足，回退到 CPU: {e}")
        if per_pixel:
            pixels = torch.empty(patch_size, dtype=dtype).normal_()
        elif rand_color:
            pixels = torch.empty((patch_size[0], 1, 1), dtype=dtype).normal_()
        else:
            pixels = torch.zeros((patch_size[0], 1, 1), dtype=dtype)
        return pixels.to(device)

class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1 / 3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        num_splits=0,
        device=device,  # 使用全局设备
        cube=True,
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.cube = cube
        
        # 模式处理
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        
        # 设备处理
        try:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
            # 验证设备
            torch.zeros(1).to(self.device)
        except Exception as e:
            print(f"====> random_erasing.py: 设备 '{device}' 无效，使用 CPU: {e}")
            self.device = torch.device("cpu")

    def _erase(self, img, chan, img_h, img_w, dtype):
        """擦除单个图像"""
        if random.random() > self.probability:
            return img
        
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        
        for _ in range(count):
            for _ in range(10):  # 最多尝试10次
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                
                if 0 < w < img_w and 0 < h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    
                    try:
                        # 尝试在 GPU 上擦除
                        img[:, top:top+h, left:left+w] = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device=self.device,
                        )
                    except RuntimeError as e:
                        # GPU 内存不足时回退到 CPU
                        print(f"====> random_erasing.py: GPU 擦除失败，回退到 CPU: {e}")
                        pixels = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device="cpu"
                        )
                        img[:, top:top+h, left:left+w] = pixels.to(img.device)
                    
                    break
        return img

    def _erase_cube(
        self,
        img,
        batch_start,
        batch_size,
        chan,
        img_h,
        img_w,
        dtype,
    ):
        """擦除一批图像（立方体模式）"""
        if random.random() > self.probability:
            return img
        
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        
        for _ in range(count):
            for _ in range(100):  # 最多尝试100次
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                
                if 0 < w < img_w and 0 < h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    
                    for i in range(batch_start, batch_size):
                        try:
                            # 尝试在 GPU 上擦除
                            img[i, :, top:top+h, left:left+w] = _get_pixels(
                                self.per_pixel,
                                self.rand_color,
                                (chan, h, w),
                                dtype=dtype,
                                device=self.device,
                            )
                        except RuntimeError as e:
                            # GPU 内存不足时回退到 CPU
                            print(f"====> random_erasing.py: GPU 擦除失败，回退到 CPU: {e}")
                            pixels = _get_pixels(
                                self.per_pixel,
                                self.rand_color,
                                (chan, h, w),
                                dtype=dtype,
                                device="cpu"
                            )
                            img[i, :, top:top+h, left:left+w] = pixels.to(img.device)
                    
                    break
        return img

    def __call__(self, input):
        """应用随机擦除"""
        try:
            if len(input.size()) == 3:
                return self._erase(input, *input.size(), input.dtype)
            else:
                batch_size, chan, img_h, img_w = input.size()
                # skip first slice of batch if num_splits is set
                batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
                
                if self.cube:
                    return self._erase_cube(
                        input,
                        batch_start,
                        batch_size,
                        chan,
                        img_h,
                        img_w,
                        input.dtype,
                    )
                else:
                    for i in range(batch_start, batch_size):
                        input[i] = self._erase(input[i], chan, img_h, img_w, input.dtype)
                    return input
        except Exception as e:
            print(f"====> random_erasing.py: 随机擦除失败: {e}")
            return input  # 出错时返回原始输入