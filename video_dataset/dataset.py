import os
import sys
from typing import Optional, Tuple
import av
import io
import numpy as np
import torch
from torchvision import transforms
import traceback

from .transform import (
    create_random_augment,
    random_resized_crop,
    random_short_side_scale_jitter,
    random_crop,
)
from .random_erasing import RandomErasing
try:
    from .load_binary_internal import load_binary
except ImportError:
    from .load_binary import load_binary

class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        list_path: str,
        data_root: str,
        num_frames: int = 8,
        sampling_rate: int = 0,
        spatial_size: int = 224,
        num_spatial_views: int = 1,
        num_temporal_views: int = 1,
        random_sample: bool = True,
        mean: torch.Tensor = torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
        std: torch.Tensor = torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
        auto_augment: Optional[str] = None,
        interpolation: str = 'bicubic',
        mirror: bool = False,
        load_labels: bool = True,
        resize_type: str = 'random_short_side_scale_jitter',
        scale_range: Tuple[float, float] = (0.08, 1.0),
        random_erasing: Optional[RandomErasing] = None,
    ):
        self.list_path = list_path
        self.data_root = data_root
        self.interpolation = interpolation
        self.spatial_size = spatial_size
        self.load_labels = load_labels
        self.scale_range = scale_range
        self.random_erasing = random_erasing
        self.resize_type = resize_type
        self.mean = mean
        self.std = std
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.random_sample = random_sample
        self.mirror = mirror
        self.auto_augment = auto_augment
        self.num_spatial_views = num_spatial_views
        self.num_temporal_views = num_temporal_views

        # === 关键修复：动态调整 scale_range 验证 ===
        if resize_type == 'random_resized_crop':
            # 对于 random_resized_crop，scale_range 应在 (0, 1] 范围内
            if scale_range[0] <= 0 or scale_range[1] > 1.0:
                print(f"警告: 对于 random_resized_crop，scale_range 应满足 0 < min <= max <= 1.0，但得到 {scale_range}")
                # 自动修正为有效值
                scale_range = (max(0.01, min(scale_range[0], 1.0)), min(max(scale_range[1], scale_range[0]), 1.0))
                print(f"已自动修正 scale_range 为: {scale_range}")
        elif resize_type == 'random_short_side_scale_jitter':
            # 对于 random_short_side_scale_jitter，scale_range 应 >= 1.0
            if scale_range[0] < 1.0:
                print(f"警告: 对于 random_short_side_scale_jitter，scale_range 应满足 min >= 1.0，但得到 {scale_range}")
                # 自动修正为有效值
                scale_range = (max(1.0, scale_range[0]), max(scale_range[1], 1.0))
                print(f"已自动修正 scale_range 为: {scale_range}")
        else:
            raise ValueError(f'resize type {resize_type} is not supported.')
        
        # 更新修正后的 scale_range
        self.scale_range = scale_range

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
        else:
            assert auto_augment is None and not mirror

        with open(list_path) as f:
            self.data_list = f.read().splitlines()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        try:
            line = self.data_list[idx]
            if self.load_labels:
                parts = line.split()
                path = parts[0]
                label = int(parts[1]) if len(parts) > 1 else 0
            else:
                path = line.split()[0]
                label = 0
            full_path = os.path.join(self.data_root, path)
            
            # 加载视频数据
            raw_data = load_binary(full_path)
            container = av.open(io.BytesIO(raw_data), metadata_encoding="ISO-8859-1")
            container.streams.video[0].thread_count = 1
            
            # 解码视频帧
            frames = {}
            for frame in container.decode(video=0):
                frames[frame.pts] = frame
            container.close()
            frames = [frames[k] for k in sorted(frames.keys())]
            
            # 确保有足够的帧
            if len(frames) < 1:
                raise ValueError(f"视频没有帧: {full_path}")
            
            # 处理训练模式
            if self.random_sample:
                # 随机采样帧索引
                frame_idx = self._random_sample_frame_idx(len(frames))
                # 提取并转换帧
                frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
                frames = torch.as_tensor(np.stack(frames)).float() / 255.0
                
                # 自动增强
                if self.auto_augment is not None:
                    frames = frames.permute(0, 3, 1, 2)  # T, C, H, W
                    frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                    aug_transform = create_random_augment(
                        input_size=(frames[0].height, frames[0].width),
                        auto_augment=self.auto_augment,
                        interpolation=self.interpolation,
                    )
                    frames = aug_transform(frames)
                    frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                    frames = frames.permute(0, 2, 3, 1)  # T, H, W, C
                
                # 归一化
                frames = (frames - self.mean) / self.std
                frames = frames.permute(3, 0, 1, 2)  # C, T, H, W
                
                # 空间调整
                if self.resize_type == 'random_resized_crop':
                    frames = random_resized_crop(
                        frames, self.spatial_size, self.spatial_size,
                        scale=self.scale_range,
                        interpolation=self.interpolation,
                    )
                elif self.resize_type == 'random_short_side_scale_jitter':
                    frames, _ = random_short_side_scale_jitter(
                        frames,
                        min_size=round(self.spatial_size * self.scale_range[0]),
                        max_size=round(self.spatial_size * self.scale_range[1]),
                        interpolation=self.interpolation,
                    )
                    frames, _ = random_crop(frames, self.spatial_size)
                else:
                    raise NotImplementedError(f"不支持的resize类型: {self.resize_type}")
                
                # 随机擦除
                if self.random_erasing is not None:
                    frames = self.random_erasing(frames.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                
                # 镜像翻转
                if self.mirror and torch.rand(1).item() < 0.5:
                    frames = frames.flip(dims=(-1,))
                
                # 确保形状为 [C, T, H, W]
                if frames.dim() != 4 or frames.size(0) != 3:
                    # 尝试自动修复形状
                    if frames.dim() == 4 and frames.size(3) == 3:  # T, H, W, C
                        frames = frames.permute(3, 0, 1, 2)  # C, T, H, W
                    elif frames.dim() == 4 and frames.size(1) == 3:  # T, C, H, W
                        frames = frames.permute(1, 0, 2, 3)  # C, T, H, W
                    else:
                        raise RuntimeError(f"无法确定通道维度: {frames.shape}")
            # 处理测试模式
            else:
                # 提取并转换帧
                frames = [frame.to_rgb().to_ndarray() for frame in frames]
                frames = torch.as_tensor(np.stack(frames)).float() / 255.0
                
                # 归一化
                frames = (frames - self.mean) / self.std
                frames = frames.permute(3, 0, 1, 2)  # C, T, H, W
                
                # 调整大小
                if frames.size(-2) < frames.size(-1):
                    new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                    new_height = self.spatial_size
                else:
                    new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                    new_width = self.spatial_size
                
                frames = torch.nn.functional.interpolate(
                    frames, size=(new_height, new_width),
                    mode=self.interpolation, align_corners=False,
                )
                
                # 空间裁剪
                spatial_crops = self._generate_spatial_crops(frames)
                
                # 时间裁剪
                temporal_crops = []
                for crop in spatial_crops:
                    temporal_crops.extend(self._generate_temporal_crops(crop))
                
                if len(temporal_crops) > 1:
                    frames = torch.stack(temporal_crops)
                else:
                    frames = temporal_crops[0]
            
            # 最终形状验证
            if frames.dim() == 4:
                if frames.size(0) != 3:
                    raise RuntimeError(f"最终形状错误: {frames.shape}, 应为 [C, T, H, W]")
            elif frames.dim() == 5:
                if frames.size(1) != 3:
                    raise RuntimeError(f"最终形状错误: {frames.shape}, 应为 [N, C, T, H, W]")
            else:
                raise RuntimeError(f"无效的视频形状: {frames.shape}")
            
            if self.load_labels:
                return frames, label
            else:
                return frames
                
        except Exception as e:
            print(f"处理视频 {full_path} 时出错: {e}")
            traceback.print_exc()
            # 返回空数据
            dummy_shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
            if not self.random_sample and (self.num_spatial_views > 1 or self.num_temporal_views > 1):
                dummy_shape = [self.num_spatial_views * self.num_temporal_views] + dummy_shape
            
            dummy_frames = torch.zeros(dummy_shape)
            if self.load_labels:
                return dummy_frames, 0
            else:
                return dummy_frames

    def _generate_temporal_crops(self, frames):
        if self.sampling_rate <= 0:
            assert self.num_temporal_views == 1, (
                'temporal multi-crop for uniform sampling is not supported.'
            )
            seg_size = (frames.size(1) - 1) / self.num_frames
            frame_indices = []
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append((start + end) // 2)

            return [frames[:, frame_indices]]

        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops

    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]
        elif self.num_spatial_views == 4:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in list((0, margin // 4, margin // 2, margin)):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError(f"不支持的num_spatial_views: {self.num_spatial_views}")

    def _random_sample_frame_idx(self, total_frames):
        frame_indices = []

        if self.sampling_rate <= 0:  # tsn sample
            seg_size = (total_frames - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))  # random
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= total_frames:
            for i in range(self.num_frames):
                idx = i * self.sampling_rate
                if idx < total_frames:
                    frame_indices.append(idx)
                else:
                    frame_indices.append(frame_indices[-1] if frame_indices else 0)
        else:
            start = np.random.randint(total_frames - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, list_path: str, num_frames: int, num_views: int, spatial_size: int):
        with open(list_path) as f:
            self.len = len(f.read().splitlines())
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0