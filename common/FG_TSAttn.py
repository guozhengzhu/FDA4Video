# fg_tsa_with_flow_model.py
# -*- coding: utf-8 -*-
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============== 可选日志（无 mmengine 时自动降级） ==============
try:
    from mmengine.logging import MMLogger
    LOGGER = MMLogger.get_current_instance()
    def log_info(msg): LOGGER.info(msg)
    def log_warn(msg): LOGGER.warning(msg)
except Exception:
    def log_info(msg): print(f"[INFO] {msg}")
    def log_warn(msg): print(f"[WARN] {msg}")

# ========================= 基础校验（与 torch API 对齐） =========================
def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    if query.dim() != 3:
        raise AssertionError(f"query should be unbatched 3D tensor, got {query.dim()}D tensor")
    if key.dim() != 3:
        raise AssertionError(f"key should be unbatched 3D tensor, got {key.dim()}D tensor")
    if value.dim() != 3:
        raise AssertionError(f"value should be unbatched 3D tensor, got {value.dim()}D tensor")
    if query.shape[-1] != key.shape[-1]:
        raise AssertionError(f"query and key must have the same embedding dimension, got {query.shape[-1]} and {key.shape[-1]}")
    if key.shape[0] != value.shape[0]:
        raise AssertionError(f"key and value must have the same sequence length, got {key.shape[0]} and {value.shape[0]}")

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != (query.shape[0], key.shape[0]):
                raise AssertionError(f"The shape of the 2D attn_mask should be {(query.shape[0], key.shape[0])}, got {attn_mask.shape}")
        elif attn_mask.dim() == 3:
            if attn_mask.shape != (query.shape[1] * num_heads, query.shape[0], key.shape[0]):
                raise AssertionError(f"The shape of the 3D attn_mask should be {(query.shape[1] * num_heads, query.shape[0], key.shape[0])}, got {attn_mask.shape}")
        else:
            raise AssertionError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None:
        if key_padding_mask.dim() != 2:
            raise AssertionError(f"key_padding_mask should be a 2D tensor, got {key_padding_mask.dim()}D tensor")
        if key_padding_mask.shape != (query.shape[1], key.shape[0]):
            raise AssertionError(f"key_padding_mask should have shape {(query.shape[1], key.shape[0])}, got {key_padding_mask.shape}")
    return True


# ========================= 时序移位（轻量，无参数） =========================
class TemporalShift(nn.Module):
    """
    轻量时序移位（支持 divide_head / 整体移位；健壮的整除/填充处理）
    输入/输出: [seq_len, batch, embed_dim]
    """
    def __init__(self, num_frames, n_head, n_div=8, divide_head=False, shift_stride=1, long_shift_div=-1):
        super().__init__()
        self.num_frames = num_frames
        self.n_div = n_div
        self.n_head = n_head
        self.divide_head = divide_head
        self.shift_stride = shift_stride
        self.long_shift_div = long_shift_div
        log_info(f"TemporalShift: T={num_frames}, heads={n_head}, n_div={n_div}, divide_head={divide_head}, "
                 f"stride={shift_stride}, long_div={long_shift_div}")

    def forward(self, x: Tensor) -> Tensor:
        seq_len, batch_size, embed_dim = x.shape
        num_frames = self.num_frames

        # 估算 segment 数
        num_batches = torch.div(batch_size + num_frames - 1, num_frames, rounding_mode='trunc').item()
        if batch_size % num_frames != 0:
            adjusted_num_frames = torch.div(batch_size, max(1, num_batches), rounding_mode='trunc').item()
            adjusted_num_frames = max(1, adjusted_num_frames)
            num_frames = adjusted_num_frames
            log_warn(f"[TemporalShift] Adjust T -> {num_frames} to match batch {batch_size}")

        # 补齐 batch 维，确保能 reshape
        if batch_size % num_frames != 0:
            padding_size = num_frames - (batch_size % num_frames)
            padded_batch_size = batch_size + padding_size
            x = F.pad(x, (0, 0, 0, padding_size, 0, 0))
        else:
            padded_batch_size = batch_size

        batch_per_segment = torch.div(padded_batch_size, num_frames, rounding_mode='trunc').item()
        x = x.view(seq_len, batch_per_segment, num_frames, embed_dim)
        out = x.clone()

        if self.divide_head:
            head_dim = int(embed_dim) // int(self.n_head)
            feat = x.view(seq_len, batch_per_segment, num_frames, self.n_head, head_dim)
            out_feat = feat.clone()
            fold = max(1, int(head_dim) // int(self.n_div))
            if num_frames > self.shift_stride:
                out_feat[:, :, self.shift_stride:, :, :fold] = feat[:, :, :-self.shift_stride, :, :fold]
                out_feat[:, :, :-self.shift_stride, :, fold:2 * fold] = feat[:, :, self.shift_stride:, :, fold:2 * fold]
            if self.long_shift_div > 0 and num_frames > 2:
                long_fold = max(1, int(head_dim) // int(self.long_shift_div))
                out_feat[:, :, 2:, :, 2 * fold:2 * fold + long_fold] = feat[:, :, :-2, :, 2 * fold:2 * fold + long_fold]
            out = out_feat.view(seq_len, batch_per_segment, num_frames, embed_dim)
        else:
            fold = max(1, int(embed_dim) // int(self.n_div))
            if num_frames > self.shift_stride:
                out[:, :, self.shift_stride:, :fold] = x[:, :, :-self.shift_stride, :fold]
                out[:, :, :-self.shift_stride, fold:2 * fold] = x[:, :, self.shift_stride:, fold:2 * fold]
            if self.long_shift_div > 0 and num_frames > 2:
                long_fold = max(1, int(embed_dim) // int(self.long_shift_div))
                out[:, :, 2:, 2 * fold:2 * fold + long_fold] = x[:, :, :-2, 2 * fold:2 * fold + long_fold]

        out = out.view(seq_len, padded_batch_size, embed_dim)
        if batch_size != padded_batch_size:
            out = out[:, :batch_size, :]
        return out


# ============ 轻量光流提取器（无可训练卷积：Sobel + 池化 + Linear） ============
class InternalFlowExtractor(nn.Module):
    """
    极轻量“光流输入”：
      - 灰度化
      - Sobel 求 Ix, Iy；帧差为 It
      - 近似 u = -It * Ix / (Ix^2 + Iy^2 + eps), v 同理
      - (u,v) 做 K×K 自适应平均池化，拼 2K^2 后 Linear 到 embed_dim
    计算/显存极小，适合在推理中以极低成本注入运动先验。
    """
    def __init__(self, embed_dim: int, num_frames: int, pool_grid: int = 4,
                 down_hw: Optional[int] = 64, eps: float = 1e-3, smooth_ks: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.pool_grid = pool_grid
        self.down_hw = down_hw
        self.eps = eps
        self.smooth_ks = smooth_ks

        # 固定 Sobel 核
        sobel_x = torch.tensor([[1., 0., -1.],
                                [2., 0., -2.],
                                [1., 0., -1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1.,  2.,  1.],
                                [0.,  0.,  0.],
                                [-1., -2., -1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 唯一可训练层
        self.proj = nn.Linear(2 * pool_grid * pool_grid, embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _to_gray(frames: Tensor) -> Tensor:
        # frames: [B,1/3,H,W]
        if frames.size(1) == 1:
            return frames
        r, g, b = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def extract_optical_flow(self, video_frames: Tensor) -> Tensor:
        """
        Args:
            video_frames: [B, T, C, H, W], float
        Returns:
            flow_feats:  [B, T, embed_dim]
        """
        B, T, C, H, W = video_frames.shape
        if T < 2:
            return torch.zeros(B, T, self.embed_dim, device=video_frames.device, dtype=video_frames.dtype)

        def ds(x: Tensor) -> Tensor:
            if self.down_hw is None:
                return x
            return F.interpolate(x, size=(self.down_hw, self.down_hw), mode="bilinear", align_corners=False)

        gray_all = [ds(self._to_gray(video_frames[:, t])) for t in range(T)]
        feats = []

        for t in range(T - 1):
            I0 = gray_all[t]
            I1 = gray_all[t + 1]

            Ix = F.conv2d(I0, self.sobel_x.to(I0.dtype), padding=1)
            Iy = F.conv2d(I0, self.sobel_y.to(I0.dtype), padding=1)
            It = I1 - I0

            denom = Ix * Ix + Iy * Iy + self.eps
            u = -It * Ix / denom
            v = -It * Iy / denom

            # 轻度均值滤波（可选）
            if self.smooth_ks and self.smooth_ks > 1:
                pad = self.smooth_ks // 2
                k = torch.ones(1, 1, self.smooth_ks, self.smooth_ks, device=u.device, dtype=u.dtype) / (self.smooth_ks * self.smooth_ks)
                u = F.conv2d(u, k, padding=pad)
                v = F.conv2d(v, k, padding=pad)

            uv = torch.cat([u, v], dim=1)                              # [B,2,h,w]
            uv_pool = F.adaptive_avg_pool2d(uv, (self.pool_grid, self.pool_grid))  # [B,2,K,K]
            vec = uv_pool.flatten(1)                                   # [B, 2*K*K]
            feat = self.proj(vec)                                      # [B, D]
            feats.append(feat)

        feats.append(feats[-1])  # 对齐最后一帧
        feats = torch.stack(feats, dim=1)  # [B, T, D]
        return feats


# ========================= 注意力（集成光流 + 时序移位） =========================
class FG_TSAttn(nn.MultiheadAttention):
    """
    增强的时空多头注意力：
      - TemporalShift（q/kv/qkv 模式）
      - 极轻量光流特征（可选），注入到 K（键）
      - 公共实现（不依赖私有 API）
    """
    def __init__(self, embed_dim, num_heads, num_frames, shift_div=4, divide_head=True,
                 shift_pattern='kv', shift_stride=1, long_shift_div=-1,
                 enable_flow_extraction=True, **kwargs) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        self.time_shift = TemporalShift(
            num_frames=num_frames,
            n_head=num_heads,
            n_div=shift_div,
            divide_head=divide_head,
            shift_stride=shift_stride,
            long_shift_div=long_shift_div
        )
        self.shift_pattern = shift_pattern
        self.enable_flow_extraction = enable_flow_extraction

        if enable_flow_extraction:
            self.flow_extractor = InternalFlowExtractor(
                embed_dim=embed_dim,
                num_frames=num_frames,
                pool_grid=4,     # 默认 4x4 池化
                down_hw=64,      # 先降到 64×64
                eps=1e-3,
                smooth_ks=3
            )

    # ---- 光流融合到 K ----
    def extract_and_fuse_optical_flow(self, video_frames: Optional[Tensor], k: Tensor) -> Tensor:
        if video_frames is None or not self.enable_flow_extraction:
            return k
        optical_flow_features = self.flow_extractor.extract_optical_flow(video_frames)  # [B,T,D]
        B, T, D = optical_flow_features.shape
        seq_len, batch_size, embed_dim = k.shape

        if batch_size != B * T:
            fuse = optical_flow_features.reshape(B * T, D).unsqueeze(0).expand(seq_len, -1, -1)
        else:
            fuse = optical_flow_features.reshape(batch_size, D).unsqueeze(0).expand(seq_len, -1, -1)
        return k + fuse.to(k.dtype)

    # ---- 兼容版 SDPA ----
    def scaled_dot_product_attention_compat(self, q: Tensor, k: Tensor, v: Tensor,
                                            attn_mask: Optional[Tensor] = None,
                                            dropout_p: float = 0.0) -> Tuple[Tensor, Tensor]:
        scale = 1.0 / math.sqrt(q.size(-1))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    # ---- 主前向（仿 torch.nn.MultiheadAttention 的内部） ----
    def FG_TSAttn_forward(
            self,
            query: Tensor, key: Tensor, value: Tensor,
            embed_dim_to_check: int, num_heads: int,
            in_proj_weight: Optional[Tensor], in_proj_bias: Optional[Tensor],
            bias_k: Optional[Tensor], bias_v: Optional[Tensor],
            add_zero_attn: bool, dropout_p: float,
            out_proj_weight: Tensor, out_proj_bias: Optional[Tensor],
            training: bool = True, key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True, attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[Tensor] = None, k_proj_weight: Optional[Tensor] = None, v_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None, static_v: Optional[Tensor] = None,
            average_attn_weights: bool = True, video_frames: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
        if torch.overrides.has_torch_function(tens_ops):
            return torch.overrides.handle_torch_function(
                self.FG_TSAttn_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads,
                in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p,
                out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight,
                static_k=static_k, static_v=static_v, average_attn_weights=average_attn_weights,
                video_frames=video_frames)

        _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == embed_dim_to_check, f"Expected embedding dim {embed_dim_to_check}, got {embed_dim}"

        head_dim = int(embed_dim) // int(num_heads)
        assert head_dim * num_heads == embed_dim, f"Embed dim {embed_dim} not divisible by num_heads {num_heads}"

        # qkv 投影
        if not use_separate_proj_weight:
            q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # 光流融合（到 K）
        if video_frames is not None and self.enable_flow_extraction:
            k = self.extract_and_fuse_optical_flow(video_frames, k)

        # 时序移位
        if self.shift_pattern == 'qkv':
            q = self.time_shift(q.contiguous()); k = self.time_shift(k.contiguous()); v = self.time_shift(v.contiguous())
        elif self.shift_pattern == 'kv':
            k = self.time_shift(k.contiguous()); v = self.time_shift(v.contiguous())
        elif self.shift_pattern == 'q':
            q = self.time_shift(q.contiguous())
        else:
            raise ValueError(f"Unsupported shift pattern: {self.shift_pattern}")

        # 掩码准备
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            if attn_mask.dim() == 2:
                if attn_mask.shape != (tgt_len, src_len):
                    raise RuntimeError(f"2D attn_mask shape {attn_mask.shape} should be {(tgt_len, src_len)}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape != (bsz * num_heads, tgt_len, src_len):
                    raise RuntimeError(f"3D attn_mask shape {attn_mask.shape} should be {(bsz * num_heads, tgt_len, src_len)}.")
            else:
                raise RuntimeError(f"attn_mask dimension {attn_mask.dim()} not supported")

        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # 偏置项（通过拼接零/重复向量实现，避免私有 pad）
        def _pad_last_dim(t: Tensor, pad_cols: int):
            if pad_cols <= 0:
                return t
            pad_shape = list(t.shape); pad_shape[-1] = pad_cols
            return torch.cat([t, t.new_zeros(pad_shape)], dim=-1)

        if bias_k is not None and bias_v is not None:
            assert static_k is None and static_v is None
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)], dim=0)
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)], dim=0)
            if attn_mask is not None:
                attn_mask = _pad_last_dim(attn_mask, 1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros((bsz, 1), dtype=key_padding_mask.dtype)], dim=1)
        else:
            assert bias_k is None and bias_v is None

        # 变换维度为 [B*H, T, D_head]
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            assert static_k.size(0) == bsz * num_heads and static_k.size(2) == head_dim
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            assert static_v.size(0) == bsz * num_heads and static_v.size(2) == head_dim
            v = static_v

        # 可选零注意力位置
        if add_zero_attn:
            zero = k.new_zeros((bsz * num_heads, 1, head_dim))
            k = torch.cat([k, zero], dim=1); v = torch.cat([v, zero], dim=1)
            if attn_mask is not None:
                attn_mask = _pad_last_dim(attn_mask, 1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros((bsz, 1), dtype=key_padding_mask.dtype)], dim=1)

        src_len = k.size(1)

        # 合并 padding 掩码
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len)
            kpm = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = kpm
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(kpm)
            else:
                attn_mask = attn_mask.masked_fill(kpm, float("-inf"))

        # bool 掩码 → float 掩码
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn

        if not training:
            dropout_p = 0.0

        # 注意力
        attn_output, attn_output_weights = self.scaled_dot_product_attention_compat(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

        # 输出投影
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if need_weights:
            if attn_output_weights is None:
                attn_output_weights = torch.zeros(bsz * num_heads, tgt_len, src_len, device=attn_output.device)
            if attn_output_weights.dim() == 3:
                attn_output_weights = attn_output_weights.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    # ---- 公共接口 ----
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None, need_weights: bool = True,
                attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True,
                video_frames: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.FG_TSAttn_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights, video_frames=video_frames
            )
        else:
            attn_output, attn_output_weights = self.FG_TSAttn_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                video_frames=video_frames
            )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


# ========================= 基于 ViT 的上层骨干（实际使用光流） =========================
class LayerNormFP32(nn.LayerNorm):
    def forward(self, x: Tensor):
        orig_type = x.dtype
        return super().forward(x.to(torch.float32)).to(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """将 video_frames 传入 FG_TSAttn，确保“实际启用光流”"""
    def __init__(self, d_model: int, n_head: int, num_frames: int,
                 shift_div=12, divide_head=False, shift_stride=1, long_shift_div=-1,
                 enable_flow_extraction=True):
        super().__init__()
        self.ln_1 = LayerNormFP32(d_model)
        self.attn = FG_TSAttn(
            embed_dim=d_model,
            num_heads=n_head,
            num_frames=num_frames,
            shift_div=shift_div,
            divide_head=divide_head,
            shift_stride=shift_stride,
            long_shift_div=long_shift_div,
            enable_flow_extraction=enable_flow_extraction
        )
        self.ln_2 = LayerNormFP32(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x_tokens: Tensor, video_frames: Optional[Tensor]) -> Tensor:
        # x_tokens: [BT, L, D]  →  MHA 期望 [L, BT, D]
        x = x_tokens
        x1 = self.ln_1(x)
        qkv = x1.permute(1, 0, 2)
        attn_out, _ = self.attn(qkv, qkv, qkv, video_frames=video_frames)
        x = x + attn_out.permute(1, 0, 2)
        x2 = self.ln_2(x)
        x = x + self.mlp(x2)
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_frames: int,
                 enable_flow_extraction=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width, n_head=heads, num_frames=num_frames,
                shift_div=12, divide_head=False, shift_stride=1, long_shift_div=-1,
                enable_flow_extraction=enable_flow_extraction
            ) for _ in range(layers)
        ])

    def forward(self, x_tokens: Tensor, video_frames: Optional[Tensor]) -> Tensor:
        for blk in self.blocks:
            x_tokens = blk(x_tokens, video_frames)
        return x_tokens

class VideoViT_FG_TSA(nn.Module):
    """
    输入: video [B, T, C, H, W]
    输出: 时序平均后的 class-token 表征（或分类 logits）
    """
    def __init__(self, image_size=224, patch_size=16, width=768, layers=12, heads=12,
                 num_frames=16, num_classes: Optional[int] = None,
                 enable_flow_extraction=True, max_frames: Optional[int] = None):
        super().__init__()
        self.image_size = image_size
        self.patch = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_frames = num_frames
        self.grid = image_size // patch_size
        self.seq_len = self.grid * self.grid + 1

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(self.seq_len, width))

        # 可选时间位置嵌入
        if max_frames is None:
            max_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, max_frames, width))

        self.ln_pre = LayerNormFP32(width)
        self.transformer = Transformer(width, layers, heads, num_frames, enable_flow_extraction)
        self.ln_post = LayerNormFP32(width)

        self.head = None
        if num_classes is not None:
            self.head = nn.Linear(width, num_classes)
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def _patchify(self, frames_btchw: Tensor) -> Tensor:
        # frames_btchw: [B*T, C, H, W] -> tokens [B*T, L, D]
        x = self.conv1(frames_btchw)                     # [B*T, D, g, g]
        x = x.flatten(2).permute(0, 2, 1)                # [B*T, L-1, D]
        cls = self.class_embedding.view(1, 1, -1).expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                   # [B*T, L, D]
        x = x + self.positional_embedding.to(x.dtype)    # 加空间位置
        return x

    def forward(self, video: Tensor) -> Tensor:
        """
        video: [B, T, C, H, W]
        返回:
            - 若设置 num_classes: logits [B, num_classes]
            - 否则: 表征 [B, D]
        """
        B, T, C, H, W = video.shape
        assert T <= self.temporal_embedding.size(1), \
            f"T={T} exceeds temporal_embedding length {self.temporal_embedding.size(1)}"

        # 给注意力的光流分支：保持 [B, T, C, H, W]
        video_frames = video

        # patch + token
        btchw = video.flatten(0, 1)                      # [B*T, C, H, W]
        tokens = self._patchify(btchw)                   # [B*T, L, D]

        # 加时间位置（广播到每个 token）
        tokens = tokens.view(B, T, self.seq_len, self.width)
        tpos = self.temporal_embedding[:, :T, :].unsqueeze(2)    # [1, T, 1, D]
        tokens = tokens + tpos
        tokens = tokens.flatten(0, 1)                    # [B*T, L, D]

        # Pre-LN
        tokens = self.ln_pre(tokens)

        # 进入 Transformer（每层注意力都能拿到原始视频帧 → 实际使用光流）
        tokens = self.transformer(tokens, video_frames)

        # 取 class-token 并做时间平均
        cls = tokens[:, 0, :].view(B, T, -1).mean(dim=1)         # [B, D]
        cls = self.ln_post(cls)

        if self.head is not None:
            return self.head(cls)                                # [B, num_classes]
        return cls                                               # [B, D]


# ========================= 测试 =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = VideoViT_FG_TSA(
        image_size=224, patch_size=16, width=384, layers=4, heads=6,
        num_frames=8, num_classes=10, enable_flow_extraction=True, max_frames=32
    ).to(device)

    B, T, C, H, W = 2, 8, 3, 224, 224
    video = torch.rand(B, T, C, H, W, device=device)
    with torch.no_grad():
        logits = model(video)
    print("logits:", logits.shape)  # [B, 10]
