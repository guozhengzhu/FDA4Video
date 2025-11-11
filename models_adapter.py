# -*- coding: utf-8 -*-
from typing import Tuple, Optional
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import sys

# === 安全获取 GPU ID ===
def get_gpu_id():
    """安全获取 GPU ID，提供多种备选方案"""
    try:
        # 方案1: 从 configs 模块获取
        import configs
        if hasattr(configs, 'gpu_id'):
            return configs.gpu_id
    except ImportError:
        pass
    try:
        # 方案2: 从 Slurm 环境变量获取
        if 'SLURM_STEP_GPUS' in os.environ:
            return int(os.environ['SLURM_STEP_GPUS'])
    except (ValueError, KeyError):
        pass
    try:
        # 方案3: 从 CUDA 可见设备获取
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            return int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    except (ValueError, IndexError):
        pass
    # 方案4: 默认值
    return 0

# 获取 GPU ID
gpu_id = get_gpu_id()
print(f"====> models_adapter.py: 使用 gpu_id = {gpu_id}")

# === 安全设备初始化 ===
try:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.zeros(1).to(device)
        print(f"====> models_adapter.py: 成功使用设备 {device}")
    else:
        device = torch.device("cpu")
        print(f"====> models_adapter.py: CUDA 不可用，使用 CPU")
except Exception as e:
    print(f"====> models_adapter.py: 设备初始化失败: {e}")
    device = torch.device("cpu")
    print(f"====> models_adapter.py: 回退到 CPU 设备")

# === 日志 ===
try:
    from mmengine.logging import MMLogger
    logger = MMLogger.get_current_instance()
except ImportError:
    class DummyLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    logger = DummyLogger()

# === configs ===
class DummyConfigs:
    CLIP_VIT_B16_PATH = None
    CLIP_VIT_L14_PATH = None

try:
    import configs
    from configs import CLIP_VIT_B16_PATH, CLIP_VIT_L14_PATH
    print("====> models_adapter.py: 成功导入 configs 模块")
except ImportError as e:
    print(f"====> models_adapter.py: 导入 configs 模块失败: {e}")
    configs = DummyConfigs()

# === FG_TSAttn ===
try:
    from common.FG_TSAttn import FG_TSAttn
    print("====> models_adapter.py: 成功导入 FG_TSAttn 模块")
except ImportError as e:
    print(f"====> models_adapter.py: 导入 FG_TSAttn 模块失败: {e}")

# ==================== 适配器配置（含近似重参数化补丁） ====================
DEFAULT_ADAPTER_CONFIG = {
    "MAX_FRAMES": 32,
    "ADAPTER_CHANNELS": 64,             # 主适配器通道数
    "AUX_ADAPTER_CHANNELS": 384,        # 辅助适配器通道数
    "ADAPTER_START_LAYER": 3,           # 主适配器起始层
    "AUX_ADAPTER_START_LAYER": 8,       # 辅助适配器起始层
    "ADAPTER_SCALE": 0.1,
    "USE_PRE_ATTN_ADAPTER": True,       # 注意力前 TSA
    "USE_POST_ATTN_ADAPTER": True,      # 注意力后 TSA
    "SHARED_CONV_KERNEL": True,         # 前/后 TSA 共享卷积核
    "ENABLE_MAIN_ADAPTER": True,        # 启用主时空适配器
    "ENABLE_AUX_ADAPTER": True,         # 启用辅助时序适配器
    "ENABLE_TEMPORAL_EMBEDDING": True,  # 时间位置嵌入

    # ===== 近似重参数化（推理折叠）开关 =====
    # 仅在 eval() 推理时生效；训练阶段无影响
    "REPARAM_TSA_INFER": False,         # 推理时把 post-TSA 的缩放折叠到 pre-TSA
    "REPARAM_USE_OMEGA": True,          # 使用可学习系数 ω
    "REPARAM_INIT_OMEGA": 1.0,          # ω 初始化值
}

# ==================== 基础层 ====================
class LayerNorm(nn.LayerNorm):
    """处理 fp16 的 LayerNorm 子类"""
    def forward(self, x: Tensor):
        orig = x.dtype
        y = super().forward(x.to(torch.float32))
        return y.to(orig)

class QuickGELU(nn.Module):
    def forward(self, x: Tensor):
        return x * torch.sigmoid(1.702 * x)

# ==================== 主时空适配器 ====================
class SpatiotemporalFeatureAdapter(nn.Module):
    """深度可分离 3D 卷积 + 轻量通道注意力"""
    def __init__(self, in_channels, adapter_config, scale=DEFAULT_ADAPTER_CONFIG["ADAPTER_SCALE"]):
        super().__init__()
        self.adapter_config = adapter_config
        self.scale = nn.Parameter(torch.tensor(scale))
        ch = adapter_config["ADAPTER_CHANNELS"]

        self.project_in = nn.Linear(in_channels, ch, bias=False)
        self.spatiotemporal_conv = nn.Sequential(
            nn.Conv3d(ch, ch, (3, 3, 3), padding=(1, 1, 1), groups=ch),
            QuickGELU()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, max(8, ch // 16), 1),
            QuickGELU(),
            nn.Conv3d(max(8, ch // 16), ch, 1),
            nn.Sigmoid()
        )
        self.project_out = nn.Linear(ch, in_channels, bias=False)

        # 初始化为近似恒等
        nn.init.constant_(self.project_out.weight, 0)
        nn.init.dirac_(self.spatiotemporal_conv[0].weight.data)
        nn.init.zeros_(self.channel_att[1].weight)
        nn.init.zeros_(self.channel_att[3].weight)
        if self.channel_att[1].bias is not None:
            self.channel_att[1].bias.data.zero_()
        if self.channel_att[3].bias is not None:
            self.channel_att[3].bias.data.zero_()

    def forward(self, x: Tensor, T: int):
        identity = x
        BT, L, C = x.shape
        B = (BT // T) if not isinstance(BT, torch.Tensor) else int(torch.div(BT, T, rounding_mode='trunc').item())
        H = W = int(math.sqrt(L - 1)) if L > 1 else 1

        cls_tok = x[:, :1, :]
        vis = x[:, 1:, :]
        if vis.numel() == 0:
            vis = torch.zeros(BT, 1, C, device=x.device, dtype=x.dtype)

        xs = self.project_in(vis)                # [BT, HW, ch]
        D = xs.size(-1)
        xs = xs.view(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # [B, D, T, H, W]
        conv_out = self.spatiotemporal_conv(xs)
        attn_w = self.channel_att(conv_out)
        conv_out = conv_out * attn_w
        conv_out = conv_out.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, D)
        conv_out = self.project_out(conv_out)
        out = torch.cat([cls_tok, conv_out], dim=1)
        return identity + self.scale * out

# ==================== 辅助时序适配器（TSA） ====================
class TemporalSequenceAdapter(nn.Module):
    """1D 卷积建模时间依赖；支持共享卷积核"""
    def __init__(self, in_channels, adapter_config, shared_conv=None):
        super().__init__()
        self.adapter_config = adapter_config
        self.scale = nn.Parameter(torch.tensor(adapter_config["ADAPTER_SCALE"]))

        if shared_conv is not None and adapter_config["SHARED_CONV_KERNEL"]:
            self.conv = shared_conv
        else:
            aux_ch = adapter_config["AUX_ADAPTER_CHANNELS"]
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, aux_ch, kernel_size=3, padding=1),
                QuickGELU(),
                nn.Conv1d(aux_ch, in_channels, kernel_size=1)
            )
            nn.init.constant_(self.conv[2].weight, 0)
            nn.init.constant_(self.conv[2].bias, 0)

    def forward(self, x: Tensor, T: int):
        identity = x
        BT, L, C = x.shape
        B = (BT // T) if not isinstance(BT, torch.Tensor) else int(torch.div(BT, T, rounding_mode='trunc').item())
        x = x.view(B, T, L, C).permute(0, 2, 1, 3)     # [B, L, T, C]
        x = x.reshape(B * L, T, C).permute(0, 2, 1)    # [B*L, C, T]
        out = self.conv(x).permute(0, 2, 1)            # [B*L, T, C]
        out = out.view(B, L, T, C).permute(0, 2, 1, 3) # [B, T, L, C]
        out = out.reshape(B * T, L, C)                 # [BT, L, C]
        return identity + self.scale * out

# ==================== 残差注意力块 ====================
class ResidualAttentionBlock(nn.Module):
    """Self-Attn + (pre/post) TSA + MLP + 主时空适配器；含推理阶段的近似重参数化折叠"""
    def __init__(self, d_model: int, n_head: int, num_frames: int,
                 layer_idx: int, total_layers: int, adapter_config):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.adapter_config = adapter_config

        # 共享的 TSA 卷积核（可选，更利于折叠）
        shared_conv = None
        if adapter_config["SHARED_CONV_KERNEL"] and adapter_config["ENABLE_AUX_ADAPTER"]:
            shared_conv = nn.Sequential(
                nn.Conv1d(d_model, adapter_config["AUX_ADAPTER_CHANNELS"], kernel_size=3, padding=1),
                QuickGELU(),
                nn.Conv1d(adapter_config["AUX_ADAPTER_CHANNELS"], d_model, kernel_size=1)
            )
            nn.init.constant_(shared_conv[2].weight, 0)
            nn.init.constant_(shared_conv[2].bias, 0)

        # 注意力前 TSA
        self.pre_aux_adapter = None
        if adapter_config["USE_PRE_ATTN_ADAPTER"] and adapter_config["ENABLE_AUX_ADAPTER"] \
           and layer_idx >= adapter_config["AUX_ADAPTER_START_LAYER"]:
            self.pre_aux_adapter = TemporalSequenceAdapter(d_model, adapter_config, shared_conv=shared_conv)

        self.ln_1 = LayerNorm(d_model)
        self.attn = FG_TSAttn(
            embed_dim=d_model, num_heads=n_head, num_frames=num_frames,
            shift_div=12, divide_head=False, shift_stride=1, long_shift_div=-1
        )

        # 注意力后 TSA
        self.post_aux_adapter = None
        if adapter_config["USE_POST_ATTN_ADAPTER"] and adapter_config["ENABLE_AUX_ADAPTER"] \
           and layer_idx >= adapter_config["AUX_ADAPTER_START_LAYER"]:
            self.post_aux_adapter = TemporalSequenceAdapter(d_model, adapter_config, shared_conv=shared_conv)

        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))

        # 主时空适配器
        self.spatiotemporal_adapter = None
        if adapter_config["ENABLE_MAIN_ADAPTER"] and layer_idx >= adapter_config["ADAPTER_START_LAYER"]:
            self.spatiotemporal_adapter = SpatiotemporalFeatureAdapter(d_model, adapter_config)

        # ===== 近似重参数化：eval() 推理时把 post-TSA 的缩放折叠进 pre-TSA =====
        self.reparam_enabled = (
            adapter_config.get("REPARAM_TSA_INFER", False)
            and self.pre_aux_adapter is not None
            and self.post_aux_adapter is not None
            and adapter_config.get("SHARED_CONV_KERNEL", True)
        )
        if self.reparam_enabled and adapter_config.get("REPARAM_USE_OMEGA", True):
            self.reparam_omega = nn.Parameter(torch.tensor(float(adapter_config.get("REPARAM_INIT_OMEGA", 1.0))))
        else:
            self.reparam_omega = None

    def forward(self, x: Tensor, num_frames: int) -> Tensor:
        # ---------- 注意力前 TSA ----------
        if self.pre_aux_adapter is not None:
            if (not self.training) and self.reparam_enabled:
                # 近似折叠：α' = α_pre + ω·α_post
                pre_scale_orig = self.pre_aux_adapter.scale
                if self.reparam_omega is None:
                    eff_scale = self.pre_aux_adapter.scale + self.post_aux_adapter.scale
                else:
                    eff_scale = self.pre_aux_adapter.scale + self.reparam_omega * self.post_aux_adapter.scale
                self.pre_aux_adapter.scale = eff_scale
                x = self.pre_aux_adapter(x, num_frames)
                self.pre_aux_adapter.scale = pre_scale_orig   # 还原
                skip_post_tsa = True
            else:
                x = self.pre_aux_adapter(x, num_frames)
                skip_post_tsa = False
        else:
            skip_post_tsa = False

        # ---------- 自注意力 ----------
        identity = x
        x = self.ln_1(x)
        x = x.permute(1, 0, 2)                      # [BT, L, D] -> [L, BT, D]
        attn_output, _ = self.attn(x, x, x)         # FG_TSAttn
        attn_output = attn_output.permute(1, 0, 2)  # [L, BT, D] -> [BT, L, D]
        x = identity + attn_output

        # ---------- 注意力后 TSA ----------
        if self.post_aux_adapter is not None:
            if (not self.training) and self.reparam_enabled and skip_post_tsa:
                pass  # 推理折叠：跳过
            else:
                x = self.post_aux_adapter(x, num_frames)

        # ---------- MLP ----------
        identity2 = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = identity2 + x

        # ---------- 主时空适配器 ----------
        if self.spatiotemporal_adapter is not None:
            x = self.spatiotemporal_adapter(x, num_frames)
        return x

# ==================== Transformer ====================
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_frames: int, adapter_config):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width, n_head=heads, num_frames=num_frames,
                layer_idx=i, total_layers=layers, adapter_config=adapter_config
            ) for i in range(layers)
        ])

    def forward(self, x: Tensor, num_frames: int) -> Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
        return x

# ==================== ViT（含适配器） ====================
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, num_classes: int,
                 adapter_config=DEFAULT_ADAPTER_CONFIG, max_frames=DEFAULT_ADAPTER_CONFIG["MAX_FRAMES"],
                 num_frames: int = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.adapter_config = adapter_config

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        grid = input_resolution // patch_size
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid * grid + 1, width))

        self.temporal_embedding = None
        if adapter_config["ENABLE_TEMPORAL_EMBEDDING"]:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, max_frames, width))

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, max_frames, adapter_config)
        self.ln_post = LayerNorm(width)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(width, num_classes)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)

        self._strict_freeze_parameters()
        self._print_parameter_info()

        if num_frames is not None:
            self._compute_flops(num_frames)
        else:
            print("====> num_frames not provided, skipping FLOPs calculation")

        self.to(device)

    def _strict_freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False
        for name, p in self.named_parameters():
            if 'temporal_embedding' in name:
                p.requires_grad = True; continue
            if 'ln_post' in name:
                p.requires_grad = True; continue
            if name in ['fc.weight', 'fc.bias']:
                p.requires_grad = True; continue
            if 'adapter' in name:
                p.requires_grad = True

    def _print_parameter_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        st_params = 0
        aux_params = 0
        for name, p in self.named_parameters():
            if 'spatiotemporal_adapter' in name and p.requires_grad:
                st_params += p.numel()
            if 'aux_adapter' in name and p.requires_grad:
                aux_params += p.numel()

        print("=" * 50)
        print("Model Configuration:")
        print(f"  Input resolution: {self.input_resolution}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Width: {self.width}")
        print(f"  Layers: {self.layers}")
        print(f"  Heads: {self.heads}")
        print(f"  Max frames: {self.adapter_config['MAX_FRAMES']}")
        print(f"  Enable main adapter: {self.adapter_config['ENABLE_MAIN_ADAPTER']}")
        print(f"  Enable aux adapter: {self.adapter_config['ENABLE_AUX_ADAPTER']}")
        print(f"  Enable temporal embedding: {self.adapter_config['ENABLE_TEMPORAL_EMBEDDING']}")
        print(f"  Adapter start layer: {self.adapter_config['ADAPTER_START_LAYER']} (total layers: {self.layers})")
        print(f"  Aux adapter start layer: {self.adapter_config['AUX_ADAPTER_START_LAYER']} (total layers: {self.layers})")
        print(f"  Adapter channels: {self.adapter_config['ADAPTER_CHANNELS']}")
        print(f"  Aux adapter channels: {self.adapter_config['AUX_ADAPTER_CHANNELS']}")
        print(f"  Pre-attention adapter: {self.adapter_config['USE_PRE_ATTN_ADAPTER']}")
        print(f"  Post-attention adapter: {self.adapter_config['USE_POST_ATTN_ADAPTER']}")
        print(f"  Shared conv kernel: {self.adapter_config['SHARED_CONV_KERNEL']}")
        print(f"  Reparam at infer: {self.adapter_config.get('REPARAM_TSA_INFER', False)}")
        print("=" * 50)
        print(f"Total Parameters: {total / 1e6:.4f}M")
        print(f"Trainable Parameters: {trainable / 1e6:.4f}M")
        print(f"Frozen Parameters: {frozen / 1e6:.4f}M")
        print(f"Main adapter params: {st_params / 1e6:.4f}M")
        print(f"Aux adapter params: {aux_params / 1e6:.4f}M")
        print("=" * 50)
        print("Trainable Components:")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"  {name}: {p.numel() / 1e6:.4f}M")
        print("=" * 50)

    def _compute_flops(self, num_frames: int):
        try:
            from fvcore.nn import FlopCountAnalysis
            original_training = self.training
            self.eval()
            self.to(device)
            input_shape = (1, 3, num_frames, self.input_resolution, self.input_resolution)
            inp = torch.randn(input_shape).to(device)
            with torch.no_grad():
                flops = FlopCountAnalysis(self, inp).total()
                params = sum(p.numel() for p in self.parameters())
            print("\n" + "=" * 50)
            print("Model FLOPs Calculation (using actual num_frames):")
            print(f"  Input shape: {input_shape}")
            print(f"  Extra FLOPs: {flops / 1e9:.2f} GFLOPs")
            print(f"  Params: {params / 1e6:.2f} MParams")
            print("=" * 50)
        except ImportError:
            print("====> fvcore not installed, skipping FLOPs calculation")
        except Exception as e:
            print(f"====> Failed to compute FLOPs: {e}")
        finally:
            if hasattr(self, 'training'):
                self.train(original_training)

    def forward(self, x: Tensor):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)           # [B*T, C, H, W]
        x = self.conv1(x)                                     # [B*T, D, g, g]
        x = x.flatten(-2).permute(0, 2, 1)                    # [B*T, L, D]
        cls = self.class_embedding.view(1, 1, -1).expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                        # [B*T, L+1, D]
        x = x + self.positional_embedding.to(x.dtype)

        if self.temporal_embedding is not None:
            x = x.view(B, T, self.positional_embedding.size(0), -1)
            temp = self.temporal_embedding[:, :T, :].unsqueeze(2)  # [1,T,1,D]
            x = x + temp.expand(B, T, x.size(2), -1)
            x = x.flatten(0, 1)                                   # [B*T, L, D]
        else:
            x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1)

        x = self.ln_pre(x)
        x = self.transformer(x, T)
        cls = x[:, 0, :].view(B, T, -1).mean(dim=1)
        x = self.ln_post(cls)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== 工厂函数 ====================
def load_pretrained_weights(model, pretrained_path):
    """加载预训练权重（兼容 JIT / 普通 state_dict）"""
    logger = MMLogger.get_current_instance()
    if not pretrained_path:
        logger.warning("No pretrained weights path provided")
        return model
    try:
        checkpoint = torch.jit.load(pretrained_path, map_location=device)
        state_dict = checkpoint.visual.state_dict()
        logger.info(f"Loaded JIT model from {pretrained_path}")
    except Exception as e:
        logger.warning(f"Failed to load JIT model: {e}")
        try:
            state = torch.load(pretrained_path, map_location=device)
            state_dict = state.get('state_dict', state)
            logger.info(f"Loaded state dict from {pretrained_path}")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            return model

    current = model.state_dict()
    loaded, skipped, mism = [], [], []
    for name, param in state_dict.items():
        new_name = name.replace('visual.', '')
        if new_name in current:
            if current[new_name].shape == param.shape:
                current[new_name].copy_(param)
                loaded.append(new_name)
            else:
                mism.append(new_name)
        else:
            skipped.append(name)

    # 初始化适配器权重
    for name, p in model.named_parameters():
        if 'adapter' in name and 'weight' in name and p.dim() >= 2:
            if 'conv' in name:
                nn.init.dirac_(p)
            else:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    model.load_state_dict(current)
    logger.info(f"Loaded {len(loaded)} pretrained parameters")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} parameters (not found in model)")
    if mism:
        logger.warning(f"Skipped {len(mism)} parameters due to shape mismatch")
    return model

def clip_vit_base_patch16_adapter(num_classes=51, adapter_config=None, **kwargs):
    """ViT-B/16 + 适配器（含可选推理折叠）"""
    logger = MMLogger.get_current_instance()
    logger.info("\n" + "=" * 50)
    logger.info("Initializing ViT-Base with configurable adapters")
    merged = DEFAULT_ADAPTER_CONFIG.copy()
    if adapter_config is not None:
        merged.update(adapter_config)

    model = VisionTransformer(
        input_resolution=224, patch_size=16, width=768,
        layers=12, heads=12, num_classes=num_classes,
        adapter_config=merged,
        max_frames=kwargs.get('max_frames', merged["MAX_FRAMES"]),
        num_frames=kwargs.get('num_frames')
    ).to(device)
    model = load_pretrained_weights(model, getattr(configs, "CLIP_VIT_B16_PATH", None))
    return model

def clip_vit_large_patch14_adapter(num_classes=51, adapter_config=None, **kwargs):
    """ViT-L/14 + 适配器（含可选推理折叠）"""
    logger = MMLogger.get_current_instance()
    logger.info("\n" + "=" * 50)
    logger.info("Initializing ViT-Large with configurable adapters")
    merged = DEFAULT_ADAPTER_CONFIG.copy()
    if adapter_config is not None:
        merged.update(adapter_config)
    if "ADAPTER_START_LAYER" not in merged:
        merged["ADAPTER_START_LAYER"] = 15  # 24 层中最后 9 层
    if "AUX_ADAPTER_START_LAYER" not in merged:
        merged["AUX_ADAPTER_START_LAYER"] = 15

    model = VisionTransformer(
        input_resolution=224, patch_size=14, width=1024,
        layers=24, heads=16, num_classes=num_classes,
        adapter_config=merged,
        max_frames=kwargs.get('max_frames', merged["MAX_FRAMES"]),
        num_frames=kwargs.get('num_frames')
    ).to(device)
    model = load_pretrained_weights(model, getattr(configs, "CLIP_VIT_L14_PATH", None))
    return model
