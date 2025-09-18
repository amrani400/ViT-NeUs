# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch

from torch import Tensor
from torch import nn

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

# Attempt to import flash_attn_func, set to None if not available
flash_attn_func = None
try:
    from flash_attn import flash_attn_func
except ImportError:
    logger.warning("flash_attn not available; FlashLinearAttention and CrossAttention with flash_attn will be disabled unless installed.")

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, context: Tensor, attn_bias=None) -> Tensor:
        """
        Args:
            x: Query input of shape (B, N, C)
            context: Key/Value input of shape (B, M, C)
            attn_bias: Optional attention bias tensor
        """
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads)
        k, v = unbind(kv, 2)

        if flash_attn_func is not None:
            x = flash_attn_func(q, k, v)  # Use flash attention if available
        else:
            # Fallback to standard attention if flash_attn is not installed
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x

class VanillaLinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=proj_bias)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        logging.info(f"Using linear attention")

    def forward(self, x, rotary_emb=None, ret_attn=False):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim)

        if rotary_emb is not None:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

        q = q.softmax(dim=-1) * self.scale
        k = k.softmax(dim=1)
        kv = torch.einsum('bnhf,bnhd->bhfd', k, v)
        y = torch.einsum('bnhf,bhfd->bnhd', q, kv)

        B, N, h, d = y.shape
        y = y.reshape(B, N, h * d)
        return self.o_proj(y)

class GateLinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.gla = GatedLinearAttention(
            hidden_size=dim,
            num_heads=num_heads,
            expand_k=1.0,
            expand_v=1.0,
        )
        logging.info(f"Using gated linear attention")

    def forward(self, x, rotary_emb=None, ret_attn=False):
        output, _, _ = self.gla(x)
        return output

class FlashLinearAttention(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        if flash_attn_func is None:
            raise ImportError("flash_attn is required for FlashLinearAttention but is not installed. "
                              "Please install it with 'pip install flash-attn' or avoid using FlashLinearAttention.")
        self.flash_linear_attn = LinearAttention(
            hidden_size=dim,
            num_heads=num_heads,
            expand_k=1.0,
            expand_v=1.0,
            feature_map=kwargs.get('attn_feature_map', 'identity'),
            output_norm='identity',
        )
        logging.info(f"Using flash linear attention")

    def forward(self, x):
        output = self.flash_linear_attn(x)
        return output