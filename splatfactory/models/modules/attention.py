# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Kernel of the fused attention path, set by set_attention_backend(): "sdpa" or "fa4"
# (FlashAttention-4, Hopper+, only for half precision without mask or dropout, else SDPA).
_BACKEND = "sdpa"


def set_attention_backend(name: str) -> None:
    global _BACKEND
    assert name in ("sdpa", "fa4"), f"Unknown attention backend {name}"
    if name == "fa4":
        # optional dependency, imported here to fail at setup
        from flash_attn.cute import flash_attn_func  # noqa: F401
    _BACKEND = name


@torch.compiler.disable
def _fa4_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """FlashAttention-4 on [B, H, N, D] tensors (graph break, its JIT kernels are opaque)."""
    from flash_attn.cute import flash_attn_func  # lazy: optional dependency

    out, _ = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


def fused_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask=None, dropout_p: float = 0.0):
    plain = attn_mask is None and dropout_p == 0.0 and q.dtype in (torch.float16, torch.bfloat16)
    if plain and _BACKEND == "fa4":
        return _fa4_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if attn_mask is not None:
            attn_mask = attn_mask[:, None].repeat(1, self.num_heads, 1, 1)

        if self.fused_attn:
            x = fused_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        attn_mode: str = "softmax",
        rope=None,  # RoPE for positional encoding
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.attn_mode = attn_mode
        valid_modes = {"softmax", "sigmoid", "linear", "slot"}
        assert attn_mode in valid_modes, f"Invalid attention mode {attn_mode}"

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        qpos=None,
        kpos=None,
        return_attention: bool = False,
    ) -> Tensor:
        """Cross-attention with separate, token-aligned key and value sources [B, N_c, *]."""
        B, N_q, C = query.shape
        N_c = key.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Slot attention can't use fused attention
        use_fused = self.fused_attn and self.attn_mode == "softmax" and not return_attention

        if use_fused:
            x = fused_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            attn_normalized = None
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)  # [B, H, N_q, N_c]

            if self.attn_mode == "slot":
                # Slot attention: softmax over slots (queries), not context
                attn_weights = attn.softmax(dim=-2)  # [B, H, N_q, N_c] - context chooses slots
                attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            elif self.attn_mode == "sigmoid":
                attn_weights = torch.sigmoid(attn)
                attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            elif self.attn_mode == "linear":
                attn_weights = F.relu(attn)
                attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            else:  # softmax
                attn_normalized = attn.softmax(dim=-1)

            attn_normalized = self.attn_drop(attn_normalized)
            x = attn_normalized @ v

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn_normalized.mean(dim=1)
        return x
