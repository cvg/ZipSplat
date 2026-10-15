"""Building blocks: PatchEmbed, RoPE2D, attention, FFNs, transformer blocks and ZipSplat's
color embedding and token merge.

Most blocks adapted from DINOv2 / timm ((c) Meta Platforms, Apache-2.0). Trimmed for
inference.

Author: Alexander Veicht
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Learnable per-channel residual gate (CaiT)."""

    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """2D image -> flat patch tokens: (B, C, H, W) -> (B, N, D)."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = _make_2tuple(img_size)
        self.patch_size = _make_2tuple(patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (
            self.img_size[1] // self.patch_size[1]
        )
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, f"image ({H},{W}) not divisible by patch ({ph},{pw})"
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Feed-forward variants
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with SiLU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        **_,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


try:
    # xformers' fused SwiGLU. The DA3-Giant checkpoint was trained with this kernel,
    # so using the fallback yields a small (but non-trivial) numerical drift per layer.
    from xformers.ops import SwiGLU as _SwiGLU  # type: ignore[import-not-found]
except ImportError:
    _SwiGLU = SwiGLUFFN


class SwiGLUFFNFused(_SwiGLU):
    """SwiGLU with hidden dim rounded to a multiple of 8; uses xformers when available."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        **_,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding
# ---------------------------------------------------------------------------


class PositionGetter:
    """Generates and caches (y, x) patch positions for a 2D grid."""

    def __init__(self) -> None:
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    @torch._dynamo.disable
    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            ys = torch.arange(height, device=device)
            xs = torch.arange(width, device=device)
            self.position_cache[height, width] = torch.cartesian_prod(ys, xs)
        cached = self.position_cache[height, width]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1)


class RotaryPositionEmbedding2D(nn.Module):
    """2D rotary position embedding, applied separately on the y and x feature halves."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0) -> None:
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch._dynamo.disable
    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate(x: Tensor) -> Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d(self, tokens: Tensor, positions: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        cos = F.embedding(positions, cos)[:, None, :, :]
        sin = F.embedding(positions, sin)[:, None, :, :]
        return tokens * cos + self._rotate(tokens) * sin

    @torch._dynamo.disable
    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        assert tokens.size(-1) % 2 == 0
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max().item()) + 1
        cos, sin = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        v_feat, h_feat = tokens.chunk(2, dim=-1)
        v_feat = self._apply_1d(v_feat, positions[..., 0], cos, sin)
        h_feat = self._apply_1d(h_feat, positions[..., 1], cos, sin)
        return torch.cat((v_feat, h_feat), dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
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
            attn_mask = attn_mask[:, None].expand(-1, self.num_heads, -1, -1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with separate query / key / value sources.

    `key` and `value` must be token-aligned: ZipSplat attends with the geometry tokens as keys
    and reads a different payload (a backbone layer or the color tokens) as values.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        B, N_q, C = query.shape
        N_c = key.shape[1]
        heads = lambda x, n: x.reshape(B, n, self.num_heads, self.head_dim).transpose(1, 2)  # noqa
        q = heads(self.q_proj(query), N_q)
        k = heads(self.k_proj(key), N_c)
        v = heads(self.v_proj(value), N_c)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N_q, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class SelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block: x -> x + ls1(attn(norm1(x))) -> x + ls2(mlp(norm2(x)))."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: Optional[float] = 0.01,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = True,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block with its own norm per source:
    q -> q + ls1(attn(norm_q(q), norm_k(k), norm_v(v))) -> . + ls2(mlp(norm2(.)))."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: Optional[float] = 0.01,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        x = query + self.ls1(self.attn(self.norm_q(query), self.norm_k(key), self.norm_v(value)))
        return x + self.ls2(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# ZipSplat token streams
# ---------------------------------------------------------------------------


class ColorEmbedMLP(nn.Module):
    """Per-patch MLP from raw RGB patches to embed_dim color tokens.

    The Conv2d with kernel = stride = patch is a per-patch Linear on the flattened pixels,
    followed by GELU and Linear.
    """

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """(N, 3, H, W) -> (N, T, embed_dim)."""
        x = self.proj1(x).flatten(2).transpose(1, 2)
        return self.proj2(self.act(x))


class TokenMerge(nn.Module):
    """Merge N token streams into one: sum_i LS_i(MLP_i(LN_i(t_i)))."""

    def __init__(self, dim: int, num_streams: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_streams)])
        self.projs = nn.ModuleList([Mlp(dim, hidden_features=dim) for _ in range(num_streams)])
        self.gates = nn.ModuleList(
            [LayerScale(dim, init_values=1 / num_streams) for _ in range(num_streams)]
        )

    def forward(self, tokens: List[Tensor]) -> Tensor:
        """List of N (..., dim) tensors -> (..., dim)."""
        out = 0
        for norm, proj, gate, t in zip(self.norms, self.projs, self.gates, tokens):
            out = out + gate(proj(norm(t)))
        return out
