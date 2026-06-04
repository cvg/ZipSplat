# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint as ckpt

from splatfactory import get_logger
from splatfactory.models import BaseModel
from splatfactory.models.modules.patch_embed import PatchEmbed
from splatfactory.models.modules.rope import PositionGetter, RotaryPositionEmbedding2D
from splatfactory.models.modules.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from splatfactory.models.modules.transformer_block import Mlp, SelfAttentionBlock

logger = get_logger(__name__)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DAV3DinoVisionTransformer(BaseModel):
    """Depth-Anything-V3 style DINO Vision Transformer with multi-view support."""

    default_conf = {
        "img_size": 518,
        "patch_size": 14,
        "in_chans": 3,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
        "init_values": 1.0,
        "act_layer": "gelu",
        "ffn_layer": "mlp",
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
        "alt_start": -1,
        "qknorm_start": -1,
        "rope_start": -1,
        "rope_freq": 100,
        "patch_start_idx": 1,
        "cat_token": True,
        "use_checkpoint": False,
        # Output settings
        "out_layers": [23],
    }

    def _init(self, conf):
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = conf.embed_dim
        self.num_tokens = 1
        self.n_blocks = conf.depth
        self.num_heads = conf.num_heads
        self.patch_size = conf.patch_size
        self.num_register_tokens = conf.num_register_tokens
        self.interpolate_antialias = conf.interpolate_antialias
        self.interpolate_offset = conf.interpolate_offset
        self.alt_start = conf.alt_start
        self.qknorm_start = conf.qknorm_start
        self.rope_start = conf.rope_start
        self.rope_freq = conf.rope_freq
        self.cat_token = conf.cat_token
        self.use_reentrant = False

        self.patch_embed = PatchEmbed(
            img_size=conf.img_size,
            patch_size=conf.patch_size,
            in_chans=conf.in_chans,
            embed_dim=conf.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, conf.embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, conf.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, conf.embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, conf.num_register_tokens, conf.embed_dim))
            if conf.num_register_tokens
            else None
        )

        if conf.drop_path_uniform:
            dpr = [conf.drop_path_rate] * conf.depth
        else:
            dpr = [x.item() for x in torch.linspace(0, conf.drop_path_rate, conf.depth)]

        ffn_layers = {
            "mlp": Mlp,
            "swiglufused": SwiGLUFFNFused,
            "swiglu": SwiGLUFFN,
            "identity": lambda *args, **kwargs: nn.Identity(),
        }
        if conf.ffn_layer not in ffn_layers:
            raise NotImplementedError(f"FFN layer {conf.ffn_layer} not implemented")
        ffn_layer = ffn_layers[conf.ffn_layer]
        logger.info(f"Using {conf.ffn_layer} as FFN layer")

        act_layers = {"gelu": nn.GELU}
        if conf.act_layer not in act_layers:
            raise NotImplementedError(f"Activation {conf.act_layer} not implemented")
        act_layer = act_layers[conf.act_layer]

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=self.rope_freq)
            self.position_getter = PositionGetter()
        else:
            self.rope = None
            self.position_getter = None

        blocks_list = [
            SelfAttentionBlock(
                dim=conf.embed_dim,
                num_heads=conf.num_heads,
                mlp_ratio=conf.mlp_ratio,
                qkv_bias=conf.qkv_bias,
                proj_bias=conf.proj_bias,
                ffn_bias=conf.ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=conf.init_values,
                qk_norm=i >= conf.qknorm_start if conf.qknorm_start != -1 else False,
                rope=self.rope if i >= conf.rope_start and conf.rope_start != -1 else None,
            )
            for i in range(conf.depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(conf.embed_dim)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # Create CLS token
        if cls_token is None:
            cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)

        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.conf.patch_start_idx > 0:
                pos = pos + 1
                pos_special = (
                    torch.zeros(B * S, self.conf.patch_start_idx, 2).to(device).to(pos.dtype)
                )
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output, local_x = [], len(self.blocks), [], None
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        use_ckpt = self.conf.use_checkpoint and self.training
        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    logger.debug("Using camera conditions provided by the user")
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                    logger.debug("Using learned camera conditions")
                x[:, :, 0] = cam_token

            is_global = self.alt_start != -1 and i >= self.alt_start and i % 2 == 1
            attn_type = "global" if is_global else "local"
            blk_pos = g_pos if is_global else l_pos
            attn_mask = kwargs.get("attn_mask", None) if is_global else None

            if use_ckpt:
                x = ckpt(
                    self.process_attention,
                    x,
                    blk,
                    attn_type,
                    blk_pos,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                x = self.process_attention(x, blk, attn_type, pos=blk_pos, attn_mask=attn_mask)

            if not is_global:
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s, n=n)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        export_feat_layers: list[int] = [],
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]]]:
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        aux_outputs = [self.norm(out) for out in aux_outputs]
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]
        return tuple(zip(outputs, camera_tokens)), aux_outputs

    def _forward(self, data):
        img = data["image"]
        cam_token = data.get("cam_token", None)
        export_feat_layers = data.get("export_feat_layers", [])
        feats, aux_feats = self.get_intermediate_layers(
            img, self.conf.out_layers, cam_token=cam_token, export_feat_layers=export_feat_layers
        )
        return {"feats": feats, "aux_feats": aux_feats}

    def loss(self, pred, data):
        raise NotImplementedError("Loss not implemented for DAV3 DINO encoder.")

    def metrics(self, pred, data):
        raise NotImplementedError("Metrics not implemented for DAV3 DINO encoder.")


def vit_small(patch_size=14, num_register_tokens=0, depth=12, **kwargs):
    return DAV3DinoVisionTransformer(
        {
            "patch_size": patch_size,
            "embed_dim": 384,
            "depth": depth,
            "num_heads": 6,
            "mlp_ratio": 4,
            "num_register_tokens": num_register_tokens,
            "out_layers": [5, 7, 9, 11],
            "alt_start": 4,
            "qknorm_start": 4,
            "rope_start": 4,
            "cat_token": True,
            **kwargs,
        }
    )


def vit_base(patch_size=14, num_register_tokens=0, depth=12, **kwargs):
    return DAV3DinoVisionTransformer(
        {
            "patch_size": patch_size,
            "embed_dim": 768,
            "depth": depth,
            "num_heads": 12,
            "mlp_ratio": 4,
            "num_register_tokens": num_register_tokens,
            "out_layers": [5, 7, 9, 11],
            "alt_start": 4,
            "qknorm_start": 4,
            "rope_start": 4,
            "cat_token": True,
            **kwargs,
        }
    )


def vit_large(patch_size=14, num_register_tokens=0, depth=24, **kwargs):
    return DAV3DinoVisionTransformer(
        {
            "patch_size": patch_size,
            "embed_dim": 1024,
            "depth": depth,
            "num_heads": 16,
            "mlp_ratio": 4,
            "num_register_tokens": num_register_tokens,
            "out_layers": [11, 15, 19, 23],
            "alt_start": 8,
            "qknorm_start": 8,
            "rope_start": 8,
            "cat_token": True,
            **kwargs,
        }
    )


def vit_giant2(patch_size=14, num_register_tokens=0, depth=40, **kwargs):
    return DAV3DinoVisionTransformer(
        {
            "patch_size": patch_size,
            "embed_dim": 1536,
            "depth": depth,
            "num_heads": 24,
            "mlp_ratio": 4,
            "num_register_tokens": num_register_tokens,
            "out_layers": [19, 27, 33, 39],
            "alt_start": 13,
            "qknorm_start": 13,
            "rope_start": 13,
            "cat_token": True,
            **kwargs,
        }
    )


if __name__ == "__main__":
    import torch
    from safetensors.torch import load_file

    from splatfactory.utils import mappings

    sd = load_file("weights/da3-small.safetensor")

    # remove leading 'model.' from state dict keys
    sd = {k[len("model.") :]: v for k, v in sd.items()}
    # remove head keys
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    torch.save(sd, "weights/da3-small.pth")
    ckpt = torch.load("weights/da3-small.pth", map_location="cpu", weights_only=True)

    model = vit_small(weights="weights/da3-small.pth")
    sample = torch.randn(2, 5, 3, 518, 518)
    out = model({"image": sample})

    mappings.print_summary(out)
    for feat in out["feats"]:
        print([f.shape for f in feat])
