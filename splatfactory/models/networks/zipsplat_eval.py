"""ZipSplat for evaluation: eval-only settings and test-time bundle adjustment (TokenBA).

TokenBA refines the feed-forward prediction against the context views, pose-free: the context
cameras come from the model's own pose head. Only the Gaussians and the context poses change.
`token_ba: null` skips the BA.

Usage:
    model = get_model("networks.zipsplat_eval")(conf)  # conf.token_ba: TokenBA overrides
    pred = model(data)

Author: Alexander Veicht
"""

import contextlib

import torch
from omegaconf import OmegaConf

from splatfactory.geometry import Camera
from splatfactory.geometry.token_ba import TokenBA
from splatfactory.models.networks.zipsplat import ZipSplat
from splatfactory.utils.image import resize_context


class ZipSplatEval(ZipSplat):
    """ZipSplat + eval settings + TokenBA refinement of the prediction on the context views."""

    default_conf = {
        "return_attention": False,
        "token_ba": {},  # TokenBA overrides (rounds, joint, verbose, ...), null = no BA
        "amp": None,  # None (fp32 or the caller's autocast) | bfloat16
        "use_priors": False,  # inject the GT context cameras into the backbone
        "encoder_resize": None,  # forward input size, TokenBA keeps the input res (None = input)
        "render_kwargs": {},  # rasterizer args of the benchmark's aligned re-render
    }

    def _init(self, conf):
        super()._init(conf)
        self.gaussian_head._render_results = lambda *a, **k: {}
        self.ba = None
        if conf.token_ba is not None:
            self.ba = TokenBA(OmegaConf.to_container(conf.token_ba, resolve=True))

    def _use_priors(self):
        return self.conf.use_priors

    def refine(self, pred, image, encoder_hw=None):
        """Refine an existing prediction with TokenBA.

        Args:
            pred: Output of the feed-forward pass (needs pose_pred, camera_pred, scene_tokens).
            image: Context images [B, V, 3, H, W] TokenBA supervises on.
            encoder_hw: Resolution the tokens were read at, when lower than the images'.

        Returns:
            pred with the refined gaussians, context poses and scene tokens.
        """
        cam, (h, w) = pred["camera_pred"], image.shape[-2:]
        camera = Camera.from_fov(cam.hfov[0], cam.vfov[0], w=w, h=h)[None]
        # the benchmark's rasterizer args, so TokenBA refines what is scored
        rk = OmegaConf.to_container(self.conf.render_kwargs, resolve=True) or {}
        ba_data = {
            "camera": camera,
            "pose": pred["pose_pred"],
            "image": image,
            "render_kwargs": rk,
            "scene_tokens": pred["scene_tokens"],
            "head": self.gaussian_head,
            "assignments": pred["assignments"],
        }
        if encoder_hw is not None and tuple(encoder_hw) != (h, w):
            eh, ew = encoder_hw
            ba_data["encoder_hw"] = (eh, ew)
            ba_data["encoder_camera"] = Camera.from_fov(cam.hfov[0], cam.vfov[0], w=ew, h=eh)[None]
        out = self.ba(ba_data)
        return {
            **pred,
            "gaussians": out["gaussians"],
            "pose_pred": out["pose"],
            "scene_tokens": out["scene_tokens"],
        }

    def _forward(self, data):
        enc_size = self.conf.encoder_resize
        enc_data = data
        if enc_size is not None and min(data["context"]["image"].shape[-2:]) != enc_size:
            enc_data = resize_context(data, enc_size)

        # Only enable: a nested autocast(enabled=False) would switch off the caller's.
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.conf.amp == "bfloat16" else None
        with amp or contextlib.nullcontext():
            pred = super()._forward(enc_data)
        if self.ba is None:
            return pred
        return self.refine(pred, data["context"]["image"], enc_data["context"]["image"].shape[-2:])
