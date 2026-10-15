import torch
from torch import nn

from splatfactory.models import BaseModel
from splatfactory.models.decoders.cam_dec import CameraDec
from splatfactory.models.encoders.dav3_dino import vit_base, vit_giant2, vit_large, vit_small
from splatfactory.models.modules.transformer_block import Mlp, SelfAttentionBlock
from splatfactory.utils.conversions import matrix_to_quaternion

# DA3 takes ImageNet-normalized images.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

encoder_map = {
    "vits": vit_small,
    "vitb": vit_base,
    "vitl": vit_large,
    "vitg": vit_giant2,
}


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=None):
    """Convert camera extrinsics and intrinsics to a compact pose encoding."""

    R = extrinsics[:, :, :3, :3]  # BxSx3x3
    T = extrinsics[:, :, :3, 3]  # BxSx3

    quat_wxyz = matrix_to_quaternion(R)
    quat = quat_wxyz[..., [1, 2, 3, 0]]  # wxyz -> xyzw

    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()

    return pose_encoding


class CameraEnc(nn.Module):
    """DA3's camera encoder: camera priors -> per-view camera tokens via a transformer trunk."""

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(
            *[
                SelfAttentionBlock(
                    dim=dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(
            in_features=dim_in, hidden_features=dim_out // 2, out_features=dim_out, drop=0
        )

    def forward(self, ext, ixt, image_size) -> tuple:
        """c2w poses [B, V, 4, 4] and intrinsics [B, V, 3, 3] -> camera tokens [B, V, D]."""
        pose_encoding = extri_intri_to_pose_encoding(ext, ixt, image_size)
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens


class DAV3Encoder(BaseModel):
    """The DA3 trunk with its camera encoder (camera priors) and camera decoder (pose readout).

    Holds the `backbone.*`, `cam_enc.*` and `cam_dec.*` modules of a DA3 weights file (the
    depth head is dropped). The forward returns the per-layer (patch, camera) tokens.
    """

    default_conf = {
        "vit_name": "vits",
        "out_layers": [5, 7, 9, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "cat_token": True,
        "use_checkpoint": False,
        "with_camera_enc": False,
    }

    def _init(self, conf):
        assert conf.vit_name in encoder_map.keys()
        encoder_fn = encoder_map[conf.vit_name]
        ffn_layer = "swiglufused" if conf.vit_name == "vitg" else "mlp"
        self.backbone = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=conf.alt_start,
            qknorm_start=conf.qknorm_start,
            rope_start=conf.rope_start,
            cat_token=conf.cat_token,
            use_checkpoint=conf.use_checkpoint,
            # the ViT is a BaseModel too, it needs its own flag to stay frozen
            trainable=conf.trainable,
        )

        self.embed_dim = self.backbone.embed_dim
        # Camera decoder on the deepest camera token (cat_token: local | global halves).
        self.cam_dec = CameraDec(dim_in=2 * self.embed_dim if conf.cat_token else self.embed_dim)

        self.cam_enc = None
        if conf.with_camera_enc:
            # Frozen: it only runs on prior-injected steps, so DDP would see unused params.
            self.cam_enc = CameraEnc(dim_out=self.backbone.embed_dim)
            self.cam_enc.requires_grad_(False)
            if hasattr(self.backbone, "camera_token"):
                self.backbone.camera_token.requires_grad_(False)

    def _forward(self, data):
        img = data["image"]

        cam_token = None
        if data.get("use_priors", False):
            ext, intr = data["pose"], data["camera"]
            cam_token = self.cam_enc(ext, intr, image_size=img.shape[-2:])

        export_feat_layers = data.get("export_feat_layers", [])
        feats, aux_feats = self.backbone.get_intermediate_layers(
            img,
            self.conf.out_layers,
            cam_token=cam_token,
            export_feat_layers=export_feat_layers,
            timestamps=data.get("timestamps", None),
        )

        return {"feats": feats, "aux_feats": aux_feats}

    def metrics(self, pred, data):
        raise NotImplementedError("Metrics not implemented for DAV3 DINO encoder.")

    def loss(self, pred, data):
        raise NotImplementedError("Loss not implemented for DAV3 DINO encoder.")


if __name__ == "__main__":
    # Convert a downloaded DA3 safetensors checkpoint into the .pth format expected by
    # DAV3Encoder (weights/da3-<size>.pth). Download the raw weights first, e.g.:
    #   huggingface-cli download depth-anything/DA3-GIANT model.safetensors \
    #       --local-dir weights && mv weights/model.safetensors weights/da3-giant.safetensor
    import argparse

    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(description="Convert a DA3 safetensors checkpoint to .pth")
    parser.add_argument("--size", choices=["small", "base", "large", "giant"], default="giant")
    parser.add_argument(
        "--input", default=None, help="Input safetensors (default: weights/da3-<size>.safetensor)"
    )
    parser.add_argument(
        "--output", default=None, help="Output .pth (default: weights/da3-<size>.pth)"
    )
    # DA3NESTED-* checkpoints hold the any-view net under `model.da3.*`, the only source of
    # the retrained DA3-GIANT-1.1 weights.
    parser.add_argument(
        "--nested", action="store_true", help="Input is a DA3NESTED-* ckpt: take model.da3.*"
    )
    args = parser.parse_args()

    src = args.input or f"weights/da3-{args.size}.safetensor"
    dst = args.output or f"weights/da3-{args.size}.pth"

    sd = load_file(src)
    if args.nested:
        pre = "model.da3."
        sd = {"model." + k[len(pre) :]: v for k, v in sd.items() if k.startswith(pre)}
    # DA3 ships keys as 'model.pretrained.<...>', strip the wrappers to match DAV3Encoder.
    sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
    sd = {k.replace(".pretrained", ""): v for k, v in sd.items()}
    torch.save(sd, dst)
    print(f"Converted {src} -> {dst} ({len(sd)} tensors)")
