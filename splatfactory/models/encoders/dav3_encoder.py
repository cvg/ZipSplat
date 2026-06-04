import torch
import torch.nn as nn
from torch import nn

from splatfactory.models import BaseModel
from splatfactory.models.encoders.dav3_dino import vit_base, vit_giant2, vit_large, vit_small
from splatfactory.models.modules.transformer_block import Mlp, SelfAttentionBlock
from splatfactory.utils.conversions import matrix_to_quaternion

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

    quat_wxyz = matrix_to_quaternion(R)  # Your function outputs WXYZ
    quat = quat_wxyz[..., [1, 2, 3, 0]]  # Convert to XYZW

    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()

    return pose_encoding


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


class CameraEnc(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

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
        c2ws = affine_inverse(ext)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, ixt, image_size)
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens


class DAV3Encoder(BaseModel):
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
        )

        self.embed_dim = self.backbone.embed_dim

        self.cam_enc = None
        if conf.with_camera_enc:
            self.cam_enc = CameraEnc(dim_out=self.backbone.embed_dim)
            if conf.get("freeze_camera_enc", False):
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
            img, self.conf.out_layers, cam_token=cam_token, export_feat_layers=export_feat_layers
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
    args = parser.parse_args()

    src = args.input or f"weights/da3-{args.size}.safetensor"
    dst = args.output or f"weights/da3-{args.size}.pth"

    sd = load_file(src)
    # DA3 ships keys as 'model.pretrained.<...>'; strip the wrappers to match DAV3Encoder.
    sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
    sd = {k.replace(".pretrained", ""): v for k, v in sd.items()}
    torch.save(sd, dst)
    print(f"Converted {src} -> {dst} ({len(sd)} tensors)")
