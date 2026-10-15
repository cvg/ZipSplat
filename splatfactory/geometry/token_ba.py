"""TokenBA: test-time bundle adjustment of a ZipSplat prediction in token space.

Self-supervised on the context views: the scene tokens and the context poses are refined so the
rendered context views match the input images. The frozen GaussianHead decodes the tokens to
Gaussians and acts as a learned prior.

Steps:
    1. Camera update. Robust PnP of every context pose on the patch-Gaussian correspondences
       (geometry/utils/pnp.py), then a short photometric polish that removes the sub-patch
       error.
    2. Token update. One Adam step on the scene tokens under the fixed poses, with the L1
       rendering loss of the context views.
    Steps 1 and 2 alternate until no camera update moves a pose by more than `round_tol_deg`.
    3. Joint refinement. Scene tokens and pose deltas in one optimizer until the loss plateaus.

Usage:
    tba = TokenBA({})
    out = tba({"scene_tokens": tokens, "head": head, "camera": cam, "pose": pose, "image": img})
    out["gaussians"], out["pose"]

Author: Alexander Veicht
"""

import contextlib
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from splatfactory.gaussians import Gaussians
from splatfactory.geometry.cameras import Camera
from splatfactory.geometry.pose_aligner import PoseAligner
from splatfactory.geometry.poses import Pose
from splatfactory.geometry.utils.pnp import patch_correspondences, robust_pnp
from splatfactory.models import BaseModel
from splatfactory.models.decoders.gaussian_head import GaussianHead

# PoseAligner conf of the photometric polish after the PnP: two shallow levels
POLISH = {
    "levels": [[0.5, 2, 0.0], [1.0, 1, 0.0]],
    "verify": False,
    "full_res_guard": False,
    "global_scale": False,
    "rel_tol": 1e-3,
}
PATCH_PX = 14  # side of a token's image patch in pixels


@contextlib.contextmanager
def tf32_matmuls():
    """TF32 matmuls inside the block, the previous precision restored after."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


class TokenBA(BaseModel):
    """Bundle adjustment of the scene tokens and context poses through the frozen decoder."""

    default_conf = {
        "rounds": 4,  # cap on the camera / token alternation
        "round_tol_deg": 0.3,  # the alternation stops once no pose moves by more than this
        "token_lr": 1e-3,  # Adam lr of the token update
        # Joint refinement: at most `steps` steps, stopped when a `window`-step block lowers the
        # mean loss by less than `tol` (relative, None runs all steps).
        "joint": {"steps": 96, "window": 8, "tol": 0.02, "token_lr": 2e-3, "pose_lr": 3e-4},
        "half_decoder": True,  # fp16 MLP body while optimizing, ~7% faster at no quality loss
        "verbose": False,
    }
    required_data_keys = ["scene_tokens", "head", "camera", "pose", "image"]

    def _init(self, conf):
        self.polish = PoseAligner(POLISH)

    def decode(self, head: GaussianHead, scene_tokens: Tensor, half: bool = False) -> Gaussians:
        """Gaussians [1, N] of the scene tokens [1, K, D] through the frozen head.

        `half` runs the head's MLP body in fp16. The readout always runs in fp32.
        """
        means, scales, quats, opacities, sh_coeffs = head.decode(scene_tokens, half_mlp=half)
        return Gaussians.from_parameters(
            means=means, scales=scales, quats=quats, opacities=opacities, sh_coeffs=sh_coeffs
        )

    def _step(
        self,
        optimizer: torch.optim.Optimizer,
        head: GaussianHead,
        scene_tokens: Tensor,
        camera: Camera,
        pose: Pose,
        image: Tensor,
        render_kwargs: Dict[str, Any],
    ) -> Tensor:
        """One Adam step on the L1 rendering loss of the context views. Returns the loss."""
        optimizer.zero_grad(set_to_none=True)
        gaussians = self.decode(head, scene_tokens, half=self.conf.half_decoder)
        render = gaussians.render_view(cameras=camera, poses=pose, **render_kwargs)["rendering"]
        loss = F.l1_loss(render, image)
        loss.backward()
        optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def camera_update(
        self,
        head: GaussianHead,
        scene_tokens: Tensor,
        camera: Camera,
        pose: Pose,
        image: Tensor,
        assignments: Optional[Tensor] = None,
        encoder_camera: Optional[Camera] = None,
        encoder_hw: Optional[Sequence[int]] = None,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Pose:
        """Robust PnP on the patch correspondences, then the photometric polish.

        Args:
            head: The frozen GaussianHead.
            scene_tokens: Scene tokens [1, K, D].
            camera: Context cameras [1, V].
            pose: Current c2w context poses [1, V].
            image: Context images [1, V, 3, H, W].
            assignments: Per-token cluster ids [1, V * T] of a compressed prediction.
            encoder_camera: Cameras of the patch grid the tokens were read from, when it is
                coarser than the images. Default: `camera`.
            encoder_hw: Size of that patch grid. Default: the image size.
            render_kwargs: Rasterizer arguments of the polish renders.

        Returns:
            The refined c2w context poses [1, V].
        """
        gaussians = self.decode(head, scene_tokens, half=self.conf.half_decoder)
        pts, obs = patch_correspondences(
            gaussians,
            image.shape[-2:] if encoder_hw is None else encoder_hw,
            image.shape[1],
            assignments=assignments,
            gaussians_per_token=gaussians.num_gaussians // scene_tokens.shape[1],
            patch_px=PATCH_PX,
        )
        pose = robust_pnp(pose, camera if encoder_camera is None else encoder_camera, pts, obs)
        data = {"gaussians": gaussians, "camera": camera, "pose": pose, "image": image}
        if render_kwargs:
            data["render_kwargs"] = render_kwargs
        return self.polish(data)["pose"].detach()

    def token_update(
        self,
        optimizer: torch.optim.Optimizer,
        head: GaussianHead,
        scene_tokens: Tensor,
        camera: Camera,
        pose: Pose,
        image: Tensor,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """One rendering-loss step on the scene tokens under fixed poses. Returns the loss."""
        return self._step(
            optimizer, head, scene_tokens, camera, pose.detach(), image, render_kwargs or {}
        )

    def joint_refinement(
        self,
        optimizer: torch.optim.Optimizer,
        head: GaussianHead,
        scene_tokens: Tensor,
        camera: Camera,
        pose: Pose,
        image: Tensor,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Pose:
        """Refine the scene tokens and context poses jointly until the loss plateaus.

        Reuses the alternation's optimizer, whose only group holds the scene tokens. The token
        moments carry over and only the token lr changes, so the hand-off starts without a
        fresh-Adam transient. The poses are refined as se3 deltas on top of `pose`.

        Returns:
            The refined c2w context poses [1, V].
        """
        conf = self.conf.joint
        pose0 = pose.detach()
        se3 = nn.Parameter(torch.zeros(*pose0.shape, 6, device=image.device))  # rot | trans
        optimizer.param_groups[0]["lr"] = conf.token_lr
        optimizer.add_param_group({"params": [se3], "lr": conf.pose_lr})
        window, prev = torch.zeros((), device=image.device), None
        pbar = tqdm(total=conf.steps, desc="TokenBA joint", disable=not self.conf.verbose)
        for step in range(conf.steps):
            pose = pose0 @ Pose.from_aa(se3[..., :3], se3[..., 3:])
            window = window + self._step(
                optimizer, head, scene_tokens, camera, pose, image, render_kwargs or {}
            )
            pbar.update()
            if (step + 1) % conf.window == 0:  # one sync per window, not per step
                cur = window.item() / conf.window
                window = torch.zeros((), device=image.device)
                pbar.set_postfix(loss=f"{cur:.4f}")
                if conf.tol is not None and prev is not None and (prev - cur) / prev < conf.tol:
                    break
                prev = cur
        pbar.close()
        with torch.no_grad():
            return pose0 @ Pose.from_aa(se3[..., :3], se3[..., 3:])

    def _forward(self, data):
        """Refine a prediction on its context views.

        Args:
            data: Batched with B == 1.
                scene_tokens: Scene tokens [1, K, D] of the prediction.
                head: The frozen GaussianHead that decoded them.
                camera: Context cameras [1, V].
                pose: Initial c2w context poses [1, V] in the Gaussians' frame.
                image: Context images [1, V, 3, H, W].
                render_kwargs: Optional rasterizer arguments.
                assignments: Optional per-token cluster ids [1, V * T] of a compressed
                    prediction.
                encoder_camera: Optional cameras of the patch grid the tokens were read from,
                    when it is coarser than the images.
                encoder_hw: Optional size of that patch grid.

        Returns:
            gaussians: The refined Gaussians [1, N], decoded in full precision.
            pose: The refined c2w context poses [1, V].
            camera: The context cameras, unchanged.
            scene_tokens: The refined scene tokens [1, K, D].
        """
        assert data["image"].dim() == 5 and data["image"].shape[0] == 1, "expects batched B == 1"
        head, camera, image = data["head"], data["camera"], data["image"]
        pose = data["pose"].detach()
        render_kwargs = data.get("render_kwargs") or {}
        grid = {k: data.get(k) for k in ("assignments", "encoder_camera", "encoder_hw")}

        # callers such as the eval export run under no_grad, the refinement needs gradients
        with tf32_matmuls(), torch.enable_grad():
            # copy: .float() is a no-op on fp32 tokens and Adam steps in place, so the caller's
            # tokens would change silently
            scene_tokens = nn.Parameter(data["scene_tokens"].detach().float().clone())
            optimizer = torch.optim.Adam([scene_tokens], lr=self.conf.token_lr)
            rounds = tqdm(
                range(self.conf.rounds), desc="TokenBA rounds", disable=not self.conf.verbose
            )
            for _ in rounds:
                new_pose = self.camera_update(
                    head, scene_tokens, camera, pose, image, render_kwargs=render_kwargs, **grid
                )
                moved = (pose.inv() @ new_pose).magnitude()[0].max().item()
                pose = new_pose
                self.token_update(optimizer, head, scene_tokens, camera, pose, image, render_kwargs)
                rounds.set_postfix(moved=f"{moved:.2f} deg")
                if moved < self.conf.round_tol_deg:
                    break
            rounds.close()
            pose = self.joint_refinement(
                optimizer, head, scene_tokens, camera, pose, image, render_kwargs
            )

        with torch.no_grad():
            gaussians = self.decode(head, scene_tokens)
        return {
            "gaussians": gaussians,
            "pose": pose,
            "camera": camera,
            "scene_tokens": scene_tokens.detach(),
        }

    def loss(self, pred, data):
        raise NotImplementedError("TokenBA has no loss.")

    def metrics(self, pred, data):
        raise NotImplementedError("TokenBA has no metrics.")
