"""TokenBA: test-time bundle adjustment of a ZipSplat prediction in token space.

Self-supervised on the input views: the scene tokens and the input poses are refined so the
rendered views match the input images. The frozen GaussianHead decodes the tokens to Gaussians
and acts as a learned prior.

Steps:
    1. Camera update. Robust PnP of every input pose on the patch-Gaussian correspondences
       (zipsplat/pnp.py), then a short photometric polish that removes the sub-patch error.
    2. Token update. One Adam step on the scene tokens under the fixed poses, with the L1
       rendering loss of the input views.
    Steps 1 and 2 alternate until no camera update moves a pose by more than `round_tol_deg`.
    3. Joint refinement. Scene tokens and pose deltas in one optimizer until the loss plateaus.

Usage:
    ba = TokenBA(verbose=True)
    out = ba(scene_tokens, head, camera, pose, image)  # batched, B == 1
    out["gaussians"], out["poses"]

Author: Alexander Veicht
"""

import contextlib
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from zipsplat.camera import Camera
from zipsplat.heads import GaussianHead
from zipsplat.pnp import patch_correspondences, robust_pnp
from zipsplat.pose import Pose
from zipsplat.pose_aligner import PoseAligner

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


class TokenBA:
    """Bundle adjustment of the scene tokens and input poses through the frozen decoder.

    Args:
        rounds: Cap on the camera / token alternation.
        round_tol_deg: The alternation stops once no pose moves by more than this.
        token_lr: Adam lr of the token update.
        joint_steps: Cap on the joint refinement steps.
        joint_window: Block length of the joint refinement's stopping test.
        joint_tol: The joint refinement stops when a block lowers the mean loss by less than
            this (relative). None runs all `joint_steps`.
        joint_token_lr: Adam lr of the scene tokens in the joint refinement.
        joint_pose_lr: Adam lr of the pose deltas in the joint refinement.
        half_decoder: fp16 MLP body while optimizing, ~7% faster at no quality loss.
        verbose: Progress bars.
    """

    def __init__(
        self,
        rounds: int = 4,
        round_tol_deg: float = 0.3,
        token_lr: float = 1e-3,
        joint_steps: int = 96,
        joint_window: int = 8,
        joint_tol: Optional[float] = 0.02,
        joint_token_lr: float = 2e-3,
        joint_pose_lr: float = 3e-4,
        half_decoder: bool = True,
        verbose: bool = False,
    ) -> None:
        self.rounds, self.round_tol_deg, self.token_lr = rounds, round_tol_deg, token_lr
        self.joint_steps, self.joint_window, self.joint_tol = joint_steps, joint_window, joint_tol
        self.joint_token_lr, self.joint_pose_lr = joint_token_lr, joint_pose_lr
        self.half_decoder, self.verbose = half_decoder, verbose
        self.polish = PoseAligner()

    def _step(self, optimizer, head, scene_tokens, camera, pose, image) -> Tensor:
        """One Adam step on the L1 rendering loss of the input views. Returns the loss."""
        optimizer.zero_grad(set_to_none=True)
        gaussians = head(scene_tokens, half_mlp=self.half_decoder)
        loss = F.l1_loss(gaussians.render(camera, pose)[0], image)
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
    ) -> Pose:
        """Robust PnP on the patch correspondences, then the photometric polish.

        Args:
            head: The frozen GaussianHead.
            scene_tokens: Scene tokens (1, K, D).
            camera: Input cameras (1, V).
            pose: Current c2w input poses (1, V).
            image: Input images (1, V, 3, H, W).
            assignments: Per-token cluster ids (1, V * T) of a compressed prediction.
            encoder_camera: Cameras of the patch grid the tokens were read from, when it is
                coarser than the images. Default: `camera`.
            encoder_hw: Size of that patch grid. Default: the image size.

        Returns:
            The refined c2w input poses (1, V).
        """
        gaussians = head(scene_tokens, half_mlp=self.half_decoder)
        pts, obs = patch_correspondences(
            gaussians,
            image.shape[-2:] if encoder_hw is None else encoder_hw,
            image.shape[1],
            assignments=assignments,
            gaussians_per_token=head.gaussians_per_token,
            patch_px=PATCH_PX,
        )
        pose = robust_pnp(pose, camera if encoder_camera is None else encoder_camera, pts, obs)
        return self.polish(gaussians, camera, pose, image)

    def token_update(
        self,
        optimizer: torch.optim.Optimizer,
        head: GaussianHead,
        scene_tokens: Tensor,
        camera: Camera,
        pose: Pose,
        image: Tensor,
    ) -> Tensor:
        """One rendering-loss step on the scene tokens under fixed poses. Returns the loss."""
        return self._step(optimizer, head, scene_tokens, camera, pose, image)

    def joint_refinement(
        self,
        optimizer: torch.optim.Optimizer,
        head: GaussianHead,
        scene_tokens: Tensor,
        camera: Camera,
        pose: Pose,
        image: Tensor,
    ) -> Pose:
        """Refine the scene tokens and input poses jointly until the loss plateaus.

        Reuses the alternation's optimizer, whose only group holds the scene tokens. The token
        moments carry over and only the token lr changes, so the hand-off starts without a
        fresh-Adam transient. The poses are refined as se3 deltas on top of `pose`.

        Returns:
            The refined c2w input poses (1, V).
        """
        pose0 = pose
        se3 = nn.Parameter(torch.zeros(*pose0.shape, 6, device=image.device))  # rot | trans

        optimizer.param_groups[0]["lr"] = self.joint_token_lr
        optimizer.add_param_group({"params": [se3], "lr": self.joint_pose_lr})

        window, prev = torch.zeros((), device=image.device), None

        pbar = tqdm(
            total=self.joint_steps, desc="TokenBA joint", disable=not self.verbose, leave=False
        )
        for step in range(self.joint_steps):
            pose = pose0 @ Pose.from_aa(se3[..., :3], se3[..., 3:])

            window = window + self._step(optimizer, head, scene_tokens, camera, pose, image)
            pbar.update()
            if (step + 1) % self.joint_window == 0:  # one sync per window, not per step
                cur = window.item() / self.joint_window
                window = torch.zeros((), device=image.device)
                pbar.set_postfix(loss=f"{cur:.4f}")

                if self.joint_tol is not None and prev is not None:
                    if (prev - cur) / prev < self.joint_tol:
                        break

                prev = cur

        pbar.close()
        return (pose0 @ Pose.from_aa(se3[..., :3], se3[..., 3:])).detach()

    def __call__(
        self,
        scene_tokens: Tensor,
        head: GaussianHead,
        camera: Camera,
        pose: Pose,
        image: Tensor,
        assignments: Optional[Tensor] = None,
        encoder_camera: Optional[Camera] = None,
        encoder_hw: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        """Refine a prediction on its input views.

        Args:
            scene_tokens: Scene tokens (1, K, D) of the prediction.
            head: The frozen GaussianHead that decoded them.
            camera: Input cameras (1, V).
            pose: Initial c2w input poses (1, V) in the Gaussians' frame.
            image: Input images (1, V, 3, H, W) in [0, 1].
            assignments: Per-token cluster ids (1, V * T) of a compressed prediction.
            encoder_camera: Cameras of the patch grid the tokens were read from, when it is
                coarser than the images.
            encoder_hw: Size of that patch grid.

        Returns:
            gaussians: The refined Gaussians (1, N), decoded in full precision.
            poses: The refined c2w input poses (1, V).
            scene_tokens: The refined scene tokens (1, K, D).
        """
        assert image.dim() == 5 and image.shape[0] == 1, "expects batched B == 1"
        pose = pose.detach()
        grid = dict(assignments=assignments, encoder_camera=encoder_camera, encoder_hw=encoder_hw)
        # the predictor runs under no_grad, the refinement needs gradients
        with tf32_matmuls(), torch.enable_grad():
            # copy: .float() is a no-op on fp32 tokens and Adam steps in place, so the caller's
            # tokens would change silently
            scene_tokens = nn.Parameter(scene_tokens.detach().float().clone())
            optimizer = torch.optim.Adam([scene_tokens], lr=self.token_lr)
            for _ in range(self.rounds):
                new_pose = self.camera_update(head, scene_tokens, camera, pose, image, **grid)
                moved = (pose.inv() @ new_pose).magnitude()[0].max().item()
                pose = new_pose
                self.token_update(optimizer, head, scene_tokens, camera, pose, image)
                if moved < self.round_tol_deg:
                    break
            pose = self.joint_refinement(optimizer, head, scene_tokens, camera, pose, image)

        with torch.no_grad():
            gaussians = head(scene_tokens)
        return {"gaussians": gaussians, "poses": pose, "scene_tokens": scene_tokens.detach()}
