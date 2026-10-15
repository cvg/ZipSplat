"""TokenBA: test-time bundle adjustment of a ZipSplat prediction in token space.

Self-supervised against the input views only: the input-view poses and the scene tokens are
refined so the rendered input views match the images. The frozen GaussianHead maps the scene
tokens to Gaussians and acts as a learned prior.

Steps:
    1. Camera update. Robust multi-start PnP of every pose on the patch-Gaussian
       correspondences (zipsplat/pnp.py), then a short photometric polish (PoseAligner)
       that removes the sub-patch error.
    2. Token update. One Adam step on the scene tokens under the fixed poses, L1 rendering
       loss on the input views through the frozen decoder.
    Steps 1-2 alternate until a camera update moves every pose less than `round_tol_deg`.
    3. Joint refinement. Scene tokens AND poses (se3 deltas) in one optimizer until the loss
       plateaus.

The refinement runs under TF32 matmuls; the decoder's MLP body runs in fp16 (`half_decoder`),
its readout always in fp32. Port of splatfactory/geometry/token_ba.py.

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

PATCH_PX = 14


@contextlib.contextmanager
def tf32_matmuls():
    """Scoped TF32 matmuls (the BA runs in TF32; the caller's setting is restored)."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


class TokenBA:
    """Bundle adjustment of the scene tokens and input poses through the frozen decoder.

    Args:
        rounds: Cap on the camera / token alternation (it exits on round_tol_deg).
        round_tol_deg: Stop the alternation once a camera update moves every pose less.
        token_lr: Adam lr of the token update.
        joint_steps: Cap on the joint refinement steps.
        joint_window, joint_tol: The joint refinement exits when a `joint_window`-step block
            improves the running loss by less than `joint_tol` (relative); None runs all
            `joint_steps`.
        joint_token_lr, joint_pose_lr: Adam lrs of the joint refinement.
        half_decoder: Decode the head's MLP body in fp16 while optimizing (the readout stays
            fp32). The returned Gaussians are always decoded in full precision.
        verbose: Progress bars for the alternation and the joint refinement.
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
            head, scene_tokens: The frozen GaussianHead and the scene tokens (1, K, D).
            camera, pose, image: Input cameras and c2w poses (1, V), images (1, V, 3, H, W).
            assignments: Per-token cluster ids (1, V * T) of a compressed prediction.
            encoder_camera, encoder_hw: The patch grid the tokens were read from, when it is
                coarser than the images (default: the images' own grid).

        Returns:
            The refined c2w poses (1, V).
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
        """Refine the scene tokens and poses jointly until the loss plateaus.

        Continues the alternation's optimizer (whose only group holds the scene tokens): the
        token moments carry over and only the token lr changes. Returns the refined c2w poses
        (1, V).
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
            head: The (frozen) GaussianHead that decoded them.
            camera: Cameras (1, V) of the images.
            pose: Initial c2w poses (1, V), in the Gaussians' frame (e.g. the predicted ones).
            image: Input views (1, V, 3, H, W) in [0, 1].
            assignments: Per-token cluster ids (1, V * T) of a compressed prediction.
            encoder_camera, encoder_hw: The cameras and size of the patch grid the tokens were
                read from, when the BA supervises at a higher resolution.

        Returns:
            gaussians (1, N), poses (1, V) and scene_tokens (1, K, D), all refined.
        """
        assert image.dim() == 5 and image.shape[0] == 1, "expects batched B == 1"
        pose = pose.detach()
        grid = dict(assignments=assignments, encoder_camera=encoder_camera, encoder_hw=encoder_hw)
        with tf32_matmuls(), torch.enable_grad():
            # clone: .float() is a no-op on fp32 and Adam steps in place
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

        with torch.no_grad():  # full-precision decode of the refined tokens
            gaussians = head(scene_tokens)
        return {"gaussians": gaussians, "poses": pose, "scene_tokens": scene_tokens.detach()}
