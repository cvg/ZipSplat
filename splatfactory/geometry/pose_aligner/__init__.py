"""Photometric pose alignment: refine camera poses so rendered Gaussians match target images.

The aligner is direct (correspondence-free): each step renders the Gaussians at the current
pose and linearises the photometric residual by brightness constancy, so a step costs ONE
forward render and no autograd.

    photometric.py  the gradients: image gradient, interaction matrix, pose Jacobian and
                    Gauss-Newton normal equations (fused Triton kernel or torch fallback).
    lm.py           PoseAligner: coarse-to-fine trust-region Levenberg-Marquardt. The
                    benchmarks' target-pose aligner and TokenBA's photometric polish.

Usage:
    from splatfactory.geometry.pose_aligner import PoseAligner
    pose = PoseAligner({})({"gaussians": g, "camera": cam, "pose": init, "image": img})["pose"]

Author: Alexander Veicht
"""

from splatfactory.geometry.pose_aligner.lm import PoseAligner

__all__ = ["PoseAligner"]
