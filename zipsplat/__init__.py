"""ZipSplat — feed-forward 3DGS from multi-view images.

Author: Alexander Veicht
"""

import logging

from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.pose import Pose
from zipsplat.predictor import ZipSplat
from zipsplat.utils import load_image, load_video

__all__ = ["ZipSplat", "Camera", "Pose", "Gaussians", "load_image", "load_video"]

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
