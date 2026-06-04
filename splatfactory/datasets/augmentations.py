"""Photometric / color augmentations for training.

ONLY color/photometric transforms are allowed here - no spatial transforms
(resize, crop, flip, rotate). Spatial transforms are handled by ImagePreprocessor
which tracks the 3x3 camera transform matrix.

All augmentations operate on HWC uint8 numpy arrays.

Author: Alexander Veicht
"""

from collections.abc import Callable

import albumentations as A
import cv2
import numpy as np
from omegaconf import OmegaConf


class RandomAdditiveShade(A.ImageOnlyTransform):
    """Overlay random shadow ellipses on the image."""

    def __init__(
        self, nb_ellipses=10, transparency_limit=(-0.5, 0.8), kernel_size_limit=(150, 350), p=0.5
    ):
        super().__init__(p=p)
        self.nb_ellipses = nb_ellipses
        self.transparency_limit = transparency_limit
        self.kernel_size_limit = kernel_size_limit

    def apply(self, img, **params):
        if img.dtype == np.uint8:
            return self._shade(img.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return self._shade(img * 255.0).clip(0, 255) / 255.0

    def _shade(self, img):
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        for _ in range(self.nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)
        transparency = np.random.uniform(*self.transparency_limit)
        ks = np.random.randint(*self.kernel_size_limit)
        if ks % 2 == 0:
            ks += 1
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
        return img * (1 - transparency * mask[..., np.newaxis] / 255.0)

    def get_transform_init_args_names(self):
        return "transparency_limit", "kernel_size_limit", "nb_ellipses"


def _build_strong(conf) -> A.Compose:
    """Strong photometric augmentation preset."""
    d = conf.get("difficulty", 1.0)
    return A.Compose(
        [
            A.MultiplicativeNoise(
                multiplier=(1.0 - 0.2 * d, 1.0 + 0.2 * d),
                per_channel=True,
                elementwise=False,
                p=conf.get("mult_noise", 0.3),
            ),
            A.OneOf(
                [
                    A.MultiplicativeNoise(
                        multiplier=(1.0 - 0.1 * d, 1.0 + 0.1 * d),
                        per_channel=True,
                        elementwise=True,
                        p=0.34,
                    ),
                    A.GaussNoise(
                        std_range=(0.01 * d, 0.05 * d),
                        p=0.66,
                    ),
                ],
                p=conf.get("noise", 0.2),
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 3 + 2 * round(6 * d)), p=0.2, allow_shifted=False),
                    A.MedianBlur(blur_limit=(3, 3 + 2 * round(2 * d)), p=0.1),
                    A.Blur(blur_limit=(3, 3 + 2 * round(2 * d)), p=0.1),
                ],
                p=conf.get("blur", 0.3),
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=(2, 8), p=1.0),
                    A.Sharpen(p=1.0),
                    A.Emboss(p=1.0),
                ],
                p=conf.get("sharpen", 0.4),
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.4 * d, -0.1 * d),
                        contrast_limit=(-0.4 * d, -0.1 * d),
                        p=0.50,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(0.1 * d, 0.2 * d),
                        contrast_limit=(0.1 * d, 0.4 * d),
                        p=0.50,
                    ),
                ],
                p=conf.get("bright_contr", 0.75),
            ),
            A.RandomGamma(
                p=conf.get("gamma", 0.15),
                gamma_limit=(100 - int(70 * d), 100 + int(50 * d)),
            ),
            A.HueSaturationValue(
                p=conf.get("hue", 0.75),
                val_shift_limit=(-20 * d, 20 * d),
                hue_shift_limit=(-30 * d, 30 * d),
                sat_shift_limit=(-50 * d, 50 * d),
            ),
            A.RandomToneCurve(p=conf.get("tone", 0.2), scale=0.7 * d),
            RandomAdditiveShade(
                p=conf.get("shade", 0.5),
                transparency_limit=(-0.2 * d, 0.8 * d),
            ),
        ],
        p=conf.get("p", 0.95),
    )


def _identity(image: np.ndarray) -> np.ndarray:
    return image


class _AlbumentationsAugmentation:
    """Picklable wrapper around an albumentations pipeline."""

    def __init__(self, pipeline: A.Compose):
        self.pipeline = pipeline

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.pipeline(image=image)["image"]


def get_augmentation(conf) -> Callable[[np.ndarray], np.ndarray]:
    """Create a color augmentation callable from config.

    Args:
        conf: OmegaConf or dict with at least "name" key.
            name="identity": no augmentation (passthrough).
            name="strong": strong photometric augmentation.

    Returns:
        Callable: HWC uint8 numpy -> HWC uint8 numpy.
    """
    if isinstance(conf, dict):
        conf = OmegaConf.create(conf)
    name = conf.get("name", "identity")

    if name == "identity":
        return _identity

    if name == "strong":
        return _AlbumentationsAugmentation(_build_strong(conf))

    raise ValueError(f"Unknown augmentation: '{name}'. Available: 'identity', 'strong'.")
