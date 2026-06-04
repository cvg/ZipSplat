"""Image operations: format conversion, spatial transforms, I/O, and preprocessing.

Single home for all image manipulation. Functions accept both numpy arrays
and torch tensors where it makes sense, dispatching internally.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image as PILImage
from torch import nn

# --- Format conversion --------------------------------------------------------


def to_tensor(image):
    """Convert image to CHW float32 tensor.

    Accepts HWC uint8 numpy, HW numpy (grayscale), or passthrough if already a tensor.
    """
    if isinstance(image, torch.Tensor):
        return image
    if image.ndim == 3:
        image = image.transpose(2, 0, 1)  # HWC -> CHW
    elif image.ndim == 2:
        image = image[None]  # HW -> 1HW
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)


def to_numpy(image):
    """Convert image to HWC uint8 numpy.

    Accepts CHW float32 tensor, 1HW tensor (grayscale), or passthrough if already numpy.
    """
    if isinstance(image, np.ndarray):
        return image
    x = image.detach().cpu()
    if x.ndim == 3:
        x = x.permute(1, 2, 0)  # CHW -> HWC
    elif x.ndim == 2:
        pass  # HW stays HW
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")
    return (x * 255.0).clamp(0, 255).to(torch.uint8).numpy()


def chw_from_hwc(x):
    """...HWC -> ...CHW. Works on both numpy arrays and torch tensors."""
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -1, -3)
    return x.transpose(-2, -1).transpose(-3, -2)


def hwc_from_chw(x):
    """...CHW -> ...HWC. Works on both numpy arrays and torch tensors."""
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -3, -1)
    return x.transpose(-3, -2).transpose(-2, -1)


# --- Spatial operations -------------------------------------------------------
# Each returns (result, transform_3x3) where transform is a numpy float64 3x3 matrix.


def _get_hw(image):
    """Get (h, w) from numpy HWC/HW or torch ...CHW/...HW."""
    if isinstance(image, np.ndarray):
        return image.shape[0], image.shape[1]
    return image.shape[-2], image.shape[-1]


def _round_even(x):
    """Round to nearest even integer. Ensures symmetric center crops for camera cx/cy."""
    return 2 * round(x / 2)


def resize_context(data: dict, target_size: int) -> dict:
    """Resize context images (and update intrinsics) so the short edge is `target_size`.

    Used by baselines to send a downscaled image to their encoder while leaving
    `data["target"]` untouched so rendering uses the original eval-res cameras.
    """
    context = data["context"]
    images = context["image"]
    B, V, C, H, W = images.shape
    if min(H, W) == target_size:
        return data

    flat = images.flatten(0, 1)
    pairs = [resize_to_cover(im, target_size, target_size) for im in flat]
    out = torch.stack([p[0] for p in pairs])
    H_new, W_new = out.shape[-2:]
    images_new = out.view(B, V, C, H_new, W_new)

    new_context = {**context, "image": images_new}
    if "camera" in context:
        s = torch.tensor([W_new / W, H_new / H], device=images.device, dtype=images.dtype)
        new_context["camera"] = context["camera"].scale(s)
    return {**data, "context": new_context}


def resize_to_cover(image, target_h, target_w, method: str = "torch_bilinear_aa"):
    """Uniformly scale so both dims >= target. Preserves aspect ratio.

    Args:
        image: HWC uint8 numpy or CHW float torch tensor.
        target_h, target_w: minimum output dimensions.
        method: "pil_lanczos" (default), "torch_bilinear_aa", or "cv2_inter_area".

    Returns:
        (image, transform): resized image, 3x3 scale matrix.
    """
    h, w = _get_hw(image)
    scale = max(target_h / h, target_w / w)
    h_new, w_new = _round_even(h * scale), _round_even(w * scale)

    if h_new == h and w_new == w:
        return image, np.eye(3)

    is_tensor = isinstance(image, torch.Tensor)

    if method == "pil_lanczos":
        np_img = to_numpy(image) if is_tensor else image
        resized = np.array(PILImage.fromarray(np_img).resize((w_new, h_new), PILImage.LANCZOS))
        if is_tensor:
            resized = to_tensor(resized).to(device=image.device, dtype=image.dtype)
    elif method == "torch_bilinear_aa":
        img_tens = to_tensor(image) if not is_tensor else image
        resized_tens = F.interpolate(
            img_tens[None], size=(h_new, w_new), mode="bilinear", antialias=True
        )[0]
        resized = resized_tens if is_tensor else to_numpy(resized_tens)
    elif method == "cv2_inter_area":
        np_img = to_numpy(image) if is_tensor else image
        resized = cv2.resize(np_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        if is_tensor:
            resized = to_tensor(resized).to(device=image.device, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown resize method: {method!r}")

    transform = np.diag([w_new / w, h_new / h, 1.0])
    return resized, transform


def crop_to_principal_point(image, cx, cy):
    """Crop largest rectangle with principal point at center, preserving original AR.

    The crop is the largest sub-image where (cx, cy) maps to the exact center,
    with the same aspect ratio as the input. Even dimensions guarantee no
    off-by-one on the principal point.

    Args:
        image: HWC numpy or CHW torch tensor.
        cx, cy: principal point in pixel coordinates.

    Returns:
        (image, transform): cropped image, 3x3 translation matrix.
    """
    h, w = _get_hw(image)
    ar = w / h

    half_w = min(cx, w - cx)
    half_h = min(cy, h - cy)

    # Constrain to original AR
    if half_w / half_h > ar:
        half_w = half_h * ar
    else:
        half_h = half_w / ar

    w_new = 2 * int(half_w)
    h_new = 2 * int(half_h)

    x0 = int(cx - w_new / 2)
    y0 = int(cy - h_new / 2)

    if w_new == w and h_new == h:
        return image, np.eye(3)

    if isinstance(image, np.ndarray):
        cropped = image[y0 : y0 + h_new, x0 : x0 + w_new]
    else:
        cropped = image[..., y0 : y0 + h_new, x0 : x0 + w_new]

    transform = np.eye(3)
    transform[0, 2] = -x0
    transform[1, 2] = -y0
    return cropped, transform


def crop_to_ar(image, aspect_ratio):
    """Center-crop to match target aspect ratio (W/H).

    ar > 1 = landscape, ar < 1 = portrait, ar = 1 = square.
    Crops whichever dimension exceeds the target AR.

    Args:
        image: HWC numpy or CHW torch tensor.
        aspect_ratio: target W/H ratio.

    Returns:
        (image, transform): cropped image, 3x3 translation matrix.
    """
    h, w = _get_hw(image)
    target_w = _round_even(h * aspect_ratio)

    if target_w <= w:
        # Crop width
        cx = (w - target_w) // 2
        cy = 0
        h_new, w_new = h, target_w
    else:
        # Crop height
        target_h = _round_even(w / aspect_ratio)
        cy = (h - target_h) // 2
        cx = 0
        h_new, w_new = target_h, w

    if cx == 0 and cy == 0 and h_new == h and w_new == w:
        return image, np.eye(3)

    if isinstance(image, np.ndarray):
        cropped = image[cy : cy + h_new, cx : cx + w_new]
    else:
        cropped = image[..., cy : cy + h_new, cx : cx + w_new]

    transform = np.eye(3)
    transform[0, 2] = -cx
    transform[1, 2] = -cy
    return cropped, transform


def crop_to_divisible(image, divisor):
    """Center-crop edges to nearest multiple of divisor.

    Removes at most (divisor - 1) pixels per edge. Never resizes.

    Args:
        image: HWC numpy or CHW torch tensor.
        divisor: integer divisor.

    Returns:
        (image, transform): cropped image, 3x3 translation matrix.
    """
    h, w = _get_hw(image)
    h_new = h // divisor * divisor
    w_new = w // divisor * divisor

    if h_new == h and w_new == w:
        return image, np.eye(3)

    cy = (h - h_new) // 2
    cx = (w - w_new) // 2

    if isinstance(image, np.ndarray):
        cropped = image[cy : cy + h_new, cx : cx + w_new]
    else:
        cropped = image[..., cy : cy + h_new, cx : cx + w_new]

    transform = np.eye(3)
    transform[0, 2] = -cx
    transform[1, 2] = -cy
    return cropped, transform


# --- Depth --------------------------------------------------------------------


def resize_depth(depth, target_hw, invalid_value=-1.0):
    """Resize a depth map with proper invalid pixel handling.

    Invalid pixels (< 0) are set to NaN before resize so they propagate
    through INTER_AREA averaging. Any output pixel whose source region
    overlaps an invalid pixel becomes invalid.

    Args:
        depth: (H, W) float32 numpy array, negative = invalid.
        target_hw: (H_target, W_target).
        invalid_value: value for invalid pixels in output.

    Returns:
        (H_target, W_target) float32 numpy array.
    """
    h_target, w_target = target_hw
    d = depth.astype(np.float32).copy()
    d[d < 0] = np.nan
    resized = cv2.resize(d, (w_target, h_target), interpolation=cv2.INTER_AREA)
    resized[np.isnan(resized)] = invalid_value
    return resized


# --- I/O ----------------------------------------------------------------------


def read_image(path, grayscale=False, as_tensor=False):
    """Read an image from disk.

    Args:
        path: file path.
        grayscale: read as grayscale.
        as_tensor: if True, return CHW float32 tensor; else HWC uint8 numpy.

    Returns:
        HWC uint8 numpy array, or CHW float32 tensor if as_tensor=True.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]  # BGR -> RGB
    if as_tensor:
        return to_tensor(image)
    return image


# --- Normalization ------------------------------------------------------------


class ImageNetNormalizer(nn.Module):
    """Normalize images with ImageNet mean and std."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x):
        return ((x.transpose(-3, -1) - self.mean) / self.std).transpose(-3, -1)


# --- Composer -----------------------------------------------------------------


class ImagePreprocessor:
    """Chains: resize_to_cover -> crop_to_ar -> crop_to_divisible -> [augment] -> to_tensor.

    Pipeline operates on numpy uint8 throughout, converts to tensor at the end.
    Uses cv2.INTER_AREA for downscaling (fast, accurate box filter on uint8).
    Depth follows the same spatial ops with nearest interpolation.
    All spatial ops compose into a single 3x3 transform for camera intrinsics update.

    conf.resize = long edge of the final output (before divisible crop).
    aspect_ratio is passed at call time (W/H).
    """

    default_conf = {
        "resize": None,  # long edge of output, None for no resize
        "edge_divisible_by": None,  # crop to nearest multiple
        "resize_method": "torch_bilinear_aa",  # "torch_bilinear_aa" or "pil_lanczos"
    }

    def __init__(self, conf, augmentation=None):
        """
        Args:
            conf: OmegaConf or dict with preprocessing config.
            augmentation: optional callable (HWC uint8 numpy -> HWC uint8 numpy).
        """
        default_conf = OmegaConf.create(self.default_conf)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        self.augmentation = augmentation

    def __call__(self, image, depth=None, aspect_ratio=1.0):
        """Preprocess an image (and optional depth map).

        Args:
            image: HWC uint8 numpy array. If CHW float tensor, converted to numpy first.
            depth: (H, W) float32 numpy or tensor, optional. Values < 0 are invalid.
            aspect_ratio: target W/H ratio. 1.0 = square.

        Returns:
            dict with:
                "image": (3, H, W) float32 tensor
                "transform": (3, 3) float64 ndarray
                "depth": (H, W) float32 tensor (if depth provided)
                "depth_mask": (H, W) bool tensor (if depth provided)
        """
        if isinstance(image, torch.Tensor):
            image = to_numpy(image)

        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth = depth.astype(np.float32).copy()
            depth[depth < 0] = np.nan

        transform = np.eye(3)

        # 1. Resize to cover target dimensions
        if self.conf.resize is not None:
            target_h, target_w = self._compute_target(aspect_ratio)
            image, t = resize_to_cover(image, target_h, target_w, method=self.conf.resize_method)
            transform = t @ transform
            if depth is not None:
                h, w = image.shape[:2]
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Crop to target AR
        image, t = crop_to_ar(image, aspect_ratio)
        transform = t @ transform
        if depth is not None:
            h, w = image.shape[:2]
            dh, dw = depth.shape
            if dh != h or dw != w:
                cy, cx = int(-t[1, 2]), int(-t[0, 2])
                depth = depth[cy : cy + h, cx : cx + w]

        # 3. Crop to divisible
        if self.conf.edge_divisible_by is not None:
            image, t = crop_to_divisible(image, self.conf.edge_divisible_by)
            transform = t @ transform
            if depth is not None:
                h, w = image.shape[:2]
                dh, dw = depth.shape
                if dh != h or dw != w:
                    cy, cx = int(-t[1, 2]), int(-t[0, 2])
                    depth = depth[cy : cy + h, cx : cx + w]

        # 4. Color augmentation (on uint8, after spatial)
        if self.augmentation is not None:
            image = self.augmentation(image)

        # 5. To tensor
        result = {
            "image": to_tensor(image),
            "transform": transform,
        }

        if depth is not None:
            depth[np.isnan(depth)] = -1.0
            depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).float()
            result["depth"] = depth_tensor
            result["depth_mask"] = depth_tensor > 0

        return result

    def _compute_target(self, aspect_ratio):
        """Compute (target_h, target_w) from resize and aspect_ratio."""
        if aspect_ratio >= 1.0:  # landscape: W >= H
            target_w = self.conf.resize
            target_h = round(self.conf.resize / aspect_ratio)
        else:  # portrait: H > W
            target_h = self.conf.resize
            target_w = round(self.conf.resize * aspect_ratio)
        return target_h, target_w
