"""Resolution-Indexed Adaptive Output Sharpening Engine.

Features:
1. Mode-tailored sharpening:
   - Sony Mode: Fine-radius single-scale unsharp mask (r=0.5px, amount=0.35) reproducing
     optical prime lens MTF curves.
   - Apple Mode: Dual-scale unsharp mask (r1=0.8px, r2=1.5px) bounded by an edge-detection
     protection mask to boost micro-contrast without sharpening noise in flat/sky regions.
2. Dynamic resolution indexing:
   - Scaled proportional to output megapixels to ensure consistent visual sharpening
     across preview (1080p) and full-res capture (12MP - 48MP).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .color_transform import luminance
from .spline_curve import smoothstep


def create_gaussian_kernel1d(radius: float) -> torch.Tensor:
    """Generates a normalized 1D Gaussian kernel for the given radius (sigma)."""
    sigma = max(radius, 0.1)
    k_size = int(math.ceil(sigma * 3.0)) * 2 + 1
    x = torch.arange(k_size, dtype=torch.float32) - (k_size // 2)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur2d(img: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable 2D Gaussian blur with reflect padding to avoid boundary darkening."""
    if radius <= 0.05:
        return img

    B, C, H, W = img.shape
    k1d = create_gaussian_kernel1d(radius).to(device=img.device, dtype=img.dtype)
    k_size = k1d.numel()
    pad = k_size // 2

    # Horizontal pass
    kernel_h = k1d.view(1, 1, 1, k_size).expand(C, 1, 1, k_size)
    padded_h = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    blurred_h = F.conv2d(padded_h, kernel_h, groups=C)

    # Vertical pass
    kernel_v = k1d.view(1, 1, k_size, 1).expand(C, 1, k_size, 1)
    padded_v = F.pad(blurred_h, (0, 0, pad, pad), mode="reflect")
    blurred_v = F.conv2d(padded_v, kernel_v, groups=C)

    return blurred_v


def compute_resolution_scale(H: int, W: int, reference_mp: float = 12.0) -> float:
    """Computes a sharpening radius scaling factor indexed to output megapixels."""
    current_mp = (H * W) / 1e6
    if current_mp <= 0:
        return 1.0
    # Scale with square root of MP ratio to track linear pixel density
    return max(0.5, min(math.sqrt(current_mp / reference_mp), 3.0))


def sony_sharpen(
    img: torch.Tensor,
    base_radius: float = 0.5,
    amount: float = 0.35,
    scale_with_resolution: bool = True,
) -> torch.Tensor:
    """Fine-radius single-scale unsharp mask for Sony Alpha look."""
    H, W = img.shape[2:]
    scale = compute_resolution_scale(H, W) if scale_with_resolution else 1.0
    r_eff = base_radius * scale

    blurred = gaussian_blur2d(img, r_eff)
    high_freq = img - blurred
    sharpened = img + amount * high_freq
    return sharpened.clamp(0.0, 1.0)


def apple_sharpen(
    img: torch.Tensor,
    r1: float = 0.8,
    r2: float = 1.5,
    amount1: float = 0.30,
    amount2: float = 0.20,
    threshold_low: float = 0.015,
    threshold_high: float = 0.08,
    scale_with_resolution: bool = True,
) -> torch.Tensor:
    """Dual-scale unsharp mask with edge-protection mask for Apple Photonic look."""
    H, W = img.shape[2:]
    scale = compute_resolution_scale(H, W) if scale_with_resolution else 1.0
    r1_eff = r1 * scale
    r2_eff = r2 * scale

    # 1. Compute multi-scale high frequencies
    blur1 = gaussian_blur2d(img, r1_eff)
    blur2 = gaussian_blur2d(img, r2_eff)
    detail1 = img - blur1
    detail2 = img - blur2

    # 2. Compute edge detection mask from luminance
    lum = luminance(img)
    lum_blur = gaussian_blur2d(lum, r1_eff)
    edge_gradient = (lum - lum_blur).abs()

    # Mask: 0.0 in flat areas (suppressing noise amplification), 1.0 along distinct edges
    edge_mask = smoothstep(edge_gradient, threshold_low, threshold_high)

    # 3. Apply edge-masked sharpening
    boost = edge_mask * (amount1 * detail1 + amount2 * detail2)
    sharpened = img + boost
    return sharpened.clamp(0.0, 1.0)


def adaptive_sharpen(
    img: torch.Tensor,
    mode: str = "sony",
    scale_with_resolution: bool = True,
) -> torch.Tensor:
    """Applies mode-tailored adaptive sharpening to output tensor."""
    if mode.lower() == "sony":
        return sony_sharpen(img, scale_with_resolution=scale_with_resolution)
    elif mode.lower() == "apple":
        return apple_sharpen(img, scale_with_resolution=scale_with_resolution)
    else:
        raise ValueError(f"Unknown sharpening mode: {mode}")
