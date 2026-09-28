"""Zero-Training Algorithmic Baseline Engine (Milestone 0).

Provides deterministic non-learning enhancement paths:
1. Normalization: Clamped gray-world WB + conservative levels.
2. Sony Alpha Mode:
   - Route (a): sRGB -> Linear -> S-Gamut3.Cine/S-Log3 -> Sony Look LUT
   - Route (b): Display-referred filmic S-curve applied via luminance-ratio scaling.
3. Apple Photonic Mode:
   - Coarse Retinex-style illumination gain map -> SubsampledGuidedFilter (s=4)
   - Highlight-protected gain modulation + soft shoulder tone roll-off.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .color_transform import luminance, to_slog3_input
from .guided_filter import SubsampledGuidedFilter
from .lut_interp import SinglePassLUTEngine
from .sharpening import adaptive_sharpen
from .spline_curve import (
    apply_curve_1d,
    build_filmic_table,
    filmic_ratio_grade,
    smoothstep,
    soft_shoulder,
)


def gray_world_wb(
    img: torch.Tensor,
    max_gain: float = 1.25,
    strength: float = 1.0,
) -> torch.Tensor:
    """Clamped Gray-World White Balance.

    Ignores extreme clipped highlights (> 0.95) and near-black shadows (< 0.05)
    to prevent sensor boundary skew. Bounds gain adjustments within [1/max_gain, max_gain].
    """
    mx = img.amax(dim=1, keepdim=True)
    mn = img.amin(dim=1, keepdim=True)
    mask = ((mx < 0.95) & (mn > 0.05)).float()

    count = mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
    # Sum valid pixels per channel
    channel_sum = (img * mask).sum(dim=(2, 3), keepdim=True)
    mean = channel_sum / count.clamp(min=1.0)  # [B, 3, 1, 1]

    # Target grey value is the average of channel means
    target = mean.mean(dim=1, keepdim=True)
    gains = (target / mean.clamp(min=1e-4)).clamp(1.0 / max_gain, max_gain)

    # Modulate by user strength; if no valid pixels exist, fallback to identity
    has_valid = (count > 0).float()
    effective_gains = 1.0 + strength * (gains - 1.0)
    final_gains = has_valid * effective_gains + (1.0 - has_valid)

    return (img * final_gains).clamp(0.0, 1.0)


def conservative_levels(
    img: torch.Tensor,
    thumb: torch.Tensor,
    black_cap: float = 0.06,
    white_floor: float = 0.85,
) -> torch.Tensor:
    """Partial black and white point fit based on thumbnail luminance percentiles.

    Prevents aggressive clipping: never clips beyond the 1st/99th percentiles,
    enforces conservative black/white threshold caps, and protects bright highlights
    from clipping blowout.
    """
    Y = luminance(thumb).flatten(start_dim=1)  # [B, H*W]
    p1 = torch.quantile(Y, 0.01, dim=1)
    p99 = torch.quantile(Y, 0.99, dim=1)

    b = p1.clamp(max=black_cap)[:, None, None, None]
    
    # Highlight protection: if image contains bright highlights (> white_floor),
    # respect peak luminance so highlights and glowing structures are not clipped.
    Y_max = luminance(img).flatten(start_dim=1).amax(dim=1)
    w_effective = torch.where(Y_max > white_floor, torch.max(p99, Y_max), p99.clamp(min=white_floor))
    w = w_effective[:, None, None, None]

    return ((img - b) / (w - b).clamp(min=1e-3)).clamp(0.0, 1.0)


def apple_shadow_lift(
    img: torch.Tensor,
    guided_filter: SubsampledGuidedFilter,
    thumb: torch.Tensor,
    target: float = 0.45,
    r_max: float = 4.0,
    gamma: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Retinex-style local illumination gain map with edge-guided upsampling.

    Computes gain at coarse illumination scale (32x32) to avoid local halos,
    upsamples using high-res luminance guide, protects specular highlights,
    and applies soft shoulder compression.
    """
    Y = luminance(img)  # Full-resolution luminance guide [B, 1, H, W]
    # Coarse illumination proxy (area-averaged blur)
    Y_s = F.adaptive_avg_pool2d(luminance(thumb), (32, 32))

    # Coarse gain computed BEFORE upsampling
    G_s = (target / (Y_s + 1e-3)).clamp(1.0, r_max) ** gamma  # [B, 1, 32, 32]

    # Edge-aware upsample to full resolution
    G = guided_filter(Y, G_s)

    # Highlight protection: pixels above 0.6 luminance gradually fade out gain lift
    protect = 1.0 - smoothstep(Y, 0.60, 0.95)
    G_effective = 1.0 + (G - 1.0) * protect

    # Soft shoulder compression on lifted luminance
    Y_out = soft_shoulder(Y * G_effective, knee=0.8)

    # Ratio-based chromaticity-preserving application
    return (img * (Y_out / (Y + eps))).clamp(0.0, 1.0)


def sony_baseline_grade(
    img: torch.Tensor,
    route: str = "b",
    lut_engine: Optional[SinglePassLUTEngine] = None,
    filmic_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Applies Sony Alpha aesthetic color grading via Route (a) or Route (b)."""
    if route.lower() == "a":
        if lut_engine is None:
            raise ValueError("Route (a) requires an initialized SinglePassLUTEngine")
        # 1. Transform sRGB to S-Gamut3.Cine / S-Log3
        slog_input = to_slog3_input(img)
        # 2. Look LUT lookup
        weights = torch.zeros(img.shape[0], lut_engine.num_luts, device=img.device)
        weights[:, 0] = 1.0  # Apply first LUT
        return lut_engine(slog_input, weights)
    elif route.lower() == "b":
        table = filmic_table if filmic_table is not None else build_filmic_table(1024)
        return filmic_ratio_grade(img, table)
    else:
        raise ValueError(f"Unknown Sony route: {route}. Expected 'a' or 'b'")
