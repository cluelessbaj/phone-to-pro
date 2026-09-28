"""Edge-Aware Guided Filter Implementations.

Implements both standard FastGuidedFilter (He et al. 2010) and
SubsampledGuidedFilter (He & Sun 2015, s=4) for joint/edge-aware upsampling.

Includes verified border-darkening fix:
Manual reflect-padding prior to avg_pool2d prevents zero-padded border darkening
vignettes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGuidedFilter(nn.Module):
    """Standard Guided Filter for edge-aware upsampling using full-resolution guidance."""

    def __init__(self, radius: int = 16, eps: float = 1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def box_filter(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """Computes local box mean using reflect padding to prevent border darkening.

        Standard avg_pool2d with zero-padding counts zeros in the denominator near
        boundaries, causing artificial darkening bands (vignetting). Reflect padding
        preserves true local boundary statistics.
        """
        # Ensure radius does not exceed spatial dimensions
        h, w = x.shape[2:]
        r_clamped = max(1, min(r, min(h, w) - 1))

        x_padded = F.pad(x, (r_clamped, r_clamped, r_clamped, r_clamped), mode="reflect")
        return F.avg_pool2d(x_padded, kernel_size=2 * r_clamped + 1, stride=1, padding=0)

    def forward(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Applies edge-aware guided filtering.

        Args:
            guide: Full-resolution luminance guide [B, 1, H, W] in [0.0, 1.0].
            src:   Low-resolution spatial map [B, C, H_s, W_s].

        Returns:
            [B, C, H, W] edge-aligned full-resolution spatial map.
        """
        # Upsample src to guide resolution as initial baseline if needed
        if src.shape[2:] != guide.shape[2:]:
            src = F.interpolate(src, size=guide.shape[2:], mode="bilinear", align_corners=False)

        r, eps = self.radius, self.eps
        mean_I = self.box_filter(guide, r)
        mean_p = self.box_filter(src, r)
        mean_Ip = self.box_filter(guide * src, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter(guide * guide, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a, r)
        mean_b = self.box_filter(b, r)

        return mean_a * guide + mean_b


class SubsampledGuidedFilter(nn.Module):
    """Subsampled Guided Filter (He & Sun 2015).

    Computes covariance statistics at 1/s resolution, then interpolates linear
    coefficients (a, b) back to guide resolution. Cuts computational complexity
    by up to s^2 while preserving sharp edge boundaries.
    """

    def __init__(self, radius: int = 16, eps: float = 1e-3, s: int = 4):
        super().__init__()
        self.radius = radius
        self.eps = eps
        self.s = s

    @staticmethod
    def _box(x: torch.Tensor, r: int) -> torch.Tensor:
        """Reflect-padded 2D box filter."""
        h, w = x.shape[2:]
        r_clamped = max(1, min(r, min(h, w) - 1))
        x_padded = F.pad(x, (r_clamped, r_clamped, r_clamped, r_clamped), mode="reflect")
        return F.avg_pool2d(x_padded, kernel_size=2 * r_clamped + 1, stride=1, padding=0)

    def forward(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Applies subsampled guided filtering.

        Args:
            guide: Full-resolution luminance guide [B, 1, H, W] in [0.0, 1.0].
            src:   Low-resolution control map [B, C, H_s, W_s].

        Returns:
            [B, C, H, W] edge-guided upsampled map.
        """
        H, W = guide.shape[2:]
        h, w = max(H // self.s, 2), max(W // self.s, 2)

        # 1. Downsample guide and align src to intermediate (h, w) grid
        I_sub = F.adaptive_avg_pool2d(guide, (h, w))
        p_sub = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=False)

        # 2. Compute statistics at subsampled resolution
        r_sub = max(1, min(self.radius // self.s, min(h, w) - 1))
        mI = self._box(I_sub, r_sub)
        mp = self._box(p_sub, r_sub)
        cov = self._box(I_sub * p_sub, r_sub) - mI * mp
        var = self._box(I_sub * I_sub, r_sub) - mI * mI

        # 3. Linear model parameters a, b
        a = cov / (var + self.eps)
        b = mp - a * mI

        # 4. Filter parameters and upsample back to (H, W)
        a_smooth = self._box(a, r_sub)
        b_smooth = self._box(b, r_sub)

        a_up = F.interpolate(a_smooth, size=(H, W), mode="bilinear", align_corners=False)
        b_up = F.interpolate(b_smooth, size=(H, W), mode="bilinear", align_corners=False)

        # 5. Output reconstructed at full resolution
        return a_up * guide + b_up
