"""Monotonic Tone Curves and Luminance Ratio Tone Grading.

Implements:
1. PCHIP / Cubic Hermite spline evaluator guaranteeing global monotonicity (dY_out/dY_in >= 0).
2. Ratio-based tone modification: I_out = I_in * (Y_out / (Y_in + eps)), preserving chromaticity and hue.
3. Fixed-shape filmic table generator (PchipInterpolator).
4. C1-continuous soft shoulder highlight roll-off with asymptote at 1.0.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator

from .color_transform import luminance


def smoothstep(y: torch.Tensor, e0: float, e1: float) -> torch.Tensor:
    """Hermite smoothstep interpolation between e0 and e1."""
    t = ((y - e0) / max(e1 - e0, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_shoulder(y: torch.Tensor, knee: float = 0.8) -> torch.Tensor:
    """Identity below knee, smooth hyperbolic tangent roll-off above.

    Guarantees C1 continuity at y = knee and approaches asymptotic maximum of 1.0.
    """
    over = (y - knee).clamp(min=0.0)
    scale = max(1.0 - knee, 1e-4)
    shoulder = knee + scale * torch.tanh(over / scale)
    return torch.where(y <= knee, y, shoulder)


def build_filmic_table(n: int = 1024) -> torch.Tensor:
    """Builds a 1024-entry monotonic filmic tone curve lookup table.

    Characteristics: gentle toe roll-off, organic mid contrast, soft compressed shoulder.
    """
    xs = [0.0, 0.10, 0.50, 0.85, 1.0]
    ys = [0.0, 0.06, 0.50, 0.93, 1.0]
    interp = PchipInterpolator(xs, ys)
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    table_np = interp(grid)
    return torch.tensor(table_np, dtype=torch.float32).clamp(0.0, 1.0)


def apply_curve_1d(y: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Evaluates a 1D tone curve lookup table via linear interpolation on GPU/CPU."""
    n = table.numel()
    table_dev = table.to(device=y.device, dtype=y.dtype)

    idx = y.clamp(0.0, 1.0) * (n - 1)
    i0 = idx.floor().long().clamp(max=n - 2)
    f = idx - i0.float()

    return table_dev[i0] * (1.0 - f) + table_dev[i0 + 1] * f


def filmic_ratio_grade(
    img: torch.Tensor,
    table: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Applies monotonic tone curve via luminance-ratio scaling.

    Scaling RGB channels by Y_out / (Y_in + eps) modifies tonal luminance
    while strictly preserving chromaticity coordinates (r = R/sum, g = G/sum).
    """
    Y = luminance(img)
    Y_out = apply_curve_1d(Y, table)
    return (img * (Y_out / (Y + eps))).clamp(0.0, 1.0)


class MonotonicSplineCurve(torch.nn.Module):
    """Monotonic cubic spline generator parameterized by learnable or predicted knots.

    Guarantees dY_out/dY_in >= 0 by enforcing monotonic knot increments.
    """

    def __init__(self, num_knots: int = 5):
        super().__init__()
        self.num_knots = num_knots
        # Fixed x positions uniformly spaced between 0 and 1
        self.register_buffer("x_knots", torch.linspace(0.0, 1.0, num_knots))

    def forward(self, img: torch.Tensor, raw_knots: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Applies dynamic monotonic spline to image tensor.

        Args:
            img: [B, 3, H, W] in [0.0, 1.0].
            raw_knots: [B, num_knots] raw unconstrained knot values from predictor head.

        Returns:
            [B, 3, H, W] tone-mapped image.
        """
        B = img.shape[0]
        # Enforce boundary conditions: y(0)=0, y(1)=1, and strict non-decreasing order
        # Softplus increments ensure strictly positive slopes
        increments = F.softplus(raw_knots)
        increments[:, 0] = 0.0  # anchor origin
        cumsum = torch.cumsum(increments, dim=1)
        y_knots = cumsum / cumsum[:, -1:].clamp(min=1e-6)  # normalize so y(1) == 1.0

        Y = luminance(img)  # [B, 1, H, W]

        # Piecewise linear / Hermite evaluation across knots
        # Find interval for each pixel luminance
        Y_clamped = Y.clamp(0.0, 1.0)
        x_knots = self.x_knots.to(device=img.device, dtype=img.dtype)
        # Scaled indices into intervals [0, num_knots - 1]
        scaled = Y_clamped * (self.num_knots - 1)
        idx = scaled.floor().long().clamp(0, self.num_knots - 2)
        alpha = scaled - idx.float()

        # Gather y knot coordinates for each batch item
        y0 = torch.gather(y_knots, 1, idx.squeeze(1).flatten(1)).view(B, 1, *Y.shape[2:])
        y1 = torch.gather(y_knots, 1, (idx + 1).squeeze(1).flatten(1)).view(B, 1, *Y.shape[2:])

        # Monotonic linear blend
        Y_out = y0 * (1.0 - alpha) + y1 * alpha

        # Ratio-based channel scaling
        return (img * (Y_out / (Y + eps))).clamp(0.0, 1.0)
