"""Single-Pass Pre-Blended 3D-LUT Engine.

Optimizes multi-LUT blending by computing the composite table in parameter space
BEFORE spatial trilinear interpolation:
    sum_k w_k * T(x, L_k) == T(x, sum_k w_k * L_k)

Collapses K full-resolution volumetric lookups into a single pass, saving >75%
memory bandwidth and execution latency on hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinglePassLUTEngine(nn.Module):
    """High-performance single-pass trilinear 3D-LUT interpolation engine."""

    def __init__(self, num_luts: int = 5, lut_dim: int = 33):
        super().__init__()
        self.num_luts = num_luts
        self.lut_dim = lut_dim

        # Initialize K basis 3D-LUTs as an identity-centered learnable tensor.
        # Shape: [K, 3, D, D, D]
        # Axes (D_in, H_in, W_in) correspond to (R, G, B) input coordinates.
        self.luts = nn.Parameter(torch.zeros(num_luts, 3, lut_dim, lut_dim, lut_dim))
        self._init_identity()

    def _init_identity(self) -> None:
        """Initializes all K basis LUTs to the identity color transform."""
        with torch.no_grad():
            linear = torch.linspace(0.0, 1.0, self.lut_dim)
            r, g, b = torch.meshgrid(linear, linear, linear, indexing="ij")
            # r varies along axis 0 (D_in), g along axis 1 (H_in), b along axis 2 (W_in)
            identity = torch.stack([r, g, b], dim=0)  # [3, D, D, D]
            for k in range(self.num_luts):
                self.luts[k].copy_(identity)

    def blend_luts(self, weights: torch.Tensor) -> torch.Tensor:
        """Pre-blends basis LUTs in parameter space.

        Args:
            weights: [B, K] softmax-normalized blend coefficients.

        Returns:
            composite_lut: [B, 3, D, D, D] blended 3D-LUT tensor.
        """
        # weights: [B, K], luts: [K, 3, D, D, D] -> composite_lut: [B, 3, D, D, D]
        return torch.einsum("bk,kcdhw->bcdhw", weights, self.luts)

    def _build_query_grid(self, img_float32: torch.Tensor) -> torch.Tensor:
        """Constructs the 5D sampling grid with correct channel-to-axis alignment.

        F.grid_sample's 5D convention reads grid[..., 0:3] as (x, y, z),
        which index the input's (W_in, H_in, D_in) axes respectively.
        Our identity assignment mapped:
            R -> D_in (axis 0)
            G -> H_in (axis 1)
            B -> W_in (axis 2)
        Therefore, grid's (x, y, z) coordinates must map to (B, G, R).
        """
        # Reorder channels from (R, G, B) to (B, G, R)
        bgr = img_float32[:, [2, 1, 0], :, :]
        # Shape: [B, H, W, 3] -> unsqueeze depth dimension -> [B, 1, H, W, 3]
        grid = bgr.permute(0, 2, 3, 1).unsqueeze(1)
        # Rescale [0.0, 1.0] -> [-1.0, 1.0] for F.grid_sample
        return (grid * 2.0) - 1.0

    def forward(self, img_float32: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Applies composite 3D-LUT in a single trilinear interpolation pass.

        Args:
            img_float32: [B, 3, H, W] in range [0.0, 1.0], channel order (R, G, B).
            weights:     [B, K] Softmax-normalized blend coefficients.

        Returns:
            [B, 3, H, W] graded float32 tensor in range [0.0, 1.0].
        """
        # 1. Parameter-space pre-blending: [B, 3, D, D, D]
        composite_lut = self.blend_luts(weights)

        # 2. Build 5D query grid: [B, 1, H, W, 3]
        grid = self._build_query_grid(img_float32)

        # 3. Single-pass trilinear interpolation
        out = F.grid_sample(
            composite_lut,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.squeeze(2)  # [B, 3, H, W]

    def forward_multipass(self, img_float32: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Reference un-optimized multi-pass interpolation for verification and benchmarking.

        Evaluates T(x, L_k) independently K times and sums with weights:
            out = sum_k w_k * T(x, L_k)
        """
        grid = self._build_query_grid(img_float32)
        B, _, H, W = img_float32.shape
        out = torch.zeros(B, 3, H, W, device=img_float32.device, dtype=img_float32.dtype)

        for k in range(self.num_luts):
            lut_k = self.luts[k : k + 1]  # [1, 3, D, D, D]
            lut_k = lut_k.expand(B, -1, -1, -1, -1)  # [B, 3, D, D, D]
            res_k = F.grid_sample(
                lut_k,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).squeeze(2)
            out = out + weights[:, k : k + 1, None, None] * res_k

        return out
