"""MobileNetV3 Dual-Head Parameter Regressor.

Predicts aesthetic grade parameters from low-resolution proxy thumbnails (256-512px):
- Sony Head: Global basis LUT blend weights (w) + Monotonic spline control points (S).
- Apple Head: Global LUT weights (w) + Edge-guided spatial weights (W_s) + Low-res gain map (G_s).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DualHeadPredictor(nn.Module):
    """Lightweight shared MobileNetV3 backbone with task-specific parameter regression heads."""

    def __init__(
        self,
        num_luts: int = 5,
        spline_points: int = 5,
        spatial_grid_size: int = 16,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_luts = num_luts
        self.spline_points = spline_points
        self.spatial_grid_size = spatial_grid_size

        # Backbone: MobileNetV3-Small (lightweight < 2.5M params, fast on CPU oneDNN)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # In-features from MobileNetV3-Small feature extractor (576 channels)
        in_dim = base.classifier[0].in_features

        # =====================================================================
        # Head A: Sony Mode (Global LUT weights + 1D Spline points)
        # =====================================================================
        self.sony_lut_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_luts),
        )
        self.sony_spline_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, spline_points),
        )

        # =====================================================================
        # Head B: Apple Mode (Global LUT weights + Spatial weight map + Gain map)
        # =====================================================================
        self.apple_lut_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_luts),
        )
        # Output: num_luts channels for spatial LUT blend + 1 channel illumination gain
        self.apple_spatial_head = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((spatial_grid_size, spatial_grid_size)),
            nn.Conv2d(64, num_luts + 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, mode: str = "sony") -> Dict[str, torch.Tensor]:
        """Predicts parametric control vectors from proxy thumbnail.

        Args:
            x: [B, 3, H, W] normalized float32 thumbnail tensor (e.g. 256x256).
            mode: 'sony' or 'apple'.

        Returns:
            Dictionary of predicted parameters:
            - 'sony': {'lut_weights': [B, K], 'spline_points': [B, S]}
            - 'apple': {'lut_weights': [B, K], 'spatial_weights': [B, K, H_s, W_s], 'gain_map': [B, 1, H_s, W_s]}
        """
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)

        mode_lower = mode.lower()
        if mode_lower == "sony":
            lut_weights = F.softmax(self.sony_lut_head(pooled), dim=1)
            spline_pts = torch.sigmoid(self.sony_spline_head(pooled))
            return {
                "lut_weights": lut_weights,
                "spline_points": spline_pts,
            }
        elif mode_lower == "apple":
            lut_weights = F.softmax(self.apple_lut_head(pooled), dim=1)
            spatial_maps = self.apple_spatial_head(feat)

            # Separate spatial LUT weights and illumination gain map
            spatial_weights = F.softmax(spatial_maps[:, : self.num_luts, :, :], dim=1)
            # Sigmoid * 2.0 allows up to 2x local shadow lift
            gain_map = torch.sigmoid(spatial_maps[:, self.num_luts :, :, :]) * 2.0

            return {
                "lut_weights": lut_weights,
                "spatial_weights": spatial_weights,
                "gain_map": gain_map,
            }
        else:
            raise ValueError(f"Unknown predictor mode: {mode}. Expected 'sony' or 'apple'")
