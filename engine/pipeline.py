"""End-to-End Photo Enhancement Pipeline Coordinator.

Orchestrates the entire multi-stage processing graph:
1. Ingestion (sRGB Phone Capture).
2. Thumbnail Extraction (256-512px).
3. Super-Resolution Core (Float32 intermediate tensor).
4. Adaptive Normalization (Clamped Gray-World WB + Conservative Levels).
5. Aesthetic Color Grading (Sony Route A/B or Apple Photonic Guided Lift).
6. Adaptive Output Sharpening (Resolution-indexed MTF profile).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.srfactory import SRCore, create_sr_core
from .color_transform import luminance
from .baseline import (
    apple_shadow_lift,
    conservative_levels,
    gray_world_wb,
    sony_baseline_grade,
)
from .guided_filter import SubsampledGuidedFilter
from .lut_interp import SinglePassLUTEngine
from .sharpening import adaptive_sharpen
from .spline_curve import build_filmic_table


class PhotoEnhancementPipeline(nn.Module):
    """Unified Photo Enhancement Pipeline supporting both Sony Alpha and Apple Photonic modes."""

    def __init__(
        self,
        sr_scale: int = 4,
        sr_denoise: float = 0.5,
        guided_radius: int = 16,
        guided_eps: float = 1e-3,
        guided_subsample: int = 4,
        sr_weights_path: Optional[str] = None,
    ):
        super().__init__()
        # 1. Super-Resolution Core
        self.sr_core = create_sr_core(
            scale=sr_scale,
            denoise_strength=sr_denoise,
            weights_path=sr_weights_path,
        )

        # 2. Guided Filter for spatial upsampling
        self.guided_filter = SubsampledGuidedFilter(
            radius=guided_radius,
            eps=guided_eps,
            s=guided_subsample,
        )

        # 3. Parametric LUT Engine (initialized with identity tables)
        self.lut_engine = SinglePassLUTEngine(num_luts=5, lut_dim=33)

        # 4. Precomputed Filmic Table
        self.register_buffer("filmic_table", build_filmic_table(1024))

        # 5. Edge-Preserving Denoise Filter for sensor noise suppression
        self.denoise_filter = SubsampledGuidedFilter(radius=4, eps=1e-3, s=4)

        self.eval()

    @torch.no_grad()
    def enhance_tensor(
        self,
        img: torch.Tensor,
        mode: str = "sony",
        sony_route: str = "b",
        enable_sr: bool = True,
        enable_denoise: bool = True,
        enable_wb: bool = True,
        enable_levels: bool = True,
        enable_sharpening: bool = True,
    ) -> torch.Tensor:
        """Executes enhancement on a torch.Tensor [B, 3, H, W] in [0.0, 1.0]."""
        # Step 1: Extract proxy thumbnail (256x256)
        thumb = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)

        # Step 2: Super-Resolution Core or Edge-Preserving Denoising
        if enable_sr:
            hr_tensor = self.sr_core(img)
        else:
            hr_tensor = img.clone()

        if enable_denoise and not enable_sr:
            guide = luminance(hr_tensor)
            hr_tensor = self.denoise_filter(guide, hr_tensor)

        # Step 3: Normalization (clamped gray-world WB + conservative levels)
        norm_tensor = hr_tensor
        if enable_wb:
            norm_tensor = gray_world_wb(norm_tensor, max_gain=1.20, strength=0.8)
        if enable_levels:
            norm_tensor = conservative_levels(norm_tensor, thumb)

        # Step 4: Mode-specific Aesthetic Color Grading
        mode_lower = mode.lower()
        if mode_lower == "sony":
            graded = sony_baseline_grade(
                norm_tensor,
                route=sony_route,
                lut_engine=self.lut_engine,
                filmic_table=self.filmic_table,
            )
        elif mode_lower == "apple":
            graded = apple_shadow_lift(
                norm_tensor,
                self.guided_filter,
                thumb,
                target=0.45,
                r_max=3.5,
                gamma=0.5,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'sony' or 'apple'")

        # Step 5: Adaptive Output Sharpening
        if enable_sharpening:
            enhanced = adaptive_sharpen(graded, mode=mode_lower)
        else:
            enhanced = graded

        return enhanced.clamp(0.0, 1.0)

    def enhance_pil(
        self,
        pil_img: Image.Image,
        mode: str = "sony",
        sony_route: str = "b",
        **kwargs,
    ) -> Image.Image:
        """Convenience method accepting and returning PIL Images."""
        # Convert PIL to planar float32 tensor [1, 3, H, W]
        img_np = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        out_tensor = self.enhance_tensor(tensor, mode=mode, sony_route=sony_route, **kwargs)

        # Convert back to PIL Image
        out_np = (out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(out_np)
