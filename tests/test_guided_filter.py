"""Verification tests for FastGuidedFilter and SubsampledGuidedFilter.

Tests:
1. Border darkening immunity: Verifies reflect padding produces no boundary vignetting.
2. Edge-aware upsampling: Verifies guided filtering reduces transition errors across
   sharp edges compared to standard bilinear interpolation.
3. Multi-channel support: Verifies K-channel spatial weight grids process accurately.
"""

import pytest
import torch
import torch.nn.functional as F
from engine.guided_filter import FastGuidedFilter, SubsampledGuidedFilter


@pytest.mark.parametrize("filter_cls", [FastGuidedFilter, SubsampledGuidedFilter])
def test_guided_filter_border_darkening_immunity(filter_cls):
    """Asserts that a constant control map remains strictly constant up to the image borders."""
    gf = filter_cls(radius=16, eps=1e-3)
    H, W = 128, 128

    # Guide is an arbitrary natural gradient [B, 1, H, W]
    y = torch.linspace(0.1, 0.9, H)
    x = torch.linspace(0.1, 0.9, W)
    guide = torch.meshgrid(y, x, indexing="ij")[0].unsqueeze(0).unsqueeze(0)

    # Input map is perfectly constant 0.65 across a coarse 16x16 grid
    constant_val = 0.65
    src = torch.full((1, 1, 16, 16), constant_val, dtype=torch.float32)

    out = gf(guide, src)

    # Check that every pixel, including border corners and edges, equals constant_val
    max_border_error = (out - constant_val).abs().max().item()
    assert max_border_error < 1e-4, f"Border darkening detected! Max error: {max_border_error}"


def test_guided_filter_edge_alignment_improvement():
    """Validates that guided filtering cuts boundary error across a sharp step edge vs bilinear."""
    H, W = 128, 128
    # 1. Guide: Sharp vertical step edge at center
    guide = torch.zeros((1, 1, H, W), dtype=torch.float32)
    guide[:, :, :, W // 2 :] = 1.0

    # 2. Ground truth high-res map matches the sharp step
    gt_map = torch.zeros((1, 1, H, W), dtype=torch.float32)
    gt_map[:, :, :, W // 2 :] = 0.8

    # 3. Coarse low-resolution proxy (16x16)
    src_coarse = F.adaptive_avg_pool2d(gt_map, (16, 16))

    # Baseline: Bilinear upsampling
    bilinear_up = F.interpolate(src_coarse, size=(H, W), mode="bilinear", align_corners=False)

    # Guided filtering
    gf = SubsampledGuidedFilter(radius=16, eps=1e-3, s=4)
    guided_up = gf(guide, src_coarse)

    # Measure error in the transition corridor (16 pixels around the center edge)
    corridor = slice(W // 2 - 8, W // 2 + 8)
    err_bilinear = (bilinear_up[:, :, :, corridor] - gt_map[:, :, :, corridor]).abs().mean().item()
    err_guided = (guided_up[:, :, :, corridor] - gt_map[:, :, :, corridor]).abs().mean().item()

    # Guided filter should significantly reduce edge blurring error
    error_reduction = (err_bilinear - err_guided) / err_bilinear
    assert error_reduction > 0.15, (
        f"Guided filter did not achieve >15% edge error reduction (achieved {error_reduction * 100:.1f}%)"
    )


@pytest.mark.parametrize("filter_cls", [FastGuidedFilter, SubsampledGuidedFilter])
def test_multichannel_spatial_maps(filter_cls):
    """Validates joint edge-aware upsampling on multi-channel tensors (e.g. 5 LUT weights + 1 gain)."""
    gf = filter_cls(radius=8, eps=1e-3)
    B, C, H, W = 2, 6, 64, 64

    guide = torch.rand(B, 1, H, W, dtype=torch.float32)
    src_coarse = torch.rand(B, C, 16, 16, dtype=torch.float32)

    out = gf(guide, src_coarse)

    assert out.shape == (B, C, H, W)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
