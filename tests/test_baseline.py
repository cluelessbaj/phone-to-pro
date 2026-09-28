"""Integration tests for the Zero-Training Baseline Engine and Pipeline Coordinator."""

import pytest
import torch
from PIL import Image
from engine.baseline import (
    apple_shadow_lift,
    conservative_levels,
    gray_world_wb,
    sony_baseline_grade,
)
from engine.color_transform import luminance
from engine.guided_filter import SubsampledGuidedFilter
from engine.lut_interp import SinglePassLUTEngine
from engine.pipeline import PhotoEnhancementPipeline


def test_gray_world_wb_cast_correction():
    """Validates that clamped gray-world WB reduces color cast."""
    # Create an image with an artificial warm / red tint
    img = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    img[:, 0] = 0.6  # High Red
    img[:, 1] = 0.4  # Moderate Green
    img[:, 2] = 0.3  # Low Blue

    balanced = gray_world_wb(img, max_gain=1.25, strength=1.0)

    # After balancing, the red-blue disparity should be reduced
    diff_before = (img[:, 0] - img[:, 2]).mean().item()
    diff_after = (balanced[:, 0] - balanced[:, 2]).mean().item()

    assert diff_after < diff_before, "WB did not reduce color disparity"
    assert (balanced >= 0.0).all() and (balanced <= 1.0).all()


def test_conservative_levels_expansion():
    """Validates conservative levels stretches low-contrast images safely."""
    # Image with compressed dynamic range in [0.25, 0.75]
    img = 0.25 + 0.5 * torch.rand((1, 3, 64, 64), dtype=torch.float32)
    thumb = img.clone()

    expanded = conservative_levels(img, thumb, black_cap=0.10, white_floor=0.85)

    # Dynamic range should be expanded
    assert expanded.min() < img.min()
    assert (expanded >= 0.0).all() and (expanded <= 1.0).all()


def test_apple_shadow_lift_selectivity():
    """Validates Apple shadow lift boosts dark tones while protecting bright highlights."""
    gf = SubsampledGuidedFilter(radius=16, eps=1e-3, s=4)

    # Create image with deep shadow on left (0.1) and bright highlight on right (0.9)
    img = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    img[:, :, :, :32] = 0.10  # Deep shadow
    img[:, :, :, 32:] = 0.90  # Bright highlight
    thumb = img.clone()

    lifted = apple_shadow_lift(img, gf, thumb, target=0.40, r_max=3.0, gamma=0.5)

    shadow_before = img[:, :, :, :32].mean().item()
    shadow_after = lifted[:, :, :, :32].mean().item()
    highlight_before = img[:, :, :, 32:].mean().item()
    highlight_after = lifted[:, :, :, 32:].mean().item()

    # Shadows must be noticeably lifted
    assert shadow_after > shadow_before * 1.5, f"Shadow was not lifted: {shadow_before} -> {shadow_after}"
    # Highlights must not be heavily blown
    assert highlight_after <= 1.0
    assert abs(highlight_after - highlight_before) < 0.10, "Highlight was modified too drastically"


@pytest.mark.parametrize("route", ["a", "b"])
def test_pipeline_end_to_end_sony(route):
    """Validates full pipeline execution in Sony Mode (Routes a & b)."""
    pipeline = PhotoEnhancementPipeline(sr_scale=4)
    # 32x32 input -> 128x128 output
    img = torch.rand((1, 3, 32, 32), dtype=torch.float32)

    enhanced = pipeline.enhance_tensor(img, mode="sony", sony_route=route)

    assert enhanced.shape == (1, 3, 128, 128)
    assert enhanced.dtype == torch.float32
    assert not torch.isnan(enhanced).any()
    assert (enhanced >= 0.0).all() and (enhanced <= 1.0).all()


def test_pipeline_end_to_end_apple():
    """Validates full pipeline execution in Apple Photonic Mode."""
    pipeline = PhotoEnhancementPipeline(sr_scale=4)
    img = torch.rand((1, 3, 32, 32), dtype=torch.float32)

    enhanced = pipeline.enhance_tensor(img, mode="apple")

    assert enhanced.shape == (1, 3, 128, 128)
    assert enhanced.dtype == torch.float32
    assert not torch.isnan(enhanced).any()
    assert (enhanced >= 0.0).all() and (enhanced <= 1.0).all()


def test_pipeline_pil_interface():
    """Validates PIL image enhancement convenience method."""
    pipeline = PhotoEnhancementPipeline(sr_scale=4)
    pil_in = Image.new("RGB", (24, 24), color=(120, 80, 60))

    pil_out = pipeline.enhance_pil(pil_in, mode="sony")

    assert isinstance(pil_out, Image.Image)
    assert pil_out.size == (96, 96)
