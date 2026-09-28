"""Verification tests for mode-tailored adaptive sharpening."""

import pytest
import torch
from engine.sharpening import (
    adaptive_sharpen,
    apple_sharpen,
    compute_resolution_scale,
    gaussian_blur2d,
    sony_sharpen,
)


def test_gaussian_blur2d_identity_and_bounds():
    """Validates Gaussian blur preserves tensor shapes and value bounds."""
    img = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    blurred = gaussian_blur2d(img, radius=1.0)

    assert blurred.shape == img.shape
    assert (blurred >= 0.0).all() and (blurred <= 1.0).all()

    # Zero radius should be an identity operation
    unblurred = gaussian_blur2d(img, radius=0.0)
    assert torch.allclose(unblurred, img, atol=1e-6)


def test_sony_sharpen_enhancement_and_clamping():
    """Validates Sony fine-radius sharpening amplifies fine details within [0.0, 1.0]."""
    # Create an image with high frequency detail
    img = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    img[:, :, 16, 16] = 0.8  # Point impulse detail

    sharpened = sony_sharpen(img, base_radius=0.5, amount=0.35)

    assert sharpened.shape == img.shape
    # Center pixel should be boosted
    assert sharpened[:, :, 16, 16].mean().item() > 0.8
    # Global bounds strictly enforced
    assert (sharpened >= 0.0).all() and (sharpened <= 1.0).all()


def test_apple_sharpen_noise_suppression_on_flat_fields():
    """Validates Apple mode's edge mask suppresses sharpening in flat regions."""
    # Flat field with tiny sensor noise
    torch.manual_seed(42)
    flat_noise = torch.full((1, 3, 64, 64), 0.5) + (torch.rand(1, 3, 64, 64) - 0.5) * 0.005

    # Sharpen flat field
    sharpened_flat = apple_sharpen(flat_noise, threshold_low=0.02, threshold_high=0.08)

    # Edge gradient on flat noise is below threshold_low (~0.002 < 0.02)
    # Output should remain practically identical to input (noise not amplified)
    noise_diff = (sharpened_flat - flat_noise).abs().max().item()
    assert noise_diff < 1e-4, f"Flat field noise was amplified by {noise_diff}"


def test_apple_sharpen_edge_enhancement():
    """Validates Apple mode boosts contrast across distinct edges."""
    # Sharp step edge
    step = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    step[:, :, :, 32:] = 1.0

    sharpened_step = apple_sharpen(step, threshold_low=0.015, threshold_high=0.08)

    # Contrast across the boundary should be slightly expanded
    val_left_before = step[:, :, :, 31].mean().item()
    val_left_after = sharpened_step[:, :, :, 31].mean().item()
    val_right_before = step[:, :, :, 32].mean().item()
    val_right_after = sharpened_step[:, :, :, 32].mean().item()

    assert val_left_after <= val_left_before
    assert val_right_after >= val_right_before


def test_resolution_scale_indexing():
    """Validates dynamic resolution scaling indexing."""
    scale_1080p = compute_resolution_scale(1080, 1920, reference_mp=12.0)
    scale_12mp = compute_resolution_scale(3000, 4000, reference_mp=12.0)
    scale_48mp = compute_resolution_scale(6000, 8000, reference_mp=12.0)

    # Preview scale < 12MP reference < 48MP capture
    assert scale_1080p < scale_12mp
    assert abs(scale_12mp - 1.0) < 1e-2
    assert scale_48mp > scale_12mp


def test_adaptive_sharpen_dispatcher():
    """Validates dispatcher correctly routes modes and rejects invalid modes."""
    img = torch.rand(1, 3, 32, 32)
    out_sony = adaptive_sharpen(img, mode="sony")
    out_apple = adaptive_sharpen(img, mode="apple")

    assert out_sony.shape == img.shape
    assert out_apple.shape == img.shape

    with pytest.raises(ValueError, match="Unknown sharpening mode"):
        adaptive_sharpen(img, mode="invalid_mode")
