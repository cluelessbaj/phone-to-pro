"""Verification tests for Core Decision 3: Float32 Intermediate Hand-Off.

Asserts that:
1. SR output tensor maintains strictly torch.float32 precision.
2. No 8-bit quantization (uint8 clamping / integer rounding) occurs across the SR boundary.
3. Output spatial resolution scales by 4x.
4. Tensor values remain strictly bounded in [0.0, 1.0] with zero NaNs or Infs.
"""

import pytest
import torch
from models.srfactory import SRCore, SRVGGNetCompact, create_sr_core


def test_sr_output_is_float32():
    """Asserts that SRCore outputs strictly float32 tensors with 4x spatial scaling."""
    sr = create_sr_core(scale=4)
    B, H, W = 1, 32, 32
    img = torch.rand(B, 3, H, W, dtype=torch.float32)

    out = sr(img)

    assert out.dtype == torch.float32, f"Expected torch.float32, got {out.dtype}"
    assert out.shape == (B, 3, H * 4, W * 4)
    assert (out >= 0.0).all() and (out <= 1.0).all()
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sr_precision_unquantized():
    """Validates that continuous float values are not truncated to discrete 1/255 uint8 bins."""
    sr = create_sr_core(scale=4, denoise_strength=0.0)

    # Construct input with arbitrary fractional values that are NOT multiples of 1/255
    # e.g., 0.12345678, 0.45678912
    img = torch.tensor(
        [[[[0.1234567]], [[0.4567891]], [[0.7891234]]]],
        dtype=torch.float32,
    ).expand(1, 3, 8, 8)

    out = sr(img)

    # In an 8-bit quantized pipeline (e.g. uint8 cast / disk write),
    # all pixel values * 255.0 would be exact integers (mod 1 == 0).
    scaled = out * 255.0
    residuals = (scaled - scaled.round()).abs()

    # The max fractional residual should be non-zero (proving unquantized float32)
    max_residual = residuals.max().item()
    assert max_residual > 1e-4, (
        f"Output appears 8-bit quantized: max residual from uint8 grid was {max_residual}"
    )


def test_sr_neural_compact_forward():
    """Validates forward pass through the SRVGGNetCompact neural architecture."""
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=32,  # Compact feature dimension for test
        num_conv=4,
        upscale=4,
    )
    img = torch.rand(2, 3, 16, 16, dtype=torch.float32)
    out = model(img)

    assert out.shape == (2, 3, 64, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()


def test_denoise_strength_parameter():
    """Validates denoise parameter modulates outputs smoothly."""
    sr_clean = create_sr_core(scale=4, denoise_strength=0.0)
    sr_denoise = create_sr_core(scale=4, denoise_strength=0.8)

    img = torch.rand(1, 3, 16, 16, dtype=torch.float32)
    out_clean = sr_clean(img)
    out_denoise = sr_denoise(img)

    assert out_clean.shape == out_denoise.shape == (1, 3, 64, 64)
    # Different denoise strength should produce modulated outputs
    diff = (out_clean - out_denoise).abs().mean().item()
    assert diff > 1e-4
