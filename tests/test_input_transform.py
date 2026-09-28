"""Verification tests for color space transformations, S-Log3 transfer function, and .cube parsing."""

import tempfile
from pathlib import Path

import pytest
import torch
from engine.color_transform import (
    linear_to_srgb,
    load_cube,
    luminance,
    slog3_decode,
    slog3_encode,
    srgb_to_linear,
    to_slog3_input,
    write_cube,
)


def test_slog3_18percent_grey():
    """Asserts that 18% scene-linear grey evaluates to S-Log3 code value 420/1023 (~0.410557)."""
    grey_18 = torch.tensor([0.18], dtype=torch.float32)
    slog_val = slog3_encode(grey_18).item()
    expected = 420.0 / 1023.0

    assert abs(slog_val - expected) < 1e-4, (
        f"18% grey evaluated to {slog_val:.6f}, expected {expected:.6f}"
    )


def test_slog3_c0_continuity():
    """Asserts C^0 continuity across the splice threshold at x = 0.01125."""
    splice = 0.01125
    delta = 1e-6
    x_left = torch.tensor([splice - delta], dtype=torch.float32)
    x_right = torch.tensor([splice + delta], dtype=torch.float32)

    y_left = slog3_encode(x_left).item()
    y_right = slog3_encode(x_right).item()

    assert abs(y_left - y_right) < 1e-4, (
        f"Discontinuity at splice threshold: left={y_left:.6f}, right={y_right:.6f}"
    )


def test_slog3_roundtrip_invertibility():
    """Validates that slog3_decode correctly inverts slog3_encode."""
    # Test values spanning shadows up to highlights
    lin_values = torch.linspace(0.001, 2.5, 100, dtype=torch.float32)
    encoded = slog3_encode(lin_values)
    decoded = slog3_decode(encoded)

    assert torch.allclose(decoded, lin_values, rtol=1e-3, atol=1e-4)


def test_srgb_linear_roundtrip():
    """Validates piecewise sRGB <-> linear conversion roundtrip."""
    srgb = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
    lin = srgb_to_linear(srgb)
    reconstructed = linear_to_srgb(lin)

    assert torch.allclose(reconstructed, srgb, atol=1e-5)


def test_to_slog3_input_pipeline():
    """Validates full sRGB -> S-Gamut3.Cine / S-Log3 conversion pipeline."""
    img = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    slog_out = to_slog3_input(img)

    assert slog_out.shape == (2, 3, 32, 32)
    assert not torch.isnan(slog_out).any()
    assert (slog_out >= 0.0).all() and (slog_out <= 1.0).all()


def test_cube_io_identity_roundtrip():
    """Validates .cube file export and import preserves identity lattice and axis indexing."""
    N = 9
    linear = torch.linspace(0.0, 1.0, N)
    r, g, b = torch.meshgrid(linear, linear, linear, indexing="ij")
    orig_lut = torch.stack([r, g, b], dim=0)  # [3, N, N, N]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cube_path = Path(tmp_dir) / "test_identity.cube"
        write_cube(cube_path, orig_lut, title="Identity9")

        loaded_lut = load_cube(cube_path)

        # Assert shape and values match
        assert loaded_lut.shape == (3, N, N, N)
        assert torch.allclose(loaded_lut, orig_lut, atol=1e-5)

        # Assert axis index conventions:
        # lut[0, i, j, k] corresponds to R_i
        # lut[1, i, j, k] corresponds to G_j
        # lut[2, i, j, k] corresponds to B_k
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    assert abs(loaded_lut[0, i, j, k].item() - linear[i].item()) < 1e-5
                    assert abs(loaded_lut[1, i, j, k].item() - linear[j].item()) < 1e-5
                    assert abs(loaded_lut[2, i, j, k].item() - linear[k].item()) < 1e-5
