"""Verification tests for tone curve monotonicity, C1 continuity, and ratio-based chromaticity preservation."""

import pytest
import torch
from engine.color_transform import luminance
from engine.spline_curve import (
    MonotonicSplineCurve,
    apply_curve_1d,
    build_filmic_table,
    filmic_ratio_grade,
    soft_shoulder,
)


def test_filmic_table_strict_monotonicity():
    """Validates that build_filmic_table produces non-decreasing values across all 1024 entries."""
    table = build_filmic_table(1024)
    diff = table[1:] - table[:-1]
    # Negative difference indicates tonal inversion
    min_diff = diff.min().item()
    assert min_diff >= -1e-6, f"Tonal reversal detected in filmic table: min diff = {min_diff}"
    assert table[0].item() == 0.0
    assert table[-1].item() == 1.0


def test_apply_curve_1d_monotonic_response():
    """Validates 1D curve evaluation maintains monotonicity over a dense input ramp."""
    table = build_filmic_table(1024)
    ramp = torch.linspace(0.0, 1.0, 5000, dtype=torch.float32)
    out = apply_curve_1d(ramp, table)
    diff = out[1:] - out[:-1]
    assert (diff >= -1e-6).all(), "Non-monotonic step found in apply_curve_1d output"


def test_monotonic_spline_curve_module():
    """Validates that MonotonicSplineCurve enforces positive slopes regardless of raw knot inputs."""
    torch.manual_seed(42)
    spline = MonotonicSplineCurve(num_knots=5)

    # Test with varying (even wild / disordered) raw knot predictions
    raw_knots = torch.randn(4, 5, dtype=torch.float32)

    # Input gradient test ramp
    x = torch.linspace(0.0, 1.0, 200).view(1, 1, 1, 200).expand(4, 3, 1, 200)
    out = spline(x, raw_knots)

    # For each batch item, check luminance along the ramp is non-decreasing
    for b in range(4):
        lum_out = luminance(out[b : b + 1]).squeeze()
        diff = lum_out[1:] - lum_out[:-1]
        assert (diff >= -1e-5).all(), f"Monotonicity violated in batch item {b}"


def test_soft_shoulder_c1_continuity_and_asymptote():
    """Validates that soft_shoulder is continuous and monotone."""
    knee = 0.8
    y = torch.linspace(0.0, 1.5, 3000, dtype=torch.float32)
    out = soft_shoulder(y, knee=knee)

    # Monotonicity check
    diff = out[1:] - out[:-1]
    assert (diff >= 0.0).all(), "soft_shoulder is not strictly non-decreasing"

    # Identity below knee check
    below_knee = y[y <= knee]
    assert torch.allclose(out[y <= knee], below_knee, atol=1e-6)

    # Upper bound check: values must remain < 1.0
    assert (out <= 1.0 + 1e-5).all()

    # C1 continuity check at knee: left derivative = 1.0, right derivative approx 1.0
    delta = 1e-4
    val_left = soft_shoulder(torch.tensor([knee - delta]), knee=knee).item()
    val_right = soft_shoulder(torch.tensor([knee + delta]), knee=knee).item()
    slope_left = (knee - val_left) / delta
    slope_right = (val_right - knee) / delta
    assert abs(slope_left - 1.0) < 1e-3
    assert abs(slope_right - 1.0) < 1e-3


def test_filmic_ratio_grade_preserves_chromaticity():
    """Validates that ratio grading changes luminance while preserving chromaticity coordinates."""
    table = build_filmic_table(1024)
    # Generate arbitrary color pixels with distinct RGB ratios
    img = torch.tensor([[[[0.8]], [[0.4]], [[0.2]]]], dtype=torch.float32)  # [1, 3, 1, 1]
    orig_sum = img.sum(dim=1, keepdim=True)
    orig_r_ratio = img[:, 0:1] / orig_sum
    orig_g_ratio = img[:, 1:2] / orig_sum

    graded = filmic_ratio_grade(img, table)
    new_sum = graded.sum(dim=1, keepdim=True)
    new_r_ratio = graded[:, 0:1] / new_sum
    new_g_ratio = graded[:, 1:2] / new_sum

    assert torch.allclose(orig_r_ratio, new_r_ratio, atol=1e-5)
    assert torch.allclose(orig_g_ratio, new_g_ratio, atol=1e-5)
