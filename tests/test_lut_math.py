"""Mathematical verification of parameter-space pre-blending vs multi-pass interpolation.

Validates that trilinear interpolation commutes with linear combination:
    sum_k w_k * T(x, L_k) == T(x, sum_k w_k * L_k)
with numerical precision < 1e-7.
"""

import pytest
import torch
import torch.nn.functional as F
from engine.lut_interp import SinglePassLUTEngine


def test_lut_preblending_mathematical_equivalence():
    """Validates single-pass pre-blending exactly equals multi-pass interpolation."""
    torch.manual_seed(1337)
    B, K, D = 2, 5, 17  # Use dimension 17 for fast, high-precision test
    H, W = 64, 64

    engine = SinglePassLUTEngine(num_luts=K, lut_dim=D)
    # Populate basis LUTs with smooth random values
    with torch.no_grad():
        engine.luts.copy_(torch.rand(K, 3, D, D, D, dtype=torch.float32))

    # Generate random test image and normalized blend weights
    img = torch.rand(B, 3, H, W, dtype=torch.float32)
    raw_weights = torch.rand(B, K, dtype=torch.float32)
    weights = F.softmax(raw_weights, dim=1)

    # 1. Compute via single-pass pre-blended kernel
    single_pass_out = engine(img, weights)

    # 2. Compute via reference multi-pass interpolation
    multi_pass_out = engine.forward_multipass(img, weights)

    # 3. Assert mathematical equivalence
    abs_diff = (single_pass_out - multi_pass_out).abs()
    max_diff = abs_diff.max().item()
    mse = F.mse_loss(single_pass_out, multi_pass_out).item()

    assert mse < 1e-7, f"MSE ({mse}) exceeds 1e-7 threshold"
    assert max_diff < 1e-4, f"Max absolute difference ({max_diff}) exceeds numerical tolerance"


@pytest.mark.parametrize("lut_dim", [9, 17, 33])
def test_lut_linearity_across_dimensions(lut_dim):
    """Verifies linearity holds across different lattice sizes."""
    torch.manual_seed(42)
    engine = SinglePassLUTEngine(num_luts=3, lut_dim=lut_dim)
    with torch.no_grad():
        engine.luts.copy_(torch.rand(3, 3, lut_dim, lut_dim, lut_dim))

    img = torch.rand(1, 3, 32, 32)
    weights = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32)

    out_single = engine(img, weights)
    out_multi = engine.forward_multipass(img, weights)

    assert torch.allclose(out_single, out_multi, atol=1e-5, rtol=1e-5)
