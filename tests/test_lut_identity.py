"""Identity roundtrip verification for SinglePassLUTEngine.

Asserts that an identity-initialized LUT yields exact identity transformation
across all individual color channels. Validates that the query grid channel
ordering (B, G, R) does NOT silently transpose Red and Blue channels.
"""

import pytest
import torch
from engine.lut_interp import SinglePassLUTEngine


def test_lut_identity_pure_channels():
    """Validates that pure primary colors (Red, Green, Blue) are preserved without swapping."""
    engine = SinglePassLUTEngine(num_luts=1, lut_dim=33)
    weights = torch.tensor([[1.0]], dtype=torch.float32)

    # 1. Pure Red pixel
    red_img = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], dtype=torch.float32)  # [1, 3, 1, 1]
    out_red = engine(red_img, weights)
    assert torch.allclose(out_red, red_img, atol=1e-5), f"Red transformed to {out_red.squeeze().tolist()}"

    # 2. Pure Green pixel
    green_img = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]], dtype=torch.float32)
    out_green = engine(green_img, weights)
    assert torch.allclose(out_green, green_img, atol=1e-5), f"Green transformed to {out_green.squeeze().tolist()}"

    # 3. Pure Blue pixel
    blue_img = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]], dtype=torch.float32)
    out_blue = engine(blue_img, weights)
    assert torch.allclose(out_blue, blue_img, atol=1e-5), f"Blue transformed to {out_blue.squeeze().tolist()}"


def test_lut_identity_full_spectrum_ramps():
    """Validates smooth color ramps across full 2D spatial dimensions."""
    engine = SinglePassLUTEngine(num_luts=3, lut_dim=33)
    # Uniform blend weights across 3 identity tables
    weights = torch.tensor([[0.3333, 0.3334, 0.3333]], dtype=torch.float32)

    H, W = 64, 64
    y = torch.linspace(0.0, 1.0, H)
    x = torch.linspace(0.0, 1.0, W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Channel 0 (Red) varies with X, Channel 1 (Green) with Y, Channel 2 (Blue) with 1 - X
    r = xx
    g = yy
    b = 1.0 - xx
    img = torch.stack([r, g, b], dim=0).unsqueeze(0)  # [1, 3, H, W]

    out = engine(img, weights)

    # Max error should be within trilinear float precision tolerance (< 1e-5)
    max_err = (out - img).abs().max().item()
    assert max_err < 1e-4, f"Identity roundtrip max error ({max_err}) exceeded tolerance"
    assert torch.allclose(out, img, atol=1e-4, rtol=1e-4)


def test_lut_identity_arbitrary_random_image():
    """Feeds random uniform image through identity engine."""
    torch.manual_seed(99)
    engine = SinglePassLUTEngine(num_luts=5, lut_dim=17)
    weights = torch.softmax(torch.randn(1, 5), dim=1)

    img = torch.rand(1, 3, 128, 128, dtype=torch.float32)
    out = engine(img, weights)

    assert torch.allclose(out, img, atol=1e-4, rtol=1e-4)
