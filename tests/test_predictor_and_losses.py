"""Verification tests for DualHeadPredictor and loss objectives."""

import pytest
import torch
from engine.lut_interp import SinglePassLUTEngine
from models.losses import (
    ColorCosineLoss,
    LUTRegularizationLoss,
    TotalEnhancementLoss,
)
from models.predictor import DualHeadPredictor


def test_predictor_sony_forward_shapes():
    """Validates predictor output shapes and normalization for Sony mode."""
    predictor = DualHeadPredictor(num_luts=5, spline_points=5)
    B, H, W = 2, 256, 256
    thumb = torch.rand(B, 3, H, W, dtype=torch.float32)

    preds = predictor(thumb, mode="sony")

    assert "lut_weights" in preds and "spline_points" in preds
    assert preds["lut_weights"].shape == (B, 5)
    assert preds["spline_points"].shape == (B, 5)

    # Softmax partition of unity
    sums = preds["lut_weights"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Sigmoid bounds [0, 1]
    assert (preds["spline_points"] >= 0.0).all() and (preds["spline_points"] <= 1.0).all()


def test_predictor_apple_forward_shapes():
    """Validates predictor output shapes and normalization for Apple mode."""
    predictor = DualHeadPredictor(num_luts=5, spatial_grid_size=16)
    B, H, W = 2, 256, 256
    thumb = torch.rand(B, 3, H, W, dtype=torch.float32)

    preds = predictor(thumb, mode="apple")

    assert "lut_weights" in preds and "spatial_weights" in preds and "gain_map" in preds
    assert preds["lut_weights"].shape == (B, 5)
    assert preds["spatial_weights"].shape == (B, 5, 16, 16)
    assert preds["gain_map"].shape == (B, 1, 16, 16)

    # Spatial weights partition of unity along channel dimension
    spatial_sums = preds["spatial_weights"].sum(dim=1)
    assert torch.allclose(spatial_sums, torch.ones_like(spatial_sums), atol=1e-5)

    # Gain map up to 2.0x
    assert (preds["gain_map"] >= 0.0).all() and (preds["gain_map"] <= 2.0).all()


def test_color_cosine_loss():
    """Validates color cosine loss behavior."""
    loss_fn = ColorCosineLoss()
    img = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    # Exact match -> 0.0 loss
    loss_zero = loss_fn(img, img)
    assert abs(loss_zero.item()) < 1e-5

    # Pure color hue shift (invert channels) -> strictly positive loss
    shifted = img[:, [2, 0, 1], :, :]
    loss_shifted = loss_fn(img, shifted)
    assert loss_shifted.item() > 0.05


def test_lut_regularization_loss():
    """Validates monotonicity and smoothness penalties on 3D-LUT lattices."""
    reg_fn = LUTRegularizationLoss(lambda_smooth=1e-4, lambda_mono=1e-2)
    D = 17

    # 1. Monotonic linear identity LUT: should incur zero monotonicity loss
    linear = torch.linspace(0.0, 1.0, D)
    r, g, b = torch.meshgrid(linear, linear, linear, indexing="ij")
    identity_lut = torch.stack([r, g, b], dim=0).unsqueeze(0)  # [1, 3, D, D, D]

    loss_ident = reg_fn(identity_lut)
    # Second order diffs of linear ramp are 0, first order diffs are positive -> total is 0.0
    assert abs(loss_ident.item()) < 1e-6

    # 2. Inverted / non-monotonic LUT
    inverted_lut = 1.0 - identity_lut
    loss_inverted = reg_fn(inverted_lut)
    assert loss_inverted.item() > 1e-4, "Inverted LUT should trigger monotonicity penalty"


def test_end_to_end_gradient_backprop():
    """Validates full gradient backpropagation from total loss to predictor and LUT weights."""
    predictor = DualHeadPredictor(num_luts=5, spline_points=5)
    lut_engine = SinglePassLUTEngine(num_luts=5, lut_dim=9)
    loss_fn = TotalEnhancementLoss()

    # Use standard batch size 2 and thumbnail resolution 64x64
    B, H, W = 2, 64, 64
    thumb = torch.rand(B, 3, H, W, requires_grad=True)
    target = torch.rand(B, 3, H, W)

    # Forward pass: thumb -> predictor -> lut_weights -> lut_engine -> graded -> loss
    preds = predictor(thumb, mode="sony")
    graded = lut_engine(thumb, preds["lut_weights"])
    loss_dict = loss_fn(graded, target, luts=lut_engine.luts)

    total_loss = loss_dict["total"]
    total_loss.backward()

    # Assert gradients exist on predictor parameters
    has_predictor_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    assert has_predictor_grad, "Gradients failed to flow back into DualHeadPredictor parameters"

    # Assert gradients exist on LUT parameters
    assert lut_engine.luts.grad is not None
    assert lut_engine.luts.grad.abs().sum() > 0, "Gradients failed to flow back into basis LUTs"
