"""Verification tests for the multi-stage training harness and metrics."""

import pytest
import torch
from scripts.train import (
    EnhancementModel,
    compute_delta_e00_sample,
    compute_psnr,
    run_dry_run,
)
from models.losses import TotalEnhancementLoss


def test_metrics_computation():
    """Validates PSNR and delta E 2000 metrics."""
    img = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    # Identical images
    psnr_exact = compute_psnr(img, img)
    assert psnr_exact >= 90.0, f"Expected near-infinite PSNR for identical images, got {psnr_exact}"

    de00_exact = compute_delta_e00_sample(img, img)
    assert de00_exact < 1e-4, f"Expected 0.0 dE00 for identical images, got {de00_exact}"

    # Distorted image
    noisy = (img + 0.05 * torch.randn_like(img)).clamp(0.0, 1.0)
    psnr_noisy = compute_psnr(img, noisy)
    de00_noisy = compute_delta_e00_sample(img, noisy)

    assert 20.0 < psnr_noisy < 45.0, f"Expected reasonable PSNR, got {psnr_noisy}"
    assert de00_noisy > 0.5, f"Expected positive color difference, got {de00_noisy}"


@pytest.mark.parametrize("mode", ["sony", "apple"])
def test_enhancement_model_modes(mode):
    """Validates EnhancementModel executes forward pass in both Sony and Apple modes."""
    model = EnhancementModel(mode=mode, num_luts=5, lut_dim=9, spline_points=5)
    B = 2
    thumb = torch.rand(B, 3, 32, 32)
    full = torch.rand(B, 3, 64, 64)

    out = model(thumb, full)

    assert out.shape == (B, 3, 64, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()


def test_microbatch_overfit_convergence():
    """Validates that backpropagation reduces loss over a micro-batch."""
    torch.manual_seed(42)
    model = EnhancementModel(mode="sony", num_luts=3, lut_dim=9, spline_points=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = TotalEnhancementLoss()

    B = 2
    thumb = torch.rand(B, 3, 32, 32)
    img_in = torch.rand(B, 3, 64, 64)
    target = torch.rand(B, 3, 64, 64)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        pred = model(thumb, img_in)
        l = criterion(pred, target, luts=model.lut_engine.luts)["total"]
        l.backward()
        optimizer.step()
        losses.append(l.item())

    # Final loss should be lower than initial loss
    assert losses[-1] < losses[0], f"Loss failed to decrease: {losses}"


def test_training_dry_run_entrypoint():
    """Validates dry-run execution completes with return code 0."""
    status = run_dry_run()
    assert status == 0
