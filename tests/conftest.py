"""Pytest configuration and shared test fixtures."""

import pytest
import torch
from engine.hardware import configure_hardware_acceleration


@pytest.fixture(scope="session", autouse=True)
def setup_hardware():
    """Initializes hardware acceleration settings across all test runs."""
    configure_hardware_acceleration()


@pytest.fixture
def synthetic_rgb_image():
    """Returns a deterministic [1, 3, 64, 64] float32 image in [0.0, 1.0]."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)


@pytest.fixture
def identity_lut_33():
    """Generates an identity 3D-LUT of dimension 33 with shape [3, 33, 33, 33]."""
    lut_dim = 33
    linear = torch.linspace(0.0, 1.0, lut_dim, dtype=torch.float32)
    r, g, b = torch.meshgrid(linear, linear, linear, indexing="ij")
    return torch.stack([r, g, b], dim=0)
