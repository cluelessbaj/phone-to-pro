"""Verification tests for Step 1: Environment, Hardware Detection, and Dependencies."""

import torch
import torchvision
import scipy
import numpy as np
import colour
import cv2
from engine.hardware import get_hardware_profile, configure_hardware_acceleration


def test_packages_importable():
    """Validates all required dependencies are installed and accessible."""
    assert torch.__version__ is not None
    assert torchvision.__version__ is not None
    assert scipy.__version__ is not None
    assert np.__version__ is not None
    assert colour.__version__ is not None
    assert cv2.__version__ is not None


def test_onednn_and_threading():
    """Validates Intel oneDNN acceleration and multi-threading."""
    profile = configure_hardware_acceleration()
    assert profile.has_onednn is True, "Intel oneDNN must be available in PyTorch CPU build"
    assert torch.get_num_threads() >= 1


def test_vulkan_detection():
    """Validates Vulkan detection runs without error."""
    profile = get_hardware_profile()
    assert isinstance(profile.vulkan_available, bool)
    if profile.vulkan_available:
        assert profile.vulkan_device_name is not None
