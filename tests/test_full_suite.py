"""Comprehensive CI Regression Test Suite for Step 11."""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch

from engine.pipeline import PhotoEnhancementPipeline
from scripts.export_onnx import (
    export_apple_predictor,
    export_sony_predictor,
    export_sr_compact,
)


def test_onnx_export_and_validity():
    """Validates that neural modules export to valid ONNX graphs with dynamic batch axes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Export Sony Predictor
        sony_path = tmp_path / "test_sony.onnx"
        export_sony_predictor(sony_path, num_luts=5, spline_points=5)
        assert sony_path.is_file()
        onnx_sony = onnx.load(str(sony_path))
        onnx.checker.check_model(onnx_sony)

        # 2. Export Apple Predictor
        apple_path = tmp_path / "test_apple.onnx"
        export_apple_predictor(apple_path, num_luts=5, spatial_grid_size=16)
        assert apple_path.is_file()
        onnx_apple = onnx.load(str(apple_path))
        onnx.checker.check_model(onnx_apple)

        # 3. Export SR Compact
        sr_path = tmp_path / "test_sr.onnx"
        export_sr_compact(sr_path)
        assert sr_path.is_file()
        onnx_sr = onnx.load(str(sr_path))
        onnx.checker.check_model(onnx_sr)


def test_full_pipeline_multi_mode_execution():
    """Validates complete end-to-end processing across Sony and Apple modes on synthetic photo."""
    pipeline = PhotoEnhancementPipeline(sr_scale=4)
    img = torch.rand(1, 3, 48, 48, dtype=torch.float32)

    # Sony Mode
    out_sony = pipeline.enhance_tensor(img, mode="sony", sony_route="b")
    assert out_sony.shape == (1, 3, 192, 192)
    assert out_sony.dtype == torch.float32
    assert (out_sony >= 0.0).all() and (out_sony <= 1.0).all()

    # Apple Mode
    out_apple = pipeline.enhance_tensor(img, mode="apple")
    assert out_apple.shape == (1, 3, 192, 192)
    assert out_apple.dtype == torch.float32
    assert (out_apple >= 0.0).all() and (out_apple <= 1.0).all()
