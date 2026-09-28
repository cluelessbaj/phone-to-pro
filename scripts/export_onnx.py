"""ONNX Graph Tracing and Export Module for Mobile & Desktop Deployment.

Exports:
1. DualHeadPredictor (Sony and Apple branches with dynamic batch dimensions).
2. SubsampledGuidedFilter (Edge-aware upsampler).
3. SRVGGNetCompact (Super-resolution core).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.guided_filter import SubsampledGuidedFilter
from models.predictor import DualHeadPredictor
from models.srfactory import SRVGGNetCompact


class SonyPredictorONNXWrapper(nn.Module):
    def __init__(self, predictor: DualHeadPredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, thumb: torch.Tensor):
        preds = self.predictor(thumb, mode="sony")
        return preds["lut_weights"], preds["spline_points"]


class ApplePredictorONNXWrapper(nn.Module):
    def __init__(self, predictor: DualHeadPredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, thumb: torch.Tensor):
        preds = self.predictor(thumb, mode="apple")
        return preds["lut_weights"], preds["spatial_weights"], preds["gain_map"]


def export_sony_predictor(output_path: Path, num_luts: int = 5, spline_points: int = 5) -> Path:
    predictor = DualHeadPredictor(num_luts=num_luts, spline_points=spline_points)
    model = SonyPredictorONNXWrapper(predictor)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["thumbnail"],
        output_names=["lut_weights", "spline_points"],
        dynamic_axes={
            "thumbnail": {0: "batch_size"},
            "lut_weights": {0: "batch_size"},
            "spline_points": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported Sony Predictor to {output_path}")
    return output_path


def export_apple_predictor(output_path: Path, num_luts: int = 5, spatial_grid_size: int = 16) -> Path:
    predictor = DualHeadPredictor(num_luts=num_luts, spatial_grid_size=spatial_grid_size)
    model = ApplePredictorONNXWrapper(predictor)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["thumbnail"],
        output_names=["lut_weights", "spatial_weights", "gain_map"],
        dynamic_axes={
            "thumbnail": {0: "batch_size"},
            "lut_weights": {0: "batch_size"},
            "spatial_weights": {0: "batch_size"},
            "gain_map": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported Apple Predictor to {output_path}")
    return output_path


def export_sr_compact(output_path: Path) -> Path:
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=32,
        num_conv=4,
        upscale=4,
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height_4x", 3: "width_4x"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported SR Compact Core to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Phone-to-Pro neural components to ONNX")
    parser.add_argument("--output-dir", type=str, default="models_onnx", help="Target export directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_sony_predictor(out_dir / "predictor_sony.onnx")
    export_apple_predictor(out_dir / "predictor_apple.onnx")
    export_sr_compact(out_dir / "sr_compact_x4.onnx")
    print(f"All components exported successfully to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
