#!/usr/bin/env bash
# ==============================================================================
# Script to convert exported ONNX models to NCNN format with FP16 Vulkan support.
#
# Prerequisites (Arch Linux):
#   sudo pacman -S ncnn
#   or build from https://github.com/Tencent/ncnn
# ==============================================================================

set -euo pipefail

ONNX_DIR="${1:-models_onnx}"
NCNN_DIR="${2:-models_ncnn}"

mkdir -p "$NCNN_DIR"

echo "=========================================================="
echo "Exporting ONNX models to NCNN with Vulkan FP16 support"
echo "Source: $ONNX_DIR -> Target: $NCNN_DIR"
echo "=========================================================="

if ! command -v onnx2ncnn &> /dev/null; then
    echo "Warning: onnx2ncnn binary not found in PATH."
    echo "Install via Arch Linux: 'sudo pacman -S ncnn' or compile from source."
    echo "Writing placeholder conversion commands for deployment."
fi

# Convert models
for onnx_model in "$ONNX_DIR"/*.onnx; do
    if [ ! -f "$onnx_model" ]; then
        continue
    fi
    base_name=$(basename "$onnx_model" .onnx)
    param_file="$NCNN_DIR/${base_name}.param"
    bin_file="$NCNN_DIR/${base_name}.bin"
    opt_param="$NCNN_DIR/${base_name}.opt.param"
    opt_bin="$NCNN_DIR/${base_name}.opt.bin"

    echo "Converting $base_name..."
    if command -v onnx2ncnn &> /dev/null; then
        onnx2ncnn "$onnx_model" "$param_file" "$bin_file"
        if command -v ncnnoptimize &> /dev/null; then
            echo "Optimizing $base_name for FP16 Vulkan execution..."
            ncnnoptimize "$param_file" "$bin_file" "$opt_param" "$opt_bin" 65536
        fi
    else
        echo "  [Mock/Dry-Run] onnx2ncnn $onnx_model $param_file $bin_file"
        echo "  [Mock/Dry-Run] ncnnoptimize $param_file $bin_file $opt_param $opt_bin 65536"
    fi
done

echo "NCNN conversion pipeline ready."
