"""Phone-to-Pro Core Processing Engine."""
from .hardware import configure_hardware_acceleration, get_hardware_profile
from .lut_interp import SinglePassLUTEngine
from .guided_filter import FastGuidedFilter, SubsampledGuidedFilter
from .color_transform import (
    luminance,
    srgb_to_linear,
    linear_to_srgb,
    slog3_encode,
    slog3_decode,
    to_slog3_input,
    load_cube,
    write_cube,
    SRGB_TO_SGAMUT3CINE_MATRIX,
)
from .spline_curve import (
    MonotonicSplineCurve,
    build_filmic_table,
    apply_curve_1d,
    filmic_ratio_grade,
    soft_shoulder,
    smoothstep,
)
from .sharpening import (
    adaptive_sharpen,
    sony_sharpen,
    apple_sharpen,
    gaussian_blur2d,
)

__all__ = [
    "configure_hardware_acceleration",
    "get_hardware_profile",
    "SinglePassLUTEngine",
    "FastGuidedFilter",
    "SubsampledGuidedFilter",
    "luminance",
    "srgb_to_linear",
    "linear_to_srgb",
    "slog3_encode",
    "slog3_decode",
    "to_slog3_input",
    "load_cube",
    "write_cube",
    "SRGB_TO_SGAMUT3CINE_MATRIX",
    "MonotonicSplineCurve",
    "build_filmic_table",
    "apply_curve_1d",
    "filmic_ratio_grade",
    "soft_shoulder",
    "smoothstep",
    "adaptive_sharpen",
    "sony_sharpen",
    "apple_sharpen",
    "gaussian_blur2d",
]
