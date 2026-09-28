"""Color Space Transformations and .cube 3D-LUT File Parser.

Provides mathematically verified conversions:
1. sRGB <-> Scene-Linear (IEC 61966-2-1 piecewise EOTF / OETF).
2. Sony S-Log3 Encoding & Decoding (with continuous 18% grey calibration at 420/1023).
3. sRGB -> S-Gamut3.Cine color matrix transformation.
4. Parsing and export of 3D .cube LUT files conforming to the [3, iR, iG, iB] lattice format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# Standard Rec.709 Luminance Weights
LUM_REC709_R = 0.2126
LUM_REC709_G = 0.7152
LUM_REC709_B = 0.0722

# Precomputed 3x3 matrix converting sRGB (Rec.709) primaries to S-Gamut3.Cine
# Calculated via colour.matrix_RGB_to_RGB(sRGB, S-Gamut3.Cine)
SRGB_TO_SGAMUT3CINE_MATRIX = torch.tensor(
    [
        [0.64572347, 0.25912590, 0.09524418],
        [0.08747217, 0.75973283, 0.15276679],
        [0.03693043, 0.12928657, 0.83373467],
    ],
    dtype=torch.float32,
)


def luminance(img: torch.Tensor) -> torch.Tensor:
    """Computes Rec.709 luminance Y from an [B, 3, H, W] RGB tensor."""
    return (
        LUM_REC709_R * img[:, 0:1, :, :]
        + LUM_REC709_G * img[:, 1:2, :, :]
        + LUM_REC709_B * img[:, 2:3, :, :]
    )


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Piecewise IEC 61966-2-1 sRGB to scene-linear conversion."""
    x_clamped = x.clamp(min=0.0)
    return torch.where(
        x_clamped <= 0.04045,
        x_clamped / 12.92,
        ((x_clamped + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(x_lin: torch.Tensor) -> torch.Tensor:
    """Piecewise IEC 61966-2-1 scene-linear to sRGB conversion."""
    x_clamped = x_lin.clamp(min=0.0)
    return torch.where(
        x_clamped <= 0.0031308,
        12.92 * x_clamped,
        1.055 * (x_clamped ** (1.0 / 2.4)) - 0.055,
    ).clamp(0.0, 1.0)


def slog3_encode(x_lin: torch.Tensor) -> torch.Tensor:
    """Encodes scene-linear float values to Sony S-Log3 code values in [0.0, 1.0].

    Properties:
    - 18% scene-linear grey (0.18) encodes exactly to 420/1023 (~0.410557).
    - C^0 continuous splice threshold at x = 0.01125.
    """
    x = x_lin.clamp(min=0.0)
    splice = 0.01125

    hi = (420.0 + torch.log10((x.clamp(min=splice) + 0.01) / 0.19) * 261.5) / 1023.0
    lo = (x * (171.2102946929 - 95.0) / splice + 95.0) / 1023.0

    return torch.where(x >= splice, hi, lo)


def slog3_decode(slog: torch.Tensor) -> torch.Tensor:
    """Decodes Sony S-Log3 code values in [0.0, 1.0] back to scene-linear."""
    slog_clamped = slog.clamp(min=0.0)
    splice_code = 171.2102946929 / 1023.0
    splice_lin = 0.01125

    # Inverse log segment: 0.19 * 10^((1023 * y - 420) / 261.5) - 0.01
    hi_lin = 0.19 * (10.0 ** ((slog_clamped * 1023.0 - 420.0) / 261.5)) - 0.01

    # Inverse linear segment: (1023 * y - 95) * 0.01125 / (171.2102946929 - 95)
    lo_lin = (slog_clamped * 1023.0 - 95.0) * splice_lin / (171.2102946929 - 95.0)

    return torch.where(slog_clamped >= splice_code, hi_lin, lo_lin).clamp(min=0.0)


def to_slog3_input(
    img_srgb: torch.Tensor,
    gamut_matrix: Optional[torch.Tensor] = None,
    exposure_offset: float = 1.0,
) -> torch.Tensor:
    """Converts display-referred sRGB image to S-Gamut3.Cine / S-Log3.

    Sony look LUTs (.cube) expect log-encoded input rather than display sRGB.
    Applying them directly to sRGB causes crushed shadows and harsh clipping.

    Args:
        img_srgb: [B, 3, H, W] tensor in [0.0, 1.0].
        gamut_matrix: 3x3 sRGB -> S-Gamut3.Cine transformation matrix.
        exposure_offset: Multiplicative linear exposure scaling factor.

    Returns:
        [B, 3, H, W] S-Log3 encoded tensor in [0.0, 1.0].
    """
    matrix = gamut_matrix if gamut_matrix is not None else SRGB_TO_SGAMUT3CINE_MATRIX
    matrix = matrix.to(device=img_srgb.device, dtype=img_srgb.dtype)

    # 1. Linearize sRGB
    lin = srgb_to_linear(img_srgb) * exposure_offset

    # 2. Transform color primaries: [3, 3] x [B, 3, H, W] -> [B, 3, H, W]
    lin_gamut = torch.einsum("ij,bjhw->bihw", matrix, lin)

    # 3. Apply S-Log3 transfer curve
    return slog3_encode(lin_gamut).clamp(0.0, 1.0)


def load_cube(path: Union[str, Path]) -> torch.Tensor:
    """Parses a 3D .cube file into the [3, iR, iG, iB] layout required by SinglePassLUTEngine.

    In standard Adobe .cube files, data lines are listed with RED varying fastest,
    then GREEN, then BLUE. When loaded sequentially, the 3D array layout is
    [iB, iG, iR, c]. We permute to [c, iR, iG, iB].
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LUT file not found: {path}")

    lut_size: Optional[int] = None
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens or tokens[0].startswith("#") or tokens[0] == "TITLE":
                continue
            if tokens[0] == "LUT_3D_SIZE":
                lut_size = int(tokens[1])
                continue
            if tokens[0] == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported by SinglePassLUTEngine")
            if tokens[0] in ("DOMAIN_MIN", "DOMAIN_MAX"):
                expected = 0.0 if tokens[0] == "DOMAIN_MIN" else 1.0
                if any(abs(float(v) - expected) > 1e-4 for v in tokens[1:4]):
                    raise ValueError(f"Non-default DOMAIN {tokens} is not supported")
                continue

            if len(tokens) >= 3:
                rows.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])

    if lut_size is None:
        raise ValueError(f"Invalid .cube format: missing LUT_3D_SIZE in {path}")

    expected_rows = lut_size**3
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {lut_size}^3 LUT, but found {len(rows)}"
        )

    # Raw layout: Blue slowest, then Green, then Red fastest -> [iB, iG, iR, c]
    arr = np.asarray(rows, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
    # Permute to [c, iR, iG, iB] for SinglePassLUTEngine
    lut_tensor = torch.from_numpy(arr).permute(3, 2, 1, 0).contiguous()
    return lut_tensor


def write_cube(path: Union[str, Path], lut_tensor: torch.Tensor, title: str = "LUT3D") -> None:
    """Exports a [3, iR, iG, iB] PyTorch tensor to standard Adobe 3D .cube file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c, n_r, n_g, n_b = lut_tensor.shape
    if c != 3 or n_r != n_g or n_g != n_b:
        raise ValueError(f"Expected [3, N, N, N] shape, got {lut_tensor.shape}")

    N = n_r
    # Inverse permutation from [c, iR, iG, iB] back to [iB, iG, iR, c]
    arr = lut_tensor.permute(3, 2, 1, 0).detach().cpu().numpy().reshape(-1, 3)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {N}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for r, g, b in arr:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
