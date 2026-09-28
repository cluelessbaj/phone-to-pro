"""Synthetic High-Fidelity Pair Generator.

Generates ground-truth target pairs for target tuning:
1. Sony Alpha Mode: High-quality neutral captures transformed via S-Gamut3.Cine/S-Log3 input
   transfer followed by Sony .cube look profiles.
2. Apple Smart HDR Mode: Exposure-bracketed (-2, 0, +2 EV) merges using Mertens local tone mapping.

Enforces Section 6.2 Guardrail:
The DSLR Photo Enhancement Dataset (DPED) is strictly excluded from Sony Alpha training.
DPED pairs mobile cameras against a Canon 70D DSLR target, which bakes in Canon's color science.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from engine.color_transform import load_cube, to_slog3_input
from engine.lut_interp import SinglePassLUTEngine


def verify_dped_exclusion(dataset_path: Union[str, Path]) -> None:
    """Enforces Section 6.2 Guardrail: strictly rejects DPED (Canon 70D) datasets."""
    p_str = str(dataset_path).lower()
    forbidden_tokens = ["dped", "canon", "canon70d", "canon_70d"]
    for token in forbidden_tokens:
        if token in p_str:
            raise ValueError(
                f"Dataset exclusion violation: '{token}' found in path '{dataset_path}'. "
                f"Per Section 6.2, DPED is strictly excluded because it targets Canon 70D color rendering, "
                f"which conflicts directly with Sony Alpha color science."
            )


class SyntheticPairGenerator:
    """Batch processor generating paired synthetic targets from neutral source captures."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir = self.output_dir / "input"
        self.target_dir = self.output_dir / "target"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def generate_sony_pairs(
        self,
        source_images: List[Path],
        cube_lut_path: Path,
        exposure_offset: float = 1.0,
    ) -> List[dict]:
        """Generates synthetic high-fidelity pairs using Sony S-Log3 and .cube look transform."""
        verify_dped_exclusion(cube_lut_path)
        lut_tensor = load_cube(cube_lut_path)

        lut_engine = SinglePassLUTEngine(num_luts=1, lut_dim=lut_tensor.shape[1])
        lut_engine.luts[0].copy_(lut_tensor)

        weights = torch.tensor([[1.0]], dtype=torch.float32)
        manifest = []

        for img_path in source_images:
            verify_dped_exclusion(img_path)
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

            # Step 1: Input log transform (sRGB -> linear -> S-Gamut3.Cine / S-Log3)
            slog_in = to_slog3_input(tensor, exposure_offset=exposure_offset)

            # Step 2: LUT evaluation
            target_tensor = lut_engine(slog_in, weights).clamp(0.0, 1.0)

            # Save input copy and target
            stem = img_path.stem
            out_in_path = self.input_dir / f"{stem}.png"
            out_tgt_path = self.target_dir / f"{stem}.png"

            pil_img.save(out_in_path)
            target_np = (target_tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(target_np).save(out_tgt_path)

            manifest.append(
                {
                    "id": stem,
                    "input": str(out_in_path.resolve()),
                    "target": str(out_tgt_path.resolve()),
                    "mode": "sony",
                }
            )

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def generate_apple_hdr_pairs(self, source_images: List[Path]) -> List[dict]:
        """Generates synthetic Smart-HDR pairs via synthetic bracketing (-2, 0, +2 EV) + Mertens merge."""
        manifest = []
        merge_mertens = cv2.createMergeMertens()

        for img_path in source_images:
            verify_dped_exclusion(img_path)
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32) / 255.0

            # Synthesize exposure brackets: -2 EV (darker), 0 EV (base), +2 EV (brighter)
            b_underexposed = np.clip(img_np * 0.25, 0.0, 1.0)
            b_base = img_np
            b_overexposed = np.clip(img_np * 2.50, 0.0, 1.0)

            brackets_uint8 = [
                (b_underexposed * 255).astype(np.uint8),
                (b_base * 255).astype(np.uint8),
                (b_overexposed * 255).astype(np.uint8),
            ]

            # Mertens exposure fusion
            fusion = merge_mertens.process(brackets_uint8)
            fusion_uint8 = np.clip(fusion * 255.0, 0.0, 255.0).astype(np.uint8)

            stem = img_path.stem
            out_in_path = self.input_dir / f"{stem}.png"
            out_tgt_path = self.target_dir / f"{stem}.png"

            pil_img.save(out_in_path)
            Image.fromarray(fusion_uint8).save(out_tgt_path)

            manifest.append(
                {
                    "id": stem,
                    "input": str(out_in_path.resolve()),
                    "target": str(out_tgt_path.resolve()),
                    "mode": "apple",
                }
            )

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
