"""MIT-Adobe FiveK Dataset Parser with Strict Expert C Filtering Rule.

Enforces Section 6.1 Constraint:
The MIT-Adobe FiveK dataset contains edits from 5 professional photographers (A, B, C, D, E).
Pretraining across differing photographer styles causes gradient cancellation and produces
muddy, desaturated outputs. This loader strictly filters and ingests ONLY Expert C targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union


@dataclass
class FiveKPair:
    image_id: str
    input_path: Path
    target_path: Path
    expert: str = "C"


class FiveKExpertCParser:
    """Parses and filters MIT-Adobe FiveK dataset strictly for Expert C."""

    VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng"}

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.pairs: List[FiveKPair] = []

    def scan_pairs(self) -> List[FiveKPair]:
        """Scans root_dir and constructs paired inputs and Expert C targets.

        Supports standard directory layouts:
        Layout 1:
            root_dir/
                input/ (or raw/)
                    a0001.jpg
                expert_c/ (or ExpertC/)
                    a0001.jpg
        Layout 2:
            Flat directory with naming convention:
                a0001.jpg (input)
                a0001-c.jpg (or a0001_c.jpg, a0001_ExpertC.jpg)
        """
        pairs: List[FiveKPair] = []
        if not self.root_dir.is_dir():
            return pairs

        input_dir = None
        for candidate in ["input", "raw", "original", "source"]:
            p = self.root_dir / candidate
            if p.is_dir():
                input_dir = p
                break

        target_dir = None
        for candidate in ["expert_c", "ExpertC", "expertC", "c"]:
            p = self.root_dir / candidate
            if p.is_dir():
                target_dir = p
                break

        if input_dir and target_dir:
            # Layout 1: Separate directories
            input_files = {f.stem: f for f in input_dir.iterdir() if f.suffix.lower() in self.VALID_IMAGE_EXTS}
            target_files = {f.stem: f for f in target_dir.iterdir() if f.suffix.lower() in self.VALID_IMAGE_EXTS}

            for stem, in_path in input_files.items():
                # Strictly match stem in Expert C directory
                if stem in target_files:
                    pairs.append(FiveKPair(image_id=stem, input_path=in_path, target_path=target_files[stem]))
        else:
            # Layout 2: Flat or search tree
            all_files = [f for f in self.root_dir.rglob("*") if f.suffix.lower() in self.VALID_IMAGE_EXTS]
            # Group by base ID
            expert_c_files = {}
            source_files = {}
            for f in all_files:
                stem_lower = f.stem.lower()
                # Strictly filter for Expert C indicator
                if stem_lower.endswith("-c") or stem_lower.endswith("_c") or "expertc" in stem_lower:
                    # Strip expert suffix to recover base ID
                    base_id = stem_lower.replace("-c", "").replace("_c", "").replace("expertc", "").strip()
                    expert_c_files[base_id] = f
                elif not any(f"expert{x}" in stem_lower or stem_lower.endswith(f"-{x}") for x in ["a", "b", "d", "e"]):
                    source_files[stem_lower] = f

            for base_id, tgt in expert_c_files.items():
                if base_id in source_files:
                    pairs.append(FiveKPair(image_id=base_id, input_path=source_files[base_id], target_path=tgt))

        # Sort deterministically
        pairs.sort(key=lambda p: p.image_id)
        self.pairs = pairs
        return pairs

    def split_train_val(self, val_ratio: float = 0.1) -> Tuple[List[FiveKPair], List[FiveKPair]]:
        """Splits pairs into deterministic train and validation sets."""
        if not self.pairs:
            self.scan_pairs()

        n_total = len(self.pairs)
        n_val = int(n_total * val_ratio)
        train_pairs = self.pairs[:-n_val] if n_val > 0 else self.pairs
        val_pairs = self.pairs[-n_val:] if n_val > 0 else []
        return train_pairs, val_pairs
