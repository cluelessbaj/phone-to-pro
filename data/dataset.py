"""Paired Image Dataset with Float32 Ingestion and Synchronized Augmentations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class PairedEnhancementDataset(Dataset):
    """Dataset providing paired (input, target) images in unquantized float32 [0.0, 1.0]."""

    def __init__(
        self,
        pairs: Optional[List[Tuple[Union[str, Path], Union[str, Path]]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        patch_size: Optional[int] = 256,
        thumb_size: int = 256,
        augment: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.thumb_size = thumb_size
        self.augment = augment
        self.items: List[Tuple[Path, Path]] = []

        if manifest_path and Path(manifest_path).is_file():
            with open(manifest_path, "r") as f:
                data = json.load(f)
                for entry in data:
                    self.items.append((Path(entry["input"]), Path(entry["target"])))
        elif pairs:
            self.items = [(Path(p[0]), Path(p[1])) for p in pairs]

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: Path) -> torch.Tensor:
        """Loads image as planar float32 tensor [3, H, W] in [0.0, 1.0]."""
        pil_img = Image.open(path).convert("RGB")
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        in_path, tgt_path = self.items[idx]
        img_in = self._load_image(in_path)
        img_tgt = self._load_image(tgt_path)

        # Ensure spatial shapes match
        if img_in.shape[1:] != img_tgt.shape[1:]:
            img_tgt = F.interpolate(
                img_tgt.unsqueeze(0),
                size=img_in.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Synchronized paired cropping if requested
        _, H, W = img_in.shape
        if self.patch_size and H >= self.patch_size and W >= self.patch_size:
            if self.augment:
                top = random.randint(0, H - self.patch_size)
                left = random.randint(0, W - self.patch_size)
            else:
                top = (H - self.patch_size) // 2
                left = (W - self.patch_size) // 2

            img_in = img_in[:, top : top + self.patch_size, left : left + self.patch_size]
            img_tgt = img_tgt[:, top : top + self.patch_size, left : left + self.patch_size]

        # Synchronized geometric flips
        if self.augment:
            if random.random() > 0.5:
                img_in = torch.flip(img_in, dims=[2])
                img_tgt = torch.flip(img_tgt, dims=[2])
            if random.random() > 0.5:
                img_in = torch.flip(img_in, dims=[1])
                img_tgt = torch.flip(img_tgt, dims=[1])

        # Generate low-res thumbnail proxy for the predictor network [3, thumb_size, thumb_size]
        thumb = F.interpolate(
            img_in.unsqueeze(0),
            size=(self.thumb_size, self.thumb_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return {
            "thumbnail": thumb.contiguous(),
            "input": img_in.contiguous(),
            "target": img_tgt.contiguous(),
            "id": in_path.stem,
        }
