"""Loss Functions and Structural Regularizers for Parametric Photo Enhancement.

Includes:
1. Pixel Loss (Smooth L1).
2. Color Cosine Loss (L_cosine): Penalizes hue and chromaticity errors independently of luminance.
3. Perceptual Loss (L_perc): Multi-scale VGG-19 feature distance (relu1_2, relu2_2, relu3_2).
4. Structural LUT Regularizers:
   - Monotonicity Regularizer (R_mono): Enforces non-negative partial derivatives across 3D lattice.
   - Smoothness Regularizer (R_smooth): Penalizes second-order spatial differences across 3D lattice.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ColorCosineLoss(nn.Module):
    """Penalizes hue and chromaticity angle errors across RGB color vectors."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:

        pred:   [B, 3, H, W] in [0.0, 1.0]
        target: [B, 3, H, W] in [0.0, 1.0]
        """
        # Dot product across channels at each pixel
        dot = (pred * target).sum(dim=1, keepdim=True)
        norm_pred = torch.norm(pred, p=2, dim=1, keepdim=True)
        norm_target = torch.norm(target, p=2, dim=1, keepdim=True)

        cosine_sim = dot / (norm_pred * norm_target + self.eps)
        # Cosine distance: 1.0 - cos(theta), clamped in [0.0, 2.0]
        loss = (1.0 - cosine_sim).clamp(min=0.0)
        return loss.mean()


class LUTRegularizationLoss(nn.Module):
    """Hard structural regularizers on 3D-LUT lattice parameters.

    Prevents color banding, noise amplification, solarization, and tone reversals.
    """

    def __init__(self, lambda_smooth: float = 1e-4, lambda_mono: float = 1e-2):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_mono = lambda_mono

    def forward(self, luts: torch.Tensor) -> torch.Tensor:
        """Args:

        luts: [K, 3, D, D, D] basis 3D-LUT tensor.
        """
        # 1. First-order forward differences for monotonicity: [K, 3, D-1, D, D]
        dx = luts[:, :, 1:, :, :] - luts[:, :, :-1, :, :]
        dy = luts[:, :, :, 1:, :] - luts[:, :, :, :-1, :]
        dz = luts[:, :, :, :, 1:] - luts[:, :, :, :, :-1]

        # Monotonicity penalty: penalize negative changes (tone inversions)
        loss_mono = (
            torch.relu(-dx).mean()
            + torch.relu(-dy).mean()
            + torch.relu(-dz).mean()
        )

        # 2. Second-order differences for surface smoothness
        d2x = dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :]
        d2y = dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]
        d2z = dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]

        loss_smooth = (
            (d2x**2).mean()
            + (d2y**2).mean()
            + (d2z**2).mean()
        )

        return (self.lambda_smooth * loss_smooth) + (self.lambda_mono * loss_mono)


class VGGPerceptualLoss(nn.Module):
    """Multi-scale feature distance from frozen VGG-19 backbone."""

    def __init__(self, layers: Optional[List[str]] = None, pretrained: bool = False):
        super().__init__()
        self.layers = layers or ["relu1_2", "relu2_2", "relu3_2"]
        self.pretrained = pretrained

        # Slice VGG-19 features
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        vgg = models.vgg19(weights=weights).features

        # Layer mapping in standard torchvision VGG-19:
        # conv1_1: 0, relu1_1: 1, conv1_2: 2, relu1_2: 3
        # conv2_1: 5, relu2_1: 6, conv2_2: 7, relu2_2: 8
        # conv3_1: 10, relu3_1: 11, conv3_2: 12, relu3_2: 13
        layer_indices = {"relu1_2": 4, "relu2_2": 9, "relu3_2": 14}
        max_idx = max(layer_indices[l] for l in self.layers)

        self.slices = nn.ModuleList()
        prev = 0
        for l in self.layers:
            curr = layer_indices[l]
            self.slices.append(vgg[prev:curr])
            prev = curr

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        x_pred, x_tgt = pred, target
        for slice_layer in self.slices:
            x_pred = slice_layer(x_pred)
            x_tgt = slice_layer(x_tgt)
            loss = loss + F.l1_loss(x_pred, x_tgt)
        return loss / len(self.slices)


class TotalEnhancementLoss(nn.Module):
    """Composite optimization objective uniting reconstruction, perceptual, and LUT smoothness losses."""

    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_cosine: float = 0.1,
        lambda_perc: float = 0.05,
        lambda_smooth: float = 1e-4,
        lambda_mono: float = 1e-2,
        use_vgg: bool = False,
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_cosine = lambda_cosine
        self.lambda_perc = lambda_perc
        self.use_vgg = use_vgg

        self.cosine_loss = ColorCosineLoss()
        self.lut_reg = LUTRegularizationLoss(lambda_smooth, lambda_mono)
        self.vgg_loss = VGGPerceptualLoss(pretrained=False) if use_vgg else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        luts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Pixel reconstruction loss (Smooth L1)
        l_pixel = F.smooth_l1_loss(pred, target)

        # 2. Color cosine loss
        l_cosine = self.cosine_loss(pred, target)

        # 3. Perceptual loss
        l_perc = self.vgg_loss(pred, target) if self.vgg_loss is not None else torch.tensor(0.0, device=pred.device)

        # 4. LUT structural regularization
        l_reg = self.lut_reg(luts) if luts is not None else torch.tensor(0.0, device=pred.device)

        total = (
            self.lambda_pixel * l_pixel
            + self.lambda_cosine * l_cosine
            + self.lambda_perc * l_perc
            + l_reg
        )

        return {
            "total": total,
            "pixel": l_pixel,
            "cosine": l_cosine,
            "perceptual": l_perc,
            "lut_reg": l_reg,
        }
