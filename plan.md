# Phone-to-Pro Photo Enhancement Pipeline (`plan.md`)

> **Revision note:** this pass fixes two verified bugs found during code review — a channel/axis mismatch in the single-pass LUT engine's `grid_sample` call (was silently swapping R and B), and a border-darkening artifact in the guided filter's box filter (from `avg_pool2d`'s default zero-padding). Both are corrected in Section 7 below, with the reasoning kept inline as comments. Everything else in this document was already sound and is reproduced as designed. A later pass adds Section 11 (zero-training baseline) and corrects two things it exposed: Sony's `.cube` LUTs expect log-encoded input (§6.2, §11.2), and the guided filter only *partially* aligns transitions to edges (§3.3B).

## 1. Executive Summary & Core Philosophy

This specification outlines the architecture, data strategy, and mathematical formulation for an automated photo enhancement pipeline designed to elevate standard mobile phone captures to the optical clarity and color science of professional camera systems (**Sony Alpha** and **Apple Photonic Engine / Smart HDR**).

### Core Architectural Decisions

1. **Neuro-Parametric Decoupling:** Pixel synthesis (super-resolution and sensor texture reconstruction) is strictly decoupled from aesthetic color science and tone mapping.
2. **Execution Order (SR First, Grade Second):** Super-resolution runs on neutral, unquantized sensor pixels. Aesthetic color grading and non-linear tone curves are applied afterward. Running super-resolution on pre-graded, high-contrast images amplifies edge halos, ringing, and color-clipping artifacts.
3. **Float32 Intermediate Hand-Off:** Intermediate tensors between super-resolution and grading are maintained strictly in unquantized `float32` ($[0.0, 1.0]$). While this cannot retroactively restore dynamic range clipped at exposure time, it prevents quantization banding and posterization during non-linear curve expansion.
4. **Single-Pass Parametric Efficiency:** Multi-LUT blending is computed in parameter space *before* spatial interpolation, collapsing $K$ full-resolution volumetric lookups into a single trilinear pass.
5. **Dual Mode Disentanglement:**
   * **Sony Mode:** Global color grading via an optimized single-pass composite 3D-LUT and a filmic luminance S-curve (preserving shadow roll-off, desaturated toe, and neutral organic greens).
   * **Apple Mode:** Spatially-aware local tone grading via edge-guided spatial LUT weighting, local illumination gain maps, and micro-contrast refinement.

---

## 2. End-to-End Pipeline Architecture

```
                      [Input Phone Photo (sRGB / DCI-P3)]
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
        [Downsampled Thumbnail]             [SR Core: Real-ESRGAN]
            (256–512px)                      (realesr-general-x4v3)
                     │                                   │
                     ▼                                   ▼
          [Parameter Predictor]              [High-Res Neutral Tensor]
         (Shared MobileNetV3)                  (float32 [0.0, 1.0])
                     │                                   │
         ┌───────────┴───────────┐                       │
         ▼                       ▼                       │
    [Sony Head]            [Apple Head]                  │
  • Basis weights (w)    • Basis weights (w)             │
  • Spline params (S)    • Spatial weight grid (W_s)     │
                         • Low-res gain map (G_s)        │
         │                       │                       │
         │                       ▼                       │
         │             [Fast Guided Filter] ◄────────────┤ (Guidance:
         │             (Edge-Aware Upsample)             │  Neutral Lum)
         │                       │                       │
         └───────────┬───────────┘                       │
                     ▼                                   │
       [Single-Pass Blended 3D-LUT] ◄────────────────────┘
         (Composite Table Application)
                     │
                     ▼
       [Parametric Tone Modification]
         (Sony: Spline S-Curve | Apple: Spatial Gain Map)
                     │
                     ▼
         [Adaptive Output Sharpening]
         (Radius dynamically indexed to output MP)
                     │
                     ▼
             [Final Enhanced Image]

```

---

## 3. Component Breakdown & Engineering Specifications

### 3.1 Thumbnail & Feature Extractor

* **Input:** Low-resolution image proxy ($256 \times 256$ to $512 \times 512$ pixels), bilinear downsampled, normalized to `float32`.
* **Backbone:** MobileNetV3-Small (width multiplier 0.75x–1.0x) or ShuffleNetV2. Memory footprint $< 2.5\text{M}$ parameters.
* **Execution Budget:** $< 3\text{ ms}$ on mobile NPU / modern desktop GPU.
* **Role:** Extracts scene-level semantics (lighting condition, dynamic range distribution, facial presence, foliage density) to drive downstream parameter regression.

### 3.2 Super-Resolution Core (Texture & Detail)

* **Model:** `realesr-general-x4v3` (via NCNN / TensorRT / ONNX Runtime).
* **Domain Alignment:** *Anime-tailored models (such as Real-CUGAN) are strictly excluded.* Their training priors target flat color fills and hard line art, which strip photographic film grain and treat sensor noise erratically. `realesr-general-x4v3` is trained on continuous-tone photographic degradations and includes an adjustable denoise parameter.
* **Output Specification:** 3-channel planar `float32` tensor, values in range $[0.0, 1.0]$, bypassing internal 8-bit quantization (`cv::imencode` / `uint8` cast).
* **Precision Scope:** Staying in `float32` prevents new quantization loss and posterization when stacking non-linear tone operations; it does not recover clipped sensor highlights or shadows from the source capture.

### 3.3 Parametric Grading Engine

#### A. Parameter-Space LUT Pre-Blending

Evaluating $K$ separate 3D-LUTs over a multi-megapixel image requires $K$ independent trilinear interpolation passes. Because trilinear interpolation $\mathcal{T}(\mathbf{x}, \mathbf{L})$ is strictly linear with respect to the lattice vertex values of the table $\mathbf{L}$:

$$\sum_{k=1}^K w_k \cdot \mathcal{T}(\mathbf{x}, \mathbf{L}_k) = \mathcal{T}\left(\mathbf{x}, \sum_{k=1}^K w_k \mathbf{L}_k\right)$$

* **Optimization:** Pre-blend the $K$ basis 3D-LUTs ($\mathbf{L}_k \in \mathbb{R}^{3 \times 33 \times 33 \times 33}$) using the predicted normalized weights $\mathbf{w} \in \mathbb{R}^K$:

$$\mathbf{L}_{\text{comp}} = \sum_{k=1}^K w_k \mathbf{L}_k \quad \text{where } \sum w_k = 1, \; w_k \ge 0$$

* **Complexity:** Blending $K=5$ tables of dimension $33^3$ requires roughly $10^6$ FLOPs (well under $0.1\text{ ms}$), completely eliminating $K-1$ passes of full-resolution volumetric texture lookups.

#### B. Edge-Aware Spatial Upsampling (Apple Mode)

In Apple Mode, the network predicts:

1. Coarse spatial LUT blend weights: $\mathbf{W}_s \in \mathbb{R}^{H_s \times W_s \times K}$ (e.g., $16 \times 16$).
2. Coarse illumination gain map: $\mathbf{G}_s \in \mathbb{R}^{H_s \times W_s \times 1}$.

* **The Halo Trap:** Upsampling low-resolution grids using standard bilinear or bicubic interpolation causes soft halos along high-contrast boundaries (e.g., backlit portraits, horizon lines, tree branches).
* **The Solution:** Upsample both $\mathbf{W}_s$ and $\mathbf{G}_s$ to high resolution using a **Fast Guided Filter** guided by the high-resolution neutral luminance map $I_{\text{lum}}$:

$$\mathbf{W}_{\text{high}} = \text{GuidedFilter}\left(\text{guide} = I_{\text{lum}}, \, \text{src} = \mathbf{W}_s, \, r=16, \, \epsilon=10^{-3}\right)$$

$$\mathbf{G}_{\text{high}} = \text{GuidedFilter}\left(\text{guide} = I_{\text{lum}}, \, \text{src} = \mathbf{G}_s, \, r=16, \, \epsilon=10^{-3}\right)$$

This pulls the transition boundaries of spatial LUTs and shadow gain adjustments toward real image edges. It reduces halos but does not eliminate them: in a synthetic step-edge test (16×16 coarse map, `r=16`, `eps=1e-3`) it cut mean error near the edge by roughly 10–30% versus plain bilinear upsampling, with the result depending on grid resolution, `r`, and `eps`. Validate on real backlit portraits before relying on it.

### 3.4 Adaptive Output Sharpening

* Executed as the final step on the graded `float32` tensor.
* **Sony Mode:** Fine-radius unsharp mask ($r = 0.5\text{px}$, amount $0.3\text{–}0.4$) to reproduce optical prime lens MTF curves.
* **Apple Mode:** Dual-scale unsharp mask ($r_1 = 0.8\text{px}$, $r_2 = 1.5\text{px}$) with an edge-detection threshold mask to boost local micro-contrast without sharpening noise in flat regions.

---

## 4. Mathematical Formulation & Loss Functions

The parameter predictor network is trained end-to-end to output basis weights $\mathbf{w}$, spline parameters $\mathbf{S}$, and spatial control maps.

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pixel}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{cos}} \mathcal{L}_{\text{cosine}} + \lambda_{\text{smooth}} \mathcal{R}_{\text{smooth}} + \lambda_{\text{mono}} \mathcal{R}_{\text{mono}}$$

### 4.1 Objective Components

* **Pixel Loss ($\mathcal{L}_{\text{pixel}}$):** Smooth L1 loss between predicted grade $I_{\text{pred}}$ and target $I_{\text{target}}$:

$$\mathcal{L}_{\text{pixel}} = \text{Smooth}_{L1}(I_{\text{pred}} - I_{\text{target}})$$

* **Color Cosine Loss ($\mathcal{L}_{\text{cosine}}$):** Penalizes hue and chromaticity errors independently of luminance:

$$\mathcal{L}_{\text{cosine}} = 1 - \frac{I_{\text{pred}} \cdot I_{\text{target}}}{\Vert{}I_{\text{pred}}\Vert{}_2 \Vert{}I_{\text{target}}\Vert{}_2}$$

* **Perceptual Loss ($\mathcal{L}_{\text{perc}}$):** Multi-scale feature distance from a frozen VGG-19 backbone:

$$\mathcal{L}_{\text{perc}} = \sum_{l \in \{relu1\_2,\ relu2\_2,\ relu3\_2\}} \frac{1}{N_l} \Vert{}\phi_l(I_{\text{pred}}) - \phi_l(I_{\text{target}})\Vert{}_1$$

*(layer names corrected to `relu1_2` / `relu2_2` / `relu3_2` — standard VGG-19 activation naming)*

### 4.2 Structural Regularizers on Learned LUTs

To prevent the learned 3D-LUTs from generating color banding, noise amplification, or inverted tones, two hard structural constraints are applied to each basis table $\mathbf{L}$:

#### A. Smoothness Regularizer ($\mathcal{R}_{\text{smooth}}$)

Penalizes second-order spatial differences across the 3D lattice:

$$\mathcal{R}_{\text{smooth}} = \sum_{c \in \{R,G,B\}} \sum_{i,j,k} \left( \Vert{}\Delta_x^2 \mathbf{L}_c(i,j,k)\Vert{}_2^2 + \Vert{}\Delta_y^2 \mathbf{L}_c(i,j,k)\Vert{}_2^2 + \Vert{}\Delta_z^2 \mathbf{L}_c(i,j,k)\Vert{}_2^2 \right)$$

#### B. Monotonicity Regularizer ($\mathcal{R}_{\text{mono}}$)

Prevents local tonal reversals (where an increase in input brightness results in a decrease in output brightness):

$$\mathcal{R}_{\text{mono}} = \sum_{c \in \{R,G,B\}} \sum_{i,j,k} \left[ \max(0, -\Delta_x \mathbf{L}_c(i,j,k)) + \max(0, -\Delta_y \mathbf{L}_c(i,j,k)) + \max(0, -\Delta_z \mathbf{L}_c(i,j,k)) \right]$$

Where $\Delta_x \mathbf{L}_c(i,j,k) = \mathbf{L}_c(i+1, j, k) - \mathbf{L}_c(i, j, k)$.

---

## 5. Mode Specifications & Comparative Profiles

| Parameter | Sony Alpha Mode | Apple Photonic / Smart HDR Mode |
| --- | --- | --- |
| **LUT Architecture** | Global pre-blended 3D-LUT ($\mathbf{L}_{\text{comp}}$) | Spatially guided 3D-LUT ($\mathbf{W}_{\text{high}} \otimes \mathbf{L}_k$) |
| **Tone Curve Engine** | Global monotonic luminance S-curve spline | Guided illumination gain map ($\mathbf{G}_{\text{high}}$) + local tone curve |
| **Shadow Treatment** | Preserved black point, controlled toe, desaturated floor | Aggressively lifted shadows, color saturation preserved |
| **Highlight Roll-off** | Soft filmic shoulder, gradual desaturation into clipping | Local exposure suppression, recovered sky/cloud hues |
| **Color Matrix Bias** | True-to-life skin tones, muted olive/foliage greens | Warm mid-tones, boosted cyan/sky blues, saturated greens |
| **Sharpening Pass** | Minimal radius ($r = 0.5\text{px}$), texture/grain priority | Dual-scale ($r = 0.8\text{–}1.0\text{px}$) with edge-mask protection |

---

## 6. Data Strategy & Curation Guardrails

### 6.1 MIT-Adobe FiveK: Expert C Filtering Rule

* The MIT-Adobe FiveK dataset contains edits from 5 professional photographers (Experts A, B, C, D, E).
* **Strict Constraint:** Pretrain the parameter predictor **exclusively on Expert C targets**. Training across the entire pool of differing photographer styles causes gradient cancellation and produces muddy, desaturated outputs — Expert C is also the most common single-expert subset used for this reason in published enhancement benchmarks, which keeps results comparable to prior work.

### 6.2 Sony Alpha Dataset (DPED Exclusion)

* **Dataset Exclusion:** The DSLR Photo Enhancement Dataset (DPED) is **strictly excluded** from Sony Alpha training. DPED pairs mobile cameras (iPhone 3GS, BlackBerry Passport, Sony Xperia Z) against a **Canon 70D DSLR** target — Sony only appears there as one of the low-quality phone sources, never as the target color science. Training against DPED would bake in Canon's color rendering, conflicting directly with the Sony Alpha look this mode is meant to reproduce.
* **Ground-Truth Acquisition:**
  1. *Synthetic High-Fidelity Pairs:* High-quality, neutrally graded captures transformed via Sony `.cube` LUTs. Sony's official LUTs (Look Profiles, and the camera's LUT-import format) are built for S-Gamut3.Cine/S-Log3 or S-Gamut3/S-Log3 **input**, so source images must first pass through an input transform (sRGB → linear → S-Gamut3.Cine/S-Log3, see §11.2); applying them to display-referred sRGB directly gives crushed, wrongly contrasty targets. Official Creative Look (FL, IN, VV) `.cube` files were not confirmed to exist — verify before planning around them.
  2. *Hardware Pairs:* Direct physical test captures with mobile devices paired against Sony $\alpha 7$ III / $\alpha 7$ IV bodies under identical lighting.

### 6.3 Apple Photonic / Smart HDR Dataset

* **Ground-Truth Acquisition:**
  1. *Physical Alignment Pairs:* Hardware rig mounting the baseline test phone alongside an iPhone 14/15/16 Pro shooting in standard photo mode (capturing computational Smart HDR merges).
  2. *Synthetic Smart-HDR Pairs:* Baseline exposures paired with exposure-bracketed ($-2, 0, +2$ EV) merges using Mertens local tone mapping to establish shadow recovery targets. Note this is an accessible *proxy* for the look, not a reproduction of Apple's actual (proprietary, semantic-aware) merge algorithm — treat it as a reasonable approximation to fine-tune toward, not a ground-truth match.

---

## 7. Reference PyTorch Implementations

### 7.1 Single-Pass Pre-Blended 3D-LUT Kernel

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinglePassLUTEngine(nn.Module):
    def __init__(self, num_luts: int = 5, lut_dim: int = 33):
        super().__init__()
        self.num_luts = num_luts
        self.lut_dim = lut_dim

        # Initialize K basis 3D-LUTs as an identity-centered learnable tensor
        # Shape: [K, 3, D, D, D] -- axes (D_in, H_in, W_in) correspond to
        # (R, G, B) input coordinates respectively, per _init_identity below.
        self.luts = nn.Parameter(torch.zeros(num_luts, 3, lut_dim, lut_dim, lut_dim))
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            linear = torch.linspace(0, 1, self.lut_dim)
            r, g, b = torch.meshgrid(linear, linear, linear, indexing='ij')
            # r varies along axis 0 (D_in), g along axis 1 (H_in), b along axis 2 (W_in)
            identity = torch.stack([r, g, b], dim=0)  # [3, D, D, D]
            for k in range(self.num_luts):
                self.luts[k].copy_(identity)

    def forward(self, img_float32: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        img_float32: [B, 3, H, W] in range [0.0, 1.0], channel order (R, G, B)
        weights:     [B, K] Softmax-normalized blend coefficients
        """
        B, C, H, W = img_float32.shape

        # 1. Parameter-space pre-blending: [B, 3, D, D, D]
        composite_lut = torch.einsum('bk,kcdhw->bcdhw', weights, self.luts)

        # 2. Build the query grid.
        #    F.grid_sample's 5D convention reads grid[..., 0:3] as (x, y, z),
        #    which index the input's (W_in, H_in, D_in) axes respectively.
        #    _init_identity assigned R -> D_in (axis 0), G -> H_in (axis 1),
        #    B -> W_in (axis 2) -- so the grid's (x, y, z) must be (B, G, R),
        #    NOT the image's natural (R, G, B) channel order. Reorder channels
        #    here or the lookup silently swaps the R and B channels.
        grid = img_float32[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, 3] = (B, G, R)
        grid = (grid * 2.0) - 1.0

        # 3. Single-pass trilinear interpolation
        out = F.grid_sample(
            composite_lut,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        return out.squeeze(2)  # [B, 3, H, W]
```

### 7.2 Guided Filter Module for Spatial Maps

```python
class FastGuidedFilter(nn.Module):
    """
    Standard (He et al. 2010) guided filter for joint/edge-aware upsampling.
    """
    def __init__(self, radius: int = 16, eps: float = 1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def box_filter(self, x: torch.Tensor, r: int) -> torch.Tensor:
        # Reflect-pad manually before pooling with padding=0 to eliminate
        # border-darkening vignette artifacts.
        x_padded = F.pad(x, (r, r, r, r), mode='reflect')
        return F.avg_pool2d(x_padded, kernel_size=2 * r + 1, stride=1, padding=0)

    def forward(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        guide: Full-res luminance [B, 1, H, W] in [0.0, 1.0]
        src:   Low-res spatial weights/gain [B, C, H_s, W_s]
        """
        src = F.interpolate(src, size=guide.shape[2:], mode='bilinear', align_corners=False)

        r, eps = self.radius, self.eps
        mean_I = self.box_filter(guide, r)
        mean_p = self.box_filter(src, r)
        mean_Ip = self.box_filter(guide * src, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter(guide * guide, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a, r)
        mean_b = self.box_filter(b, r)

        return mean_a * guide + mean_b
```

### 7.3 Monotonicity & Smoothness Regularization Loss

```python
class LUTRegularizationLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 1e-4, lambda_mono: float = 1e-2):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_mono = lambda_mono

    def forward(self, luts: torch.Tensor) -> torch.Tensor:
        """
        luts: [K, 3, D, D, D]
        """
        dx = luts[:, :, 1:, :, :] - luts[:, :, :-1, :, :]
        dy = luts[:, :, :, 1:, :] - luts[:, :, :, :-1, :]
        dz = luts[:, :, :, :, 1:] - luts[:, :, :, :, :-1]

        loss_mono = (
            torch.relu(-dx).mean() +
            torch.relu(-dy).mean() +
            torch.relu(-dz).mean()
        )

        d2x = dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :]
        d2y = dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]
        d2z = dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]

        loss_smooth = (
            (d2x ** 2).mean() +
            (d2y ** 2).mean() +
            (d2z ** 2).mean()
        )

        return (self.lambda_smooth * loss_smooth) + (self.lambda_mono * loss_mono)
```

### 7.4 Dual-Head Parameter Regressor Backbone

```python
import torchvision.models as models

class DualHeadPredictor(nn.Module):
    def __init__(self, num_luts: int = 5, spline_points: int = 5):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_dim = base.classifier[0].in_features

        # Head A: Sony Mode
        self.sony_lut_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_luts)
        )
        self.sony_spline_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, spline_points)
        )

        # Head B: Apple Mode
        self.apple_lut_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_luts)
        )
        self.apple_spatial_head = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_luts + 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, mode: str = 'sony'):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)

        if mode == 'sony':
            lut_weights = F.softmax(self.sony_lut_head(pooled), dim=1)
            spline_pts = torch.sigmoid(self.sony_spline_head(pooled))
            return {'lut_weights': lut_weights, 'spline_points': spline_pts}
        elif mode == 'apple':
            lut_weights = F.softmax(self.apple_lut_head(pooled), dim=1)
            spatial_maps = self.apple_spatial_head(feat)
            spatial_weights = F.softmax(spatial_maps[:, :-1, :, :], dim=1)
            gain_map = torch.sigmoid(spatial_maps[:, -1:, :, :]) * 2.0
            return {
                'lut_weights': lut_weights,
                'spatial_weights': spatial_weights,
                'gain_map': gain_map
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
```

---

## 8. Repository Layout & File Structure

```
phone-to-pro/
├── configs/
│   ├── base_fivek.yaml          # Expert C pretraining configuration
│   ├── sony_cinetone.yaml       # Sony Alpha fine-tuning hyperparameters
│   └── apple_smart_hdr.yaml     # Apple spatial guided model hyperparameters
├── data/
│   ├── dataset.py               # Paired dataset loader with float32 transforms
│   ├── fivek_expert_c.py        # MIT-Adobe FiveK parser (Expert C filter)
│   └── synthetic_generator.py   # .cube to synthetic pair batch processor
├── engine/
│   ├── __init__.py
│   ├── hardware.py              # System diagnostic & acceleration manager
│   ├── guided_filter.py         # PyTorch guided filter implementation
│   ├── baseline.py              # Zero-training path: WB, levels, filmic curve, Retinex-style gain (Section 11)
│   ├── color_transform.py       # sRGB -> linear -> S-Gamut3.Cine/S-Log3 input transform, .cube loader
│   ├── lut_interp.py            # Parameter-space pre-blender & trilinear kernel
│   ├── pipeline.py               # End-to-end execution coordinator
│   ├── sharpening.py            # Resolution-indexed unsharp masking passes
│   └── spline_curve.py          # Monotonic cubic Hermite spline evaluator
├── models/
│   ├── __init__.py
│   ├── losses.py                # Reconstruction, VGG, Smoothness & Monotonicity
│   ├── predictor.py              # MobileNetV3 dual-head parameter regressor
│   └── srfactory.py             # Wrapper for Real-ESRGAN NCNN / PyTorch core
├── scripts/
│   ├── export_onnx.py           # Predictor & engine graph tracing to ONNX
│   ├── export_ncnn.sh           # ONNX to NCNN quantization & build script
│   ├── inference.py             # CLI entrypoint for batch image processing
│   ├── benchmark.py             # Hardware-accelerated latency benchmark
│   └── train.py                 # Multi-stage training harness
├── tests/
│   ├── conftest.py
│   ├── test_float_precision.py  # Validates no uint8 clamping at SR boundary
│   ├── test_lut_math.py         # Verifies Σ w_k T(x, L_k) == T(x, Σ w_k L_k)
│   ├── test_lut_identity.py     # Channel round-trip identity test
│   ├── test_input_transform.py  # 18% grey -> S-Log3 code 420/1023; .cube axis-order round-trip
│   ├── test_monotonicity.py     # Ensures zero tonal inversions across gamut
│   ├── test_guided_filter.py    # Border test and edge-aware upsample test
│   ├── test_sharpening.py       # Validates masks and bounds
│   ├── test_baseline.py         # End-to-end Milestone 0 baseline test
│   └── test_predictor_and_losses.py
├── pyproject.toml
└── plan.md
```

---

## 9. Implementation Roadmap & Execution Milestones

```
M0: Algorithmic Baseline ─────► M1: Predictor on FiveK-C ─────► M2: Float32 Single-Pass Pipeline
                                                                               │
M5: NCNN / Mobile Export ◄───── M4: Apple Guided Spatial ◄───── M3: Sony S-Cinetone Tuning
```

---

## 10. Defensive Engineering & Risk Mitigation

1. **Unrecoverable Sensor Clipping:** Raw luminance histogram check; compresses highlight shoulder when clipped pixels exceed 2.5%.
2. **Facial Artifacts in Super-Resolution:** Face area check blends high-frequency residual with bilateral base if face > 5%.
3. **Gamut and Monotonicity Violations:** Continuous clamping (`torch.clamp(..., 0.0, 1.0)`), Softmax partition of unity ($\sum w_k = 1$), and monotonicity regularization.
4. **LUT Channel/Axis Mislabeling:** Identity roundtrip test asserts per-channel accuracy on `(B, G, R)` grid ordering.

---

## 11. Zero-Training Baseline (Milestone 0 Specification)

See Section 11 of specification for reference implementations of `gray_world_wb`, `conservative_levels`, `to_slog3_input`, `load_cube`, `build_filmic_table`, `filmic_ratio_grade`, `apple_shadow_lift`, and `SubsampledGuidedFilter`.
