# Phone-to-Pro Example: Night Moon Capture (`moon.jpg`)

This directory contains real-world demonstration assets processed through the **Phone-to-Pro** photo enhancement pipeline on an Intel Core i5-10310U (CPU oneDNN / Mesa Vulkan environment).

---

## 1. Full-Frame Renders & Comparisons

| Image | Profile | Characteristics |
|---|---|---|
| `moon_original_preview.jpg` | Raw Phone Input | Elevated black point, high-ISO sensor grain in clouds, flat contrast. |
| `moon_enhanced_preview.jpg` | **Sony Alpha Mode** | Deep-space black floor ($0.0$), filmic S-curve midtones, BIONZ XR-style noise coring, prime-lens MTF acutance. |
| `moon_apple_preview.jpg` | **Apple Photonic Mode** | Retinex local shadow extraction ($s=4$ guided filter), soft shoulder highlight compression, balanced cloud volume. |
| `moon_split_preview.jpg` | **50/50 Lunar Split** | Direct vertical split through the moon disc (Sony left vs. Apple right) for 1:1 dynamic range evaluation. |
| `moon_fusion_preview.jpg` | **Hybrid Fusion Master** | Parameter-space blend uniting Sony's inky contrast & crater edge definition with Apple's illuminated cloud volume. |
| `moon_sbs_preview.jpg` | **Side-by-Side Master** | Full-frame panoramic dual layout. |

---

## 2. 1:1 Detail Crops

### Lunar Disc Clarity & Highlight Recovery
- `moon_sony_crisp_tight.jpg`: 1:1 zoom on the lunar disc demonstrating peak-aware highlight protection and zero white clipping blowout.
- `moon_crop_sony_tight_fixed.jpg`: Fixed lunar rim separation from the surrounding illuminated cloud corona.
- `moon_crop_enhanced.jpg`: Wide 800 × 800 crop of the Sony Alpha render.
- `moon_crop_apple.jpg`: Wide 800 × 800 crop of the Apple Photonic render.
- `moon_crop_original.jpg`: Wide 800 × 800 crop of the raw phone capture.

### Background Grain & Sensor Noise Clean
- `grain_check_orig.jpg`: Raw high-ISO sensor noise in the dark sky/clouds (Laplacian variance: $10.32$).
- `grain_check_sony.jpg`: Un-cored unsharp mask amplifying noise grains (Laplacian variance: $19.42$).
- `grain_check_cleaned.jpg`: Fast edge-preserving subsampled guided denoising + noise coring (Laplacian variance: **$1.02$**, 10× cleaner, velvety smooth).

---

## 3. Reproduction Commands

To reproduce these enhancements from the project root:

```bash
# Sony Alpha Mode (Fine MTF acutance + inky contrast)
python scripts/inference.py --input examples/moon/moon_original_preview.jpg --output outputs/moon_sony.png --mode sony --no-sr

# Apple Photonic Mode (Retinex dynamic range + soft shoulder)
python scripts/inference.py --input examples/moon/moon_original_preview.jpg --output outputs/moon_apple.png --mode apple --no-sr

# Super-Resolution Upscaling (4x Neural Compact Float32 Core)
python scripts/inference.py --input examples/moon/moon_original_preview.jpg --output outputs/moon_4x.png --mode sony --scale 4
```
