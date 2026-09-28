"""Command-line inference entrypoint for Phone-to-Pro photo enhancement.

Usage:
    # Enhance single image in Sony Alpha Mode:
    python scripts/inference.py --input input.jpg --output enhanced.png --mode sony

    # Enhance directory of photos in Apple Photonic Mode:
    python scripts/inference.py --input ./photos --output ./enhanced --mode apple

    # Fast preview without 4x super-resolution:
    python scripts/inference.py --input input.jpg --output preview.png --no-sr
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import PhotoEnhancementPipeline, configure_hardware_acceleration, load_cube


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phone-to-Pro Photo Enhancement Pipeline (Sony Alpha & Apple Photonic color science)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input image file or directory of images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output image file or output directory",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="sony",
        choices=["sony", "apple"],
        help="Enhancement mode: 'sony' (Alpha look) or 'apple' (Photonic Smart HDR look)",
    )
    parser.add_argument(
        "--sony-route",
        type=str,
        default="b",
        choices=["a", "b"],
        help="Sony grading route: 'a' (S-Log3 input + .cube LUT) or 'b' (filmic luminance S-curve)",
    )
    parser.add_argument(
        "--lut",
        type=str,
        default=None,
        help="Optional path to a custom .cube 3D-LUT file",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.5,
        help="Denoise strength for super-resolution [0.0 - 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Super-resolution upscale factor (default: 4)",
    )
    parser.add_argument(
        "--no-sr",
        action="store_true",
        help="Skip super-resolution upscaling (run color grading & tone mapping at 1x resolution)",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Skip output sharpening pass",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU worker threads (default: auto)",
    )
    return parser.parse_args()


def process_image(
    pipeline: PhotoEnhancementPipeline,
    img_path: Path,
    out_path: Path,
    mode: str,
    sony_route: str,
    enable_sr: bool,
    enable_denoise: bool,
    enable_sharpen: bool,
):
    print(f"Enhancing: {img_path.name} -> {out_path.name} [Mode: {mode.upper()}]")
    pil_in = Image.open(img_path)
    w_orig, h_orig = pil_in.size

    t0 = time.perf_counter()
    pil_out = pipeline.enhance_pil(
        pil_in,
        mode=mode,
        sony_route=sony_route,
        enable_sr=enable_sr,
        enable_denoise=enable_denoise,
        enable_sharpening=enable_sharpen,
    )
    elapsed = time.perf_counter() - t0

    w_out, h_out = pil_out.size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_out.save(out_path, quality=95)
    print(
        f"  Done in {elapsed:.3f}s: {w_orig}x{h_orig} -> {w_out}x{h_out} "
        f"({(w_out * h_out) / 1e6:.1f} MP)"
    )


def main():
    args = parse_args()
    profile = configure_hardware_acceleration(num_threads=args.threads)
    print(f"Initialized Phone-to-Pro on {profile.cpu_model} ({profile.logical_threads} threads, oneDNN active)")

    pipeline = PhotoEnhancementPipeline(
        sr_scale=args.scale,
        sr_denoise=args.denoise,
    )

    # Load custom LUT if provided
    if args.lut:
        print(f"Loading custom .cube LUT: {args.lut}")
        lut_tensor = load_cube(args.lut)
        with torch.no_grad():
            pipeline.lut_engine.luts[0].copy_(lut_tensor)

    in_path = Path(args.input)
    out_path = Path(args.output)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    if in_path.is_file():
        # Single file
        if out_path.is_dir():
            target_out = out_path / f"enhanced_{in_path.stem}.png"
        else:
            target_out = out_path
        process_image(
            pipeline,
            in_path,
            target_out,
            mode=args.mode,
            sony_route=args.sony_route,
            enable_sr=not args.no_sr,
            enable_denoise=args.denoise > 0,
            enable_sharpen=not args.no_sharpen,
        )
    elif in_path.is_dir():
        # Batch directory
        out_path.mkdir(parents=True, exist_ok=True)
        image_files = [f for f in sorted(in_path.iterdir()) if f.suffix.lower() in valid_exts]
        print(f"Found {len(image_files)} images in {in_path}")
        for img_file in image_files:
            target_out = out_path / f"{img_file.stem}_enhanced.png"
            process_image(
                pipeline,
                img_file,
                target_out,
                mode=args.mode,
                sony_route=args.sony_route,
                enable_sr=not args.no_sr,
                enable_denoise=args.denoise > 0,
                enable_sharpen=not args.no_sharpen,
            )
    else:
        print(f"Error: input path {in_path} does not exist", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
