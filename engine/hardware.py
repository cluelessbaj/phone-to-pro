"""Hardware Acceleration and Diagnostic Utilities.

Optimized for Intel Core i5-10310U (4C/8T with AVX2 & oneDNN)
and Intel UHD Graphics (CML GT2 via Mesa ANV Vulkan driver) on Arch Linux.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HardwareProfile:
    cpu_model: str
    physical_cores: int
    logical_threads: int
    has_avx2: bool
    has_onednn: bool
    vulkan_available: bool
    vulkan_device_name: Optional[str]
    vulkan_driver: Optional[str]
    optimal_backend: str


def detect_vulkan_device() -> tuple[bool, Optional[str], Optional[str]]:
    """Inspects Vulkan physical devices via vulkaninfo if available."""
    vulkaninfo_bin = shutil.which("vulkaninfo")
    if not vulkaninfo_bin:
        return False, None, None

    try:
        proc = subprocess.run(
            [vulkaninfo_bin, "--summary"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if proc.returncode != 0:
            return False, None, None

        device_name = None
        driver_name = None
        for line in proc.stdout.splitlines():
            line_str = line.strip()
            if "deviceName" in line_str and "=" in line_str and not device_name:
                device_name = line_str.split("=")[-1].strip()
            if "driverName" in line_str and "=" in line_str and not driver_name:
                driver_name = line_str.split("=")[-1].strip()

        return True, device_name or "Vulkan GPU", driver_name
    except Exception:
        return False, None, None


def detect_cpu_features() -> tuple[str, bool]:
    """Inspects /proc/cpuinfo on Linux for CPU model and AVX2 flag."""
    model_name = "x86_64 CPU"
    has_avx2 = False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name") and model_name == "x86_64 CPU":
                    model_name = line.split(":", 1)[1].strip()
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip().split()
                    if "avx2" in flags:
                        has_avx2 = True
    except Exception:
        pass
    return model_name, has_avx2


def get_hardware_profile() -> HardwareProfile:
    """Builds a complete hardware profile for the host system."""
    cpu_model, has_avx2 = detect_cpu_features()
    physical_cores = os.cpu_count() or 4
    logical_threads = os.cpu_count() or 4

    has_onednn = bool(torch.backends.mkldnn.is_available())
    vulkan_ok, vulkan_device, vulkan_driver = detect_vulkan_device()

    # Determine optimal execution backend
    if vulkan_ok and shutil.which("realesrgan-ncnn-vulkan"):
        optimal_backend = "vulkan"
    elif has_onednn:
        optimal_backend = "onednn_cpu"
    else:
        optimal_backend = "cpu"

    return HardwareProfile(
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_threads=logical_threads,
        has_avx2=has_avx2,
        has_onednn=has_onednn,
        vulkan_available=vulkan_ok,
        vulkan_device_name=vulkan_device,
        vulkan_driver=vulkan_driver,
        optimal_backend=optimal_backend,
    )


def configure_hardware_acceleration(num_threads: Optional[int] = None) -> HardwareProfile:
    """Applies optimal hardware threading and vector acceleration settings."""
    profile = get_hardware_profile()

    # Optimal OpenMP / PyTorch thread scheduling for i5-10310U
    threads = num_threads or profile.logical_threads
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, threads // 2))
    except (RuntimeError, Exception):
        pass

    return profile


def print_hardware_summary() -> None:
    """Prints a structured hardware acceleration diagnosis."""
    profile = configure_hardware_acceleration()
    print("=" * 60)
    print("PHONE-TO-PRO HARDWARE ACCELERATION PROFILE")
    print("=" * 60)
    print(f" CPU Model:          {profile.cpu_model}")
    print(f" Logical Threads:    {profile.logical_threads} (Active PyTorch threads: {torch.get_num_threads()})")
    print(f" AVX2 Vector SIMD:   {'Enabled' if profile.has_avx2 else 'Not Detected'}")
    print(f" Intel oneDNN/MKL:   {'Active' if profile.has_onednn else 'Unavailable'}")
    print(f" Vulkan Support:     {'Available' if profile.vulkan_available else 'Unavailable'}")
    if profile.vulkan_available:
        print(f" Vulkan Device:      {profile.vulkan_device_name}")
        print(f" Vulkan Driver:      {profile.vulkan_driver}")
    print(f" Optimal Backend:    {profile.optimal_backend}")
    print("=" * 60)
