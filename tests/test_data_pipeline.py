"""Verification tests for dataset parsers, Expert C filtering, DPED exclusion, and synthetic generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from data.dataset import PairedEnhancementDataset
from data.fivek_expert_c import FiveKExpertCParser
from data.synthetic_generator import SyntheticPairGenerator, verify_dped_exclusion
from engine.color_transform import write_cube


def test_fivek_expert_c_filter_exclusion():
    """Validates that FiveKExpertCParser filters exclusively for Expert C targets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create mock inputs and multiple expert targets
        (tmp_path / "a001.jpg").touch()
        (tmp_path / "a001-a.jpg").touch()  # Expert A
        (tmp_path / "a001-b.jpg").touch()  # Expert B
        (tmp_path / "a001-c.jpg").touch()  # Expert C (ONLY valid target)
        (tmp_path / "a001-d.jpg").touch()  # Expert D
        (tmp_path / "a001-e.jpg").touch()  # Expert E

        parser = FiveKExpertCParser(tmp_path)
        pairs = parser.scan_pairs()

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.image_id == "a001"
        assert pair.target_path.name == "a001-c.jpg"


def test_dped_exclusion_rule():
    """Validates Section 6.2 guardrail: DPED (Canon 70D) dataset paths must be rejected."""
    # Forbidden paths must raise ValueError
    with pytest.raises(ValueError, match="Dataset exclusion violation"):
        verify_dped_exclusion("/datasets/DPED_iphone_canon70d")

    with pytest.raises(ValueError, match="Dataset exclusion violation"):
        verify_dped_exclusion("path/to/canon_camera_pairs")

    # Legitimate non-DPED paths must pass
    verify_dped_exclusion("data/sony_synthetic_pairs")
    verify_dped_exclusion("data/fivek_expert_c")


def test_synthetic_pair_generator_sony():
    """Validates synthetic high-fidelity pair generator for Sony Alpha mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"

        # 1. Create a dummy source image
        img = Image.fromarray(np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8))
        img_path = src_dir / "test_shot.png"
        img.save(img_path)

        # 2. Create a dummy identity .cube LUT
        lut_tensor = torch.zeros(3, 9, 9, 9)
        cube_path = tmp_path / "dummy.cube"
        write_cube(cube_path, lut_tensor)

        # 3. Generate synthetic pairs
        gen = SyntheticPairGenerator(out_dir)
        manifest = gen.generate_sony_pairs([img_path], cube_path)

        assert len(manifest) == 1
        assert (out_dir / "manifest.json").is_file()
        assert Path(manifest[0]["input"]).is_file()
        assert Path(manifest[0]["target"]).is_file()


def test_synthetic_pair_generator_apple():
    """Validates synthetic Smart-HDR bracketed pair generator for Apple mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"

        img = Image.fromarray(np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8))
        img_path = src_dir / "test_hdr.png"
        img.save(img_path)

        gen = SyntheticPairGenerator(out_dir)
        manifest = gen.generate_apple_hdr_pairs([img_path])

        assert len(manifest) == 1
        assert Path(manifest[0]["target"]).is_file()


def test_paired_dataset_loader():
    """Validates PairedEnhancementDataset ingestion, cropping, and float32 types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        in_file = tmp_path / "in.png"
        tgt_file = tmp_path / "tgt.png"

        img_in = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img_tgt = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img_in.save(in_file)
        img_tgt.save(tgt_file)

        dataset = PairedEnhancementDataset(
            pairs=[(in_file, tgt_file)],
            patch_size=64,
            thumb_size=32,
            augment=True,
        )

        assert len(dataset) == 1
        item = dataset[0]

        assert "thumbnail" in item and "input" in item and "target" in item
        assert item["thumbnail"].shape == (3, 32, 32)
        assert item["input"].shape == (3, 64, 64)
        assert item["target"].shape == (3, 64, 64)
        assert item["input"].dtype == torch.float32
        assert (item["input"] >= 0.0).all() and (item["input"] <= 1.0).all()


@pytest.mark.parametrize("config_name", ["base_fivek.yaml", "sony_cinetone.yaml", "apple_smart_hdr.yaml"])
def test_yaml_configs_structure(config_name):
    """Validates all hyperparameter configuration files are valid and contain required fields."""
    config_path = Path("configs") / config_name
    assert config_path.is_file(), f"Missing config: {config_path}"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "stage" in cfg
    assert "model" in cfg
    assert "loss" in cfg
    assert "training" in cfg
    assert cfg["model"]["num_luts"] == 5
