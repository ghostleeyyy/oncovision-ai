"""
Unit tests for src/dataset.py

These tests use a small temporary image set generated with the same
logic as data/generate_data.py, so they run in isolation without
depending on the full dataset being present on disk.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from PIL import Image

from src.dataset import (
    LABEL_MAP,
    class_distribution,
    load_image_dataset,
    normalize_images,
    stratified_split,
)


@pytest.fixture
def tiny_dataset_dir(tmp_path):
    """Create a minimal benign/malignant image folder structure for testing."""
    rng = np.random.default_rng(0)

    for label, n_images in [("benign", 6), ("malignant", 4)]:
        class_dir = tmp_path / label
        class_dir.mkdir()
        for i in range(n_images):
            arr = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"{label}_{i}.png")

    return str(tmp_path)


class TestLoadImageDataset:
    def test_loads_correct_number_of_images(self, tiny_dataset_dir):
        X, y, paths = load_image_dataset(tiny_dataset_dir, img_size=32)
        assert X.shape[0] == 10  # 6 benign + 4 malignant
        assert len(y) == 10
        assert len(paths) == 10

    def test_output_shape_has_channel_dim(self, tiny_dataset_dir):
        X, _, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        assert X.shape == (10, 32, 32, 1)

    def test_labels_match_label_map(self, tiny_dataset_dir):
        _, y, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        assert set(np.unique(y)) == {LABEL_MAP["benign"], LABEL_MAP["malignant"]}

    def test_pixel_values_normalized_to_unit_range(self, tiny_dataset_dir):
        X, _, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_missing_directory_raises_clear_error(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_image_dataset("this/path/does/not/exist", img_size=32)

    def test_resizes_images_to_requested_size(self, tiny_dataset_dir):
        X, _, _ = load_image_dataset(tiny_dataset_dir, img_size=64)
        assert X.shape[1:3] == (64, 64)


class TestNormalizeImages:
    def test_scales_0_255_to_0_1(self):
        arr = np.array([[0, 128, 255]], dtype=np.float32)
        normalized = normalize_images(arr)
        assert normalized.max() == pytest.approx(1.0)
        assert normalized.min() == pytest.approx(0.0)

    def test_leaves_already_normalized_data_unchanged(self):
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        normalized = normalize_images(arr)
        np.testing.assert_array_almost_equal(normalized, arr)


class TestStratifiedSplit:
    def test_split_sizes_sum_to_total(self, tiny_dataset_dir):
        X, y, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        splits = stratified_split(X, y, val_size=0.2, test_size=0.2, seed=42)
        total = (
            len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"])
        )
        assert total == len(y)

    def test_split_preserves_class_balance(self, tiny_dataset_dir):
        X, y, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        splits = stratified_split(X, y, val_size=0.2, test_size=0.2, seed=42)

        # Each split should contain both classes, not just one.
        for split_name in ["y_train", "y_val", "y_test"]:
            assert len(np.unique(splits[split_name])) == 2

    def test_split_is_reproducible_with_same_seed(self, tiny_dataset_dir):
        X, y, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        split_a = stratified_split(X, y, val_size=0.2, test_size=0.2, seed=7)
        split_b = stratified_split(X, y, val_size=0.2, test_size=0.2, seed=7)
        np.testing.assert_array_equal(split_a["y_train"], split_b["y_train"])


class TestClassDistribution:
    def test_counts_match_input(self, tiny_dataset_dir):
        _, y, _ = load_image_dataset(tiny_dataset_dir, img_size=32)
        dist = class_distribution(y)
        assert dist["benign"] == 6
        assert dist["malignant"] == 4
