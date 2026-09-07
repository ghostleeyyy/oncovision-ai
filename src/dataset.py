"""
Dataset loading and preprocessing for the OncoVision AI pipeline.

Deliberately kept free of any deep-learning framework import — this
module only depends on numpy and PIL, so its logic (loading, resizing,
normalizing, splitting, label encoding) is fully unit-testable without
installing TensorFlow, and works the same whether the model ends up
built in TensorFlow, PyTorch, or anything else.

To swap in a real dataset (e.g. BreakHis, IDC histopathology), point
`load_image_dataset` at a directory with the same structure:

    <root>/benign/*.png
    <root>/malignant/*.png

No other code needs to change.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_MAP = {"benign": 0, "malignant": 1}
CLASS_NAMES = ["benign", "malignant"]


def load_image_dataset(root_dir: str, img_size: int = 96) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all images under root_dir/<class_name>/*.png into arrays.

    Returns:
        X: float32 array of shape (N, img_size, img_size, 1), values in [0, 1]
        y: int array of shape (N,), 0 = benign, 1 = malignant
        paths: list of source file paths, same order as X/y (useful for
               tracing a misclassified sample back to its source image)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory '{root_dir}' not found. "
            f"Run `python data/generate_data.py` first, or point this at your own dataset."
        )

    images, labels, paths = [], [], []

    for class_name, label in LABEL_MAP.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class subdirectory '{class_dir}' not found.")

        file_paths = sorted(class_dir.glob("*.png"))
        if not file_paths:
            raise ValueError(f"No .png images found in '{class_dir}'.")

        for fp in file_paths:
            img = Image.open(fp).convert("L").resize((img_size, img_size))
            images.append(np.array(img, dtype=np.float32))
            labels.append(label)
            paths.append(str(fp))

    X = np.stack(images)[..., np.newaxis]  # add channel dim: (N, H, W, 1)
    X = normalize_images(X)
    y = np.array(labels, dtype=np.int64)

    return X, y, paths


def normalize_images(X: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1]."""
    if X.max() > 1.0:
        return X.astype(np.float32) / 255.0
    return X.astype(np.float32)


def stratified_split(
    X: np.ndarray, y: np.ndarray, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> dict:
    """Split into train/val/test, preserving class balance in every split.

    Done manually (rather than calling sklearn twice) so the exact same
    function works identically whether the caller is using this for a
    TensorFlow, PyTorch, or sklearn model downstream.
    """
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []

    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        rng.shuffle(class_indices)

        n = len(class_indices)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))

        test_idx.extend(class_indices[:n_test])
        val_idx.extend(class_indices[n_test : n_test + n_val])
        train_idx.extend(class_indices[n_test + n_val :])

    train_idx, val_idx, test_idx = (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
    )

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
    }


def class_distribution(y: np.ndarray) -> dict:
    """Return a {class_name: count} summary — useful for sanity-checking splits."""
    values, counts = np.unique(y, return_counts=True)
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    return {inv_map[int(v)]: int(c) for v, c in zip(values, counts)}
