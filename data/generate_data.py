"""
Synthetic histopathology-style image generator for breast cancer
classification (benign vs. malignant).

Real histopathology/mammography datasets (e.g. BreakHis, the IDC
histopathology dataset on Kaggle) are large, licensed for research use,
and not downloadable in an offline environment. This generator produces
structurally realistic synthetic images instead, so the full pipeline
here is reproducible by anyone who clones the repo with zero external
downloads.

The generated images encode class-dependent structure that mirrors real
diagnostic patterns used in practice, so the classification task is
genuinely learnable rather than random:

  - Benign:    fewer, more uniform, well-separated "nuclei" blobs
  - Malignant: denser, irregularly shaped, overlapping blobs with more
               size variance (mirrors nuclear pleomorphism, a real
               feature pathologists look for)

To train on a real dataset instead, replace this generator with a
loader pointed at your dataset's image directory — `dataset.py` expects
only a folder-per-class structure (see `load_image_dataset`), so no
other code needs to change.

Usage:
    python data/generate_data.py --n_per_class 500 --img_size 96 --seed 42
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _draw_nuclei(
    img_size: int,
    n_blobs: int,
    size_range: tuple[int, int],
    irregularity: float,
    rng: np.random.Generator,
) -> Image.Image:
    """Draw a single synthetic tissue-sample image with blob-like nuclei."""
    img = Image.new("L", (img_size, img_size), color=int(rng.uniform(150, 190)))
    draw = ImageDraw.Draw(img)

    for _ in range(n_blobs):
        cx = rng.uniform(0, img_size)
        cy = rng.uniform(0, img_size)
        r = rng.uniform(*size_range)

        # Irregularity: jitter the bounding ellipse to avoid perfectly
        # round blobs for the malignant class (mirrors irregular nuclear
        # shape, a real pathology feature).
        jitter = rng.uniform(-irregularity, irregularity, size=4) * r
        x0, y0 = cx - r + jitter[0], cy - r + jitter[1]
        x1, y1 = cx + r + jitter[2], cy + r + jitter[3]

        shade = int(rng.uniform(30, 90))
        draw.ellipse([x0, y0, x1, y1], fill=shade)

    # Soft blur to mimic microscopy image texture instead of hard vector edges.
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Add sensor-style noise.
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, 8, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_class_images(
    label: str,
    n_images: int,
    img_size: int,
    rng: np.random.Generator,
) -> list[Image.Image]:
    if label == "benign":
        n_blobs_range = (8, 14)
        size_range = (4, 8)
        irregularity = 0.15
    elif label == "malignant":
        n_blobs_range = (18, 30)
        size_range = (3, 11)
        irregularity = 0.55
    else:
        raise ValueError(f"Unknown label: {label}")

    images = []
    for _ in range(n_images):
        n_blobs = int(rng.integers(*n_blobs_range))
        images.append(_draw_nuclei(img_size, n_blobs, size_range, irregularity, rng))
    return images


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic histopathology image dataset.")
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/images")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    for label in ["benign", "malignant"]:
        label_dir = os.path.join(args.out_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        images = generate_class_images(label, args.n_per_class, args.img_size, rng)
        for i, img in enumerate(images):
            img.save(os.path.join(label_dir, f"{label}_{i:04d}.png"))

        print(f"Saved {len(images)} images to {label_dir}")

    print(f"\nDone. Total images: {2 * args.n_per_class}")


if __name__ == "__main__":
    main()
