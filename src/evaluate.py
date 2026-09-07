"""
Evaluate the trained OncoVision CNN and visualize what it's learned.

NOTE: requires TensorFlow — see the note in src/train.py. Not
execute-verified in the sandbox this project was built in.

Produces:
  1. Test-set classification report + confusion matrix
  2. ROC curve
  3. Grad-CAM overlays — visualizes which image regions the CNN relied
     on for its prediction, i.e. what "feature extraction" the
     convolutional layers actually learned. This is the evidence a
     hiring manager or interviewer will want to see behind the phrase
     "feature extraction techniques" on your resume.

Usage:
    python -m src.evaluate --data_dir data/images --model models/oncovision_cnn.keras
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)

from src.dataset import CLASS_NAMES, load_image_dataset, stratified_split


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image.

    Standard Grad-CAM (Selvaraju et al., 2017): take the gradient of the
    predicted class score with respect to the last conv layer's feature
    maps, global-average-pool those gradients into per-channel weights,
    and combine with the feature maps to get a coarse localization map
    of what the network attended to.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array[np.newaxis, ...])
        class_channel = predictions[:, 0]  # single sigmoid output

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam_examples(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    last_conv_layer_name: str = "conv_block_3",
    n_examples: int = 4,
    out_path: str = "outputs/gradcam_examples.png",
):
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=n_examples, replace=False)

    for col, idx in enumerate(indices):
        img = X[idx]
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], img.shape[:2]).numpy().squeeze()

        axes[0, col].imshow(img.squeeze(), cmap="gray")
        axes[0, col].set_title(f"True: {CLASS_NAMES[y[idx]]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(img.squeeze(), cmap="gray")
        axes[1, col].imshow(heatmap_resized, cmap="jet", alpha=0.45)
        axes[1, col].set_title("Grad-CAM")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Grad-CAM visualization to {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path: str = "outputs/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, ax=ax, cmap="Blues"
    )
    plt.title("Confusion Matrix — Benign vs. Malignant")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def plot_roc_curve(y_true, y_proba, out_path: str = "outputs/roc_curve.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the OncoVision CNN.")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--model", type=str, default="models/oncovision_cnn.keras")
    parser.add_argument("--img_size", type=int, default=96)
    args = parser.parse_args()

    X, y, _ = load_image_dataset(args.data_dir, img_size=args.img_size)
    splits = stratified_split(X, y, val_size=0.15, test_size=0.15, seed=42)
    X_test, y_test = splits["X_test"], splits["y_test"]

    model = tf.keras.models.load_model(args.model)

    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_gradcam_examples(model, X_test, y_test)


if __name__ == "__main__":
    main()
