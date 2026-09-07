"""
Train the OncoVision CNN on the benign/malignant image dataset.

NOTE: requires TensorFlow — could not be executed in the sandbox this
project was built in (no internet access to install it). Run locally
with `pip install -r requirements.txt` to train and reproduce results.

Usage:
    python -m src.train --data_dir data/images --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from src.dataset import load_image_dataset, stratified_split, class_distribution
from src.model import build_cnn, get_augmentation_layer

try:
    import mlflow
    import mlflow.tensorflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def build_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, augment: bool, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.batch(batch_size)

    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description="Train the OncoVision breast cancer classification CNN.")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model_out", type=str, default="models/oncovision_cnn.keras")
    parser.add_argument("--metrics_out", type=str, default="outputs/metrics.json")
    parser.add_argument("--use_mlflow", action="store_true")
    args = parser.parse_args()

    print(f"Loading images from {args.data_dir} ...")
    X, y, _ = load_image_dataset(args.data_dir, img_size=args.img_size)
    print(f"Loaded {len(X)} images. Class distribution: {class_distribution(y)}")

    splits = stratified_split(X, y, val_size=0.15, test_size=0.15, seed=42)
    print(
        f"Train: {len(splits['y_train'])} | "
        f"Val: {len(splits['y_val'])} | "
        f"Test: {len(splits['y_test'])}"
    )

    train_ds = build_tf_dataset(splits["X_train"], splits["y_train"], args.batch_size, augment=True, shuffle=True)
    val_ds = build_tf_dataset(splits["X_val"], splits["y_val"], args.batch_size, augment=False, shuffle=False)

    model = build_cnn(input_shape=(args.img_size, args.img_size, 1))
    model.summary()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_out, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    run_mlflow = args.use_mlflow and MLFLOW_AVAILABLE
    if args.use_mlflow and not MLFLOW_AVAILABLE:
        print("[warn] --use_mlflow was set but mlflow is not installed. Continuing without it.")

    if run_mlflow:
        mlflow.tensorflow.autolog()
        mlflow.start_run(run_name="oncovision_cnn")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if run_mlflow:
        mlflow.end_run()

    # Persist final training history for reference alongside the model.
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({"history": history_dict}, f, indent=2)

    print(f"\nSaved best model to {args.model_out}")
    print(f"Saved training history to {args.metrics_out}")


if __name__ == "__main__":
    main()
