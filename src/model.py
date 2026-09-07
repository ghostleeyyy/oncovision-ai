"""
CNN architecture for benign vs. malignant classification.

NOTE ON THIS FILE: this project was built in a sandbox without internet
access, so TensorFlow could not be installed to execute-verify this
specific file (unlike src/dataset.py, which is fully tested — see
tests/test_dataset.py). The architecture below uses only standard,
well-documented tf.keras layers and patterns. Run
`python -m src.train` locally with TensorFlow installed to train and
confirm results end-to-end.

Architecture: a compact CNN (3 conv blocks + dense head) — deep enough
to learn the class-distinguishing texture/density patterns in the
data, small enough to train in minutes on a single GPU or even CPU,
which matches the scale of an academic/portfolio project rather than
a production model with millions of images.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape: tuple[int, int, int] = (96, 96, 1)) -> tf.keras.Model:
    """Build and return a compiled binary-classification CNN.

    Three convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    act as the feature extractor, followed by a dense classification
    head with dropout for regularization. Output is a single sigmoid
    unit (benign=0, malignant=1).
    """
    model = models.Sequential(name="oncovision_cnn")
    model.add(layers.Input(shape=input_shape))

    # --- Feature extraction blocks ---
    # Layers are explicitly named so evaluate.py's Grad-CAM can reliably
    # target the last conv layer ("conv_block_3") regardless of Keras's
    # auto-naming, which isn't stable across model rebuilds in the same
    # session.
    model.add(layers.Conv2D(32, (3, 3), padding="same", name="conv_block_1"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding="same", name="conv_block_2"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding="same", name="conv_block_3"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Classification head ---
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid", name="malignancy_output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def get_augmentation_layer() -> tf.keras.Sequential:
    """Light data augmentation appropriate for microscopy-style images.

    Kept conservative — small rotations/flips only, since histopathology
    images have no single "upright" orientation but shouldn't be
    distorted in ways that would destroy the texture patterns the model
    relies on.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
