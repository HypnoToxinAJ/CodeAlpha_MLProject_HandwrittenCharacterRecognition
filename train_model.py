import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

from image_preprocessing import preprocess_dataset


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.h5"
AUGMENTATION_ROTATION = 0.04
AUGMENTATION_TRANSLATION = 0.08
AUGMENTATION_ZOOM = 0.08


def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.RandomRotation(
                factor=AUGMENTATION_ROTATION,
                fill_mode="constant",
                fill_value=0.0,
            ),
            layers.RandomTranslation(
                height_factor=AUGMENTATION_TRANSLATION,
                width_factor=AUGMENTATION_TRANSLATION,
                fill_mode="constant",
                fill_value=0.0,
            ),
            layers.RandomZoom(
                height_factor=AUGMENTATION_ZOOM,
                width_factor=AUGMENTATION_ZOOM,
                fill_mode="constant",
                fill_value=0.0,
            ),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(42)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print("Preprocessing MNIST images...")
    x_train = preprocess_dataset(train_images)
    x_test = preprocess_dataset(test_images)

    model = build_model()
    history = model.fit(
        x_train,
        train_labels,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, test_labels, verbose=0)
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    confusion = tf.math.confusion_matrix(
        test_labels, predictions, num_classes=10
    ).numpy()

    model.save(MODEL_PATH)
    (PROJECT_DIR / "test_metrics.json").write_text(
        json.dumps(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "seed": 42,
                "test_samples": int(len(test_labels)),
                "augmentation": {
                    "rotation_fraction": AUGMENTATION_ROTATION,
                    "translation_fraction": AUGMENTATION_TRANSLATION,
                    "zoom_fraction": AUGMENTATION_ZOOM,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savetxt(
        PROJECT_DIR / "confusion_matrix.csv",
        confusion,
        fmt="%d",
        delimiter=",",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(PROJECT_DIR / "training_history.png", dpi=150)
    plt.close(figure)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
