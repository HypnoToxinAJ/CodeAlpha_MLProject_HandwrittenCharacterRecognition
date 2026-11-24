import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from image_preprocessing import preprocess_image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate labeled handwritten digits in folders named 0 through 9."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "custom_samples",
        help="Folder containing one subfolder per digit (default: custom_samples).",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Sample folder not found: {data_dir}. See README.md for its layout."
        )

    model_path = PROJECT_DIR / "model.h5"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )

    image_tensors = []
    labels = []
    image_paths = []
    for digit in range(10):
        digit_dir = data_dir / str(digit)
        if not digit_dir.is_dir():
            continue
        for image_path in sorted(digit_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                with Image.open(image_path) as source:
                    image = np.asarray(source.convert("L"))
                image_tensors.append(preprocess_image(image))
                labels.append(digit)
                image_paths.append(image_path)

    if not image_tensors:
        raise ValueError(
            f"No labeled images found under {data_dir}. Add images to folders 0-9."
        )

    probabilities = model_predictions(model_path, np.stack(image_tensors))
    predictions = np.argmax(probabilities, axis=1)
    scores = np.max(probabilities, axis=1)
    label_array = np.asarray(labels)
    correct = predictions == label_array

    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (label_array, predictions), 1)
    failures_path = PROJECT_DIR / "custom_test_failures.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image", "actual", "predicted", "model_score"])
        for index in np.flatnonzero(~correct):
            writer.writerow(
                [
                    image_paths[index],
                    labels[index],
                    int(predictions[index]),
                    f"{scores[index]:.6f}",
                ]
            )

    print(f"Custom samples: {len(labels)}")
    print(f"Accuracy: {np.mean(correct):.4f}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion)
    print(f"Saved misclassified examples to {failures_path}")


def model_predictions(model_path: Path, images: np.ndarray) -> np.ndarray:
    model = load_model(model_path)
    return model.predict(images, batch_size=128, verbose=0)


if __name__ == "__main__":
    main()
