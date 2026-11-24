from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from image_preprocessing import preprocess_image


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.h5"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )

    image_path = Path(input("Enter image path: ").strip()).expanduser()
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("L"))
    prepared = preprocess_image(image)[np.newaxis, ...]

    model = load_model(MODEL_PATH)
    probabilities = model.predict(prepared, verbose=0)[0]
    top_digits = np.argsort(probabilities)[-3:][::-1]

    print(f"Predicted digit: {int(top_digits[0])}")
    print("Top predictions (model scores; not calibrated probabilities):")
    for digit in top_digits:
        print(f"  {int(digit)}: {probabilities[digit] * 100:.2f}%")


if __name__ == "__main__":
    main()
