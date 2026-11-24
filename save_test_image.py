from tensorflow.keras.datasets import mnist
from pathlib import Path
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_path = Path(__file__).resolve().with_name("digit.png")
Image.fromarray(x_test[0]).save(image_path)
print(f"Image saved as {image_path}")
print("Actual digit:", y_test[0])