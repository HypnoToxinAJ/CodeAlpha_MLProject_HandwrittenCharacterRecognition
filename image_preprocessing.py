from collections.abc import Sequence

import numpy as np
from PIL import Image


IMAGE_SIZE = 28
DIGIT_SIZE = 20


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert a digit image to a normalized, centered MNIST-shaped tensor."""
    pixels = np.asarray(image)
    if pixels.size == 0:
        raise ValueError("The input image is empty.")

    if pixels.ndim == 3:
        if pixels.shape[2] == 1:
            pixels = pixels[:, :, 0]
        else:
            pixels = np.asarray(Image.fromarray(pixels).convert("L"))
    if pixels.ndim != 2:
        raise ValueError("Expected a grayscale or RGB image.")

    if np.issubdtype(pixels.dtype, np.floating):
        finite_pixels = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)
        if finite_pixels.size and finite_pixels.max() <= 1.0:
            finite_pixels = finite_pixels * 255.0
        pixels = finite_pixels
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )
    if np.median(border) > 127:
        dark_y, dark_x = np.where(pixels < 127)
        if dark_x.size:
            left = int(dark_x.min())
            right = int(dark_x.max()) + 1
            top = int(dark_y.min())
            bottom = int(dark_y.max()) + 1
            dark_region = pixels[top:bottom, left:right]
            is_framed_dark_region = (
                dark_region.shape[0] >= pixels.shape[0] * 0.3
                and dark_region.shape[1] >= pixels.shape[1] * 0.3
                and np.mean(dark_region < 127) > 0.5
            )
            if is_framed_dark_region:
                pixels = dark_region
            else:
                pixels = 255 - pixels
        else:
            pixels = 255 - pixels

    threshold = max(10, int(pixels.max()) // 10)
    foreground_y, foreground_x = np.where(pixels > threshold)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if foreground_x.size == 0:
        return canvas.astype(np.float32)[..., np.newaxis] / 255.0

    left = int(foreground_x.min())
    right = int(foreground_x.max()) + 1
    top = int(foreground_y.min())
    bottom = int(foreground_y.max()) + 1
    cropped = pixels[top:bottom, left:right]

    side = max(cropped.shape)
    margin = max(1, int(round(side * 0.2)))
    side += margin * 2
    square = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - cropped.shape[0]) // 2
    offset_x = (side - cropped.shape[1]) // 2
    square[
        offset_y : offset_y + cropped.shape[0],
        offset_x : offset_x + cropped.shape[1],
    ] = cropped

    resized = Image.fromarray(square).resize(
        (DIGIT_SIZE, DIGIT_SIZE),
        resample=Image.Resampling.LANCZOS,
    )
    canvas[4:24, 4:24] = np.asarray(resized)
    return canvas.astype(np.float32)[..., np.newaxis] / 255.0


def preprocess_dataset(images: Sequence[np.ndarray]) -> np.ndarray:
    """Apply the same image preparation used at prediction time to a dataset."""
    processed = np.empty((len(images), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    for index, image in enumerate(images):
        processed[index] = preprocess_image(image)
    return processed
