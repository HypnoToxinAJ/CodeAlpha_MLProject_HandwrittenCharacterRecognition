import unittest

import numpy as np

from image_preprocessing import preprocess_image


class PreprocessImageTests(unittest.TestCase):
    def test_inverts_white_background_but_preserves_white_ink(self) -> None:
        white_ink = np.zeros((28, 28), dtype=np.uint8)
        white_ink[5:23, 12:16] = 255
        black_ink = 255 - white_ink

        white_ink_result = preprocess_image(white_ink)
        black_ink_result = preprocess_image(black_ink)

        np.testing.assert_allclose(white_ink_result, black_ink_result)
        self.assertEqual(white_ink_result.shape, (28, 28, 1))
        self.assertEqual(white_ink_result.dtype, np.float32)

    def test_blank_input_returns_zero_tensor(self) -> None:
        result = preprocess_image(np.zeros((40, 60), dtype=np.uint8))

        self.assertEqual(result.shape, (28, 28, 1))
        self.assertFalse(np.any(result))

    def test_preserves_digit_aspect_ratio(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        image[13:15, 3:25] = 255

        result = preprocess_image(image)[:, :, 0]
        foreground_y, foreground_x = np.where(result > 0.5)

        self.assertGreater(np.ptp(foreground_x), np.ptp(foreground_y) * 2)

    def test_trims_a_dark_digit_panel_from_a_light_margin(self) -> None:
        digit = np.zeros((28, 28), dtype=np.uint8)
        digit[5:23, 12:16] = 255
        framed_image = np.full((48, 48), 255, dtype=np.uint8)
        framed_image[10:38, 10:38] = digit

        result = preprocess_image(framed_image)
        expected = preprocess_image(digit)

        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
