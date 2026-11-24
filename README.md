# Handwritten Digit Recognition

A Python experiment that trains a convolutional neural network (CNN) to
classify grayscale handwritten digits from 0 to 9. It uses MNIST for training
and benchmark evaluation, supports prediction on individual images, and
includes a separate evaluator for labeled samples of your own handwriting.

## Features

- Downloads MNIST through Keras on the first training run.
- Trains a CNN with augmentation, a validation split, and early stopping.
- Uses the same image-preparation function for training, prediction, and custom
  evaluation.
- Predicts one image and displays the three highest model scores.
- Reports held-out test accuracy/loss and a confusion matrix.
- Evaluates user-provided, labeled images and records misclassifications.
- Includes focused tests for image polarity, blank inputs, aspect ratio, and
  images with margins.

This project recognizes **single digits**, not letters, words, or sentences.

## Experiment process

1. **Dataset:** Keras MNIST provides 60,000 training images and 10,000 test
   images. Images are grayscale, 28 x 28 pixels, and labeled 0-9.
2. **Preprocessing:** [image_preprocessing.py](./image_preprocessing.py)
   detects whether the digit/background colors need inversion, crops around the
   digit, pads it to a square without stretching, resizes it to fit a 20 x 20
   area, centers it on a 28 x 28 canvas, and scales pixels to 0-1.
3. **Training and validation:** The training set uses a 10% validation split.
   The CNN is trained for up to 15 epochs with batch size 128, Adam,
   sparse-categorical cross-entropy, and early stopping on validation loss.
   Seed 42 is used for reproducibility.
4. **Augmentation:** Random rotations (factor 0.04), translations (0.08), and
   zooms (0.08) are applied during training only. Images are never flipped.
   Validation and test images are not augmented.
5. **Final evaluation:** The separate 10,000-image MNIST test set is evaluated
   after training. Its results are reported in `test_metrics.json` and
   `confusion_matrix.csv`.
6. **Custom evaluation:** Optional, separately labeled personal images can be
   evaluated with `evaluate_custom.py`. These samples should not be used for
   training if they are also being used to measure generalization.

## Results

The latest recorded run used Python 3.13.14 and TensorFlow 2.21.0 and achieved:

| Evaluation | Samples | Accuracy | Loss |
| --- | ---: | ---: | ---: |
| MNIST held-out test set | 10,000 | 99.03% | 0.0327 |

The saved confusion matrix shows which actual digits were mistaken for other
digits. The training/validation curves are in `training_history.png`. A
smoke-test prediction on the included `digit.png` was **7**.

**Interpretation:** 99.03% is performance on the MNIST test set, not a
guarantee of the same accuracy on phone photos or new handwriting. No custom
handwriting dataset was present for this run, so real-world generalization
still needs to be measured with your own labeled examples. A no-augmentation
control run was not recorded, so this result does not isolate the effect of
augmentation. Training was run on CPU in the current native Windows
environment.

## Setup

Python 3.13 was used for the recorded run. From this project directory, create
and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, run commands directly with the virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

## Train the model

```powershell
python train_model.py
```

MNIST is downloaded automatically if it is not already cached. Training
replaces `model.h5` and generates:

- `model.h5` - trained Keras model.
- `test_metrics.json` - test loss, accuracy, sample count, seed, and
  augmentation settings.
- `confusion_matrix.csv` - 10 x 10 matrix (rows are actual digits; columns are
  predicted digits).
- `training_history.png` - training and validation accuracy/loss plots.

## Predict a single image

```powershell
python predict.py
```

Enter an image path when prompted, for example `digit.png`. The program
grayscales and preprocesses the image, then prints the predicted digit and top
three model scores. Scores are not calibrated probabilities.

## Evaluate your own handwriting

Create a `custom_samples` folder with one subfolder for each digit. Each image's
folder name is its correct label:

```text
custom_samples/
  0/
    sample1.png
    sample2.jpg
  1/
    sample1.png
  ...
  9/
    sample1.png
```

Use varied samples where possible: different writers, pen thicknesses, image
sizes, and backgrounds. Then run:

```powershell
python evaluate_custom.py
```

To select a different folder:

```powershell
python evaluate_custom.py --data-dir path\to\labeled_samples
```

The evaluator reports the number of images, accuracy, and confusion matrix. It
writes `custom_test_failures.csv` with each mistaken image's path, actual
digit, predicted digit, and model score. Supported formats include PNG, JPEG,
BMP, and TIFF.

## Tests

Run the preprocessing unit tests with:

```powershell
python -m unittest test_image_preprocessing.py
```

## Project files

- `train_model.py` - builds, trains, and evaluates the CNN.
- `image_preprocessing.py` - shared image preprocessing used by training and
  inference.
- `predict.py` - predicts one image.
- `evaluate_custom.py` - evaluates labeled custom images and records mistakes.
- `test_image_preprocessing.py` - preprocessing unit tests.
- `save_test_image.py` - saves the first MNIST test image as `digit.png`.
- `requirements.txt` - Python dependencies.
