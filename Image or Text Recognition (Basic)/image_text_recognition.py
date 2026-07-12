"""Image or Text Recognition (Basic) - OCR pipeline.

DecodeLabs Internship Program | Batch: 2026

This module follows the IPO model:
- Input: read an image from disk
- Process: preprocess the image and extract text with pytesseract
- Output: print and persist the recognized text

The implementation is intentionally modular and production-oriented so it can be
used as a beginner-friendly reference project.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import pytesseract


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_PSM = 6
DEFAULT_OEM = 3
DEFAULT_CONF = "--oem 3 --psm 6"


@dataclass(slots=True)
class OCRResult:
    """Structured OCR output."""

    image_path: Path
    text: str
    output_path: Path


def configure_logging() -> None:
    """Configure console logging for human-friendly error reporting."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_image_path(image_path: Path) -> None:
    """Validate that the input image exists and is a supported format."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{image_path.suffix}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def load_image(image_path: Path) -> np.ndarray:
    """Load the image using OpenCV."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert the BGR image to grayscale."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image: np.ndarray) -> np.ndarray:
    """Reduce high-frequency noise before thresholding."""

    return cv2.GaussianBlur(gray_image, (5, 5), 0)


def apply_adaptive_threshold(blurred_image: np.ndarray) -> np.ndarray:
    """Create a high-contrast text mask for OCR."""

    return cv2.adaptiveThreshold(
        blurred_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def remove_noise(binary_image: np.ndarray) -> np.ndarray:
    """Apply a light morphological cleanup step to reduce speckle noise."""

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


def deskew_image(binary_image: np.ndarray) -> np.ndarray:
    """Deskew the image when text is rotated slightly off-axis.

    The function finds the minimum area rectangle around foreground pixels and
    rotates the image to align the text horizontally.
    """

    coords = np.column_stack(np.where(binary_image > 0))
    if coords.size == 0:
        return binary_image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (height, width) = binary_image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(image: np.ndarray, enable_deskew: bool = True) -> np.ndarray:
    """Run the full OCR preprocessing pipeline."""

    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale)
    thresholded = apply_adaptive_threshold(blurred)
    cleaned = remove_noise(thresholded)
    if enable_deskew:
        return deskew_image(cleaned)
    return cleaned


def read_text_with_tesseract(preprocessed_image: np.ndarray, psm: int = DEFAULT_PSM) -> str:
    """Extract text from the processed image using pytesseract."""

    if not isinstance(psm, int):
        raise TypeError("psm must be an integer.")

    config = f"--oem {DEFAULT_OEM} --psm {psm}"
    pil_image = Image.fromarray(preprocessed_image)
    return pytesseract.image_to_string(pil_image, config=config)


def save_text(output_text: str, output_dir: Path, source_image: Path) -> Path:
    """Persist OCR text output to a UTF-8 text file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "recognized_text.txt"
    output_path.write_text(output_text.strip() + "\n", encoding="utf-8")
    return output_path


def run_ocr(image_path: Path, output_dir: Path, psm: int = DEFAULT_PSM, enable_deskew: bool = True) -> OCRResult:
    """Execute the OCR workflow and return a structured result."""

    validate_image_path(image_path)
    image = load_image(image_path)
    preprocessed = preprocess_image(image, enable_deskew=enable_deskew)
    recognized_text = read_text_with_tesseract(preprocessed, psm=psm)
    output_path = save_text(recognized_text, output_dir, image_path)
    return OCRResult(image_path=image_path, text=recognized_text, output_path=output_path)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="OCR pipeline for the Image or Text Recognition (Basic) project."
    )
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where the recognized text file will be written.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=DEFAULT_PSM,
        help="Tesseract Page Segmentation Mode (e.g. 3, 6, or 11).",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable the optional deskewing step.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point for OCR recognition."""

    configure_logging()
    args = parse_args(argv or sys.argv[1:])

    try:
        result = run_ocr(
            image_path=Path(args.image_path),
            output_dir=Path(args.output_dir),
            psm=args.psm,
            enable_deskew=not args.no_deskew,
        )
    except pytesseract.TesseractNotFoundError:
        logging.error(
            "Tesseract is not installed or not on PATH. Install Tesseract OCR and retry."
        )
        return 1
    except (FileNotFoundError, ValueError, PermissionError, OSError, TypeError) as exc:
        logging.error(str(exc))
        return 1

    print("Recognized Text:")
    print("-" * 40)
    print(result.text.strip() or "[No readable text detected]")
    print("-" * 40)
    print(f"Saved text output to: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
