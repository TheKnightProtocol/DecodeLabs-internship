"""Image or Text Recognition (Basic) - object detection pipeline.

DecodeLabs Internship Program | Batch: 2026

This module demonstrates transfer learning with the pre-trained MobileNet SSD
model. The model is already trained on a large dataset, so the project can focus
on inference, confidence filtering, and visual output instead of model training.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CONFIDENCE_THRESHOLD = 0.80
MODEL_DIR = Path(__file__).with_name("models")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("output")
PROTOTXT_PATH = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

MODEL_URLS = {
    "prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
}


@dataclass(slots=True)
class DetectionResult:
    """Structured object detection output."""

    image_path: Path
    output_path: Path
    detections: list[dict[str, object]]


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_image_path(image_path: Path) -> None:
    """Validate supported image input."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{image_path.suffix}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def download_model_if_needed() -> None:
    """Download the pre-trained MobileNet SSD model if it is missing.

    Transfer learning is used here because the network has already learned
    general object features from a much larger dataset. That reduces training
    time, compute requirements, and complexity for this basic project.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not PROTOTXT_PATH.exists():
        logging.info("Downloading MobileNet SSD prototxt...")
        urlretrieve(MODEL_URLS["prototxt"], PROTOTXT_PATH)

    if not CAFFEMODEL_PATH.exists():
        logging.info("Downloading MobileNet SSD caffemodel...")
        urlretrieve(MODEL_URLS["caffemodel"], CAFFEMODEL_PATH)


def load_image(image_path: Path) -> np.ndarray:
    """Load the image using OpenCV."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")
    return image


def detect_objects(
    image_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> DetectionResult:
    """Run MobileNet SSD object detection and save the annotated image."""

    validate_image_path(image_path)
    download_model_if_needed()

    image = load_image(image_path)
    (height, width) = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))

    # Create the blob to normalize and resize the image for the network.
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5,
    )
    net.setInput(blob)
    detections_tensor = net.forward()

    detections: list[dict[str, object]] = []

    for index in range(detections_tensor.shape[2]):
        confidence = float(detections_tensor[0, 0, index, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections_tensor[0, 0, index, 1])
        box = detections_tensor[0, 0, index, 3:7] * np.array([width, height, width, height])
        (start_x, start_y, end_x, end_y) = box.astype(int)

        label = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"class_{class_id}"
        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "box": (start_x, start_y, end_x, end_y),
            }
        )

        text = f"{label}: {confidence * 100:.1f}%"
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        y_position = start_y - 15 if start_y - 15 > 15 else start_y + 15
        cv2.putText(
            image,
            text,
            (start_x, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detected_image.jpg"
    cv2.imwrite(str(output_path), image)

    return DetectionResult(image_path=image_path, output_path=output_path, detections=detections)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Object detection pipeline for the Image or Text Recognition (Basic) project."
    )
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where the annotated image will be written.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold for accepted detections.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point for object detection."""

    configure_logging()
    args = parse_args(argv or sys.argv[1:])

    try:
        result = detect_objects(
            image_path=Path(args.image_path),
            output_dir=Path(args.output_dir),
            confidence_threshold=args.confidence,
        )
    except (FileNotFoundError, ValueError, PermissionError, OSError) as exc:
        logging.error(str(exc))
        return 1

    print("Detected Objects:")
    print("-" * 40)
    if not result.detections:
        print("No detections met the confidence threshold.")
    for detection in result.detections:
        print(
            f"{detection['label']} | confidence={detection['confidence'] * 100:.1f}% | box={detection['box']}"
        )
    print("-" * 40)
    print(f"Saved annotated image to: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
