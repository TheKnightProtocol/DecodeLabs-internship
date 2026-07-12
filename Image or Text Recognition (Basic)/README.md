# Image or Text Recognition (Basic)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Pytesseract](https://img.shields.io/badge/Pytesseract-OCR-orange.svg)](https://github.com/madmaze/pytesseract)
[![MobileNet SSD](https://img.shields.io/badge/MobileNet%20SSD-Transfer%20Learning-red.svg)](https://github.com/chuanqi305/MobileNet-SSD)

## Project Overview

This project demonstrates how artificial intelligence can interpret unstructured visual data. It provides two practical recognition workflows:

- Optical Character Recognition for extracting text from images
- Object detection using a pre-trained MobileNet SSD model

The project follows the IPO model and is designed to be modular, beginner-friendly, and production-oriented.

## Features

- OCR pipeline with image preprocessing
- Gaussian blur, adaptive thresholding, and optional deskewing
- Tesseract integration with configurable PSM modes
- Object detection with MobileNet SSD
- Confidence filtering at 80% or higher
- Automatic output file creation
- Clear error handling for missing files, invalid images, and missing dependencies

## Technologies

- Python 3.11+
- OpenCV
- pytesseract
- Pillow
- NumPy
- Matplotlib
- MobileNet SSD for transfer-learning-based object detection

## Project Structure

```text
Image or Text Recognition (Basic)/
├── images/
│   ├── sample1.jpg
│   └── sample2.png
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── output/
│   ├── detected_image.jpg
│   └── recognized_text.txt
├── image_text_recognition.py
├── object_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

If Tesseract OCR is not installed on your system, install it separately and ensure the executable is available on your PATH.

## How to Run

### OCR Mode

Run OCR on an image stored locally:

```bash
python image_text_recognition.py images/sample1.jpg --psm 6
```

The script will preprocess the image, extract text, print the result, and save the text to `output/recognized_text.txt`.

### Object Detection Mode

Run object detection on an image stored locally:

```bash
python object_detection.py images/sample2.png
```

The script will download the MobileNet SSD model files automatically if they are missing, detect objects with confidence values, and save the annotated image to `output/`.

## IPO Model

### Input

- Accepts JPG, JPEG, and PNG files from the local `images/` folder.

### Process

#### OCR

1. Load the image with OpenCV.
2. Convert the image to grayscale.
3. Apply Gaussian blur.
4. Apply adaptive thresholding.
5. Optionally deskew the text.
6. Extract text using `pytesseract.image_to_string()`.

#### Object Detection

1. Load the MobileNet SSD model.
2. Build a blob using `cv2.dnn.blobFromImage()`.
3. Run the forward pass.
4. Extract detections.
5. Apply the confidence filter.
6. Draw bounding boxes and labels.

### Output

- OCR produces machine-readable text in `output/recognized_text.txt`.
- Object detection produces an annotated image in `output/`.

## Screenshots Placeholder

Add your final screenshots here after running the project locally:

- OCR input image
- OCR extracted text output
- Object detection annotated image

## Testing Suggestions

- Clean printed text
- Blurred text
- Rotated text
- Book page
- Invoice
- Street sign
- Object images

## Future Improvements

- Add a Streamlit or Flask interface
- Support batch processing for multiple images
- Add layout analysis for complex documents
- Add more object detection classes or model options
- Add OCR language selection for multilingual inputs

## License

Refer to the repository license file for usage terms.

## Author

DecodeLabs Internship Program | Batch 2026
