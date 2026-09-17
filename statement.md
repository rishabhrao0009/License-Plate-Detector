# Project Statement

## Title
License Plate Localizer & OCR — A Classical Computer Vision Pipeline

## Problem Statement
Automatic License Plate Recognition (ALPR) is a common real-world computer
vision task with applications in traffic monitoring, parking systems, toll
collection, and law enforcement. Most modern ALPR systems rely on deep
learning models that require large labeled datasets, GPU resources, and
significant training time. This project addresses the problem of localizing
and reading a vehicle's license plate from a static image using only
classical image-processing techniques — without any model training, GPU, or
graphical interface — making the solution lightweight, interpretable, and
easy to reproduce.

## Objective
To design and implement a CLI-based pipeline that:
1. Detects the region of an image most likely to contain a license plate.
2. Extracts and reads the plate's text using OCR.
3. Reports results for a batch of images in a structured, reviewable format.

## Scope
- Works on static images only (no video stream or live camera feed).
- No GUI — driven entirely from the command line.
- No deep learning framework or pretrained detection model is used.
- No GPU or model training required.
- Processes a single image or an entire folder of images in one run.

## Methodology
The pipeline is organized into three functional stages:

1. **Preprocessing** — Convert the image to grayscale, apply a bilateral
   filter to reduce noise while preserving edges, and run Canny edge
   detection to produce a binary edge map.
2. **Plate Localization** — Detect contours in the edge map and filter them
   by aspect ratio (approximately 2:1 to 5:1, typical of license plates) and
   area, then rank and select the most plate-like candidate region(s).
3. **OCR & Reporting** — Crop the selected region, binarize it using Otsu's
   thresholding for better contrast, and run Tesseract OCR (restricted to an
   alphanumeric character set) to extract the plate text. Results are
   written to a bounding-box overlay image per input image and consolidated
   into a CSV/JSON report for the full batch.

## Tools & Technologies
- **Language:** Python 3
- **Libraries:** OpenCV, NumPy, pytesseract
- **OCR Engine:** Tesseract OCR
- No deep learning frameworks (e.g. TensorFlow, PyTorch) are used.

## Expected Outcome
Given a folder of vehicle images, the system outputs:
- An annotated image per input, showing the detected plate region and
  recognized text.
- A consolidated report (CSV and JSON) summarizing detection status,
  bounding box coordinates, aspect ratio, recognized text, and notes for
  every processed image.

## Limitations
- Aspect-ratio-based filtering assumes a roughly front-facing, axis-aligned
  plate; angled or tilted plates may be missed or misdetected.
- Canny edge detection can fail on low-contrast, dirty, or poorly lit
  plates, since it depends on a clear intensity gradient at the plate
  border.
- OCR accuracy is inherently limited without a trained recognition model;
  non-standard fonts, motion blur, and partial occlusion reduce read
  accuracy. This is treated as an expected, documented constraint rather
  than a defect, and each result is annotated with a confidence note
  reflecting this.
