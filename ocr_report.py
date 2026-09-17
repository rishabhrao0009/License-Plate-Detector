"""
ocr_report.py
-------------
Module 3: OCR / Reporting

Given the original image and the localized plate candidate(s):
  1. Crop the best candidate region.
  2. Binarize it (Otsu threshold) to give Tesseract a clean,
     high-contrast input.
  3. Run pytesseract OCR restricted to a plate-like character
     whitelist.
  4. Draw the bounding box + recognized text on the full image
     (overlay) for visual QA.
  5. Provide helpers to log per-image results to CSV/JSON so a
     batch run produces a flat-file report.

OCR is treated as best-effort: classical localization + Tesseract
(no deep learning) will not match commercial ANPR accuracy, and
that limitation is documented in the README rather than hidden.
"""

import json
import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

from localize import PlateCandidate


# Plates are (almost always) uppercase letters + digits + occasional
# hyphen/space -- restricting Tesseract's charset measurably improves
# accuracy versus letting it guess from its full dictionary.
TESSERACT_CONFIG = (
    "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


@dataclass
class OCRResult:
    image_name: str
    plate_found: bool
    bbox: Optional[tuple]
    aspect_ratio: Optional[float]
    recognized_text: str
    confidence_note: str


def crop_candidate(image: np.ndarray, candidate: PlateCandidate,
                    pad_frac: float = 0.05) -> np.ndarray:
    """Crop the candidate bbox out of the full image, with small padding."""
    h_img, w_img = image.shape[:2]
    pad_x = int(candidate.w * pad_frac)
    pad_y = int(candidate.h * pad_frac)

    x1 = max(0, candidate.x - pad_x)
    y1 = max(0, candidate.y - pad_y)
    x2 = min(w_img, candidate.x + candidate.w + pad_x)
    y2 = min(h_img, candidate.y + candidate.h + pad_y)

    return image[y1:y2, x1:x2]


def binarize_plate(crop: np.ndarray) -> np.ndarray:
    """
    Convert a cropped plate region to a clean black/white image using
    Otsu's method -- gives Tesseract a high-contrast, noise-reduced
    input rather than raw grayscale.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 15, 15)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def ocr_plate(binary_crop: np.ndarray) -> str:
    """Run Tesseract OCR on a binarized plate crop and clean the result."""
    try:
        raw = pytesseract.image_to_string(binary_crop, config=TESSERACT_CONFIG)
    except pytesseract.TesseractError:
        return ""
    # strip whitespace/newlines Tesseract tends to add
    return "".join(raw.split())


def process_image(
    image: np.ndarray,
    image_name: str,
    candidates: List[PlateCandidate],
) -> OCRResult:
    """
    Run OCR on the best candidate (if any) and return a structured
    result ready to log/report.
    """
    if not candidates:
        return OCRResult(
            image_name=image_name,
            plate_found=False,
            bbox=None,
            aspect_ratio=None,
            recognized_text="",
            confidence_note="no plate-shaped region found",
        )

    best = candidates[0]
    crop = crop_candidate(image, best)
    binary = binarize_plate(crop)
    text = ocr_plate(binary)

    note = "ok" if text else "region found but OCR returned no text " \
                              "(likely low contrast / non-standard font)"

    return OCRResult(
        image_name=image_name,
        plate_found=True,
        bbox=best.bbox,
        aspect_ratio=round(best.aspect_ratio, 2),
        recognized_text=text,
        confidence_note=note,
    )


def draw_result_overlay(
    image: np.ndarray, result: OCRResult
) -> np.ndarray:
    """Draw the bbox + recognized text on a copy of the full image."""
    out = image.copy()
    if result.bbox:
        x, y, w, h = result.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = result.recognized_text if result.recognized_text else "?"
        cv2.putText(
            out, label, (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )
    return out


def write_csv_report(results: List[OCRResult], out_path: str) -> None:
    fieldnames = ["image_name", "plate_found", "bbox", "aspect_ratio",
                  "recognized_text", "confidence_note"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def write_json_report(results: List[OCRResult], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image
    from localize import localize_plates

    if len(sys.argv) != 2:
        print("Usage: python ocr_report.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    name = os.path.basename(sys.argv[1])
    stages = preprocess_image(img)
    cands = localize_plates(stages["edges"], img.shape)
    result = process_image(img, name, cands)
    print(result)

    overlay = draw_result_overlay(img, result)
    cv2.imwrite("/tmp/ocr_overlay.png", overlay)
    print("Saved overlay -> /tmp/ocr_overlay.png")
