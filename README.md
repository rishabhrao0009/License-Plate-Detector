# License Plate Localizer

A classical (no deep learning, no GUI) computer vision pipeline that
localizes license plates in static images and reads them with OCR.
Runs entirely on CPU, batch-processes a folder of images, and writes
a flat-file (CSV/JSON) report plus a bounding-box overlay per image.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1. Preprocessing | `preprocess.py` | Grayscale conversion → bilateral filter (denoise while preserving edges) → Canny edge detection |
| 2. Plate Localization | `localize.py` | Contour detection on the edge map → filter by aspect ratio (2.0–5.5, typical plate width:height) and area → greedy NMS → rank and return best candidate(s) |
| 3. OCR / Reporting | `ocr_report.py` | Crop the best region → upscale + Otsu binarization → Tesseract OCR (restricted A–Z0–9 whitelist) → draw overlay → log to CSV/JSON |

`main.py` wires the three stages together into a CLI batch driver.

## Requirements

- Python 3.9+
- OpenCV, NumPy, pytesseract (see `requirements.txt`)
- The **Tesseract OCR binary** installed separately (not a pip package — see `requirements.txt` for install commands)

```bash
pip install -r requirements.txt
```

## Usage

Run on a folder of images:

```bash
python main.py --input sample_images/ --output output/
```

Run on a single image:

```bash
python main.py --input path/to/car.jpg --output output/
```

Add `--debug` to also save the raw Canny edge map and *all* surviving
candidate boxes (not just the best one) per image — useful when tuning
the aspect-ratio/area thresholds against your own dataset.

```bash
python main.py --input sample_images/ --output output/ --debug
```

### Output

For each image `foo.jpg`, `output/` will contain:

- `foo_overlay.png` — original image with the best plate bbox + recognized text drawn on it
- `foo_edges.png`, `foo_candidates.png` — debug artifacts (only with `--debug`)

Plus, for the whole batch:

- `output/report.csv`
- `output/report.json`

Each report row/entry has: `image_name`, `plate_found`, `bbox`,
`aspect_ratio`, `recognized_text`, `confidence_note`.

## Project structure

```
license-plate-localizer/
├── preprocess.py       # Module 1: grayscale, bilateral filter, Canny
├── localize.py          # Module 2: contour detection + aspect/area filtering
├── ocr_report.py         # Module 3: crop, binarize, Tesseract OCR, report writers
├── main.py               # CLI batch driver (no GUI)
├── tests/                # unit tests (synthetic images, no dataset required)
├── sample_images/        # drop your dataset images here
├── requirements.txt
└── README.md
```

Each of `preprocess.py`, `localize.py`, and `ocr_report.py` can also be
run standalone on a single image for quick manual inspection, e.g.:

```bash
python preprocess.py sample_images/car1.jpg
python localize.py sample_images/car1.jpg
python ocr_report.py sample_images/car1.jpg
```

## Testing

Unit tests use synthetically generated images (drawn rectangles), so
they don't depend on the real dataset and run anywhere:

```bash
python -m unittest discover -s tests -v
```

## Design notes

- **Bilateral filter over Gaussian blur** in preprocessing: it smooths
  flat/noisy regions while keeping the plate's rectangular border
  sharp, which matters a lot for the downstream Canny + contour step.
- **Aspect-ratio + area filtering** rather than a trained detector:
  real-world plates cluster tightly around 2:1–5:1 width:height, so
  this simple geometric filter is enough to find good candidates
  without any model training.
- **Edge dilation before `findContours`**: Canny output on real
  photos is often broken into fragments; a small dilation reconnects
  plate borders into a single closed contour.
- **Otsu binarization before OCR**: gives Tesseract a clean
  black/white image instead of raw grayscale, which noticeably helps
  recognition on low-contrast crops.
- **Character whitelist** (`A–Z0–9`) passed to Tesseract: plates
  don't contain arbitrary punctuation/lowercase letters, so
  restricting the charset reduces misreads.

## Challenges / Limitations

- **Angled or tilted plates** break the axis-aligned aspect-ratio
  filter — a plate photographed at an angle projects to a bounding
  box with a different (often smaller) aspect ratio and can be
  rejected or mis-scored. A perspective-correction step (e.g. via
  `cv2.minAreaRect` + warp) would help but was left out to keep the
  pipeline minimal.
- **Low-contrast or dirty plates** frequently fail Canny edge
  detection entirely, since Canny depends on a clear intensity
  gradient at the plate border.
- **OCR accuracy** is inherently limited without a deep-learning
  recognizer: non-standard fonts, motion blur, poor lighting, or
  partial occlusion all degrade Tesseract's output. This project
  treats OCR as best-effort and reports `confidence_note` per image
  so failures are visible rather than silently wrong.
- **Multiple/no plates per image**: the pipeline reports up to
  `top_k` ranked candidates but only OCRs the single best-scoring one
  by default: images with multiple vehicles may need the candidate
  list (via `--debug`) inspected manually.
