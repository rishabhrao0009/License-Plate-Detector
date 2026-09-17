"""
main.py
-------
CLI entry point. No GUI -- runs on a folder (or single file) of static
images, saves a bounding-box overlay per image, and writes a batch
CSV + JSON report.

Usage:
    python main.py --input sample_images/ --output output/
    python main.py --input sample_images/car1.jpg --output output/
"""

import argparse
import glob
import os
import sys
import time

import cv2

from preprocess import preprocess_image
from localize import localize_plates, draw_candidates
from ocr_report import (
    process_image,
    draw_result_overlay,
    write_csv_report,
    write_json_report,
)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_image_paths(input_path: str) -> list:
    if os.path.isdir(input_path):
        paths = []
        for ext in VALID_EXTS:
            paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(paths)
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        return []


def run_pipeline(image_path: str, output_dir: str, save_debug: bool):
    name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [skip] could not read {image_path}")
        return None

    stages = preprocess_image(img)
    candidates = localize_plates(stages["edges"], img.shape)
    result = process_image(img, name, candidates)

    overlay = draw_result_overlay(img, result)
    base, _ = os.path.splitext(name)
    overlay_path = os.path.join(output_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    if save_debug:
        debug_path = os.path.join(output_dir, f"{base}_candidates.png")
        cv2.imwrite(debug_path, draw_candidates(img, candidates))
        cv2.imwrite(os.path.join(output_dir, f"{base}_edges.png"), stages["edges"])

    status = f'"{result.recognized_text}"' if result.plate_found else "NOT FOUND"
    print(f"  {name}: {status}  ({result.confidence_note})")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="License plate localization + OCR pipeline (no GUI, CLI only)."
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Path to an image file or a directory of images.")
    parser.add_argument("--output", "-o", default="output",
                         help="Directory to write overlays + reports to.")
    parser.add_argument("--debug", action="store_true",
                         help="Also save intermediate edge maps and all "
                              "candidate boxes (not just the best one).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        print(f"No images found at: {args.input}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s). Running pipeline...")
    start = time.time()

    results = []
    for path in image_paths:
        r = run_pipeline(path, args.output, args.debug)
        if r is not None:
            results.append(r)

    csv_path = os.path.join(args.output, "report.csv")
    json_path = os.path.join(args.output, "report.json")
    write_csv_report(results, csv_path)
    write_json_report(results, json_path)

    elapsed = time.time() - start
    found = sum(1 for r in results if r.plate_found)
    print("\n--- Summary ---")
    print(f"Processed: {len(results)} image(s) in {elapsed:.2f}s")
    print(f"Plate region found: {found}/{len(results)}")
    print(f"Report written to: {csv_path}, {json_path}")
    print(f"Overlays written to: {args.output}/")


if __name__ == "__main__":
    main()
