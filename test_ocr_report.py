"""
tests/test_ocr_report.py

Tests for the crop/binarize/report-writing helpers in ocr_report.py.
OCR accuracy itself is not asserted (font-rendering/Tesseract output
is environment-dependent) -- these tests check the plumbing: that
cropping, binarizing, and report writing behave correctly.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localize import PlateCandidate
from ocr_report import (
    crop_candidate,
    binarize_plate,
    process_image,
    write_csv_report,
    write_json_report,
    OCRResult,
)


class TestOCRReport(unittest.TestCase):
    def test_crop_candidate_stays_in_bounds(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cand = PlateCandidate(x=280, y=190, w=50, h=30, area=1500,
                               aspect_ratio=1.67, score=1.0)
        crop = crop_candidate(img, cand)
        self.assertLessEqual(crop.shape[0], 200)
        self.assertLessEqual(crop.shape[1], 300)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_binarize_plate_is_single_channel_binary(self):
        crop = np.random.randint(0, 255, (40, 150, 3), dtype=np.uint8)
        binary = binarize_plate(crop)
        self.assertEqual(len(binary.shape), 2)
        unique_vals = set(np.unique(binary).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_process_image_no_candidates(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = process_image(img, "blank.png", [])
        self.assertFalse(result.plate_found)
        self.assertEqual(result.recognized_text, "")

    def test_process_image_with_candidate_runs_without_crash(self):
        img = np.full((200, 400, 3), 255, dtype=np.uint8)
        cand = PlateCandidate(x=100, y=80, w=180, h=45, area=8100,
                               aspect_ratio=4.0, score=2.0)
        result = process_image(img, "white.png", [cand])
        self.assertTrue(result.plate_found)
        self.assertIsNotNone(result.bbox)

    def test_csv_and_json_report_roundtrip(self):
        results = [
            OCRResult("img1.png", True, (1, 2, 3, 4), 3.5, "ABC123", "ok"),
            OCRResult("img2.png", False, None, None, "", "no plate found"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "report.csv")
            json_path = os.path.join(tmp, "report.json")
            write_csv_report(results, csv_path)
            write_json_report(results, json_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))

            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["recognized_text"], "ABC123")


if __name__ == "__main__":
    unittest.main()
