"""
tests/test_localize.py

Lightweight unit tests that don't depend on real photos -- they draw
synthetic rectangles onto a blank canvas and check that the
localization module's filtering logic behaves as expected. Run with:

    python -m pytest tests/
or
    python tests/test_localize.py
"""

import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localize import find_candidates, localize_plates, non_max_suppress, PlateCandidate
from preprocess import preprocess_image


def make_synthetic_plate_image(
    canvas_size=(400, 600), plate_wh=(180, 45), plate_xy=(210, 250)
):
    """Draw a filled white rectangle (a 'plate') on a noisy gray background."""
    h, w = canvas_size
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 20, size=img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    pw, ph = plate_wh
    px, py = plate_xy
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 0, 0), 2)
    return img, (px, py, pw, ph)


class TestLocalize(unittest.TestCase):
    def test_finds_plate_shaped_region(self):
        img, expected_bbox = make_synthetic_plate_image()
        stages = preprocess_image(img)
        candidates = localize_plates(stages["edges"], img.shape)

        self.assertGreater(len(candidates), 0, "should find at least one candidate")

        # best candidate should roughly overlap the synthetic plate
        best = candidates[0]
        ex, ey, ew, eh = expected_bbox
        self.assertAlmostEqual(best.x, ex, delta=15)
        self.assertAlmostEqual(best.y, ey, delta=15)

    def test_rejects_extreme_aspect_ratios(self):
        # a near-square region should NOT pass the plate aspect filter
        h, w = 200, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        stages = preprocess_image(img)
        candidates = find_candidates(stages["edges"], img.shape,
                                      min_aspect=2.0, max_aspect=5.5)
        self.assertEqual(len(candidates), 0,
                          "square region should be rejected by aspect filter")

    def test_non_max_suppression_dedupes_overlaps(self):
        c1 = PlateCandidate(x=10, y=10, w=100, h=30, area=3000,
                             aspect_ratio=3.33, score=5.0)
        c2 = PlateCandidate(x=12, y=11, w=98, h=29, area=2842,
                             aspect_ratio=3.38, score=4.9)  # near-duplicate
        c3 = PlateCandidate(x=300, y=300, w=100, h=30, area=3000,
                             aspect_ratio=3.33, score=4.5)  # distinct

        kept = non_max_suppress([c1, c2, c3])
        self.assertEqual(len(kept), 2)
        self.assertIn(c1, kept)
        self.assertIn(c3, kept)

    def test_empty_edge_map_returns_no_candidates(self):
        blank_edges = np.zeros((300, 400), dtype=np.uint8)
        candidates = localize_plates(blank_edges, (300, 400, 3))
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
