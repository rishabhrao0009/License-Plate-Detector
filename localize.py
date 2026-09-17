"""
localize.py
-----------
Module 2: Plate Localization

Given the Canny edge map from preprocess.py, this module:
  1. Finds contours in the edge map.
  2. Filters candidate contours by aspect ratio (plates are
     roughly 2:1 to 5:1, width:height) and by minimum/maximum area
     relative to the image, to throw out noise and implausible
     shapes.
  3. Ranks the surviving candidates and returns the best one(s) as
     bounding boxes (x, y, w, h).

Classical contour-geometry approach -- no ML/DL involved.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class PlateCandidate:
    x: int
    y: int
    w: int
    h: int
    area: float
    aspect_ratio: float
    score: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def find_candidates(
    edges: np.ndarray,
    image_shape: Tuple[int, int],
    min_aspect: float = 2.0,
    max_aspect: float = 5.5,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.20,
) -> List[PlateCandidate]:
    """
    Find plausible plate bounding boxes from an edge map.

    Args:
        edges: binary edge map (output of Canny).
        image_shape: (height, width) of the original image, used to
            normalize the area thresholds so they scale with image size.
        min_aspect/max_aspect: acceptable width:height ratio range.
        min_area_frac/max_area_frac: acceptable contour area as a
            fraction of the total image area (filters out both tiny
            noise specks and implausibly large regions).

    Returns:
        List of PlateCandidate, sorted best-first (highest score).
    """
    h_img, w_img = image_shape[:2]
    img_area = float(h_img * w_img)

    # Dilate edges slightly so broken/fragmented plate borders (common
    # with Canny on real-world images) connect into closed contours.
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: List[PlateCandidate] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        area = float(w * h)
        area_frac = area / img_area
        aspect_ratio = w / float(h)

        if not (min_area_frac <= area_frac <= max_area_frac):
            continue
        if not (min_aspect <= aspect_ratio <= max_aspect):
            continue

        # Score candidates: prefer boxes whose aspect ratio is close
        # to a "typical" plate ratio (~3.5) and that are reasonably
        # large (larger candidates are usually more reliable than
        # tiny fragments that happened to pass the filters).
        target_aspect = 3.5
        aspect_penalty = abs(aspect_ratio - target_aspect)
        score = area_frac * 10.0 - aspect_penalty

        candidates.append(
            PlateCandidate(
                x=x, y=y, w=w, h=h,
                area=area, aspect_ratio=aspect_ratio, score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def non_max_suppress(
    candidates: List[PlateCandidate], iou_thresh: float = 0.3
) -> List[PlateCandidate]:
    """
    Simple greedy NMS so overlapping boxes around the same plate
    don't all get reported as separate candidates.
    """
    def iou(a: PlateCandidate, b: PlateCandidate) -> float:
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx2, by2 = b.x + b.w, b.y + b.h
        ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    kept: List[PlateCandidate] = []
    for cand in candidates:
        if all(iou(cand, k) < iou_thresh for k in kept):
            kept.append(cand)
    return kept


def localize_plates(
    edges: np.ndarray,
    image_shape: Tuple[int, int],
    top_k: int = 3,
) -> List[PlateCandidate]:
    """Full localization step: find candidates, dedupe, return top_k."""
    candidates = find_candidates(edges, image_shape)
    candidates = non_max_suppress(candidates)
    return candidates[:top_k]


def draw_candidates(
    image: np.ndarray, candidates: List[PlateCandidate]
) -> np.ndarray:
    """Return a copy of `image` with bounding boxes drawn (for the report)."""
    out = image.copy()
    for i, c in enumerate(candidates):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(out, (c.x, c.y), (c.x + c.w, c.y + c.h), color, 2)
        cv2.putText(
            out, f"#{i+1} ar={c.aspect_ratio:.2f}",
            (c.x, max(0, c.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out


if __name__ == "__main__":
    import sys
    from preprocess import preprocess_image

    if len(sys.argv) != 2:
        print("Usage: python localize.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    stages = preprocess_image(img)
    cands = localize_plates(stages["edges"], img.shape)
    print(f"Found {len(cands)} candidate(s):")
    for c in cands:
        print(f"  bbox={c.bbox} aspect={c.aspect_ratio:.2f} score={c.score:.3f}")

    overlay = draw_candidates(img, cands)
    cv2.imwrite("/tmp/localize_overlay.png", overlay)
    print("Saved overlay -> /tmp/localize_overlay.png")
