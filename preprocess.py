"""
preprocess.py
-------------
Module 1: Preprocessing

Takes a raw BGR image and produces:
  - grayscale version
  - noise-reduced version (bilateral filter: smooths flat regions,
    preserves edges -- important because plate borders/characters
    are exactly the edges we need to keep sharp)
  - Canny edge map (used downstream for contour-based localization)

Kept deliberately simple / classical (no deep learning), per the
project scope: OpenCV only.
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, d: int = 11, sigma_color: int = 17,
            sigma_space: int = 17) -> np.ndarray:
    """
    Apply a bilateral filter.

    Unlike Gaussian blur, bilateral filtering smooths noise/texture
    while keeping strong edges intact -- which is exactly what we
    want before edge detection, since we don't want to blur away
    the plate's rectangular border.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def detect_edges(denoised: np.ndarray, low_thresh: int = 30,
                  high_thresh: int = 200) -> np.ndarray:
    """Run Canny edge detection on the denoised grayscale image."""
    return cv2.Canny(denoised, low_thresh, high_thresh)


def preprocess_image(image: np.ndarray) -> dict:
    """
    Run the full preprocessing pipeline on a single image.

    Returns a dict with each intermediate stage so callers (and the
    debug/report step) can inspect or save every step if needed.
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)
    edges = detect_edges(denoised)

    return {
        "original": image,
        "gray": gray,
        "denoised": denoised,
        "edges": edges,
    }


if __name__ == "__main__":
    # quick manual smoke test
    import sys

    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    stages = preprocess_image(img)
    for name, stage_img in stages.items():
        out_path = f"/tmp/preprocess_{name}.png"
        cv2.imwrite(out_path, stage_img)
        print(f"Saved {name} -> {out_path}")
