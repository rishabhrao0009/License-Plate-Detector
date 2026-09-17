# Licence-Plate-Detector
A no-GUI Python pipeline using OpenCV and Tesseract to detect and read license plates from images. It applies grayscale/bilateral filtering, Canny edge detection, and contour-based aspect-ratio filtering to locate plates, then OCRs the crop. Runs as a CLI batch tool, outputting overlays plus a CSV/JSON report. No GPU or training needed.
