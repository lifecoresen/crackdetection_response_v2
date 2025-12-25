import argparse
import csv
import os
from dataclasses import dataclass
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np

# Reuse the detector internals from crack_yolo.py
from crack_yolo import (
    _compute_crack_response,
    _threshold_by_percentile,
)


@dataclass
class ImageMetrics:
    image: str
    std_gray: float
    percentile: float
    threshold_value: int
    resp_mean: float
    resp_p95: float
    resp_p99: float
    resp_max: int

    binary_pixel_ratio: float
    binary_components: int
    binary_largest_area: int
    binary_median_area: float

    contour_count: int
    contour_perimeter_sum: float


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def _make_panel(images: List[np.ndarray], labels: List[str], height: int = 360) -> np.ndarray:
    assert len(images) == len(labels)

    tiles: List[np.ndarray] = []
    for img, label in zip(images, labels):
        tile = _to_bgr(img)
        tile = _resize_to_height(tile, height)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)

    return cv2.hconcat(tiles)


def analyze_one(image_path: str) -> Tuple[np.ndarray, np.ndarray, ImageMetrics, np.ndarray]:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"Could not read: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    percentile = 98.5 if std > 20 else 96.5

    resp = _compute_crack_response(gray)

    # Mirror the exact thresholding logic so we can report the used threshold.
    thr = int(np.percentile(resp.reshape(-1), percentile))
    thr = max(thr, 10)
    binary = _threshold_by_percentile(resp, percentile)

    # Minimal cleanup only (keep detections as-is).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    resp_mean = float(np.mean(resp))
    resp_p95 = float(np.percentile(resp.reshape(-1), 95))
    resp_p99 = float(np.percentile(resp.reshape(-1), 99))
    resp_max = int(np.max(resp))

    binary_ratio = float(np.mean(binary_clean > 0))
    # Connected components on binary to describe how fragmented/noisy it is.
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats((binary_clean > 0).astype(np.uint8), connectivity=8)
    # stats[0] is background
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if n_lbl > 1 else np.array([], dtype=np.int64)
    binary_components = int(areas.size)
    binary_largest_area = int(np.max(areas)) if areas.size else 0
    binary_median_area = float(np.median(areas)) if areas.size else 0.0

    contours, _ = cv2.findContours(binary_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_count = int(len(contours))
    contour_perimeter_sum = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else 0.0

    overlay = bgr.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    metrics = ImageMetrics(
        image=os.path.basename(image_path),
        std_gray=std,
        percentile=percentile,
        threshold_value=int(thr),
        resp_mean=resp_mean,
        resp_p95=resp_p95,
        resp_p99=resp_p99,
        resp_max=resp_max,
        binary_pixel_ratio=binary_ratio,
        binary_components=binary_components,
        binary_largest_area=binary_largest_area,
        binary_median_area=binary_median_area,
        contour_count=contour_count,
        contour_perimeter_sum=contour_perimeter_sum,
    )

    return resp, binary_clean, metrics, overlay


def main() -> None:
    ap = argparse.ArgumentParser(description="Better, comparable crack-detection tests")
    ap.add_argument("--images", default="images", help="Image folder")
    ap.add_argument("--out", default="results/tests", help="Output folder")
    ap.add_argument("--pattern", default="*.jpg", help="Glob pattern inside images folder")
    args = ap.parse_args()

    image_paths = sorted(glob(os.path.join(args.images, args.pattern)))
    if not image_paths:
        # Try common extensions
        image_paths = sorted(glob(os.path.join(args.images, "*.jpeg")))
    if not image_paths:
        image_paths = sorted(glob(os.path.join(args.images, "*.png")))

    if not image_paths:
        raise SystemExit(f"No images found in {args.images} with pattern {args.pattern}")

    _ensure_dir(args.out)

    metrics_rows: List[ImageMetrics] = []

    for p in image_paths:
        print(f"Processing: {p}")
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"  Skipping unreadable: {p}")
            continue

        resp, binary, metrics, overlay = analyze_one(p)
        metrics_rows.append(metrics)

        # Pretty response heatmap (debug)
        heat = cv2.applyColorMap(resp, cv2.COLORMAP_TURBO)

        panel = _make_panel(
            [bgr, heat, binary, overlay],
            ["original", "response", "binary", "overlay"],
            height=360,
        )

        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(args.out, f"{base}_panel.jpg"), panel)

    # CSV summary
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "std_gray",
                "percentile",
                "threshold_value",
                "resp_mean",
                "resp_p95",
                "resp_p99",
                "resp_max",
                "binary_pixel_ratio",
                "binary_components",
                "binary_largest_area",
                "binary_median_area",
                "contour_count",
                "contour_perimeter_sum",
            ]
        )
        for m in metrics_rows:
            w.writerow(
                [
                    m.image,
                    f"{m.std_gray:.3f}",
                    f"{m.percentile:.2f}",
                    m.threshold_value,
                    f"{m.resp_mean:.3f}",
                    f"{m.resp_p95:.3f}",
                    f"{m.resp_p99:.3f}",
                    m.resp_max,
                    f"{m.binary_pixel_ratio:.6f}",
                    m.binary_components,
                    m.binary_largest_area,
                    f"{m.binary_median_area:.3f}",
                    m.contour_count,
                    f"{m.contour_perimeter_sum:.3f}",
                ]
            )

    print(f"\nWrote panels + debug to: {args.out}")
    print(f"Wrote summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
