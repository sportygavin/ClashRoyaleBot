"""
Analyze a Clash Royale screenshot to extract calibration data.

Implements (first pass - minimal):
  - Viewport detection (crop away desktop/ads)
  - Card row band detection (bottom strip)
  - Four card centers (x positions via 1D projection)

Outputs to --out directory:
  - viewport.png (yellow box overlay)
  - card_row.png (row band overlay)
  - cards_overlay.png (centers overlay)
  - calibration.json (ratios relative to viewport)

Usage:
  python3 src/vision/analyze_screenshot.py --image /path/to/screenshot.png --out ./cv_out

Notes:
  - This script focuses on steps B, C, D of the planned pipeline.
  - Elixir OCR/template matching can be added after visual verification.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ----------------------------- Datatypes -----------------------------

@dataclass
class Viewport:
    x: int
    y: int
    w: int
    h: int

    def to_ratios(self, W: int, H: int) -> Dict[str, float]:
        return {
            "x_r": self.x / float(W),
            "y_r": self.y / float(H),
            "w_r": self.w / float(W),
            "h_r": self.h / float(H),
        }


# ----------------------------- Helpers ------------------------------

def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def snap_to_aspect(x: int, y: int, w: int, h: int, aspect: float, W: int, H: int) -> Tuple[int, int, int, int]:
    """Snap a rectangle to a target aspect ratio while keeping it inside image bounds."""
    current = w / max(h, 1)
    nx, ny, nw, nh = x, y, w, h
    if current > aspect:
        # Too wide → reduce width
        nw = int(h * aspect)
        nx = x + (w - nw) // 2
    else:
        # Too tall → reduce height
        nh = int(w / aspect)
        ny = y + (h - nh) // 2
    # Clamp to image
    nx = max(0, min(nx, W - 1))
    ny = max(0, min(ny, H - 1))
    nw = max(1, min(nw, W - nx))
    nh = max(1, min(nh, H - ny))
    return nx, ny, nw, nh


def detect_viewport(img: np.ndarray) -> Viewport:
    """Heuristic viewport detection using HSV masks for arena/wood tones.

    Returns a rectangle approximating the inner Clash Royale canvas.
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green-ish arena mask
    lower_green = np.array([30, 60, 50], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Brown-ish wood mask (broad, to catch borders)
    lower_brown = np.array([5, 50, 40], dtype=np.uint8)
    upper_brown = np.array([25, 255, 255], dtype=np.uint8)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    mask = cv2.bitwise_or(mask_green, mask_brown)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: assume centered 16:9 occupying height
        aspect = 16 / 9
        w_guess = int(H * aspect)
        w_guess = min(w_guess, W)
        x = (W - w_guess) // 2
        return Viewport(x=x, y=0, w=w_guess, h=H)

    # Largest contour bbox
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Snap to 16:9 that fits inside image
    x, y, w, h = snap_to_aspect(x, y, w, h, aspect=16 / 9, W=W, H=H)
    return Viewport(x=x, y=y, w=w, h=h)


def score_row_band(roi_bgr: np.ndarray, y0: int, y1: int) -> float:
    """Score a horizontal band using edge density + color variation."""
    band = roi_bgr[y0:y1, :]
    if band.size == 0:
        return -1.0
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_score = float(np.count_nonzero(edges)) / (edges.shape[0] * edges.shape[1] + 1e-6)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1])) / 255.0
    val_std = float(np.std(hsv[:, :, 2])) / 255.0
    return edge_score + 0.5 * (sat_std + val_std)


def detect_card_row(roi_bgr: np.ndarray) -> Tuple[int, int]:
    """Find the card row band near the bottom of the viewport ROI.

    Returns (row_top_px, row_bottom_px) in ROI coordinates.
    """
    H, W = roi_bgr.shape[:2]
    band_h = max(6, int(0.06 * H))  # ~6% of ROI height
    search_top = int(0.70 * H)      # start searching near bottom 30%
    best_score = -1.0
    best_y0, best_y1 = int(0.85 * H), int(0.95 * H)

    for y0 in range(search_top, H - band_h, max(4, band_h // 3)):
        y1 = y0 + band_h
        s = score_row_band(roi_bgr, y0, y1)
        if s > best_score:
            best_score = s
            best_y0, best_y1 = y0, y1
    return best_y0, best_y1


def smooth_1d(arr: np.ndarray, k: int = 21) -> np.ndarray:
    k = max(3, k | 1)  # odd
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def detect_card_centers(roi_bgr: np.ndarray, row_top: int, row_bottom: int) -> List[int]:
    """Detect 4 card x-centers inside the card row band using vertical projection."""
    band = roi_bgr[row_top:row_bottom, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    proj = edges.sum(axis=0)  # sum over y → length W
    proj_s = smooth_1d(proj, k=max(11, edges.shape[1] // 80))
    W = edges.shape[1]

    # initial guesses at 1/8, 3/8, 5/8, 7/8
    guesses = [int(W * r) for r in (1 / 8, 3 / 8, 5 / 8, 7 / 8)]
    half_window = max(4, int(0.05 * W))
    centers = []
    for g in guesses:
        x0 = max(0, g - half_window)
        x1 = min(W - 1, g + half_window)
        local = proj_s[x0:x1]
        if local.size == 0:
            centers.append(g)
            continue
        local_max_idx = int(np.argmax(local))
        centers.append(x0 + local_max_idx)

    # Enforce ordering and spacing
    centers = sorted(centers)
    return centers


def draw_viewport_overlay(img: np.ndarray, vp: Viewport) -> np.ndarray:
    canvas = img.copy()
    cv2.rectangle(canvas, (vp.x, vp.y), (vp.x + vp.w, vp.y + vp.h), (0, 255, 255), 3)
    return canvas


def draw_card_row_overlay(roi_bgr: np.ndarray, row_top: int, row_bottom: int) -> np.ndarray:
    canvas = roi_bgr.copy()
    H, W = canvas.shape[:2]
    cv2.line(canvas, (0, row_top), (W - 1, row_top), (0, 255, 0), 2)
    cv2.line(canvas, (0, row_bottom), (W - 1, row_bottom), (0, 255, 0), 2)
    return canvas


def draw_centers_overlay(roi_bgr: np.ndarray, row_top: int, row_bottom: int, centers: List[int]) -> np.ndarray:
    canvas = roi_bgr.copy()
    for cx in centers:
        cv2.line(canvas, (cx, row_top), (cx, row_bottom), (255, 0, 0), 2)
        cv2.circle(canvas, (cx, (row_top + row_bottom) // 2), 6, (0, 0, 255), -1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Clash Royale screenshot for calibration")
    ap.add_argument("--image", required=True, help="Absolute path to screenshot PNG")
    ap.add_argument("--out", default="./cv_out", help="Output directory for overlays/calibration")
    args = ap.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out)
    ensure_out_dir(out_dir)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    # B) Viewport detection
    vp = detect_viewport(img)
    overlay_vp = draw_viewport_overlay(img, vp)
    save_image(out_dir / "viewport.png", overlay_vp)

    roi = img[vp.y : vp.y + vp.h, vp.x : vp.x + vp.w]

    # C) Card row band detection
    row_top, row_bottom = detect_card_row(roi)
    overlay_row = draw_card_row_overlay(roi, row_top, row_bottom)
    save_image(out_dir / "card_row.png", overlay_row)

    # D) Card centers
    centers = detect_card_centers(roi, row_top, row_bottom)
    overlay_centers = draw_centers_overlay(roi, row_top, row_bottom, centers)
    save_image(out_dir / "cards_overlay.png", overlay_centers)

    # Build calibration JSON (ratios relative to viewport)
    vp_ratios = vp.to_ratios(W, H)
    top_r = row_top / float(vp.h)
    bottom_r = row_bottom / float(vp.h)
    centers_r = [c / float(vp.w) for c in centers]

    calib = {
        "metadata": {
            "image_W": W,
            "image_H": H,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "notes": "First-pass calibration (viewport, card row, centers)"
        },
        "viewport": vp_ratios,
        "card_row": {"top_r": top_r, "bottom_r": bottom_r},
        "cards": {
            "centers_x_r": centers_r,
            # Placeholders for future steps
            "width_r": 0.12,              # tune visually
            "top_offset_r": 0.10,         # 10% inside row from top
            "bottom_offset_r": 0.10       # 10% inside row from bottom
        },
        "elixir_roi": {
            # Relative to card box; tune later during elixir stage
            "x_off_r": 0.08,
            "y_off_r": 0.62,
            "w_r": 0.22,
            "h_r": 0.28
        }
    }

    with open(out_dir / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    # Console summary
    print("=== Calibration Summary ===")
    print(f"Image: {image_path}")
    print(f"Viewport (px): x={vp.x} y={vp.y} w={vp.w} h={vp.h}")
    print(f"Viewport (r): {vp_ratios}")
    print(f"Row (r): top={top_r:.4f} bottom={bottom_r:.4f}")
    print(f"Centers (r): {[round(c, 4) for c in centers_r]}")
    print(f"Overlays saved to: {out_dir}")


if __name__ == "__main__":
    main()


