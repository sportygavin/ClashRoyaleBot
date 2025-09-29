#!/usr/bin/env python3
"""
Quick calibration editor - edit values and immediately see results.

This lets you manually adjust the calibration.json values and see the overlay
without having to click through an interactive tool.
"""

import json
import cv2
import numpy as np
from pathlib import Path

def load_calibration(calib_path: str):
    """Load calibration data."""
    with open(calib_path, 'r') as f:
        return json.load(f)

def save_calibration(calib: dict, calib_path: str):
    """Save calibration data."""
    with open(calib_path, 'w') as f:
        json.dump(calib, f, indent=2)

def draw_overlay(img_path: str, calib: dict, output_path: str):
    """Draw calibration overlay."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    H, W = img.shape[:2]
    overlay = img.copy()
    
    # Viewport
    vp = calib['viewport']
    vp_x = int(vp['x_r'] * W)
    vp_y = int(vp['y_r'] * H)
    vp_w = int(vp['w_r'] * W)
    vp_h = int(vp['h_r'] * H)
    cv2.rectangle(overlay, (vp_x, vp_y), (vp_x + vp_w, vp_y + vp_h), (0, 255, 255), 3)
    
    # Card row
    row = calib['card_row']
    row_top_y = vp_y + int(row['top_r'] * vp_h)
    row_bottom_y = vp_y + int(row['bottom_r'] * vp_h)
    cv2.line(overlay, (vp_x, row_top_y), (vp_x + vp_w, row_top_y), (0, 255, 0), 2)
    cv2.line(overlay, (vp_x, row_bottom_y), (vp_x + vp_w, row_bottom_y), (0, 255, 0), 2)
    
    # Card centers
    centers = calib['cards']['centers_x_r']
    for i, center_x_r in enumerate(centers):
        center_x = vp_x + int(center_x_r * vp_w)
        cv2.line(overlay, (center_x, row_top_y), (center_x, row_bottom_y), (255, 0, 0), 2)
        center_y = (row_top_y + row_bottom_y) // 2
        cv2.circle(overlay, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.putText(overlay, f"C{i+1}", (center_x - 15, center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Card boxes
    card_width_r = calib['cards']['width_r']
    top_offset_r = calib['cards']['top_offset_r']
    bottom_offset_r = calib['cards']['bottom_offset_r']
    
    for center_x_r in centers:
        center_x = vp_x + int(center_x_r * vp_w)
        card_w = int(card_width_r * vp_w)
        card_x1 = center_x - card_w // 2
        card_x2 = center_x + card_w // 2
        
        card_top_y = row_top_y + int(top_offset_r * (row_bottom_y - row_top_y))
        card_bottom_y = row_bottom_y - int(bottom_offset_r * (row_bottom_y - row_top_y))
        
        cv2.rectangle(overlay, (card_x1, card_top_y), (card_x2, card_bottom_y), (255, 0, 255), 2)
    
    cv2.imwrite(output_path, overlay)
    return overlay

def main():
    calib_path = "cv_out/calibration.json"
    img_path = "test_screenshot.png"
    overlay_path = "calibration_overlay.png"
    
    print("=== Quick Calibration Editor ===")
    print("Current calibration values:")
    
    calib = load_calibration(calib_path)
    
    # Show current values
    vp = calib['viewport']
    row = calib['card_row']
    centers = calib['cards']['centers_x_r']
    
    print(f"Viewport: x_r={vp['x_r']:.4f}, y_r={vp['y_r']:.4f}, w_r={vp['w_r']:.4f}, h_r={vp['h_r']:.4f}")
    print(f"Card row: top_r={row['top_r']:.4f}, bottom_r={row['bottom_r']:.4f}")
    print(f"Card centers: {[f'{c:.4f}' for c in centers]}")
    
    print("\nSuggested adjustments based on your feedback:")
    print("- Viewport is too big: reduce w_r and h_r")
    print("- Cards are lower: increase top_r and bottom_r")
    print()
    
    # Suggest corrections
    print("Suggested corrected values:")
    print("Try these adjustments:")
    print(f"  w_r: {vp['w_r']:.4f} -> {vp['w_r'] - 0.05:.4f}  (reduce width)")
    print(f"  h_r: {vp['h_r']:.4f} -> {vp['h_r'] - 0.05:.4f}  (reduce height)")
    print(f"  top_r: {row['top_r']:.4f} -> {row['top_r'] + 0.02:.4f}  (move cards up)")
    print(f"  bottom_r: {row['bottom_r']:.4f} -> {row['bottom_r'] + 0.02:.4f}  (move cards up)")
    
    # Apply suggested corrections
    calib['viewport']['w_r'] -= 0.05
    calib['viewport']['h_r'] -= 0.05
    calib['card_row']['top_r'] += 0.02
    calib['card_row']['bottom_r'] += 0.02
    
    # Save corrected calibration
    save_calibration(calib, "cv_out/calibration_corrected.json")
    
    # Generate overlay
    draw_overlay(img_path, calib, overlay_path)
    
    print(f"\nCorrected calibration saved to: cv_out/calibration_corrected.json")
    print(f"Overlay saved to: {overlay_path}")
    print("\nCheck the overlay to see if it's better!")
    print("If not, edit cv_out/calibration_corrected.json manually and run:")
    print("  python3 view_calibration_results.py --calib cv_out/calibration_corrected.json")

if __name__ == "__main__":
    main()
