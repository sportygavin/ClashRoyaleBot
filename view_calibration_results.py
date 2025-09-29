#!/usr/bin/env python3
"""
View calibration results from analyze_screenshot.py

This script loads the calibration.json and shows you the detected regions
overlaid on your screenshot so you can verify accuracy.
"""

import json
import cv2
import numpy as np
from pathlib import Path

def load_calibration(calib_path: Path):
    """Load calibration data from JSON file."""
    with open(calib_path, 'r') as f:
        return json.load(f)

def draw_calibration_overlay(img_path: str, calib_path: str, output_path: str):
    """Draw all calibration regions on the original screenshot."""
    # Load image and calibration
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    calib = load_calibration(Path(calib_path))
    H, W = img.shape[:2]
    
    # Create overlay canvas
    overlay = img.copy()
    
    # Draw viewport (yellow rectangle)
    vp = calib['viewport']
    vp_x = int(vp['x_r'] * W)
    vp_y = int(vp['y_r'] * H)
    vp_w = int(vp['w_r'] * W)
    vp_h = int(vp['h_r'] * H)
    cv2.rectangle(overlay, (vp_x, vp_y), (vp_x + vp_w, vp_y + vp_h), (0, 255, 255), 3)
    
    # Draw card row (green horizontal lines)
    row = calib['card_row']
    row_top_y = vp_y + int(row['top_r'] * vp_h)
    row_bottom_y = vp_y + int(row['bottom_r'] * vp_h)
    cv2.line(overlay, (vp_x, row_top_y), (vp_x + vp_w, row_top_y), (0, 255, 0), 2)
    cv2.line(overlay, (vp_x, row_bottom_y), (vp_x + vp_w, row_bottom_y), (0, 255, 0), 2)
    
    # Draw card centers (red vertical lines and circles)
    centers = calib['cards']['centers_x_r']
    for i, center_x_r in enumerate(centers):
        center_x = vp_x + int(center_x_r * vp_w)
        cv2.line(overlay, (center_x, row_top_y), (center_x, row_bottom_y), (255, 0, 0), 2)
        center_y = (row_top_y + row_bottom_y) // 2
        cv2.circle(overlay, (center_x, center_y), 8, (0, 0, 255), -1)
        
        # Add card number label
        cv2.putText(overlay, f"C{i+1}", (center_x - 15, center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw individual card boxes (blue rectangles)
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
    
    # Save result
    cv2.imwrite(output_path, overlay)
    print(f"Calibration overlay saved to: {output_path}")
    
    # Print summary
    print("\n=== Calibration Summary ===")
    print(f"Image size: {W}x{H}")
    print(f"Viewport: ({vp_x}, {vp_y}) {vp_w}x{vp_h}")
    print(f"Card row: top={row_top_y}, bottom={row_bottom_y}")
    print(f"Card centers: {[vp_x + int(c * vp_w) for c in centers]}")
    
    return overlay

def main():
    import argparse
    parser = argparse.ArgumentParser(description="View calibration results")
    parser.add_argument("--image", default="test_screenshot.png", help="Original screenshot")
    parser.add_argument("--calib", default="cv_out/calibration.json", help="Calibration JSON file")
    parser.add_argument("--out", default="calibration_overlay.png", help="Output overlay image")
    
    args = parser.parse_args()
    
    try:
        draw_calibration_overlay(args.image, args.calib, args.out)
        print(f"\nOpen {args.out} to verify the calibration accuracy!")
        print("Legend:")
        print("  Yellow rectangle = Game viewport")
        print("  Green lines = Card row boundaries") 
        print("  Red lines + circles = Card centers")
        print("  Purple rectangles = Individual card boxes")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
