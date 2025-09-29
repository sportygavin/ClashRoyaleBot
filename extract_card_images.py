#!/usr/bin/env python3
"""
Extract individual card images from screenshot using calibration data.

This helps you see exactly what the bot would detect for each card.
"""

import json
import cv2
import numpy as np
from pathlib import Path

def extract_cards(img_path: str, calib_path: str, output_dir: str):
    """Extract individual card images based on calibration."""
    # Load image and calibration
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    calib = load_calibration(Path(calib_path))
    H, W = img.shape[:2]
    
    # Create output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Get viewport and card parameters
    vp = calib['viewport']
    vp_x = int(vp['x_r'] * W)
    vp_y = int(vp['y_r'] * H)
    vp_w = int(vp['w_r'] * W)
    vp_h = int(vp['h_r'] * H)
    
    row = calib['card_row']
    row_top_y = vp_y + int(row['top_r'] * vp_h)
    row_bottom_y = vp_y + int(row['bottom_r'] * vp_h)
    
    centers = calib['cards']['centers_x_r']
    card_width_r = calib['cards']['width_r']
    top_offset_r = calib['cards']['top_offset_r']
    bottom_offset_r = calib['cards']['bottom_offset_r']
    
    # Extract each card
    for i, center_x_r in enumerate(centers):
        center_x = vp_x + int(center_x_r * vp_w)
        card_w = int(card_width_r * vp_w)
        card_x1 = center_x - card_w // 2
        card_x2 = center_x + card_w // 2
        
        card_top_y = row_top_y + int(top_offset_r * (row_bottom_y - row_top_y))
        card_bottom_y = row_bottom_y - int(bottom_offset_r * (row_bottom_y - row_top_y))
        
        # Crop the card
        card_img = img[card_top_y:card_bottom_y, card_x1:card_x2]
        
        # Save the card
        card_path = out_dir / f"card_{i+1}.png"
        cv2.imwrite(str(card_path), card_img)
        
        print(f"Card {i+1}: {card_img.shape[1]}x{card_img.shape[0]} pixels")
        print(f"  Saved to: {card_path}")
    
    # Also save the card row for reference
    row_img = img[row_top_y:row_bottom_y, vp_x:vp_x + vp_w]
    cv2.imwrite(str(out_dir / "card_row.png"), row_img)
    print(f"\nCard row: {row_img.shape[1]}x{row_img.shape[0]} pixels")
    print(f"  Saved to: {out_dir / 'card_row.png'}")

def load_calibration(calib_path: Path):
    """Load calibration data from JSON file."""
    with open(calib_path, 'r') as f:
        return json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract individual card images")
    parser.add_argument("--image", default="test_screenshot.png", help="Original screenshot")
    parser.add_argument("--calib", default="cv_out/calibration.json", help="Calibration JSON file")
    parser.add_argument("--out", default="extracted_cards", help="Output directory for card images")
    
    args = parser.parse_args()
    
    try:
        extract_cards(args.image, args.calib, args.out)
        print(f"\nExtracted cards saved to: {args.out}/")
        print("Check these images to verify the card detection is accurate!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
