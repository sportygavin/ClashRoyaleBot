#!/usr/bin/env python3
"""
Test the updated calibration with corrected card boxes and elixir region.

This script will:
1. Generate new overlay with bigger card boxes
2. Extract new card images
3. Extract elixir regions from bottom center of cards
4. Show the results
"""

import cv2
import numpy as np
import json
from pathlib import Path

def load_calibration(calib_path: str):
    """Load calibration data."""
    with open(calib_path, 'r') as f:
        return json.load(f)

def draw_calibration_overlay(img_path: str, calib_path: str, output_path: str):
    """Draw all calibration regions on the original screenshot."""
    # Load image and calibration
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    calib = load_calibration(calib_path)
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
    
    # Draw individual card boxes (purple rectangles)
    card_width_r = calib['cards']['width_r']
    top_offset_r = calib['cards']['top_offset_r']
    bottom_offset_r = calib['cards']['bottom_offset_r']
    
    for i, center_x_r in enumerate(centers):
        center_x = vp_x + int(center_x_r * vp_w)
        card_w = int(card_width_r * vp_w)
        card_x1 = center_x - card_w // 2
        card_x2 = center_x + card_w // 2
        
        card_top_y = row_top_y + int(top_offset_r * (row_bottom_y - row_top_y))
        card_bottom_y = row_bottom_y - int(bottom_offset_r * (row_bottom_y - row_top_y))
        
        cv2.rectangle(overlay, (card_x1, card_top_y), (card_x2, card_bottom_y), (255, 0, 255), 2)
        
        # Draw elixir region (cyan rectangle)
        elixir_roi = calib['elixir_roi']
        elixir_x1 = card_x1 + int(elixir_roi['x_off_r'] * card_w)
        elixir_y1 = card_top_y + int(elixir_roi['y_off_r'] * (card_bottom_y - card_top_y))
        elixir_x2 = elixir_x1 + int(elixir_roi['w_r'] * card_w)
        elixir_y2 = elixir_y1 + int(elixir_roi['h_r'] * (card_bottom_y - card_top_y))
        
        cv2.rectangle(overlay, (elixir_x1, elixir_y1), (elixir_x2, elixir_y2), (255, 255, 0), 2)
        cv2.putText(overlay, f"E{i+1}", (elixir_x1, elixir_y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Save result
    cv2.imwrite(output_path, overlay)
    print(f"Updated calibration overlay saved to: {output_path}")
    
    return overlay

def extract_cards_with_new_calibration(img_path: str, calib_path: str, output_dir: str):
    """Extract cards and elixir regions with updated calibration."""
    # Load image and calibration
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    calib = load_calibration(calib_path)
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
    
    # Extract each card using the EXACT same coordinates as the overlay
    for i, center_x_r in enumerate(centers):
        # Use the same coordinate calculation as the overlay
        center_x = vp_x + int(center_x_r * vp_w)
        card_w = int(card_width_r * vp_w)
        card_x1 = center_x - card_w // 2
        card_x2 = center_x + card_w // 2
        
        # Use the same coordinate calculation as the overlay
        card_top_y = row_top_y + int(top_offset_r * (row_bottom_y - row_top_y))
        card_bottom_y = row_bottom_y - int(bottom_offset_r * (row_bottom_y - row_top_y))
        
        # Crop the card from the FULL image (not ROI)
        card_img = img[card_top_y:card_bottom_y, card_x1:card_x2]
        
        # Save the card
        card_path = out_dir / f"card_{i+1}_updated.png"
        cv2.imwrite(str(card_path), card_img)
        
        print(f"Card {i+1}: {card_img.shape[1]}x{card_img.shape[0]} pixels")
        print(f"  Saved to: {card_path}")
        
        # Extract elixir region from bottom center of card
        elixir_roi = calib['elixir_roi']
        elixir_x1 = int(elixir_roi['x_off_r'] * card_img.shape[1])
        elixir_y1 = int(elixir_roi['y_off_r'] * card_img.shape[0])
        elixir_x2 = elixir_x1 + int(elixir_roi['w_r'] * card_img.shape[1])
        elixir_y2 = elixir_y1 + int(elixir_roi['h_r'] * card_img.shape[0])
        
        elixir_region = card_img[elixir_y1:elixir_y2, elixir_x1:elixir_x2]
        
        # Save elixir region
        elixir_path = out_dir / f"elixir_{i+1}_updated.png"
        cv2.imwrite(str(elixir_path), elixir_region)
        
        print(f"  Elixir region: {elixir_region.shape[1]}x{elixir_region.shape[0]} pixels")
        print(f"  Saved to: {elixir_path}")
    
    # Also save the card row for reference
    roi = img[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
    # Convert absolute coordinates to ROI-relative coordinates
    row_top_roi = row_top_y - vp_y
    row_bottom_roi = row_bottom_y - vp_y
    row_img = roi[row_top_roi:row_bottom_roi, :]
    cv2.imwrite(str(out_dir / "card_row_updated.png"), row_img)
    print(f"\nCard row: {row_img.shape[1]}x{row_img.shape[0]} pixels")
    print(f"  Saved to: {out_dir / 'card_row_updated.png'}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test updated calibration")
    parser.add_argument("--image", default="calibration/test_screenshot.png", help="Original screenshot")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration JSON file")
    parser.add_argument("--out", default="updated_cards", help="Output directory for updated cards")
    
    args = parser.parse_args()
    
    try:
        # Generate updated overlay
        overlay_path = "updated_calibration_overlay.png"
        draw_calibration_overlay(args.image, args.calib, overlay_path)
        
        # Extract updated cards and elixir regions
        extract_cards_with_new_calibration(args.image, args.calib, args.out)
        
        print(f"\n=== Updated Calibration Test Complete ===")
        print(f"Check {overlay_path} to see the updated card boxes and elixir regions")
        print(f"Check {args.out}/ for the new card images and elixir regions")
        print("\nLegend:")
        print("  Yellow rectangle = Game viewport")
        print("  Green lines = Card row boundaries") 
        print("  Red lines + circles = Card centers")
        print("  Purple rectangles = Individual card boxes (now bigger)")
        print("  Cyan rectangles = Elixir regions (now bottom center)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
