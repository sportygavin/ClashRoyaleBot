#!/usr/bin/env python3
"""
Interactive manual calibration tool.

Click to define:
1. Viewport corners (top-left, bottom-right)
2. Card row boundaries (top, bottom)
3. Card centers (4 clicks)

This will generate a corrected calibration.json.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional

class ManualCalibrator:
    def __init__(self, image_path: str):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.H, self.W = self.img.shape[:2]
        self.display_img = self.img.copy()
        
        # Calibration data
        self.viewport_tl: Optional[Tuple[int, int]] = None
        self.viewport_br: Optional[Tuple[int, int]] = None
        self.card_row_top: Optional[int] = None
        self.card_row_bottom: Optional[int] = None
        self.card_centers: List[int] = []
        
        # State
        self.state = "viewport_tl"  # viewport_tl -> viewport_br -> card_row_top -> card_row_bottom -> card_centers
        self.current_card = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "viewport_tl":
                self.viewport_tl = (x, y)
                self.state = "viewport_br"
                print(f"Viewport top-left: ({x}, {y})")
                print("Click bottom-right corner of game viewport...")
                
            elif self.state == "viewport_br":
                self.viewport_br = (x, y)
                self.state = "card_row_top"
                print(f"Viewport bottom-right: ({x}, {y})")
                print("Click top edge of card row...")
                
            elif self.state == "card_row_top":
                self.card_row_top = y
                self.state = "card_row_bottom"
                print(f"Card row top: y={y}")
                print("Click bottom edge of card row...")
                
            elif self.state == "card_row_bottom":
                self.card_row_bottom = y
                self.state = "card_centers"
                print(f"Card row bottom: y={y}")
                print("Click center of card 1 (leftmost)...")
                
            elif self.state == "card_centers":
                if self.current_card < 4:
                    self.card_centers.append(x)
                    self.current_card += 1
                    if self.current_card < 4:
                        print(f"Card {self.current_card} center: x={x}")
                        print(f"Click center of card {self.current_card + 1}...")
                    else:
                        print(f"Card 4 center: x={x}")
                        print("Calibration complete! Press 's' to save.")
            
            self.redraw()
    
    def redraw(self):
        """Redraw the display with current calibration data."""
        self.display_img = self.img.copy()
        
        # Draw viewport
        if self.viewport_tl and self.viewport_br:
            cv2.rectangle(self.display_img, self.viewport_tl, self.viewport_br, (0, 255, 255), 3)
        
        # Draw card row
        if self.viewport_tl and self.viewport_br and self.card_row_top is not None:
            cv2.line(self.display_img, (self.viewport_tl[0], self.card_row_top), 
                    (self.viewport_br[0], self.card_row_top), (0, 255, 0), 2)
        
        if self.viewport_tl and self.viewport_br and self.card_row_bottom is not None:
            cv2.line(self.display_img, (self.viewport_tl[0], self.card_row_bottom), 
                    (self.viewport_br[0], self.card_row_bottom), (0, 255, 0), 2)
        
        # Draw card centers
        for i, center_x in enumerate(self.card_centers):
            if self.card_row_top is not None and self.card_row_bottom is not None:
                cv2.line(self.display_img, (center_x, self.card_row_top), 
                        (center_x, self.card_row_bottom), (255, 0, 0), 2)
                center_y = (self.card_row_top + self.card_row_bottom) // 2
                cv2.circle(self.display_img, (center_x, center_y), 8, (0, 0, 255), -1)
                cv2.putText(self.display_img, f"C{i+1}", (center_x - 15, center_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def save_calibration(self, output_path: str):
        """Save calibration to JSON file."""
        if not all([self.viewport_tl, self.viewport_br, self.card_row_top is not None, 
                   self.card_row_bottom is not None, len(self.card_centers) == 4]):
            print("Incomplete calibration!")
            return
        
        # Calculate ratios
        vp_x, vp_y = self.viewport_tl
        vp_w = self.viewport_br[0] - self.viewport_tl[0]
        vp_h = self.viewport_br[1] - self.viewport_tl[1]
        
        calib = {
            "metadata": {
                "image_W": self.W,
                "image_H": self.H,
                "created_at": "2025-09-29T18:00:00.000000Z",
                "notes": "Manual calibration - corrected viewport and card positions"
            },
            "viewport": {
                "x_r": vp_x / self.W,
                "y_r": vp_y / self.H,
                "w_r": vp_w / self.W,
                "h_r": vp_h / self.H
            },
            "card_row": {
                "top_r": (self.card_row_top - vp_y) / vp_h,
                "bottom_r": (self.card_row_bottom - vp_y) / vp_h
            },
            "cards": {
                "centers_x_r": [(cx - vp_x) / vp_w for cx in self.card_centers],
                "width_r": 0.12,  # You can adjust this
                "top_offset_r": 0.1,
                "bottom_offset_r": 0.1
            },
            "elixir_roi": {
                "x_off_r": 0.08,
                "y_off_r": 0.62,
                "w_r": 0.22,
                "h_r": 0.28
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"Calibration saved to: {output_path}")
        print(f"Viewport: ({vp_x}, {vp_y}) {vp_w}x{vp_h}")
        print(f"Card row: top={self.card_row_top}, bottom={self.card_row_bottom}")
        print(f"Card centers: {self.card_centers}")
    
    def run(self):
        """Run the interactive calibration."""
        cv2.namedWindow('Manual Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Manual Calibration', self.mouse_callback)
        
        print("=== Manual Calibration ===")
        print("Click top-left corner of game viewport...")
        
        while True:
            cv2.imshow('Manual Calibration', self.display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and self.state == "card_centers" and len(self.card_centers) == 4:
                self.save_calibration('cv_out/calibration_manual.json')
                break
            elif key == ord('r'):
                # Reset
                self.__init__('test_screenshot.png')
                print("Reset! Click top-left corner of game viewport...")
        
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manual calibration tool")
    parser.add_argument("--image", default="test_screenshot.png", help="Screenshot to calibrate")
    
    args = parser.parse_args()
    
    try:
        calibrator = ManualCalibrator(args.image)
        calibrator.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
