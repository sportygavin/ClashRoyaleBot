#!/usr/bin/env python3
"""
Interactive Calibration Editor

Allows you to visually adjust calibration coordinates by clicking on the screen.
Shows current calibration overlays and lets you adjust:
- Viewport boundaries (drag corners)
- Card row boundaries (click top/bottom)
- Card centers (click each card)
- Opponent region (drag rectangle)

Usage:
    python tools/interactive_calibration_editor.py --calib cv_out/calibration.json
"""

import argparse
import cv2
import numpy as np
import pyautogui
import json
import sys
import os
from pathlib import Path
from typing import Tuple, Optional, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import load_calibration, default_viewport, screen_bgr


class InteractiveCalibrationEditor:
    def __init__(self, calib_path: str):
        self.calib_path = calib_path
        self.calib = load_calibration(calib_path)
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Calculate current viewport
        self.viewport = default_viewport(self.calib)
        vx, vy, vw, vh = self.viewport
        
        # Mouse interaction state
        self.dragging = False
        self.drag_type = None  # 'viewport_tl', 'viewport_br', 'card_row_top', 'card_row_bottom', 'card_center', 'opponent'
        self.drag_start = None
        self.selected_card_index = None
        
        # Current calibration values (will be updated)
        self.viewport_tl = (vx, vy)
        self.viewport_br = (vx + vw, vy + vh)
        
        # Card row
        if 'card_row' in self.calib:
            card_row = self.calib['card_row']
            row_top_r = card_row.get('top_r', 0.85)
            row_bottom_r = card_row.get('bottom_r', 1.0)
            self.card_row_top_y = vy + int(row_top_r * vh)
            self.card_row_bottom_y = vy + int(row_bottom_r * vh)
        else:
            self.card_row_top_y = vy + int(0.85 * vh)
            self.card_row_bottom_y = vy + vh
        
        # Card centers
        if 'cards' in self.calib and 'centers_x_r' in self.calib['cards']:
            centers_x_r = self.calib['cards']['centers_x_r']
            self.card_centers = [vx + int(cx_r * vw) for cx_r in centers_x_r]
        else:
            self.card_centers = []
        
        # Opponent region
        if 'opponent_region_roi' in self.calib:
            roi = self.calib['opponent_region_roi']
            self.opponent_x = vx + int(roi['x_r'] * vw)
            self.opponent_y = vy + int(roi['y_r'] * vh)
            self.opponent_w = int(roi['w_r'] * vw)
            self.opponent_h = int(roi['h_r'] * vh)
        else:
            self.opponent_x = vx
            self.opponent_y = vy
            self.opponent_w = vw
            self.opponent_h = vh // 2
        
        # Capture screen
        self.screen = screen_bgr()
        if self.screen is None:
            raise ValueError("Could not capture screen")
        
        self.display_img = self.screen.copy()
        self.redraw()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check what was clicked
            clicked_item = self._get_clicked_item(x, y)
            
            if clicked_item == 'viewport_tl':
                self.dragging = True
                self.drag_type = 'viewport_tl'
                self.drag_start = (x, y)
                print(f"📌 Dragging viewport top-left from ({x}, {y})")
            
            elif clicked_item == 'viewport_br':
                self.dragging = True
                self.drag_type = 'viewport_br'
                self.drag_start = (x, y)
                print(f"📌 Dragging viewport bottom-right from ({x}, {y})")
            
            elif clicked_item == 'card_row_top':
                self.card_row_top_y = y
                print(f"📌 Card row top set to y={y}")
                self.redraw()
            
            elif clicked_item == 'card_row_bottom':
                self.card_row_bottom_y = y
                print(f"📌 Card row bottom set to y={y}")
                self.redraw()
            
            elif clicked_item and clicked_item.startswith('card_'):
                card_idx = int(clicked_item.split('_')[1])
                self.card_centers[card_idx] = x
                print(f"📌 Card {card_idx + 1} center set to x={x}")
                self.redraw()
            
            elif clicked_item == 'opponent':
                self.dragging = True
                self.drag_type = 'opponent'
                self.drag_start = (x, y)
                print(f"📌 Dragging opponent region from ({x}, {y})")
            
            else:
                # Clicked on empty space - check if near viewport
                vx, vy, vw, vh = self.viewport
                if vx <= x <= vx + vw and vy <= y <= vy + vh:
                    # Clicked inside viewport - could be setting card center
                    if len(self.card_centers) < 4:
                        self.card_centers.append(x)
                        print(f"📌 Card {len(self.card_centers)} center set to x={x}")
                        self.redraw()
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Update position while dragging
            if self.drag_type == 'viewport_tl':
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.viewport_tl = (self.viewport_tl[0] + dx, self.viewport_tl[1] + dy)
                self.drag_start = (x, y)
                self.redraw()
            
            elif self.drag_type == 'viewport_br':
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.viewport_br = (self.viewport_br[0] + dx, self.viewport_br[1] + dy)
                self.drag_start = (x, y)
                self.redraw()
            
            elif self.drag_type == 'opponent':
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.opponent_x += dx
                self.opponent_y += dy
                self.drag_start = (x, y)
                self.redraw()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_type = None
            self.drag_start = None
    
    def _get_clicked_item(self, x: int, y: int) -> Optional[str]:
        """Determine what was clicked"""
        click_radius = 15
        
        # Check viewport corners
        if abs(x - self.viewport_tl[0]) < click_radius and abs(y - self.viewport_tl[1]) < click_radius:
            return 'viewport_tl'
        if abs(x - self.viewport_br[0]) < click_radius and abs(y - self.viewport_br[1]) < click_radius:
            return 'viewport_br'
        
        # Check card row boundaries
        if abs(y - self.card_row_top_y) < click_radius:
            vx, vy, vw, vh = self.viewport
            if vx <= x <= vx + vw:
                return 'card_row_top'
        if abs(y - self.card_row_bottom_y) < click_radius:
            vx, vy, vw, vh = self.viewport
            if vx <= x <= vx + vw:
                return 'card_row_bottom'
        
        # Check card centers
        for i, cx in enumerate(self.card_centers):
            if abs(x - cx) < click_radius:
                cy = (self.card_row_top_y + self.card_row_bottom_y) // 2
                if abs(y - cy) < 30:
                    return f'card_{i}'
        
        # Check opponent region
        if (self.opponent_x <= x <= self.opponent_x + self.opponent_w and
            self.opponent_y <= y <= self.opponent_y + self.opponent_h):
            return 'opponent'
        
        return None
    
    def redraw(self):
        """Redraw the display with current calibration"""
        self.display_img = self.screen.copy()
        
        # Update viewport from corners
        vx = min(self.viewport_tl[0], self.viewport_br[0])
        vy = min(self.viewport_tl[1], self.viewport_br[1])
        vw = abs(self.viewport_br[0] - self.viewport_tl[0])
        vh = abs(self.viewport_br[1] - self.viewport_tl[1])
        self.viewport = (vx, vy, vw, vh)
        
        # Draw viewport rectangle (yellow)
        cv2.rectangle(self.display_img, (vx, vy), (vx + vw, vy + vh), (0, 255, 255), 3)
        cv2.putText(self.display_img, "Viewport", (vx, vy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw viewport corners (draggable)
        cv2.circle(self.display_img, self.viewport_tl, 10, (0, 255, 255), -1)
        cv2.circle(self.display_img, self.viewport_br, 10, (0, 255, 255), -1)
        
        # Draw card row (magenta)
        cv2.line(self.display_img, (vx, self.card_row_top_y), (vx + vw, self.card_row_top_y),
                (255, 0, 255), 2)
        cv2.line(self.display_img, (vx, self.card_row_bottom_y), (vx + vw, self.card_row_bottom_y),
                (255, 0, 255), 2)
        cv2.putText(self.display_img, "Card Row", (vx, self.card_row_top_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw card centers (green)
        cy = (self.card_row_top_y + self.card_row_bottom_y) // 2
        for i, cx in enumerate(self.card_centers):
            cv2.circle(self.display_img, (cx, cy), 15, (0, 255, 0), 2)
            cv2.line(self.display_img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(self.display_img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
            cv2.putText(self.display_img, f"C{i+1}", (cx - 15, cy - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw opponent region (cyan)
        cv2.rectangle(self.display_img, 
                     (self.opponent_x, self.opponent_y),
                     (self.opponent_x + self.opponent_w, self.opponent_y + self.opponent_h),
                     (255, 255, 0), 2)
        cv2.putText(self.display_img, "Opponent Region", (self.opponent_x, self.opponent_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "INSTRUCTIONS:",
            "Drag yellow circles to adjust viewport",
            "Click magenta lines to adjust card row",
            "Click green crosshairs to adjust card centers",
            "Drag cyan rectangle to move opponent region",
            "Press 's' to save, 'r' to reset, 'q' to quit"
        ]
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(self.display_img, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(self.display_img, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def save_calibration(self):
        """Save updated calibration to file"""
        vx, vy, vw, vh = self.viewport
        
        # Calculate relative coordinates
        viewport_r = {
            'x_r': vx / self.screen_w,
            'y_r': vy / self.screen_h,
            'w_r': vw / self.screen_w,
            'h_r': vh / self.screen_h
        }
        
        # Card row (relative to viewport)
        card_row = {
            'top_r': (self.card_row_top_y - vy) / vh,
            'bottom_r': (self.card_row_bottom_y - vy) / vh
        }
        
        # Card centers (relative to viewport)
        centers_x_r = [(cx - vx) / vw for cx in self.card_centers]
        
        # Opponent region (relative to viewport)
        opponent_roi = {
            'x_r': (self.opponent_x - vx) / vw,
            'y_r': (self.opponent_y - vy) / vh,
            'w_r': self.opponent_w / vw,
            'h_r': self.opponent_h / vh
        }
        
        # Update calibration
        self.calib['viewport'] = viewport_r
        self.calib['card_row'] = card_row
        if 'cards' not in self.calib:
            self.calib['cards'] = {}
        self.calib['cards']['centers_x_r'] = centers_x_r
        self.calib['opponent_region_roi'] = opponent_roi
        
        # Save to file
        with open(self.calib_path, 'w') as f:
            json.dump(self.calib, f, indent=2)
        
        print(f"\n✅ Calibration saved to {self.calib_path}")
        print(f"   Viewport: ({vx}, {vy}) {vw}x{vh}")
        print(f"   Card row: y={self.card_row_top_y}-{self.card_row_bottom_y}")
        print(f"   Card centers: {self.card_centers}")
        print(f"   Opponent region: ({self.opponent_x}, {self.opponent_y}) {self.opponent_w}x{self.opponent_h}")
    
    def reset(self):
        """Reset to original calibration"""
        self.__init__(self.calib_path)
        print("🔄 Reset to original calibration")
    
    def run(self):
        """Run the interactive editor"""
        cv2.namedWindow('Interactive Calibration Editor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Interactive Calibration Editor', self.mouse_callback)
        
        print("=" * 70)
        print("Interactive Calibration Editor")
        print("=" * 70)
        print("\nControls:")
        print("  - Drag yellow circles to adjust viewport boundaries")
        print("  - Click magenta lines to adjust card row top/bottom")
        print("  - Click green crosshairs to adjust card centers")
        print("  - Drag cyan rectangle to move opponent region")
        print("  - Press 's' to save")
        print("  - Press 'r' to reset")
        print("  - Press 'q' to quit")
        print()
        
        while True:
            cv2.imshow('Interactive Calibration Editor', self.display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_calibration()
            elif key == ord('r'):
                self.reset()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Interactive calibration editor')
    parser.add_argument('--calib', default='cv_out/calibration.json',
                       help='Path to calibration JSON file to edit')
    args = parser.parse_args()
    
    # Resolve path
    if not os.path.isabs(args.calib):
        project_root = Path(__file__).parent.parent
        calib_path = project_root / args.calib
    else:
        calib_path = Path(args.calib)
    
    if not calib_path.exists():
        print(f"❌ Calibration file not found: {calib_path}")
        return
    
    try:
        editor = InteractiveCalibrationEditor(str(calib_path))
        editor.run()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

