#!/usr/bin/env python3
"""
Real-time Card and Elixir Monitor

This script integrates with your existing bot infrastructure to provide
real-time monitoring of cards and elixir every second.

Usage:
  python3 tools/realtime_monitor.py --calib cv_out/calibration_manual_fixed.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import sys
import os

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.vision.game_vision import ClashRoyaleVision
    from core import GameState
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Warning: Could not import vision system. Install dependencies or check imports.")

class RealTimeCardMonitor:
    def __init__(self, calibration_path: str):
        """Initialize with calibration data and vision system."""
        with open(calibration_path, 'r') as f:
            self.calib = json.load(f)
        
        # Initialize vision system if available
        if VISION_AVAILABLE:
            self.vision = ClashRoyaleVision()
        else:
            self.vision = None
        
        self.last_update = 0
        self.update_interval = 1.0  # seconds
        self.last_state = {}
    
    def extract_cards_from_screenshot(self, screenshot: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract individual card images from screenshot using calibration."""
        H, W = screenshot.shape[:2]
        
        # Get viewport
        vp = self.calib['viewport']
        vp_x = int(vp['x_r'] * W)
        vp_y = int(vp['y_r'] * H)
        vp_w = int(vp['w_r'] * W)
        vp_h = int(vp['h_r'] * H)
        
        roi = screenshot[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
        
        # Extract card row
        row = self.calib['card_row']
        row_top = int(row['top_r'] * vp_h)
        row_bottom = int(row['bottom_r'] * vp_h)
        
        # Extract individual cards
        centers = self.calib['cards']['centers_x_r']
        card_width_r = self.calib['cards']['width_r']
        top_offset_r = self.calib['cards']['top_offset_r']
        bottom_offset_r = self.calib['cards']['bottom_offset_r']
        
        cards = {}
        for i, center_x_r in enumerate(centers):
            center_x = int(center_x_r * vp_w)
            card_w = int(card_width_r * vp_w)
            card_x1 = center_x - card_w // 2
            card_x2 = center_x + card_w // 2
            
            card_top = row_top + int(top_offset_r * (row_bottom - row_top))
            card_bottom = row_bottom - int(bottom_offset_r * (row_bottom - row_top))
            
            card_img = roi[card_top:card_bottom, card_x1:card_x2]
            
            if card_img.size > 0:
                cards[f"card_{i+1}"] = card_img
        
        return cards
    
    def detect_elixir_cost_simple(self, card_img: np.ndarray) -> Optional[int]:
        """Fixed elixir cost detection using template matching."""
        h, w = card_img.shape[:2]
        
        # Extract elixir region
        elixir_roi = self.calib['elixir_roi']
        x1 = int(elixir_roi['x_off_r'] * w)
        y1 = int(elixir_roi['y_off_r'] * h)
        x2 = int((elixir_roi['x_off_r'] + elixir_roi['w_r']) * w)
        y2 = int((elixir_roi['y_off_r'] + elixir_roi['h_r']) * h)
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        elixir_region = card_img[y1:y2, x1:x2]
        
        if elixir_region.size == 0:
            return None
        
        # Use the fixed detection method
        return self._recognize_elixir_with_templates(elixir_region)
    
    def _recognize_elixir_with_templates(self, elixir_img: np.ndarray) -> Optional[int]:
        """Recognize elixir cost using template matching."""
        # Load templates (this should be cached in __init__ for efficiency)
        templates = self._load_templates()
        mapping = self._get_elixir_mapping()
        
        if not templates:
            return None
        
        # Extract digit region
        digit_region = self._extract_digit_region(elixir_img)
        if digit_region is None:
            return None
        
        # Find best template match
        best_match = None
        best_score = 0
        
        for template_name, template in templates.items():
            result = cv2.matchTemplate(digit_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = template_name
        
        # Convert to elixir cost
        if best_match and best_score > 0.1:
            return mapping.get(best_match, None)
        
        return None
    
    def _load_templates(self) -> dict:
        """Load digit templates."""
        templates = {}
        for i in range(1, 5):
            template_path = f"elixir_analysis/template_card_{i}.png"
            try:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[f"card_{i}"] = template
            except:
                pass
        return templates
    
    def _get_elixir_mapping(self) -> dict:
        """Get mapping from templates to elixir costs."""
        return {
            "card_1": 4,
            "card_2": 3, 
            "card_3": 4,
            "card_4": 2
        }
    
    def _extract_digit_region(self, elixir_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract digit region from elixir image."""
        # Convert to grayscale
        if len(elixir_img.shape) == 3:
            gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = elixir_img
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        
        # Resize to standard size
        return cv2.resize(digit_region, (20, 30))
    
    def detect_current_elixir(self, screenshot: np.ndarray) -> Optional[float]:
        """Detect current elixir amount from screenshot."""
        H, W = screenshot.shape[:2]
        
        # Get viewport
        vp = self.calib['viewport']
        vp_x = int(vp['x_r'] * W)
        vp_y = int(vp['y_r'] * H)
        vp_w = int(vp['w_r'] * W)
        vp_h = int(vp['h_r'] * H)
        
        # Current elixir region - adjust these coordinates
        elixir_x = vp_x + int(0.05 * vp_w)  # Near left edge
        elixir_y = vp_y + int(0.92 * vp_h)  # Near bottom
        elixir_w = int(0.12 * vp_w)  # Small region
        elixir_h = int(0.06 * vp_h)  # Small height
        
        elixir_region = screenshot[elixir_y:elixir_y+elixir_h, elixir_x:elixir_x+elixir_w]
        
        if elixir_region.size == 0:
            return None
        
        # For now, return None - would need OCR for decimal numbers
        return None
    
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot using available methods."""
        if self.vision:
            try:
                # Use the existing vision system
                screen = self.vision.capture_screen()
                if screen is not None:
                    return screen
            except Exception as e:
                print(f"Vision system error: {e}")
        
        # Fallback: try to capture using pyautogui
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except ImportError:
            print("pyautogui not available")
        except Exception as e:
            print(f"Screenshot capture error: {e}")
        
        return None
    
    def analyze_current_state(self) -> Dict:
        """Analyze current game state."""
        screenshot = self.capture_screenshot()
        
        if screenshot is None:
            return {"error": "Could not capture screenshot"}
        
        # Check if we're in game
        if self.vision:
            try:
                game_state = self.vision.detect_game_state(screenshot)
                if game_state != GameState.IN_GAME:
                    return {"game_state": str(game_state), "message": "Not in game"}
            except Exception as e:
                print(f"Game state detection error: {e}")
        
        # Extract cards
        cards = self.extract_cards_from_screenshot(screenshot)
        
        # Analyze each card
        card_analysis = {}
        for card_id, card_img in cards.items():
            elixir_cost = self.detect_elixir_cost_simple(card_img)
            card_analysis[card_id] = {
                "elixir_cost": elixir_cost,
                "image_shape": card_img.shape
            }
        
        # Detect current elixir
        current_elixir = self.detect_current_elixir(screenshot)
        
        return {
            "timestamp": time.time(),
            "cards": card_analysis,
            "current_elixir": current_elixir,
            "screenshot_shape": screenshot.shape
        }
    
    def should_update(self) -> bool:
        """Check if it's time for an update."""
        return time.time() - self.last_update >= self.update_interval
    
    def update(self) -> Dict:
        """Perform update if needed."""
        if self.should_update():
            self.last_update = time.time()
            return self.analyze_current_state()
        return {}
    
    def print_state(self, state: Dict):
        """Print the current state in a nice format."""
        if "error" in state:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {state['error']}")
            return
        
        if "message" in state:
            print(f"[{time.strftime('%H:%M:%S')}] {state['message']}")
            return
        
        timestamp = time.strftime('%H:%M:%S')
        cards = state.get('cards', {})
        current_elixir = state.get('current_elixir', 'Unknown')
        
        # Format card info
        card_info = []
        for card_id, card_data in cards.items():
            elixir = card_data.get('elixir_cost', '?')
            card_info.append(f"{card_id}:{elixir}")
        
        print(f"[{timestamp}] Elixir: {current_elixir} | Cards: {' '.join(card_info)}")
    
    def run_monitoring(self, duration: Optional[int] = None):
        """Run real-time monitoring."""
        start_time = time.time()
        
        print("=== Real-time Card and Elixir Monitor ===")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                if self.should_update():
                    state = self.update()
                    self.print_state(state)
                    self.last_state = state
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print(f"\nMonitoring completed after {duration} seconds")
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Real-time card and elixir monitoring")
    parser.add_argument("--calib", required=True, help="Calibration JSON file")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    parser.add_argument("--duration", type=int, help="Monitoring duration in seconds (default: infinite)")
    parser.add_argument("--test", action="store_true", help="Test with single screenshot")
    
    args = parser.parse_args()
    
    monitor = RealTimeCardMonitor(args.calib)
    monitor.update_interval = args.interval
    
    if args.test:
        print("=== Single Test ===")
        state = monitor.analyze_current_state()
        monitor.print_state(state)
    else:
        monitor.run_monitoring(args.duration)

if __name__ == "__main__":
    main()
