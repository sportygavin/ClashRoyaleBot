#!/usr/bin/env python3
"""
Pure visual card detection system that relies only on visual recognition.
"""

import cv2
import numpy as np
import pyautogui
import json
import time
from pathlib import Path
from datetime import datetime

class PureVisualMonitor:
    def __init__(self, calibration_file="cv_out/calibration_manual_fixed.json"):
        self.calib = self._load_calibration(calibration_file)
        self.templates = self._load_all_templates()
        
        print(f"✅ Pure Visual Monitor Ready!")
        print(f"✅ Loaded {len(self.templates)} templates")
        print("✅ No hardcoded values - pure visual recognition")
    
    def _load_calibration(self, calibration_file):
        """Load calibration data."""
        with open(calibration_file) as f:
            return json.load(f)
    
    def _load_all_templates(self):
        """Load all available templates."""
        templates = {}
        
        # Load original templates
        for digit in range(1, 10):
            template_path = f"digit_templates/{digit}.png"
            try:
                original = cv2.imread(template_path)
                if original is not None:
                    processed = self._preprocess_template(original)
                    if processed is not None:
                        templates[f"orig_{digit}"] = processed
            except:
                pass
        
        # Load perfect templates
        for digit in [2, 3, 4]:
            template_path = f"perfect_templates/{digit}.png"
            try:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[f"perfect_{digit}"] = template
            except:
                pass
        
        return templates
    
    def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess template."""
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            digit_region = thresh[y:y+h, x:x+w]
            return cv2.resize(digit_region, (40, 50))
        
        return cv2.resize(thresh, (40, 50))
    
    def capture_and_analyze(self):
        """Capture screen and analyze cards once."""
        # Capture live screen
        screenshot = pyautogui.screenshot()
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Extract cards
        cards = self.extract_cards_from_screen(screenshot_cv)
        
        if not cards:
            return None
        
        # Detect elixir costs using pure visual recognition
        results = {}
        for card_id, card_img in cards.items():
            elixir_cost = self.detect_elixir_cost_pure_visual(card_img)
            results[card_id] = elixir_cost
        
        return results
    
    def extract_cards_from_screen(self, screenshot):
        """Extract cards from screen using calibration data."""
        # Calculate viewport coordinates
        viewport = self.calib['viewport']
        vp_x = int(viewport['x_r'] * screenshot.shape[1])
        vp_y = int(viewport['y_r'] * screenshot.shape[0])
        vp_w = int(viewport['w_r'] * screenshot.shape[1])
        vp_h = int(viewport['h_r'] * screenshot.shape[0])
        
        # Extract viewport
        roi = screenshot[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
        
        # Calculate card row
        card_row = self.calib['card_row']
        row_top_y = int(card_row['top_r'] * vp_h)
        row_bottom_y = int(card_row['bottom_r'] * vp_h)
        
        # Extract card row
        row_img = roi[row_top_y:row_bottom_y, :]
        
        # Extract individual cards
        cards = {}
        cards_config = self.calib['cards']
        
        for i in range(4):
            center_x_r = cards_config['centers_x_r'][i]
            center_x = int(center_x_r * vp_w)
            
            card_w = int(cards_config['width_r'] * vp_w)
            card_h = row_bottom_y - row_top_y
            card_top_offset = int(cards_config['top_offset_r'] * vp_h)
            card_bottom_offset = int(cards_config['bottom_offset_r'] * vp_h)
            
            # Calculate card boundaries
            card_x1 = center_x - card_w // 2
            card_y1 = row_top_y - card_top_offset
            card_x2 = center_x + card_w // 2
            card_y2 = row_bottom_y + card_bottom_offset
            
            # Extract card
            card_img = roi[card_y1:card_y2, card_x1:card_x2]
            
            if card_img.size > 0:
                cards[f"card_{i+1}"] = card_img
        
        return cards
    
    def detect_elixir_cost_pure_visual(self, card_img):
        """Detect elixir cost using pure visual recognition."""
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
        
        # Extract digit region
        digit_region = self._extract_digit_region(elixir_region)
        if digit_region is None:
            return None
        
        # Test against all templates
        scores = {}
        for template_name, template in self.templates.items():
            result = cv2.matchTemplate(digit_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            scores[template_name] = max_val
        
        # Find best visual match
        best_template = max(scores, key=scores.get)
        best_score = scores[best_template]
        
        # Extract digit from template name
        if best_template.startswith("orig_"):
            visual_digit = int(best_template.split("_")[1])
        elif best_template.startswith("perfect_"):
            visual_digit = int(best_template.split("_")[1])
        else:
            return None
        
        # Only return if confidence is high enough
        if best_score > 0.3:  # Lower threshold for pure visual
            return visual_digit
        
        return None
    
    def _extract_digit_region(self, elixir_img: np.ndarray) -> np.ndarray:
        """Extract digit region from elixir image."""
        gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        
        return cv2.resize(digit_region, (40, 50))
    
    def run_continuous_monitor(self, duration_seconds=60):
        """Run continuous monitoring with pure visual recognition."""
        print(f"\n=== PURE VISUAL CARD MONITOR STARTING ===")
        print(f"Duration: {duration_seconds} seconds")
        print("Press Ctrl+C to stop early")
        print("\nMonitoring your cards with pure visual recognition...")
        print("Play cards to see the detection in action!")
        print("-" * 60)
        
        start_time = time.time()
        last_cards = {}
        detection_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                # Capture and analyze
                current_cards = self.capture_and_analyze()
                
                if current_cards is None:
                    print("❌ No cards detected")
                    time.sleep(1)
                    continue
                
                # Check if cards changed
                if current_cards != last_cards:
                    detection_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"\n[{timestamp}] Detection #{detection_count}")
                    print(f"🎯 Cards detected: {current_cards}")
                    
                    # Show changes
                    if last_cards:
                        changes = []
                        for card_id in current_cards:
                            if card_id not in last_cards or current_cards[card_id] != last_cards[card_id]:
                                old_val = last_cards.get(card_id, '?')
                                new_val = current_cards[card_id]
                                changes.append(f"{card_id}: {old_val} → {new_val}")
                        
                        if changes:
                            print(f"🔄 Changes: {', '.join(changes)}")
                    
                    last_cards = current_cards
                
                # Small delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        
        print(f"\n=== MONITORING COMPLETE ===")
        print(f"Total detections: {detection_count}")
        print(f"Duration: {time.time() - start_time:.1f} seconds")

def main():
    """Main function for pure visual monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pure visual card monitor")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration file")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = PureVisualMonitor(args.calib)
    
    # Run monitoring
    monitor.run_continuous_monitor(duration_seconds=args.duration)

if __name__ == "__main__":
    main()
