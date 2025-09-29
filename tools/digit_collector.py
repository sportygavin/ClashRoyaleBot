#!/usr/bin/env python3
"""
Digit collector for elixir recognition.
"""

import cv2
import numpy as np
import json
import pyautogui
from pathlib import Path

class DigitCollector:
    def __init__(self, calibration_file="cv_out/calibration_manual_fixed.json"):
        self.calib = self._load_calibration(calibration_file)
        self.digits_dir = Path("templates/digits")
        self.digits_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Digit Collector Ready!")
        print(f"✅ Digits directory: {self.digits_dir}")
    
    def _load_calibration(self, calibration_file):
        """Load calibration data."""
        with open(calibration_file) as f:
            return json.load(f)
    
    def _get_viewport_roi(self, screenshot: np.ndarray):
        """Return (roi, vp_x, vp_y, vp_w, vp_h) for convenience."""
        viewport = self.calib['viewport']
        vp_x = int(viewport['x_r'] * screenshot.shape[1])
        vp_y = int(viewport['y_r'] * screenshot.shape[0])
        vp_w = int(viewport['w_r'] * screenshot.shape[1])
        vp_h = int(viewport['h_r'] * screenshot.shape[0])
        roi = screenshot[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
        return roi, vp_x, vp_y, vp_w, vp_h
    
    def extract_current_elixir_region(self, screenshot: np.ndarray):
        """Extract the current elixir number region at bottom-left under card 1."""
        roi, _, _, vp_w, vp_h = self._get_viewport_roi(screenshot)
        
        # Use same logic as the recognition system
        if 'current_elixir_roi' in self.calib:
            cfg = self.calib['current_elixir_roi']
            x1 = int(cfg['x_r'] * vp_w)
            y1 = int(cfg['y_r'] * vp_h)
            x2 = int((cfg['x_r'] + cfg['w_r']) * vp_w)
            y2 = int((cfg['y_r'] + cfg['h_r']) * vp_h)
            region = roi[y1:y2, x1:x2]
            return region if region.size > 0 else None
        
        # Fallback: derive from card row and card_1 center
        card_row = self.calib['card_row']
        row_top_y = int(card_row['top_r'] * vp_h)
        row_bottom_y = int(card_row['bottom_r'] * vp_h)
        cards_cfg = self.calib['cards']
        center_x_r = cards_cfg['centers_x_r'][0]
        center_x = int(center_x_r * vp_w)
        card_w = int(cards_cfg['width_r'] * vp_w)
        
        # Define a small box below left side of card 1
        box_w = max(int(card_w * 0.22), 16)
        box_h = max(int((row_bottom_y - row_top_y) * 0.28), 16)
        x1 = max(center_x - card_w // 2 + int(card_w * 0.06), 0)
        y1 = min(row_bottom_y + int(0.02 * vp_h), vp_h - box_h)
        x2 = min(x1 + box_w, vp_w)
        y2 = min(y1 + box_h, vp_h)
        region = roi[y1:y2, x1:x2]
        return region if region.size > 0 else None
    
    def collect_digits_interactive(self):
        """Collect digit templates interactively."""
        print(f"\n=== Interactive Digit Collection ===")
        print("This will help you collect digit templates for elixir recognition.")
        print("Make sure you have different elixir values (0-10) visible!")
        print("-" * 50)
        
        collected_digits = set()
        
        while len(collected_digits) < 11:  # 0-10
            print(f"\nCurrent progress: {sorted(collected_digits)}")
            print("Press Enter when you have a new elixir value visible, or 'done' to finish...")
            user_input = input().strip()
            
            if user_input.lower() == 'done':
                break
            
            # Capture current elixir region
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            elixir_region = self.extract_current_elixir_region(screenshot)
            
            if elixir_region is None:
                print("❌ Could not extract elixir region")
                continue
            
            # Show the region
            cv2.imshow("Current Elixir Region", elixir_region)
            cv2.waitKey(1)
            
            # Ask user for digit value
            while True:
                try:
                    digit = input("What digit is this? (0-10, or 'skip'): ").strip()
                    if digit.lower() == 'skip':
                        break
                    digit_val = int(digit)
                    if 0 <= digit_val <= 10:
                        if digit_val in collected_digits:
                            print(f"Digit {digit_val} already collected. Try a different value.")
                            continue
                        
                        # Save the template
                        template_path = self.digits_dir / f"{digit_val}.png"
                        cv2.imwrite(str(template_path), elixir_region)
                        collected_digits.add(digit_val)
                        print(f"💾 Saved digit {digit_val}: {template_path}")
                        break
                    else:
                        print("Please enter a digit between 0 and 10")
                except ValueError:
                    print("Please enter a valid number")
            
            cv2.destroyAllWindows()
        
        print(f"\n✅ Digit collection complete!")
        print(f"Collected digits: {sorted(collected_digits)}")
        print(f"💾 Templates saved to: {self.digits_dir}")
    
    def create_simple_digits(self):
        """Create simple digit templates programmatically."""
        print(f"\n=== Creating Simple Digit Templates ===")
        
        # Create simple digit images
        for digit in range(11):  # 0-10
            # Create a simple image with the digit
            img = np.ones((32, 24), dtype=np.uint8) * 255  # White background
            
            # Draw the digit
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 0, 0)  # Black
            thickness = 2
            
            text = str(digit)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = (img.shape[0] + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Save the template
            template_path = self.digits_dir / f"{digit}.png"
            cv2.imwrite(str(template_path), img)
            print(f"💾 Created simple digit {digit}: {template_path}")
        
        print(f"\n✅ Simple digit templates created!")
        print(f"💾 Templates saved to: {self.digits_dir}")
    
    def test_digit_recognition(self):
        """Test digit recognition with current templates."""
        print(f"\n=== Testing Digit Recognition ===")
        
        # Load existing templates
        templates = {}
        for template_file in self.digits_dir.glob("*.png"):
            digit = template_file.stem
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                templates[digit] = template
                print(f"  Loaded template: {digit}")
        
        if not templates:
            print("❌ No digit templates found")
            return
        
        print(f"Found {len(templates)} digit templates")
        
        # Capture current elixir region
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        elixir_region = self.extract_current_elixir_region(screenshot)
        
        if elixir_region is None:
            print("❌ Could not extract elixir region")
            return
        
        # Show the region
        cv2.imshow("Elixir Region for Testing", elixir_region)
        cv2.waitKey(1)
        
        # Preprocess the region
        gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # Test against all templates
        best_match = None
        best_score = 0.0
        
        for digit, template in templates.items():
            # Resize template to match region size
            if template.shape[0] > processed.shape[0] or template.shape[1] > processed.shape[1]:
                scale = min(processed.shape[0] / template.shape[0], 
                           processed.shape[1] / template.shape[1])
                new_h = int(template.shape[0] * scale)
                new_w = int(template.shape[1] * scale)
                template_resized = cv2.resize(template, (new_w, new_h))
            else:
                template_resized = template
            
            # Template matching
            result = cv2.matchTemplate(processed, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            print(f"Digit {digit}: {max_val:.3f}")
            
            if max_val > best_score:
                best_score = max_val
                best_match = digit
        
        print(f"\nBest match: {best_match} (confidence: {best_score:.3f})")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main function for digit collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale digit collector")
    parser.add_argument("--mode", choices=["interactive", "simple", "test"], default="interactive", help="Collection mode")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration file")
    
    args = parser.parse_args()
    
    # Create collector
    collector = DigitCollector(args.calib)
    
    if args.mode == "interactive":
        collector.collect_digits_interactive()
    elif args.mode == "simple":
        collector.create_simple_digits()
    elif args.mode == "test":
        collector.test_digit_recognition()

if __name__ == "__main__":
    main()
