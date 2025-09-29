#!/usr/bin/env python3
"""
Template improvement tool for better card recognition.
"""

import cv2
import numpy as np
import json
import pyautogui
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class TemplateImprover:
    def __init__(self, calibration_file="cv_out/calibration_manual_fixed.json"):
        self.calib = self._load_calibration(calibration_file)
        self.templates_dir = Path("templates/cards")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Template Improver Ready!")
        print(f"✅ Templates directory: {self.templates_dir}")
    
    def _load_calibration(self, calibration_file):
        """Load calibration data."""
        with open(calibration_file) as f:
            return json.load(f)
    
    def extract_cards_from_screen(self, screenshot=None):
        """Extract cards from screen using calibration data."""
        if screenshot is None:
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
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
    
    def auto_crop_content(self, image: np.ndarray, edge_thresh: float = 10.0, margin: int = 2) -> np.ndarray:
        """Automatically crop away uniform padding using edge energy."""
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Sobel to capture edges
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)

        # Sum energy along axes
        row_energy = mag.sum(axis=1)
        col_energy = mag.sum(axis=0)

        # Find bounds where energy exceeds threshold
        def find_bounds(energy: np.ndarray) -> Tuple[int, int]:
            if energy.size == 0:
                return 0, 0
            max_e = float(energy.max())
            if max_e <= 1e-6:
                return 0, energy.size
            thresh = max_e * 0.02  # 2% of max energy
            indices = np.where(energy > max(thresh, edge_thresh))[0]
            if indices.size == 0:
                return 0, energy.size
            start = max(int(indices[0]) - margin, 0)
            end = min(int(indices[-1]) + 1 + margin, energy.size)
            return start, end

        r0, r1 = find_bounds(row_energy)
        c0, c1 = find_bounds(col_energy)

        cropped = image[r0:r1, c0:c1]
        return cropped if cropped.size > 0 else image
    
    def create_focused_template(self, card_img, card_name):
        """Create a focused template by cropping to the central artwork area."""
        print(f"\n=== Creating Focused Template for {card_name} ===")
        
        # Show original card
        cv2.imshow(f"Original {card_name}", card_img)
        cv2.waitKey(1)
        
        # Auto-crop to remove padding
        cropped = self.auto_crop_content(card_img)
        cv2.imshow(f"Auto-cropped {card_name}", cropped)
        cv2.waitKey(1)
        
        print("Instructions:")
        print("1. Look at the auto-cropped version")
        print("2. If it looks good, press 'y' to save")
        print("3. If you want to manually crop, press 'm'")
        print("4. If you want to skip, press 's'")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                # Save auto-cropped version
                template_path = self.templates_dir / f"{card_name}.png"
                cv2.imwrite(str(template_path), cropped)
                print(f"💾 Saved auto-cropped template: {template_path}")
                break
            elif key == ord('m'):
                # Manual cropping
                manual_cropped = self.manual_crop(card_img, card_name)
                if manual_cropped is not None:
                    template_path = self.templates_dir / f"{card_name}.png"
                    cv2.imwrite(str(template_path), manual_cropped)
                    print(f"💾 Saved manually cropped template: {template_path}")
                break
            elif key == ord('s'):
                print(f"Skipped {card_name}")
                break
            else:
                print("Please press 'y', 'm', or 's'")
        
        cv2.destroyAllWindows()
    
    def manual_crop(self, image, card_name):
        """Allow user to manually select crop region."""
        print(f"\nManual cropping for {card_name}")
        print("Click and drag to select the crop region, then press 'Enter'")
        
        # Create a copy for cropping
        crop_img = image.copy()
        roi = cv2.selectROI(f"Select crop region for {card_name}", crop_img, False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:  # Check if valid selection
            x, y, w, h = roi
            cropped = image[y:y+h, x:x+w]
            cv2.imshow(f"Cropped {card_name}", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return cropped
        else:
            print("No valid selection made")
            return None
    
    def improve_existing_templates(self):
        """Improve existing templates by re-cropping them."""
        print(f"\n=== Improving Existing Templates ===")
        
        template_files = list(self.templates_dir.glob("*.png"))
        
        if not template_files:
            print("❌ No template files found")
            return
        
        print(f"Found {len(template_files)} existing templates:")
        for i, template_file in enumerate(template_files):
            print(f"  {i+1}. {template_file.stem}")
        
        print("\nSelect templates to improve (comma-separated numbers, or 'all'): ", end="")
        choice = input().strip()
        
        if choice.lower() == 'all':
            selected_files = template_files
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected_files = [template_files[i] for i in indices if 0 <= i < len(template_files)]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
        
        for template_file in selected_files:
            card_name = template_file.stem
            print(f"\n--- Improving {card_name} ---")
            
            # Load existing template
            existing_template = cv2.imread(str(template_file))
            if existing_template is not None:
                self.create_focused_template(existing_template, card_name)
            else:
                print(f"❌ Could not load {template_file}")
    
    def collect_new_templates(self):
        """Collect new templates with better cropping."""
        print(f"\n=== Collecting New Templates ===")
        print("Make sure you have the cards you want to collect in your hand!")
        print("Press Enter when ready...")
        input()
        
        # Capture current hand
        cards = self.extract_cards_from_screen()
        
        if not cards:
            print("❌ No cards detected")
            return
        
        print(f"Detected {len(cards)} cards in your hand")
        
        # Process each card
        for card_id, card_img in cards.items():
            card_number = int(card_id.split("_")[1])
            
            # Get card name from user
            while True:
                card_name = input(f"\nEnter name for card {card_number} (or 'skip'): ").strip()
                
                if card_name.lower() == 'skip':
                    print(f"Skipped card {card_number}")
                    break
                
                if not card_name:
                    print("Please enter a valid card name")
                    continue
                
                # Create focused template
                self.create_focused_template(card_img, card_name)
                break
    
    def compare_templates(self, card_name1, card_name2):
        """Compare two templates to see why they might be confused."""
        template1_path = self.templates_dir / f"{card_name1}.png"
        template2_path = self.templates_dir / f"{card_name2}.png"
        
        if not template1_path.exists() or not template2_path.exists():
            print("❌ One or both templates not found")
            return
        
        template1 = cv2.imread(str(template1_path))
        template2 = cv2.imread(str(template2_path))
        
        if template1 is None or template2 is None:
            print("❌ Could not load templates")
            return
        
        print(f"\n=== Comparing {card_name1} vs {card_name2} ===")
        
        # Show both templates
        cv2.imshow(f"{card_name1}", template1)
        cv2.imshow(f"{card_name2}", template2)
        
        # Resize to same size for comparison
        h1, w1 = template1.shape[:2]
        h2, w2 = template2.shape[:2]
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        resized1 = cv2.resize(template1, (target_w, target_h))
        resized2 = cv2.resize(template2, (target_w, target_h))
        
        # Show side by side
        comparison = np.hstack([resized1, resized2])
        cv2.imshow(f"Comparison: {card_name1} | {card_name2}", comparison)
        
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Calculate similarity
        gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        print(f"Similarity score: {max_val:.3f}")
        if max_val > 0.7:
            print("⚠️  High similarity - these templates might be confused!")
        elif max_val > 0.5:
            print("⚠️  Medium similarity - potential confusion")
        else:
            print("✅ Low similarity - should be distinguishable")

def main():
    """Main function for template improvement."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale template improver")
    parser.add_argument("--mode", choices=["improve", "collect", "compare"], default="improve", help="Operation mode")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration file")
    parser.add_argument("--card1", help="First card for comparison")
    parser.add_argument("--card2", help="Second card for comparison")
    
    args = parser.parse_args()
    
    # Create improver
    improver = TemplateImprover(args.calib)
    
    if args.mode == "improve":
        improver.improve_existing_templates()
    elif args.mode == "collect":
        improver.collect_new_templates()
    elif args.mode == "compare":
        if args.card1 and args.card2:
            improver.compare_templates(args.card1, args.card2)
        else:
            print("❌ Please provide --card1 and --card2 for comparison")

if __name__ == "__main__":
    main()
