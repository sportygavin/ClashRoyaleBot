#!/usr/bin/env python3
"""
Card recognition system for Clash Royale.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyautogui
import time

class CardRecognitionSystem:
    def __init__(self, 
                 calibration_file="cv_out/calibration_manual_fixed.json",
                 database_file="database/clash_royale_cards.json",
                 templates_dir="templates/cards"):
        self.calib = self._load_calibration(calibration_file)
        self.db = self._load_database(database_file)
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load card templates if they exist
        self.card_templates = self._load_card_templates()
        
        print(f"✅ Card Recognition System Ready!")
        print(f"✅ Database: {len(self.db['cards'])} cards")
        print(f"✅ Templates: {len(self.card_templates)} loaded")
    
    def _load_calibration(self, calibration_file):
        """Load calibration data."""
        with open(calibration_file) as f:
            return json.load(f)
    
    def _load_database(self, database_file):
        """Load card database."""
        with open(database_file) as f:
            return json.load(f)
    
    def _load_card_templates(self) -> Dict[str, np.ndarray]:
        """Load card templates from templates directory."""
        templates = {}
        
        if not self.templates_dir.exists():
            print(f"⚠️  Templates directory not found: {self.templates_dir}")
            return templates
        
        for template_file in self.templates_dir.glob("*.png"):
            card_name = template_file.stem
            template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
            if template is not None:
                templates[card_name] = template
                print(f"  Loaded template: {card_name}")
        
        return templates
    
    def _auto_crop_content(self, image: np.ndarray, edge_thresh: float = 10.0, margin: int = 2) -> np.ndarray:
        """Automatically crop away uniform padding (top/bottom/left/right) using edge energy.
        Keeps a small margin to avoid over-cropping rounded borders.
        """
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Sobel to capture edges; more robust than Canny for low-contrast
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)

        # Sum energy along axes
        row_energy = mag.sum(axis=1)
        col_energy = mag.sum(axis=0)

        # Find bounds where energy exceeds a small threshold of the max
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

    def _preprocess_for_match(self, image: np.ndarray) -> np.ndarray:
        """Normalize contrast and suppress color to make template matching more robust."""
        if image is None or image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CLAHE for local contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(gray)
        # Light blur to reduce noise
        norm = cv2.GaussianBlur(norm, (3, 3), 0)
        return norm

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
    
    def extract_elixir_region(self, card_img):
        """Extract elixir region from card image."""
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
        
        return elixir_region if elixir_region.size > 0 else None
    
    def recognize_card_by_template(self, card_img, threshold=0.3):
        """Recognize card using template matching."""
        best_match = None
        best_score = 0.0

        # Focus the card on its central artwork and normalize
        card_cropped = self._auto_crop_content(card_img)
        card_proc = self._preprocess_for_match(card_cropped)

        # Try multiple template scales to be robust to slight size variance
        scales = [0.9, 0.95, 1.0, 1.05, 1.1]

        for card_name, template in self.card_templates.items():
            # Auto-trim padding from the template as well
            tmpl_cropped = self._auto_crop_content(template)
            tmpl_proc_base = self._preprocess_for_match(tmpl_cropped)

            for s in scales:
                h = int(tmpl_proc_base.shape[0] * s)
                w = int(tmpl_proc_base.shape[1] * s)
                if h < 8 or w < 8:
                    continue
                tmpl_proc = cv2.resize(tmpl_proc_base, (w, h), interpolation=cv2.INTER_LINEAR)

                # Template must be <= card image for cv2.matchTemplate
                if tmpl_proc.shape[0] > card_proc.shape[0] or tmpl_proc.shape[1] > card_proc.shape[1]:
                    continue

                result = cv2.matchTemplate(card_proc, tmpl_proc, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    best_match = card_name

        # Always return the best match, even if below threshold
        return best_match, best_score
    
    def recognize_card_by_features(self, card_img):
        """Recognize card using feature matching."""
        # This is a placeholder for more sophisticated feature matching
        # In practice, you'd use SIFT, ORB, or other feature detectors
        
        # For now, return None - feature matching would require more complex implementation
        return None, 0
    
    def recognize_card(self, card_img, method="template"):
        """Recognize card using specified method."""
        if method == "template":
            return self.recognize_card_by_template(card_img)
        elif method == "features":
            return self.recognize_card_by_features(card_img)
        else:
            raise ValueError(f"Unknown recognition method: {method}")
    
    def get_card_info(self, card_name):
        """Get card information from database."""
        for card_id, card_data in self.db['cards'].items():
            if card_data['name'].lower() == card_name.lower():
                return card_data
        return None
    
    def analyze_hand(self, screenshot=None, method="template"):
        """Analyze the current hand and return card information."""
        print(f"\n=== Analyzing Hand ===")
        
        # Extract cards
        cards = self.extract_cards_from_screen(screenshot)
        
        if not cards:
            print("❌ No cards detected")
            return None
        
        hand_analysis = {}
        
        for card_id, card_img in cards.items():
            card_number = int(card_id.split("_")[1])
            
            # Extract elixir region
            elixir_region = self.extract_elixir_region(card_img)
            
            # Recognize card
            card_name, confidence = self.recognize_card(card_img, method)
            
            # Get card info from database
            card_info = self.get_card_info(card_name) if card_name else None
            
            hand_analysis[card_id] = {
                "card_number": card_number,
                "card_name": card_name,
                "confidence": confidence,
                "card_info": card_info,
                "elixir_region": elixir_region
            }
            
            if card_name:
                confidence_indicator = "✓" if confidence >= 0.6 else "?" if confidence >= 0.3 else "⚠"
                print(f"Card {card_number}: {card_name} {confidence_indicator} (confidence: {confidence:.2f})")
                if card_info:
                    print(f"  Elixir Cost: {card_info['elixir_cost']}")
                    print(f"  Rarity: {card_info['rarity']}")
                    print(f"  Type: {card_info['type']}")
            else:
                print(f"Card {card_number}: Unknown (confidence: {confidence:.2f})")
        
        return hand_analysis
    
    def save_card_template(self, card_img, card_name):
        """Save card image as template for future recognition."""
        template_path = self.templates_dir / f"{card_name}.png"
        cv2.imwrite(str(template_path), card_img)
        print(f"💾 Saved template: {template_path}")
        
        # Reload templates
        self.card_templates = self._load_card_templates()
    
    def create_templates_from_hand(self, screenshot=None):
        """Create templates from current hand for future recognition."""
        print(f"\n=== Creating Templates from Hand ===")
        
        cards = self.extract_cards_from_screen(screenshot)
        
        if not cards:
            print("❌ No cards detected")
            return
        
        for card_id, card_img in cards.items():
            card_number = int(card_id.split("_")[1])
            
            # Ask user for card name
            while True:
                card_name = input(f"Enter name for card {card_number} (or 'skip'): ").strip()
                if card_name.lower() == 'skip':
                    break
                if card_name:
                    self.save_card_template(card_img, card_name)
                    break
                print("Please enter a valid card name")
    
    def live_card_monitoring(self, duration=30, method="template"):
        """Monitor cards in real-time."""
        print(f"\n=== Live Card Monitoring ===")
        print(f"Duration: {duration} seconds")
        print(f"Method: {method}")
        print("Press Ctrl+C to stop early")
        print("-" * 50)
        
        start_time = time.time()
        last_hand = None
        
        try:
            while time.time() - start_time < duration:
                # Analyze current hand
                hand_analysis = self.analyze_hand(method=method)
                
                if hand_analysis:
                    # Check if hand changed
                    current_hand = [card['card_name'] for card in hand_analysis.values()]
                    
                    if last_hand != current_hand:
                        print(f"\n🔄 Hand changed at {time.time() - start_time:.1f}s:")
                        for card_id, card_data in hand_analysis.items():
                            card_name = card_data['card_name']
                            confidence = card_data['confidence']
                            if card_name:
                                confidence_indicator = "✓" if confidence >= 0.6 else "?" if confidence >= 0.3 else "⚠"
                                print(f"  {card_id}: {card_name} {confidence_indicator} ({confidence:.2f})")
                            else:
                                print(f"  {card_id}: Unknown ({confidence:.2f})")
                        last_hand = current_hand
                
                time.sleep(1)  # Check every second
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
        
        print(f"\n✅ Monitoring complete!")

def main():
    """Test the card recognition system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale card recognition system")
    parser.add_argument("--mode", choices=["analyze", "create-templates", "monitor"], default="analyze", help="Operation mode")
    parser.add_argument("--method", choices=["template", "features"], default="template", help="Recognition method")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in seconds")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration file")
    parser.add_argument("--db", default="database/clash_royale_cards.json", help="Database file")
    
    args = parser.parse_args()
    
    # Create recognition system
    recognizer = CardRecognitionSystem(args.calib, args.db)
    
    if args.mode == "analyze":
        recognizer.analyze_hand(method=args.method)
    elif args.mode == "create-templates":
        recognizer.create_templates_from_hand()
    elif args.mode == "monitor":
        recognizer.live_card_monitoring(args.duration, args.method)

if __name__ == "__main__":
    main()
