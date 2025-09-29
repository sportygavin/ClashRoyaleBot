#!/usr/bin/env python3
"""
Card and Elixir Analyzer

This script analyzes extracted card images to:
1. Detect elixir cost for each card
2. Recognize card types
3. Detect current elixir amount
4. Provide real-time monitoring capabilities

Usage:
  python3 tools/card_analyzer.py --cards calibration/extracted_cards_manual/ --calib cv_out/calibration_manual_fixed.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

class CardAnalyzer:
    def __init__(self, calibration_path: str):
        """Initialize with calibration data."""
        with open(calibration_path, 'r') as f:
            self.calib = json.load(f)
        
        # Load elixir digit templates (we'll create these)
        self.elixir_templates = self._load_elixir_templates()
        
        # Card type templates (we'll create these)
        self.card_templates = self._load_card_templates()
    
    def _load_elixir_templates(self) -> Dict[int, np.ndarray]:
        """Load elixir digit templates (1-9)."""
        templates = {}
        # For now, we'll create simple templates
        # In practice, you'd extract these from clean card images
        for digit in range(1, 10):
            # Create a simple template - in reality, extract from actual cards
            template = np.zeros((20, 15), dtype=np.uint8)
            cv2.putText(template, str(digit), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            templates[digit] = template
        return templates
    
    def _load_card_templates(self) -> Dict[str, np.ndarray]:
        """Load card type templates."""
        # For now, return empty dict - we'll build this from actual cards
        return {}
    
    def extract_elixir_region(self, card_img: np.ndarray) -> np.ndarray:
        """Extract the elixir cost region from a card image."""
        h, w = card_img.shape[:2]
        
        # Elixir region is typically in the lower-left quadrant
        # Based on calibration: x_off_r=0.08, y_off_r=0.62, w_r=0.22, h_r=0.28
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
        
        return card_img[y1:y2, x1:x2]
    
    def detect_elixir_cost(self, card_img: np.ndarray) -> Optional[int]:
        """Detect elixir cost from card image."""
        elixir_region = self.extract_elixir_region(card_img)
        
        if elixir_region.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for better recognition
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try template matching for each digit
        best_match = 0
        best_score = 0
        
        for digit, template in self.elixir_templates.items():
            # Resize template to match elixir region size
            template_resized = cv2.resize(template, (elixir_region.shape[1], elixir_region.shape[0]))
            
            # Template matching
            result = cv2.matchTemplate(thresh, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = digit
        
        # Return digit if confidence is high enough
        if best_score > 0.3:  # Adjust threshold as needed
            return best_match
        
        return None
    
    def detect_card_type(self, card_img: np.ndarray) -> Optional[str]:
        """Detect card type from card image."""
        # For now, return None - we'll implement this with actual card templates
        # This would involve:
        # 1. Extracting the card art region
        # 2. Comparing with known card templates
        # 3. Using feature matching or deep learning
        return None
    
    def analyze_cards(self, cards_dir: str) -> Dict[str, Dict]:
        """Analyze all cards in a directory."""
        cards_path = Path(cards_dir)
        results = {}
        
        for i in range(1, 5):  # Cards 1-4
            card_file = cards_path / f"card_{i}.png"
            if card_file.exists():
                card_img = cv2.imread(str(card_file))
                if card_img is not None:
                    elixir_cost = self.detect_elixir_cost(card_img)
                    card_type = self.detect_card_type(card_img)
                    
                    results[f"card_{i}"] = {
                        "elixir_cost": elixir_cost,
                        "card_type": card_type,
                        "image_shape": card_img.shape
                    }
                    
                    # Save elixir region for debugging
                    elixir_region = self.extract_elixir_region(card_img)
                    cv2.imwrite(str(cards_path / f"elixir_{i}.png"), elixir_region)
        
        return results
    
    def detect_current_elixir(self, screenshot: np.ndarray) -> Optional[float]:
        """Detect current elixir amount from full screenshot."""
        H, W = screenshot.shape[:2]
        
        # Get viewport
        vp = self.calib['viewport']
        vp_x = int(vp['x_r'] * W)
        vp_y = int(vp['y_r'] * H)
        vp_w = int(vp['w_r'] * W)
        vp_h = int(vp['h_r'] * H)
        
        # Current elixir is typically shown at the bottom, under the first card
        # It's usually a number like "4.2" or "7.8"
        # Let's define a region for it
        elixir_x = vp_x + int(0.1 * vp_w)  # Near left edge
        elixir_y = vp_y + int(0.95 * vp_h)  # Near bottom
        elixir_w = int(0.15 * vp_w)  # Small region
        elixir_h = int(0.05 * vp_h)  # Small height
        
        elixir_region = screenshot[elixir_y:elixir_y+elixir_h, elixir_x:elixir_x+elixir_w]
        
        if elixir_region.size == 0:
            return None
        
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # For now, return None - we'd need OCR or template matching for digits
        # This would involve detecting "4.2" style numbers
        return None

class RealTimeMonitor:
    """Real-time monitoring system for cards and elixir."""
    
    def __init__(self, calibration_path: str):
        self.analyzer = CardAnalyzer(calibration_path)
        self.last_update = 0
        self.update_interval = 1.0  # seconds
    
    def capture_and_analyze(self) -> Dict:
        """Capture screenshot and analyze cards/elixir."""
        # This would integrate with your existing screen capture system
        # For now, return mock data
        return {
            "timestamp": time.time(),
            "cards": {
                "card_1": {"elixir_cost": 3, "card_type": "Knight"},
                "card_2": {"elixir_cost": 2, "card_type": "Archers"},
                "card_3": {"elixir_cost": 4, "card_type": "Fireball"},
                "card_4": {"elixir_cost": 5, "card_type": "Giant"}
            },
            "current_elixir": 6.2
        }
    
    def should_update(self) -> bool:
        """Check if it's time for an update."""
        return time.time() - self.last_update >= self.update_interval
    
    def update(self) -> Dict:
        """Perform update if needed."""
        if self.should_update():
            self.last_update = time.time()
            return self.capture_and_analyze()
        return {}

def main():
    parser = argparse.ArgumentParser(description="Analyze cards and elixir")
    parser.add_argument("--cards", required=True, help="Directory containing card images")
    parser.add_argument("--calib", required=True, help="Calibration JSON file")
    parser.add_argument("--screenshot", help="Full screenshot for current elixir detection")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time monitoring")
    
    args = parser.parse_args()
    
    analyzer = CardAnalyzer(args.calib)
    
    print("=== Card Analysis ===")
    results = analyzer.analyze_cards(args.cards)
    
    for card_id, data in results.items():
        print(f"{card_id}:")
        print(f"  Elixir cost: {data['elixir_cost']}")
        print(f"  Card type: {data['card_type']}")
        print(f"  Image shape: {data['image_shape']}")
    
    if args.screenshot:
        print("\n=== Current Elixir Detection ===")
        screenshot = cv2.imread(args.screenshot)
        if screenshot is not None:
            current_elixir = analyzer.detect_current_elixir(screenshot)
            print(f"Current elixir: {current_elixir}")
    
    if args.realtime:
        print("\n=== Real-time Monitoring ===")
        monitor = RealTimeMonitor(args.calib)
        print("Starting real-time monitoring (press Ctrl+C to stop)...")
        
        try:
            while True:
                if monitor.should_update():
                    data = monitor.update()
                    if data:
                        print(f"[{time.strftime('%H:%M:%S')}] Elixir: {data['current_elixir']}, Cards: {[c['elixir_cost'] for c in data['cards'].values()]}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
