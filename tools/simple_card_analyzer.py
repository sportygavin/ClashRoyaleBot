#!/usr/bin/env python3
"""
Simple Card and Elixir Analyzer using OpenCV

This script uses OpenCV's built-in capabilities to detect:
1. Elixir costs from card images
2. Current elixir amount from screenshot
3. Real-time monitoring capabilities

No external OCR dependencies required.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

class SimpleCardAnalyzer:
    def __init__(self, calibration_path: str):
        """Initialize with calibration data."""
        with open(calibration_path, 'r') as f:
            self.calib = json.load(f)
    
    def extract_elixir_region(self, card_img: np.ndarray) -> np.ndarray:
        """Extract the elixir cost region from a card image (bottom center)."""
        h, w = card_img.shape[:2]
        
        # Elixir region is now in the bottom center of the card
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
    
    def preprocess_for_digit_detection(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for digit detection."""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_digit_from_contour(self, contour: np.ndarray, img_shape: Tuple[int, int]) -> Optional[int]:
        """Try to identify a digit from contour properties."""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size and aspect ratio
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Digits typically have certain properties
        if area < 20 or area > 500:  # Too small or too large
            return None
        
        if aspect_ratio < 0.2 or aspect_ratio > 2.0:  # Wrong shape
            return None
        
        if h < 8 or w < 5:  # Too small
            return None
        
        # Extract the digit region
        digit_region = img_shape[y:y+h, x:x+w]
        
        # Analyze the digit region
        white_pixels = np.sum(digit_region == 255)
        total_pixels = digit_region.size
        white_ratio = white_pixels / total_pixels
        
        # Very basic heuristic - this is where you'd add actual digit recognition
        # For now, we'll use a simple approach based on region properties
        
        # Count horizontal and vertical lines (very basic digit recognition)
        horizontal_lines = 0
        vertical_lines = 0
        
        # Check for horizontal lines
        for i in range(h):
            row = digit_region[i, :]
            if np.sum(row == 255) > w * 0.6:  # Mostly white line
                horizontal_lines += 1
        
        # Check for vertical lines
        for j in range(w):
            col = digit_region[:, j]
            if np.sum(col == 255) > h * 0.6:  # Mostly white line
                vertical_lines += 1
        
        # Very basic digit classification based on line counts
        # This is extremely simplified - real digit recognition would be much more complex
        if horizontal_lines >= 2 and vertical_lines >= 1:
            return 8  # Complex digit
        elif horizontal_lines >= 1 and vertical_lines >= 1:
            return 4  # Medium complexity
        elif horizontal_lines >= 1:
            return 1  # Simple digit
        else:
            return 3  # Default guess
    
    def detect_elixir_cost(self, card_img: np.ndarray) -> Optional[int]:
        """Detect elixir cost using contour analysis."""
        elixir_region = self.extract_elixir_region(card_img)
        
        if elixir_region.size == 0:
            return None
        
        # Preprocess for digit detection
        processed = self.preprocess_for_digit_detection(elixir_region)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to identify digits from contours
        for contour in contours:
            digit = self.detect_digit_from_contour(contour, processed)
            if digit is not None:
                return digit
        
        return None
    
    def detect_current_elixir(self, screenshot: np.ndarray) -> Optional[float]:
        """Detect current elixir amount from full screenshot."""
        H, W = screenshot.shape[:2]
        
        # Get viewport
        vp = self.calib['viewport']
        vp_x = int(vp['x_r'] * W)
        vp_y = int(vp['y_r'] * H)
        vp_w = int(vp['w_r'] * W)
        vp_h = int(vp['h_r'] * H)
        
        # Current elixir region - adjust these coordinates based on your screenshot
        elixir_x = vp_x + int(0.05 * vp_w)  # Near left edge
        elixir_y = vp_y + int(0.92 * vp_h)  # Near bottom
        elixir_w = int(0.12 * vp_w)  # Small region
        elixir_h = int(0.06 * vp_h)  # Small height
        
        elixir_region = screenshot[elixir_y:elixir_y+elixir_h, elixir_x:elixir_x+elixir_w]
        
        if elixir_region.size == 0:
            return None
        
        # For now, return None - would need more sophisticated digit recognition
        # This would involve detecting decimal numbers like "4.2"
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
                    
                    results[f"card_{i}"] = {
                        "elixir_cost": elixir_cost,
                        "image_shape": card_img.shape
                    }
                    
                    # Save elixir region for debugging
                    elixir_region = self.extract_elixir_region(card_img)
                    cv2.imwrite(str(cards_path / f"elixir_{i}_processed.png"), 
                               self.preprocess_for_digit_detection(elixir_region))
        
        return results

class RealTimeMonitor:
    """Real-time monitoring system for cards and elixir."""
    
    def __init__(self, calibration_path: str):
        self.analyzer = SimpleCardAnalyzer(calibration_path)
        self.last_update = 0
        self.update_interval = 1.0  # seconds
    
    def capture_screenshot(self) -> np.ndarray:
        """Capture screenshot - integrate with your existing system."""
        # This would integrate with your existing screen capture
        # For now, return None
        return None
    
    def analyze_current_state(self, screenshot: np.ndarray) -> Dict:
        """Analyze current game state."""
        if screenshot is None:
            return {}
        
        # Extract cards from screenshot using calibration
        H, W = screenshot.shape[:2]
        vp = self.analyzer.calib['viewport']
        vp_x = int(vp['x_r'] * W)
        vp_y = int(vp['y_r'] * H)
        vp_w = int(vp['w_r'] * W)
        vp_h = int(vp['h_r'] * H)
        
        roi = screenshot[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
        
        # Extract card row
        row = self.analyzer.calib['card_row']
        row_top = int(row['top_r'] * vp_h)
        row_bottom = int(row['bottom_r'] * vp_h)
        
        card_row = roi[row_top:row_bottom, :]
        
        # Extract individual cards
        centers = self.analyzer.calib['cards']['centers_x_r']
        card_width_r = self.analyzer.calib['cards']['width_r']
        top_offset_r = self.analyzer.calib['cards']['top_offset_r']
        bottom_offset_r = self.analyzer.calib['cards']['bottom_offset_r']
        
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
                elixir_cost = self.analyzer.detect_elixir_cost(card_img)
                cards[f"card_{i+1}"] = {"elixir_cost": elixir_cost}
        
        # Detect current elixir
        current_elixir = self.analyzer.detect_current_elixir(screenshot)
        
        return {
            "timestamp": time.time(),
            "cards": cards,
            "current_elixir": current_elixir
        }
    
    def should_update(self) -> bool:
        """Check if it's time for an update."""
        return time.time() - self.last_update >= self.update_interval
    
    def update(self) -> Dict:
        """Perform update if needed."""
        if self.should_update():
            screenshot = self.capture_screenshot()
            if screenshot is not None:
                self.last_update = time.time()
                return self.analyze_current_state(screenshot)
        return {}

def main():
    parser = argparse.ArgumentParser(description="Simple card and elixir analysis")
    parser.add_argument("--cards", required=True, help="Directory containing card images")
    parser.add_argument("--calib", required=True, help="Calibration JSON file")
    parser.add_argument("--screenshot", help="Full screenshot for current elixir detection")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time monitoring")
    
    args = parser.parse_args()
    
    analyzer = SimpleCardAnalyzer(args.calib)
    
    print("=== Simple Card Analysis ===")
    results = analyzer.analyze_cards(args.cards)
    
    for card_id, data in results.items():
        print(f"{card_id}:")
        print(f"  Elixir cost: {data['elixir_cost']}")
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
        print("Note: Screenshot capture not implemented yet - integrate with your existing system")
        
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
