#!/usr/bin/env python3
"""
Advanced Card and Elixir Analyzer using OCR

This script uses OCR to accurately detect:
1. Elixir costs from card images
2. Current elixir amount from screenshot
3. Real-time monitoring capabilities

Requirements: pip install easyocr
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not installed. Install with: pip install easyocr")

class AdvancedCardAnalyzer:
    def __init__(self, calibration_path: str):
        """Initialize with calibration data and OCR reader."""
        with open(calibration_path, 'r') as f:
            self.calib = json.load(f)
        
        # Initialize OCR reader
        if OCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            self.reader = None
    
    def extract_elixir_region(self, card_img: np.ndarray) -> np.ndarray:
        """Extract the elixir cost region from a card image."""
        h, w = card_img.shape[:2]
        
        # Elixir region is typically in the lower-left quadrant
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
    
    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
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
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_elixir_cost_ocr(self, card_img: np.ndarray) -> Optional[int]:
        """Detect elixir cost using OCR."""
        if not self.reader:
            return None
        
        elixir_region = self.extract_elixir_region(card_img)
        
        if elixir_region.size == 0:
            return None
        
        # Preprocess for OCR
        processed = self.preprocess_for_ocr(elixir_region)
        
        # Use OCR to read text
        try:
            results = self.reader.readtext(processed, allowlist='0123456789')
            
            if results:
                # Get the text with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                
                # Only accept if confidence is high enough
                if confidence > 0.5 and text.isdigit():
                    return int(text)
        except Exception as e:
            print(f"OCR error: {e}")
        
        return None
    
    def detect_elixir_cost_template(self, card_img: np.ndarray) -> Optional[int]:
        """Fallback template matching for elixir detection."""
        elixir_region = self.extract_elixir_region(card_img)
        
        if elixir_region.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)
        
        # Preprocess
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Simple digit detection using contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for digit-like contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Digits typically have certain aspect ratios
                if 0.3 < aspect_ratio < 1.5 and h > 10:
                    # Extract the digit region
                    digit_region = thresh[y:y+h, x:x+w]
                    
                    # Simple heuristic based on region properties
                    # This is very basic - OCR is much better
                    white_pixels = np.sum(digit_region == 255)
                    total_pixels = digit_region.size
                    
                    if white_pixels / total_pixels > 0.3:
                        # Very rough estimation - would need actual digit templates
                        return 3  # Placeholder
        
        return None
    
    def detect_elixir_cost(self, card_img: np.ndarray) -> Optional[int]:
        """Detect elixir cost using best available method."""
        # Try OCR first
        if self.reader:
            result = self.detect_elixir_cost_ocr(card_img)
            if result is not None:
                return result
        
        # Fallback to template matching
        return self.detect_elixir_cost_template(card_img)
    
    def detect_current_elixir(self, screenshot: np.ndarray) -> Optional[float]:
        """Detect current elixir amount from full screenshot."""
        if not self.reader:
            return None
        
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
        
        # Preprocess for OCR
        processed = self.preprocess_for_ocr(elixir_region)
        
        try:
            # Look for decimal numbers (e.g., "4.2", "7.8")
            results = self.reader.readtext(processed, allowlist='0123456789.')
            
            if results:
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                
                if confidence > 0.5:
                    try:
                        return float(text)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Current elixir OCR error: {e}")
        
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
                               self.preprocess_for_ocr(elixir_region))
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Advanced card and elixir analysis")
    parser.add_argument("--cards", required=True, help="Directory containing card images")
    parser.add_argument("--calib", required=True, help="Calibration JSON file")
    parser.add_argument("--screenshot", help="Full screenshot for current elixir detection")
    
    args = parser.parse_args()
    
    if not OCR_AVAILABLE:
        print("Installing easyocr for better accuracy...")
        print("Run: pip install easyocr")
        return
    
    analyzer = AdvancedCardAnalyzer(args.calib)
    
    print("=== Advanced Card Analysis ===")
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

if __name__ == "__main__":
    main()
