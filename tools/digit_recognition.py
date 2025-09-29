#!/usr/bin/env python3
"""
Simple digit recognition using uploaded digit templates.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse

class DigitRecognizer:
    def __init__(self, digit_templates_dir="digit_templates"):
        self.digit_templates_dir = Path(digit_templates_dir)
        self.templates = self._load_digit_templates()
    
    def _load_digit_templates(self):
        """Load digit templates (1.png, 2.png, ..., 9.png)"""
        templates = {}
        
        for digit in range(1, 10):
            template_path = self.digit_templates_dir / f"{digit}.png"
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[digit] = template
                    print(f"Loaded template for digit {digit}: {template.shape}")
                else:
                    print(f"Failed to load template for digit {digit}")
            else:
                print(f"Template not found: {template_path}")
        
        return templates
    
    def recognize_digit(self, elixir_img: np.ndarray) -> int:
        """Recognize digit in elixir image using template matching."""
        if not self.templates:
            print("No digit templates loaded!")
            return None
        
        # Preprocess elixir image
        processed_img = self._preprocess_elixir_image(elixir_img)
        
        if processed_img is None:
            return None
        
        # Try matching against each digit template
        best_match = None
        best_score = 0
        
        for digit, template in self.templates.items():
            # Resize template to match processed image size
            resized_template = cv2.resize(template, (processed_img.shape[1], processed_img.shape[0]))
            
            # Template matching
            result = cv2.matchTemplate(processed_img, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = digit
        
        # Return result if confidence is high enough
        if best_match and best_score > 0.3:  # Adjust threshold as needed
            return best_match
        
        return None
    
    def _preprocess_elixir_image(self, elixir_img: np.ndarray) -> np.ndarray:
        """Preprocess elixir image for digit recognition."""
        # Convert to grayscale if needed
        if len(elixir_img.shape) == 3:
            gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = elixir_img
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours to extract digit region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (should be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:  # Filter out noise
            return None
        
        # Extract bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        
        # Resize to standard size for consistent matching
        return cv2.resize(digit_region, (30, 40))

def test_digit_recognition():
    """Test digit recognition on current elixir images."""
    recognizer = DigitRecognizer()
    
    if not recognizer.templates:
        print("❌ No digit templates found!")
        print("Please create a 'digit_templates' folder and add digit images:")
        print("  digit_templates/1.png")
        print("  digit_templates/2.png")
        print("  digit_templates/3.png")
        print("  ...")
        print("  digit_templates/9.png")
        return
    
    print(f"✅ Loaded {len(recognizer.templates)} digit templates")
    
    # Test on current elixir images
    elixir_dir = Path("updated_cards")
    expected_costs = [4, 3, 4, 2]
    
    print("\n=== Testing Digit Recognition ===")
    
    for i in range(1, 5):
        elixir_path = elixir_dir / f"elixir_{i}_updated.png"
        if elixir_path.exists():
            elixir_img = cv2.imread(str(elixir_path))
            detected = recognizer.recognize_digit(elixir_img)
            expected = expected_costs[i-1]
            
            status = "✓" if detected == expected else "✗"
            print(f"Card {i}: Expected {expected}, Detected {detected} {status}")
        else:
            print(f"Card {i}: Elixir image not found")

def main():
    parser = argparse.ArgumentParser(description="Digit recognition for elixir detection")
    parser.add_argument("--test", action="store_true", help="Test digit recognition")
    parser.add_argument("--templates", default="digit_templates", help="Directory containing digit templates")
    
    args = parser.parse_args()
    
    if args.test:
        test_digit_recognition()
    else:
        print("Digit Recognition Tool")
        print("Usage: python3 digit_recognition.py --test")
        print("\nFirst, create digit templates:")
        print("1. Create folder: mkdir digit_templates")
        print("2. Add digit images: 1.png, 2.png, ..., 9.png")
        print("3. Run test: python3 digit_recognition.py --test")

if __name__ == "__main__":
    main()
