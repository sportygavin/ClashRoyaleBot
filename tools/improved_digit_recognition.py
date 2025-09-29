#!/usr/bin/env python3
"""
Improved digit recognition with better preprocessing and lower threshold.
"""

import cv2
import numpy as np
from pathlib import Path

class ImprovedDigitRecognizer:
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
                    # Preprocess templates too
                    template = self._preprocess_template(template)
                    templates[digit] = template
                    print(f"Loaded template for digit {digit}: {template.shape}")
                else:
                    print(f"Failed to load template for digit {digit}")
            else:
                print(f"Template not found: {template_path}")
        
        return templates
    
    def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess template for consistent matching."""
        # Apply threshold to make it binary
        _, thresh = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours and extract largest
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            digit_region = thresh[y:y+h, x:x+w]
            return cv2.resize(digit_region, (30, 40))
        
        return cv2.resize(thresh, (30, 40))
    
    def recognize_digit(self, elixir_img: np.ndarray) -> int:
        """Recognize digit in elixir image using template matching."""
        if not self.templates:
            print("No digit templates loaded!")
            return None
        
        # Try multiple preprocessing approaches
        processed_images = self._preprocess_elixir_image_multiple(elixir_img)
        
        best_match = None
        best_score = 0
        
        for processed_img in processed_images:
            if processed_img is None:
                continue
                
            for digit, template in self.templates.items():
                # Template matching
                result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = digit
        
        # Lower threshold for better recognition
        if best_match and best_score > 0.15:  # Lowered from 0.3
            return best_match
        
        return None
    
    def _preprocess_elixir_image_multiple(self, elixir_img: np.ndarray) -> list:
        """Try multiple preprocessing approaches."""
        processed_images = []
        
        # Convert to grayscale if needed
        if len(elixir_img.shape) == 3:
            gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = elixir_img
        
        # Approach 1: CLAHE + OTSU
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Approach 2: Simple threshold
        _, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Approach 3: Adaptive threshold
        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Process each thresholded image
        for i, thresh in enumerate([thresh1, thresh2, thresh3]):
            processed = self._extract_digit_region(thresh)
            if processed is not None:
                processed_images.append(processed)
                # Save for debugging
                cv2.imwrite(f"debug_thresh_{i+1}.png", thresh)
                cv2.imwrite(f"debug_processed_{i+1}.png", processed)
        
        return processed_images
    
    def _extract_digit_region(self, thresh: np.ndarray) -> np.ndarray:
        """Extract digit region from thresholded image."""
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:
            return None
        
        # Extract bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        
        # Resize to standard size
        return cv2.resize(digit_region, (30, 40))

def test_improved_recognition():
    """Test improved digit recognition."""
    recognizer = ImprovedDigitRecognizer()
    
    if not recognizer.templates:
        print("❌ No digit templates found!")
        return
    
    print(f"✅ Loaded {len(recognizer.templates)} digit templates")
    
    # Test on current elixir images
    elixir_dir = Path("updated_cards")
    expected_costs = [4, 3, 4, 2]
    
    print("\n=== Testing Improved Digit Recognition ===")
    
    correct = 0
    total = 0
    
    for i in range(1, 5):
        elixir_path = elixir_dir / f"elixir_{i}_updated.png"
        if elixir_path.exists():
            elixir_img = cv2.imread(str(elixir_path))
            detected = recognizer.recognize_digit(elixir_img)
            expected = expected_costs[i-1]
            
            status = "✓" if detected == expected else "✗"
            if detected == expected:
                correct += 1
            total += 1
            
            print(f"Card {i}: Expected {expected}, Detected {detected} {status}")
        else:
            print(f"Card {i}: Elixir image not found")
    
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    test_improved_recognition()
