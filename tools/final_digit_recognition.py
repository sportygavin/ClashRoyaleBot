#!/usr/bin/env python3
"""
Final improved digit recognition using processed templates.
"""

import cv2
import numpy as np
from pathlib import Path

class FinalDigitRecognizer:
    def __init__(self, digit_templates_dir="digit_templates"):
        self.digit_templates_dir = Path(digit_templates_dir)
        self.templates = self._load_processed_templates()
    
    def _load_processed_templates(self):
        """Load processed digit templates."""
        templates = {}
        
        # Try to load pre-processed templates first
        for digit in range(1, 10):
            processed_path = Path(f"processed_template_{digit}.png")
            if processed_path.exists():
                template = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[digit] = template
                    print(f"Loaded processed template for digit {digit}: {template.shape}")
                    continue
        
        # If no processed templates, create them
        if not templates:
            print("Creating processed templates...")
            for digit in range(1, 10):
                template_path = self.digit_templates_dir / f"{digit}.png"
                if template_path.exists():
                    original = cv2.imread(str(template_path))
                    processed = self._preprocess_template(original)
                    if processed is not None:
                        templates[digit] = processed
                        # Save for future use
                        cv2.imwrite(f"processed_template_{digit}.png", processed)
                        print(f"Created processed template for digit {digit}: {processed.shape}")
        
        return templates
    
    def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess template for consistent matching."""
        # Convert to grayscale
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
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
        
        # Preprocess elixir image
        processed_img = self._preprocess_elixir_image(elixir_img)
        
        if processed_img is None:
            return None
        
        # Try matching against each digit template
        best_match = None
        best_score = 0
        
        for digit, template in self.templates.items():
            # Template matching
            result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = digit
        
        # Return result if confidence is high enough
        if best_match and best_score > 0.1:  # Very low threshold
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

def test_final_recognition():
    """Test final digit recognition."""
    recognizer = FinalDigitRecognizer()
    
    if not recognizer.templates:
        print("❌ No digit templates found!")
        return
    
    print(f"✅ Loaded {len(recognizer.templates)} digit templates")
    
    # Test on current elixir images
    elixir_dir = Path("updated_cards")
    expected_costs = [4, 3, 4, 2]
    
    print("\n=== Testing Final Digit Recognition ===")
    
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
    
    if correct/total >= 0.75:
        print("🎉 Great accuracy! The digit recognition is working well.")
    elif correct/total >= 0.5:
        print("👍 Good accuracy! Some improvements possible.")
    else:
        print("⚠️  Low accuracy. May need better preprocessing or templates.")

if __name__ == "__main__":
    test_final_recognition()
