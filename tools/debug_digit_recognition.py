#!/usr/bin/env python3
"""
Debug version of digit recognition to see what's happening.
"""

import cv2
import numpy as np
from pathlib import Path

def debug_elixir_preprocessing():
    """Debug the preprocessing steps for elixir images."""
    recognizer = DigitRecognizer()
    
    if not recognizer.templates:
        print("❌ No digit templates found!")
        return
    
    print(f"✅ Loaded {len(recognizer.templates)} digit templates")
    
    # Test on current elixir images
    elixir_dir = Path("updated_cards")
    expected_costs = [4, 3, 4, 2]
    
    print("\n=== Debugging Elixir Preprocessing ===")
    
    for i in range(1, 5):
        elixir_path = elixir_dir / f"elixir_{i}_updated.png"
        if elixir_path.exists():
            print(f"\n--- Card {i} (Expected: {expected_costs[i-1]}) ---")
            
            # Load original image
            elixir_img = cv2.imread(str(elixir_path))
            print(f"Original image shape: {elixir_img.shape}")
            
            # Show preprocessing steps
            processed = recognizer._preprocess_elixir_image(elixir_img)
            
            if processed is not None:
                print(f"Processed image shape: {processed.shape}")
                
                # Save processed image for inspection
                debug_path = f"debug_processed_card_{i}.png"
                cv2.imwrite(debug_path, processed)
                print(f"Saved processed image: {debug_path}")
                
                # Try matching against templates
                best_match = None
                best_score = 0
                
                for digit, template in recognizer.templates.items():
                    resized_template = cv2.resize(template, (processed.shape[1], processed.shape[0]))
                    result = cv2.matchTemplate(processed, resized_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    print(f"  Digit {digit}: score = {max_val:.3f}")
                    
                    if max_val > best_score:
                        best_score = max_val
                        best_match = digit
                
                print(f"  Best match: {best_match} (score: {best_score:.3f})")
                
            else:
                print("  Failed to preprocess image")

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
    
    def _preprocess_elixir_image(self, elixir_img: np.ndarray) -> np.ndarray:
        """Preprocess elixir image for digit recognition."""
        # Convert to grayscale if needed
        if len(elixir_img.shape) == 3:
            gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = elixir_img
        
        print(f"  Grayscale shape: {gray.shape}")
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save intermediate steps
        cv2.imwrite(f"debug_gray.png", gray)
        cv2.imwrite(f"debug_enhanced.png", enhanced)
        cv2.imwrite(f"debug_thresh.png", thresh)
        
        # Find contours to extract digit region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  Found {len(contours)} contours")
        
        if not contours:
            return None
        
        # Find largest contour (should be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        print(f"  Largest contour area: {area}")
        
        if area < 50:  # Filter out noise
            print("  Contour too small, filtering out")
            return None
        
        # Extract bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        
        print(f"  Digit region: {w}x{h} at ({x}, {y})")
        
        # Save digit region
        cv2.imwrite(f"debug_digit_region.png", digit_region)
        
        # Resize to standard size for consistent matching
        resized = cv2.resize(digit_region, (30, 40))
        cv2.imwrite(f"debug_resized.png", resized)
        
        return resized

if __name__ == "__main__":
    debug_elixir_preprocessing()
