#!/usr/bin/env python3
"""
Template-based digit recognition for elixir detection.

This script creates digit templates from the elixir regions and uses
template matching to identify digits.
"""

import cv2
import numpy as np
from pathlib import Path

def create_digit_templates():
    """Create simple digit templates."""
    templates = {}
    
    # Create templates for digits 0-9
    # These are very basic templates - in practice you'd extract from actual images
    
    # Digit 0: Circle-like shape
    template_0 = np.zeros((30, 20), dtype=np.uint8)
    cv2.circle(template_0, (10, 15), 8, 255, 2)
    templates[0] = template_0
    
    # Digit 1: Vertical line
    template_1 = np.zeros((30, 20), dtype=np.uint8)
    cv2.line(template_1, (10, 5), (10, 25), 255, 3)
    cv2.line(template_1, (8, 7), (10, 5), 255, 2)
    templates[1] = template_1
    
    # Digit 2: Two horizontal lines with connecting lines
    template_2 = np.zeros((30, 20), dtype=np.uint8)
    cv2.line(template_2, (5, 8), (15, 8), 255, 2)  # Top
    cv2.line(template_2, (5, 15), (15, 15), 255, 2)  # Bottom
    cv2.line(template_2, (15, 8), (15, 15), 255, 2)  # Right side
    cv2.line(template_2, (5, 8), (5, 12), 255, 2)  # Left top
    cv2.line(template_2, (5, 12), (15, 15), 255, 2)  # Diagonal
    templates[2] = template_2
    
    # Digit 3: Two horizontal lines with right side
    template_3 = np.zeros((30, 20), dtype=np.uint8)
    cv2.line(template_3, (5, 8), (15, 8), 255, 2)  # Top
    cv2.line(template_3, (5, 15), (15, 15), 255, 2)  # Bottom
    cv2.line(template_3, (15, 8), (15, 15), 255, 2)  # Right side
    cv2.line(template_3, (5, 8), (5, 12), 255, 2)  # Left top
    cv2.line(template_3, (5, 12), (15, 12), 255, 2)  # Middle
    templates[3] = template_3
    
    # Digit 4: Vertical lines with horizontal
    template_4 = np.zeros((30, 20), dtype=np.uint8)
    cv2.line(template_4, (5, 8), (5, 15), 255, 2)  # Left vertical
    cv2.line(template_4, (15, 8), (15, 15), 255, 2)  # Right vertical
    cv2.line(template_4, (5, 12), (15, 12), 255, 2)  # Horizontal
    templates[4] = template_4
    
    # Digit 5: Similar to 2 but different
    template_5 = np.zeros((30, 20), dtype=np.uint8)
    cv2.line(template_5, (5, 8), (15, 8), 255, 2)  # Top
    cv2.line(template_5, (5, 15), (15, 15), 255, 2)  # Bottom
    cv2.line(template_5, (5, 8), (5, 12), 255, 2)  # Left top
    cv2.line(template_5, (5, 12), (15, 12), 255, 2)  # Middle
    cv2.line(template_5, (15, 12), (15, 15), 255, 2)  # Right bottom
    templates[5] = template_5
    
    # Digit 6: Circle with opening
    template_6 = np.zeros((30, 20), dtype=np.uint8)
    cv2.ellipse(template_6, (10, 15), (8, 8), 0, 0, 360, 255, 2)
    cv2.line(template_6, (2, 15), (10, 15), 255, 2)  # Bottom line
    templates[6] = template_6
    
    # Digit 7: Top line with diagonal
    template_7 = np.zeros((30, 20), dtype=np.uint8)
    cv2.line(template_7, (5, 8), (15, 8), 255, 2)  # Top
    cv2.line(template_7, (15, 8), (5, 15), 255, 2)  # Diagonal
    templates[7] = template_7
    
    # Digit 8: Two circles
    template_8 = np.zeros((30, 20), dtype=np.uint8)
    cv2.circle(template_8, (10, 10), 6, 255, 2)  # Top circle
    cv2.circle(template_8, (10, 20), 6, 255, 2)  # Bottom circle
    templates[8] = template_8
    
    # Digit 9: Circle with bottom line
    template_9 = np.zeros((30, 20), dtype=np.uint8)
    cv2.ellipse(template_9, (10, 10), (8, 8), 0, 0, 360, 255, 2)
    cv2.line(template_9, (2, 10), (10, 10), 255, 2)  # Top line
    templates[9] = template_9
    
    return templates

def preprocess_for_template_matching(img: np.ndarray) -> np.ndarray:
    """Preprocess image for template matching."""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def recognize_digit_template(elixir_img: np.ndarray, templates: dict) -> int:
    """Recognize digit using template matching."""
    processed = preprocess_for_template_matching(elixir_img)
    
    best_match = None
    best_score = 0
    
    for digit, template in templates.items():
        # Resize template to match elixir region size
        h, w = processed.shape
        template_resized = cv2.resize(template, (w, h))
        
        # Template matching
        result = cv2.matchTemplate(processed, template_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = digit
    
    # Return digit if confidence is high enough
    if best_score > 0.3:  # Adjust threshold as needed
        return best_match
    
    return None

def analyze_with_templates():
    """Analyze elixir regions using template matching."""
    templates = create_digit_templates()
    
    print("=== Template-based Digit Recognition ===")
    print("Expected elixir costs: 4, 3, 4, 2")
    print()
    
    results = []
    expected = [4, 3, 4, 2]
    
    for i in range(1, 5):
        elixir_file = Path(f"updated_cards/elixir_{i}_updated.png")
        if elixir_file.exists():
            print(f"Analyzing Card {i}:")
            elixir_img = cv2.imread(str(elixir_file))
            
            if elixir_img is not None:
                digit = recognize_digit_template(elixir_img, templates)
                results.append(digit)
                
                status = "✓" if digit == expected[i-1] else "✗"
                print(f"  Expected: {expected[i-1]}, Detected: {digit} {status}")
                
                # Save processed image for debugging
                processed = preprocess_for_template_matching(elixir_img)
                cv2.imwrite(f"elixir_analysis/card_{i}_processed.png", processed)
            else:
                results.append(None)
                print(f"  Could not load {elixir_file}")
        else:
            results.append(None)
            print(f"  File not found: {elixir_file}")
    
    # Summary
    print("\n=== Summary ===")
    correct = sum(1 for i, (exp, det) in enumerate(zip(expected, results)) if exp == det)
    print(f"Accuracy: {correct}/4")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Template-based digit recognition")
    parser.add_argument("--cards", default="updated_cards", help="Directory containing elixir images")
    
    args = parser.parse_args()
    
    # Create output directory
    Path("elixir_analysis").mkdir(exist_ok=True)
    
    results = analyze_with_templates()

if __name__ == "__main__":
    main()
