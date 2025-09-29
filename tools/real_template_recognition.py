#!/usr/bin/env python3
"""
Real template-based digit recognition using actual digit regions.

This script uses the templates created from actual elixir regions
to recognize digits in new images.
"""

import cv2
import numpy as np
from pathlib import Path

def load_real_templates():
    """Load templates created from actual digit regions."""
    templates = {}
    
    # Load the templates we created
    for i in range(1, 5):
        template_file = Path(f"elixir_analysis/template_card_{i}.png")
        if template_file.exists():
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                templates[f"card_{i}"] = template
                print(f"Loaded template for card {i}: {template.shape}")
    
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

def extract_digit_region(elixir_img: np.ndarray) -> np.ndarray:
    """Extract the digit region from elixir image."""
    processed = preprocess_for_template_matching(elixir_img)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 50:  # Too small
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract the digit region
    digit_region = processed[y:y+h, x:x+w]
    
    # Resize to standard size
    digit_resized = cv2.resize(digit_region, (20, 30))
    
    return digit_resized

def recognize_digit_with_real_templates(elixir_img: np.ndarray, templates: dict) -> tuple:
    """Recognize digit using real templates."""
    digit_region = extract_digit_region(elixir_img)
    
    if digit_region is None:
        return None, 0
    
    best_match = None
    best_score = 0
    
    for template_name, template in templates.items():
        # Template matching
        result = cv2.matchTemplate(digit_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = template_name
    
    return best_match, best_score

def analyze_with_real_templates():
    """Analyze elixir regions using real templates."""
    templates = load_real_templates()
    
    if not templates:
        print("No templates found! Run view_elixir_regions.py --create-templates first.")
        return
    
    print("\n=== Real Template-based Digit Recognition ===")
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
                match, score = recognize_digit_with_real_templates(elixir_img, templates)
                
                # Convert template name to digit
                if match:
                    digit = int(match.split('_')[1])  # Extract number from "card_X"
                else:
                    digit = None
                
                results.append(digit)
                
                status = "✓" if digit == expected[i-1] else "✗"
                print(f"  Expected: {expected[i-1]}, Detected: {digit} (score: {score:.3f}) {status}")
                
                # Save processed image for debugging
                processed = preprocess_for_template_matching(elixir_img)
                cv2.imwrite(f"elixir_analysis/card_{i}_processed_final.png", processed)
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
    parser = argparse.ArgumentParser(description="Real template-based digit recognition")
    parser.add_argument("--cards", default="updated_cards", help="Directory containing elixir images")
    
    args = parser.parse_args()
    
    # Create output directory
    Path("elixir_analysis").mkdir(exist_ok=True)
    
    results = analyze_with_real_templates()

if __name__ == "__main__":
    main()
