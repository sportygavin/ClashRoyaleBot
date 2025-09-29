#!/usr/bin/env python3
"""
Comprehensive analysis of digit recognition failures to achieve 100% accuracy.
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_all_failures():
    """Analyze why recognition is failing and create perfect templates."""
    print("=== COMPREHENSIVE DIGIT RECOGNITION ANALYSIS ===")
    
    # Load all templates
    templates = {}
    for digit in range(1, 10):
        template_path = Path(f"digit_templates/{digit}.png")
        if template_path.exists():
            original = cv2.imread(str(template_path))
            processed = preprocess_template_perfect(original)
            templates[digit] = processed
            print(f"✓ Loaded template {digit}: {processed.shape}")
    
    if not templates:
        print("❌ No templates found!")
        return
    
    print(f"\n✅ Loaded {len(templates)} templates")
    
    # Analyze each elixir image with multiple approaches
    expected_costs = [4, 3, 4, 2]
    
    for i in range(1, 5):
        print(f"\n--- ANALYZING CARD {i} (Expected: {expected_costs[i-1]}) ---")
        
        elixir_path = Path(f"updated_cards/elixir_{i}_updated.png")
        if not elixir_path.exists():
            print(f"❌ Elixir image not found: {elixir_path}")
            continue
        
        elixir_img = cv2.imread(str(elixir_path))
        
        # Try multiple preprocessing approaches
        approaches = [
            ("CLAHE + OTSU", preprocess_approach_1),
            ("Simple Threshold", preprocess_approach_2),
            ("Adaptive Threshold", preprocess_approach_3),
            ("Morphological", preprocess_approach_4),
            ("Perfect", preprocess_approach_perfect)
        ]
        
        best_results = []
        
        for approach_name, preprocess_func in approaches:
            processed = preprocess_func(elixir_img)
            if processed is None:
                continue
            
            # Save processed image
            cv2.imwrite(f"analysis_card_{i}_{approach_name.replace(' ', '_').lower()}.png", processed)
            
            # Test against all templates
            scores = {}
            for digit, template in templates.items():
                result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                scores[digit] = max_val
            
            # Find best match
            best_digit = max(scores, key=scores.get)
            best_score = scores[best_digit]
            
            best_results.append((approach_name, best_digit, best_score, scores))
            
            print(f"  {approach_name}: Best = {best_digit} (score: {best_score:.3f})")
        
        # Find the approach that gives the correct result
        correct_approach = None
        for approach_name, best_digit, best_score, scores in best_results:
            if best_digit == expected_costs[i-1]:
                correct_approach = (approach_name, best_score, scores)
                break
        
        if correct_approach:
            print(f"  ✅ CORRECT APPROACH: {correct_approach[0]} (score: {correct_approach[1]:.3f})")
        else:
            print(f"  ❌ NO CORRECT APPROACH FOUND")
            # Show all scores for debugging
            print("  All scores:")
            for approach_name, best_digit, best_score, scores in best_results:
                print(f"    {approach_name}: {dict(scores)}")

def preprocess_template_perfect(template: np.ndarray) -> np.ndarray:
    """Perfect preprocessing for templates."""
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours and extract largest
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        
        # Resize to standard size
        return cv2.resize(digit_region, (40, 50))
    
    return cv2.resize(thresh, (40, 50))

def preprocess_approach_1(img: np.ndarray) -> np.ndarray:
    """CLAHE + OTSU approach."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return extract_and_resize(thresh)

def preprocess_approach_2(img: np.ndarray) -> np.ndarray:
    """Simple threshold approach."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return extract_and_resize(thresh)

def preprocess_approach_3(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold approach."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return extract_and_resize(thresh)

def preprocess_approach_4(img: np.ndarray) -> np.ndarray:
    """Morphological operations approach."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return extract_and_resize(thresh)

def preprocess_approach_perfect(img: np.ndarray) -> np.ndarray:
    """Perfect preprocessing approach."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Multiple preprocessing steps
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return extract_and_resize(thresh)

def extract_and_resize(thresh: np.ndarray) -> np.ndarray:
    """Extract digit region and resize to standard size."""
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
    return cv2.resize(digit_region, (40, 50))

if __name__ == "__main__":
    analyze_all_failures()
