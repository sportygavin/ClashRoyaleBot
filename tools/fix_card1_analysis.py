#!/usr/bin/env python3
"""
Detailed analysis to fix card 1 detection - should be 4, not 3.
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_card1_detailed():
    """Detailed analysis of card 1 to understand why it's detected as 3 instead of 4."""
    print("=== DETAILED CARD 1 ANALYSIS ===")
    print("Expected: 4, Currently Detected: 3")
    
    # Load card 1 elixir image
    elixir_path = Path("updated_cards/elixir_1_updated.png")
    if not elixir_path.exists():
        print("❌ Card 1 elixir image not found!")
        return
    
    elixir_img = cv2.imread(str(elixir_path))
    print(f"Original elixir image shape: {elixir_img.shape}")
    
    # Save original for inspection
    cv2.imwrite("debug_card1_original_elixir.png", elixir_img)
    print("Saved original elixir: debug_card1_original_elixir.png")
    
    # Check if we're extracting the right region
    print("\n--- Checking Elixir Region Extraction ---")
    
    # Load calibration data
    calib_path = Path("cv_out/calibration_manual_fixed.json")
    if calib_path.exists():
        import json
        with open(calib_path) as f:
            calib = json.load(f)
        
        # Calculate elixir region based on calibration
        elixir_roi = calib['elixir_roi']
        h, w = elixir_img.shape[:2]
        
        x1 = int(elixir_roi['x_off_r'] * w)
        y1 = int(elixir_roi['y_off_r'] * h)
        x2 = int((elixir_roi['x_off_r'] + elixir_roi['w_r']) * w)
        y2 = int((elixir_roi['y_off_r'] + elixir_roi['h_r']) * h)
        
        print(f"Elixir ROI: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Elixir region size: {x2-x1}x{y2-y1}")
        
        # Extract the region
        extracted_region = elixir_img[y1:y2, x1:x2]
        cv2.imwrite("debug_card1_extracted_region.png", extracted_region)
        print("Saved extracted region: debug_card1_extracted_region.png")
    
    # Try different preprocessing approaches specifically for card 1
    print("\n--- Testing Different Preprocessing for Card 1 ---")
    
    approaches = [
        ("CLAHE + OTSU", preprocess_clahe_otsu),
        ("Simple Threshold", preprocess_simple),
        ("Adaptive Threshold", preprocess_adaptive),
        ("Morphological", preprocess_morphological),
        ("Perfect", preprocess_perfect),
        ("High Contrast", preprocess_high_contrast),
        ("Inverted", preprocess_inverted)
    ]
    
    best_approach = None
    best_score_for_4 = 0
    
    for approach_name, preprocess_func in approaches:
        print(f"\n--- {approach_name} ---")
        processed = preprocess_func(elixir_img)
        
        if processed is not None:
            # Save processed image
            filename = f"debug_card1_{approach_name.replace(' ', '_').lower()}.png"
            cv2.imwrite(filename, processed)
            print(f"Saved: {filename}")
            
            # Test specifically against digit 4 templates
            score_for_4 = test_against_digit_4(processed, approach_name)
            
            if score_for_4 > best_score_for_4:
                best_score_for_4 = score_for_4
                best_approach = approach_name
            
            print(f"Best score for digit 4: {score_for_4:.3f}")
        else:
            print("Failed to process")

def test_against_digit_4(processed_img, approach_name):
    """Test processed image specifically against digit 4 templates."""
    # Load digit 4 templates
    templates_4 = {}
    
    # Original template 4
    template_path = Path("digit_templates/4.png")
    if template_path.exists():
        original = cv2.imread(str(template_path))
        processed_template = preprocess_template(original)
        templates_4["orig_4"] = processed_template
    
    # Perfect template 4
    perfect_path = Path("perfect_templates/4.png")
    if perfect_path.exists():
        template = cv2.imread(str(perfect_path), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates_4["perfect_4"] = template
    
    if not templates_4:
        print("  No digit 4 templates found!")
        return 0
    
    # Test against digit 4 templates
    best_score = 0
    for template_name, template in templates_4.items():
        result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
        
        print(f"  {template_name}: {max_val:.3f}")
    
    return best_score

def preprocess_template(template: np.ndarray) -> np.ndarray:
    """Preprocess template."""
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_region = thresh[y:y+h, x:x+w]
        return cv2.resize(digit_region, (40, 50))
    
    return cv2.resize(thresh, (40, 50))

def preprocess_clahe_otsu(img: np.ndarray) -> np.ndarray:
    """CLAHE + OTSU preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return extract_and_resize(thresh)

def preprocess_simple(img: np.ndarray) -> np.ndarray:
    """Simple threshold preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return extract_and_resize(thresh)

def preprocess_adaptive(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return extract_and_resize(thresh)

def preprocess_morphological(img: np.ndarray) -> np.ndarray:
    """Morphological operations preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return extract_and_resize(thresh)

def preprocess_perfect(img: np.ndarray) -> np.ndarray:
    """Perfect preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return extract_and_resize(thresh)

def preprocess_high_contrast(img: np.ndarray) -> np.ndarray:
    """High contrast preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    alpha = 2.0  # Contrast control
    beta = 0     # Brightness control
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    _, thresh = cv2.threshold(contrasted, 127, 255, cv2.THRESH_BINARY)
    return extract_and_resize(thresh)

def preprocess_inverted(img: np.ndarray) -> np.ndarray:
    """Inverted preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return extract_and_resize(thresh)

def extract_and_resize(thresh: np.ndarray) -> np.ndarray:
    """Extract digit region and resize."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 50:
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    digit_region = thresh[y:y+h, x:x+w]
    
    return cv2.resize(digit_region, (40, 50))

if __name__ == "__main__":
    analyze_card1_detailed()
