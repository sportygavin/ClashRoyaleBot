#!/usr/bin/env python3
"""
Check if we're extracting the right elixir region from card 1.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def check_card1_region():
    """Check if we're extracting the correct elixir region from card 1."""
    print("=== CHECKING CARD 1 ELIXIR REGION EXTRACTION ===")
    
    # Load the original screenshot
    screenshot_path = Path("calibration/test_screenshot.png")
    if not screenshot_path.exists():
        print("❌ Original screenshot not found!")
        return
    
    screenshot = cv2.imread(str(screenshot_path))
    print(f"Original screenshot shape: {screenshot.shape}")
    
    # Load calibration data
    calib_path = Path("cv_out/calibration_manual_fixed.json")
    if not calib_path.exists():
        print("❌ Calibration data not found!")
        return
    
    with open(calib_path) as f:
        calib = json.load(f)
    
    # Calculate card 1 position
    viewport = calib['viewport']
    cards = calib['cards']
    
    # Calculate absolute coordinates
    vp_x = int(viewport['x_r'] * screenshot.shape[1])
    vp_y = int(viewport['y_r'] * screenshot.shape[0])
    vp_w = int(viewport['w_r'] * screenshot.shape[1])
    vp_h = int(viewport['h_r'] * screenshot.shape[0])
    
    # Card 1 center
    card1_center_x = int(cards['centers_x_r'][0] * vp_w) + vp_x
    # Y position is calculated from card row
    card_row_top = int(calib['card_row']['top_r'] * vp_h) + vp_y
    card_row_bottom = int(calib['card_row']['bottom_r'] * vp_h) + vp_y
    card1_center_y = (card_row_top + card_row_bottom) // 2
    
    # Card 1 box
    card_w = int(cards['width_r'] * vp_w)
    card_h = card_row_bottom - card_row_top
    card_top_offset = int(cards['top_offset_r'] * vp_h)
    card_bottom_offset = int(cards['bottom_offset_r'] * vp_h)
    
    card1_x1 = card1_center_x - card_w // 2
    card1_y1 = card1_center_y - card_top_offset
    card1_x2 = card1_center_x + card_w // 2
    card1_y2 = card1_center_y + card_bottom_offset
    
    print(f"Card 1 box: ({card1_x1}, {card1_y1}) to ({card1_x2}, {card1_y2})")
    
    # Extract card 1
    card1_img = screenshot[card1_y1:card1_y2, card1_x1:card1_x2]
    cv2.imwrite("debug_card1_full_card.png", card1_img)
    print("Saved full card 1: debug_card1_full_card.png")
    
    # Calculate elixir region within card 1
    elixir_roi = calib['elixir_roi']
    card_h, card_w = card1_img.shape[:2]
    
    elixir_x1 = int(elixir_roi['x_off_r'] * card_w)
    elixir_y1 = int(elixir_roi['y_off_r'] * card_h)
    elixir_x2 = int((elixir_roi['x_off_r'] + elixir_roi['w_r']) * card_w)
    elixir_y2 = int((elixir_roi['y_off_r'] + elixir_roi['h_r']) * card_h)
    
    print(f"Elixir region in card 1: ({elixir_x1}, {elixir_y1}) to ({elixir_x2}, {elixir_y2})")
    
    # Extract elixir region
    elixir_region = card1_img[elixir_y1:elixir_y2, elixir_x1:elixir_x2]
    cv2.imwrite("debug_card1_elixir_from_full_card.png", elixir_region)
    print("Saved elixir region from full card: debug_card1_elixir_from_full_card.png")
    
    # Compare with the current extracted elixir
    current_elixir_path = Path("updated_cards/elixir_1_updated.png")
    if current_elixir_path.exists():
        current_elixir = cv2.imread(str(current_elixir_path))
        cv2.imwrite("debug_card1_current_elixir.png", current_elixir)
        print("Saved current elixir: debug_card1_current_elixir.png")
        
        # Compare sizes
        print(f"Current elixir size: {current_elixir.shape}")
        print(f"New elixir size: {elixir_region.shape}")
        
        # Test recognition on the new elixir region
        test_recognition_on_new_region(elixir_region)

def test_recognition_on_new_region(elixir_region):
    """Test digit recognition on the newly extracted elixir region."""
    print("\n--- Testing Recognition on New Elixir Region ---")
    
    # Try different preprocessing approaches
    approaches = [
        ("CLAHE + OTSU", preprocess_clahe_otsu),
        ("Simple Threshold", preprocess_simple),
        ("Adaptive Threshold", preprocess_adaptive),
        ("Morphological", preprocess_morphological),
        ("Perfect", preprocess_perfect)
    ]
    
    best_score_for_4 = 0
    best_approach = None
    
    for approach_name, preprocess_func in approaches:
        processed = preprocess_func(elixir_region)
        
        if processed is not None:
            # Save processed image
            filename = f"debug_card1_new_{approach_name.replace(' ', '_').lower()}.png"
            cv2.imwrite(filename, processed)
            
            # Test against digit 4 templates
            score_for_4 = test_against_digit_4(processed)
            
            if score_for_4 > best_score_for_4:
                best_score_for_4 = score_for_4
                best_approach = approach_name
            
            print(f"{approach_name}: Best score for digit 4: {score_for_4:.3f}")
    
    print(f"\nBest approach for digit 4: {best_approach} (score: {best_score_for_4:.3f})")

def test_against_digit_4(processed_img):
    """Test processed image against digit 4 templates."""
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
        return 0
    
    # Test against digit 4 templates
    best_score = 0
    for template_name, template in templates_4.items():
        result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
    
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
    check_card1_region()
