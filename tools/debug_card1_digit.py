#!/usr/bin/env python3
"""
Debug card 1 digit specifically to understand why it's not being recognized as 4.
"""

import cv2
import numpy as np
from pathlib import Path

def debug_card1_specifically():
    """Debug card 1 digit in detail."""
    print("=== DEBUGGING CARD 1 DIGIT SPECIFICALLY ===")
    
    # Load card 1 elixir image
    elixir_path = Path("updated_cards/elixir_1_updated.png")
    if not elixir_path.exists():
        print("❌ Card 1 elixir image not found!")
        return
    
    elixir_img = cv2.imread(str(elixir_path))
    print(f"Original image shape: {elixir_img.shape}")
    
    # Save original for inspection
    cv2.imwrite("debug_card1_original.png", elixir_img)
    print("Saved original: debug_card1_original.png")
    
    # Try multiple preprocessing approaches
    approaches = [
        ("CLAHE + OTSU", preprocess_clahe_otsu),
        ("Simple Threshold", preprocess_simple),
        ("Adaptive Threshold", preprocess_adaptive),
        ("Morphological", preprocess_morphological),
        ("Perfect", preprocess_perfect),
        ("Inverted", preprocess_inverted),
        ("High Contrast", preprocess_high_contrast)
    ]
    
    for approach_name, preprocess_func in approaches:
        print(f"\n--- {approach_name} ---")
        processed = preprocess_func(elixir_img)
        
        if processed is not None:
            # Save processed image
            filename = f"debug_card1_{approach_name.replace(' ', '_').lower()}.png"
            cv2.imwrite(filename, processed)
            print(f"Saved: {filename}")
            
            # Test against all templates
            test_against_templates(processed, approach_name)
        else:
            print("Failed to process")

def test_against_templates(processed_img, approach_name):
    """Test processed image against all available templates."""
    # Load all templates
    templates = {}
    
    # Original templates
    for digit in range(1, 10):
        template_path = Path(f"digit_templates/{digit}.png")
        if template_path.exists():
            original = cv2.imread(str(template_path))
            processed_template = preprocess_template(original)
            templates[f"orig_{digit}"] = processed_template
    
    # Perfect templates
    perfect_dir = Path("perfect_templates")
    for digit in [2, 3, 4]:
        template_path = perfect_dir / f"{digit}.png"
        if template_path.exists():
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                templates[f"perfect_{digit}"] = template
    
    # Test against all templates
    scores = {}
    for template_name, template in templates.items():
        result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        scores[template_name] = max_val
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 matches:")
    for i, (template_name, score) in enumerate(sorted_scores[:5]):
        print(f"  {i+1}. {template_name}: {score:.3f}")

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

def preprocess_inverted(img: np.ndarray) -> np.ndarray:
    """Inverted preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
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
    debug_card1_specifically()
