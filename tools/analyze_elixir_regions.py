#!/usr/bin/env python3
"""
Analyze elixir regions to improve digit detection.

This script loads the extracted elixir regions and tries different
preprocessing techniques to better detect the digits.
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_elixir_region(elixir_img: np.ndarray, card_id: str) -> dict:
    """Analyze a single elixir region with multiple techniques."""
    results = {}
    
    # Convert to grayscale if needed
    if len(elixir_img.shape) == 3:
        gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = elixir_img
    
    results['original_shape'] = elixir_img.shape
    results['gray_shape'] = gray.shape
    
    # Method 1: Simple threshold
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    results['simple_thresh'] = analyze_digit_contours(thresh1, "simple")
    
    # Method 2: Otsu threshold
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['otsu_thresh'] = analyze_digit_contours(thresh2, "otsu")
    
    # Method 3: Adaptive threshold
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    results['adaptive_thresh'] = analyze_digit_contours(thresh3, "adaptive")
    
    # Method 4: CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['clahe_otsu'] = analyze_digit_contours(thresh4, "clahe_otsu")
    
    # Method 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel)
    results['morph_clean'] = analyze_digit_contours(cleaned, "morph")
    
    return results

def analyze_digit_contours(thresh_img: np.ndarray, method_name: str) -> dict:
    """Analyze contours in thresholded image."""
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"method": method_name, "contours": 0, "largest_area": 0, "digit_guess": None}
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    largest_contour = contours[0]
    area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Extract the digit region
    digit_region = thresh_img[y:y+h, x:x+w]
    
    # Analyze the digit region
    white_pixels = np.sum(digit_region == 255)
    total_pixels = digit_region.size
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Count horizontal and vertical lines
    horizontal_lines = 0
    vertical_lines = 0
    
    # Check for horizontal lines
    for i in range(h):
        row = digit_region[i, :]
        if np.sum(row == 255) > w * 0.6:
            horizontal_lines += 1
    
    # Check for vertical lines
    for j in range(w):
        col = digit_region[:, j]
        if np.sum(col == 255) > h * 0.6:
            vertical_lines += 1
    
    # Simple digit classification
    digit_guess = classify_digit(area, aspect_ratio, white_ratio, horizontal_lines, vertical_lines)
    
    return {
        "method": method_name,
        "contours": len(contours),
        "largest_area": area,
        "aspect_ratio": aspect_ratio,
        "white_ratio": white_ratio,
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "digit_guess": digit_guess,
        "bbox": (x, y, w, h)
    }

def classify_digit(area: float, aspect_ratio: float, white_ratio: float, 
                  horizontal_lines: int, vertical_lines: int) -> int:
    """Classify digit based on features."""
    
    # Very basic classification - this needs improvement
    if area < 50:
        return None  # Too small
    
    if aspect_ratio < 0.3 or aspect_ratio > 2.0:
        return None  # Wrong shape
    
    # Simple heuristics based on line counts and area
    if horizontal_lines >= 2 and vertical_lines >= 2:
        if area > 200:
            return 8  # Complex digit, large
        else:
            return 4  # Complex digit, medium
    elif horizontal_lines >= 1 and vertical_lines >= 1:
        if area > 150:
            return 6  # Medium complexity, large
        else:
            return 3  # Medium complexity, small
    elif horizontal_lines >= 1:
        return 1  # Simple horizontal
    elif vertical_lines >= 1:
        return 1  # Simple vertical
    else:
        return 2  # Default guess

def save_analysis_images(elixir_img: np.ndarray, results: dict, output_dir: Path, card_id: str):
    """Save analysis images for visual inspection."""
    card_dir = output_dir / f"analysis_{card_id}"
    card_dir.mkdir(exist_ok=True)
    
    # Save original
    cv2.imwrite(str(card_dir / "original.png"), elixir_img)
    
    # Convert to grayscale
    if len(elixir_img.shape) == 3:
        gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = elixir_img
    
    # Save different thresholding methods
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(card_dir / "simple_thresh.png"), thresh1)
    
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(card_dir / "otsu_thresh.png"), thresh2)
    
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(str(card_dir / "adaptive_thresh.png"), thresh3)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(card_dir / "clahe_otsu.png"), thresh4)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(str(card_dir / "morph_clean.png"), cleaned)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze elixir regions")
    parser.add_argument("--cards", default="updated_cards", help="Directory containing elixir images")
    parser.add_argument("--out", default="elixir_analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    cards_dir = Path(args.cards)
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)
    
    print("=== Elixir Region Analysis ===")
    print("Expected elixir costs: 4, 3, 4, 2")
    print()
    
    all_results = {}
    
    for i in range(1, 5):
        elixir_file = cards_dir / f"elixir_{i}_updated.png"
        if elixir_file.exists():
            print(f"Analyzing Card {i}:")
            elixir_img = cv2.imread(str(elixir_file))
            
            if elixir_img is not None:
                results = analyze_elixir_region(elixir_img, f"card_{i}")
                all_results[f"card_{i}"] = results
                
                # Print results
                print(f"  Original shape: {results['original_shape']}")
                print(f"  Gray shape: {results['gray_shape']}")
                
                best_method = None
                best_confidence = 0
                
                for method_name in ['simple_thresh', 'otsu_thresh', 'adaptive_thresh', 'clahe_otsu', 'morph_clean']:
                    if method_name in results:
                        method_result = results[method_name]
                        digit = method_result['digit_guess']
                        area = method_result['largest_area']
                        
                        print(f"  {method_name}: digit={digit}, area={area:.1f}, contours={method_result['contours']}")
                        
                        # Score this method
                        if digit is not None and area > 50:
                            confidence = area / 200.0  # Simple confidence based on area
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_method = method_name
                
                print(f"  Best method: {best_method} (confidence: {best_confidence:.2f})")
                
                # Save analysis images
                save_analysis_images(elixir_img, results, output_dir, f"card_{i}")
                print(f"  Analysis images saved to: {output_dir}/analysis_card_{i}/")
            else:
                print(f"  Could not load {elixir_file}")
        
        print()
    
    # Summary
    print("=== Summary ===")
    expected = [4, 3, 4, 2]
    detected = []
    
    for i, expected_elixir in enumerate(expected, 1):
        card_key = f"card_{i}"
        if card_key in all_results:
            # Find best detection
            best_digit = None
            best_confidence = 0
            
            for method_name in ['simple_thresh', 'otsu_thresh', 'adaptive_thresh', 'clahe_otsu', 'morph_clean']:
                if method_name in all_results[card_key]:
                    method_result = all_results[card_key][method_name]
                    digit = method_result['digit_guess']
                    area = method_result['largest_area']
                    
                    if digit is not None and area > 50:
                        confidence = area / 200.0
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_digit = digit
            
            detected.append(best_digit)
            status = "✓" if best_digit == expected_elixir else "✗"
            print(f"Card {i}: Expected {expected_elixir}, Detected {best_digit} {status}")
        else:
            detected.append(None)
            print(f"Card {i}: Expected {expected_elixir}, Detected None ✗")
    
    print(f"\nAccuracy: {sum(1 for i, (exp, det) in enumerate(zip(expected, detected)) if exp == det)}/4")

if __name__ == "__main__":
    main()
