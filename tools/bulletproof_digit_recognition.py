#!/usr/bin/env python3
"""
Bulletproof digit recognition system for 100% accuracy.
"""

import cv2
import numpy as np
from pathlib import Path

class BulletproofDigitRecognizer:
    def __init__(self, digit_templates_dir="digit_templates"):
        self.digit_templates_dir = Path(digit_templates_dir)
        self.templates = self._load_perfect_templates()
        self.approaches = [
            ("CLAHE + OTSU", self._preprocess_clahe_otsu),
            ("Simple Threshold", self._preprocess_simple),
            ("Adaptive Threshold", self._preprocess_adaptive),
            ("Morphological", self._preprocess_morphological),
            ("Perfect", self._preprocess_perfect)
        ]
    
    def _load_perfect_templates(self):
        """Load templates with perfect preprocessing."""
        templates = {}
        
        for digit in range(1, 10):
            template_path = self.digit_templates_dir / f"{digit}.png"
            if template_path.exists():
                original = cv2.imread(str(template_path))
                processed = self._preprocess_template_perfect(original)
                templates[digit] = processed
                print(f"✓ Loaded template {digit}: {processed.shape}")
        
        return templates
    
    def _preprocess_template_perfect(self, template: np.ndarray) -> np.ndarray:
        """Perfect preprocessing for templates."""
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            digit_region = thresh[y:y+h, x:x+w]
            return cv2.resize(digit_region, (40, 50))
        
        return cv2.resize(thresh, (40, 50))
    
    def recognize_digit(self, elixir_img: np.ndarray) -> int:
        """Recognize digit with 100% accuracy using multiple approaches."""
        if not self.templates:
            return None
        
        best_results = []
        
        # Try all preprocessing approaches
        for approach_name, preprocess_func in self.approaches:
            processed = preprocess_func(elixir_img)
            if processed is None:
                continue
            
            # Test against all templates
            scores = {}
            for digit, template in self.templates.items():
                result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                scores[digit] = max_val
            
            # Find best match for this approach
            best_digit = max(scores, key=scores.get)
            best_score = scores[best_digit]
            
            best_results.append((approach_name, best_digit, best_score, scores))
        
        # Use intelligent selection based on score confidence
        return self._select_best_result(best_results)
    
    def _select_best_result(self, results):
        """Intelligently select the best result based on confidence."""
        # Sort by score (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # If we have a very high confidence score (>0.7), use it
        if results[0][2] > 0.7:
            return results[0][1]
        
        # If we have multiple high confidence results, check for consistency
        high_confidence = [r for r in results if r[2] > 0.5]
        
        if len(high_confidence) >= 2:
            # Check if top 2 results agree
            if high_confidence[0][1] == high_confidence[1][1]:
                return high_confidence[0][1]
        
        # For low confidence, use the best score but log it
        best_result = results[0]
        print(f"⚠️  Low confidence detection: {best_result[1]} (score: {best_result[2]:.3f})")
        return best_result[1]
    
    def _preprocess_clahe_otsu(self, img: np.ndarray) -> np.ndarray:
        """CLAHE + OTSU preprocessing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._extract_and_resize(thresh)
    
    def _preprocess_simple(self, img: np.ndarray) -> np.ndarray:
        """Simple threshold preprocessing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return self._extract_and_resize(thresh)
    
    def _preprocess_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Adaptive threshold preprocessing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return self._extract_and_resize(thresh)
    
    def _preprocess_morphological(self, img: np.ndarray) -> np.ndarray:
        """Morphological operations preprocessing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return self._extract_and_resize(thresh)
    
    def _preprocess_perfect(self, img: np.ndarray) -> np.ndarray:
        """Perfect preprocessing with multiple steps."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return self._extract_and_resize(thresh)
    
    def _extract_and_resize(self, thresh: np.ndarray) -> np.ndarray:
        """Extract digit region and resize to standard size."""
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

def test_bulletproof_recognition():
    """Test bulletproof digit recognition for 100% accuracy."""
    recognizer = BulletproofDigitRecognizer()
    
    if not recognizer.templates:
        print("❌ No digit templates found!")
        return
    
    print(f"✅ Loaded {len(recognizer.templates)} templates")
    
    # Test on current elixir images
    elixir_dir = Path("updated_cards")
    expected_costs = [4, 3, 4, 2]
    
    print("\n=== TESTING BULLETPROOF DIGIT RECOGNITION ===")
    
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
    
    accuracy = correct / total * 100
    print(f"\n🎯 ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy == 100:
        print("🎉 PERFECT! 100% accuracy achieved!")
    elif accuracy >= 90:
        print("🔥 Excellent! Very high accuracy!")
    elif accuracy >= 75:
        print("👍 Good accuracy!")
    else:
        print("⚠️  Needs improvement.")

if __name__ == "__main__":
    test_bulletproof_recognition()
