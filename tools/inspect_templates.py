#!/usr/bin/env python3
"""
Inspect digit templates to see what they look like.
"""

import cv2
import numpy as np
from pathlib import Path

def inspect_templates():
    """Inspect all digit templates."""
    print("=== Inspecting Digit Templates ===")
    
    for digit in range(1, 10):
        template_path = Path(f"digit_templates/{digit}.png")
        if template_path.exists():
            # Load original
            original = cv2.imread(str(template_path))
            print(f"\nDigit {digit}:")
            print(f"  Original shape: {original.shape}")
            print(f"  File size: {template_path.stat().st_size} bytes")
            
            # Convert to grayscale
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  Contours found: {len(contours)}")
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                print(f"  Largest contour area: {area}")
                
                x, y, w, h = cv2.boundingRect(largest_contour)
                print(f"  Bounding box: {w}x{h} at ({x}, {y})")
                
                # Extract and resize
                digit_region = thresh[y:y+h, x:x+w]
                resized = cv2.resize(digit_region, (30, 40))
                
                # Save processed template
                cv2.imwrite(f"processed_template_{digit}.png", resized)
                print(f"  Saved processed template: processed_template_{digit}.png")
            
            # Save intermediate steps
            cv2.imwrite(f"template_{digit}_gray.png", gray)
            cv2.imwrite(f"template_{digit}_thresh.png", thresh)

if __name__ == "__main__":
    inspect_templates()
