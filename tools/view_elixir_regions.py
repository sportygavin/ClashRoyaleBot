#!/usr/bin/env python3
"""
View elixir regions to understand what we're working with.

This script displays the elixir regions and their processed versions
so we can see what the digits actually look like.
"""

import cv2
import numpy as np
from pathlib import Path

def display_elixir_regions():
    """Display elixir regions for manual inspection."""
    print("=== Elixir Region Inspection ===")
    print("Expected elixir costs: 4, 3, 4, 2")
    print()
    
    for i in range(1, 5):
        elixir_file = Path(f"updated_cards/elixir_{i}_updated.png")
        if elixir_file.exists():
            print(f"Card {i} elixir region:")
            elixir_img = cv2.imread(str(elixir_file))
            
            if elixir_img is not None:
                print(f"  Original shape: {elixir_img.shape}")
                
                # Convert to grayscale
                gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Apply threshold
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Save all versions
                cv2.imwrite(f"elixir_analysis/card_{i}_original.png", elixir_img)
                cv2.imwrite(f"elixir_analysis/card_{i}_gray.png", gray)
                cv2.imwrite(f"elixir_analysis/card_{i}_enhanced.png", enhanced)
                cv2.imwrite(f"elixir_analysis/card_{i}_thresh.png", thresh)
                
                print(f"  Saved to elixir_analysis/card_{i}_*.png")
                
                # Analyze the thresholded image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"  Contours found: {len(contours)}")
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    print(f"  Largest contour: area={area:.1f}, bbox=({x},{y},{w},{h})")
                    
                    # Extract the digit region
                    digit_region = thresh[y:y+h, x:x+w]
                    cv2.imwrite(f"elixir_analysis/card_{i}_digit_region.png", digit_region)
                    print(f"  Digit region saved to elixir_analysis/card_{i}_digit_region.png")
            else:
                print(f"  Could not load {elixir_file}")
        else:
            print(f"  File not found: {elixir_file}")
        
        print()

def create_simple_templates_from_regions():
    """Create templates from the actual digit regions."""
    print("=== Creating Templates from Actual Regions ===")
    
    templates = {}
    
    for i in range(1, 5):
        digit_file = Path(f"elixir_analysis/card_{i}_digit_region.png")
        if digit_file.exists():
            digit_img = cv2.imread(str(digit_file), cv2.IMREAD_GRAYSCALE)
            if digit_img is not None:
                # Resize to standard size
                digit_resized = cv2.resize(digit_img, (20, 30))
                templates[f"card_{i}"] = digit_resized
                cv2.imwrite(f"elixir_analysis/template_card_{i}.png", digit_resized)
                print(f"Created template for card {i}")
    
    return templates

def main():
    import argparse
    parser = argparse.ArgumentParser(description="View elixir regions")
    parser.add_argument("--create-templates", action="store_true", help="Create templates from regions")
    
    args = parser.parse_args()
    
    # Create output directory
    Path("elixir_analysis").mkdir(exist_ok=True)
    
    display_elixir_regions()
    
    if args.create_templates:
        create_simple_templates_from_regions()

if __name__ == "__main__":
    main()
