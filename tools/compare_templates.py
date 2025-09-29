#!/usr/bin/env python3
"""
Compare digit templates with processed elixir images visually.
"""

import cv2
import numpy as np
from pathlib import Path

def create_comparison_grid():
    """Create a visual comparison grid."""
    # Load templates
    templates = {}
    for digit in range(1, 10):
        template_path = Path(f"digit_templates/{digit}.png")
        if template_path.exists():
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                # Resize to standard size
                template = cv2.resize(template, (60, 80))
                templates[digit] = template
    
    # Load processed elixir images
    processed_images = {}
    for i in range(1, 5):
        for j in range(1, 4):  # Try different preprocessing approaches
            path = f"debug_processed_{j}.png"
            if Path(path).exists():
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to match template size
                    img = cv2.resize(img, (60, 80))
                    processed_images[f"card_{i}_method_{j}"] = img
    
    # Create comparison grid
    if templates and processed_images:
        # Create a grid showing templates vs processed images
        rows = []
        
        # Template row
        template_row = np.hstack([templates.get(i, np.zeros((80, 60), dtype=np.uint8)) for i in range(1, 10)])
        rows.append(template_row)
        
        # Add labels
        label_row = np.zeros((30, template_row.shape[1]), dtype=np.uint8)
        for i, digit in enumerate(range(1, 10)):
            cv2.putText(label_row, str(digit), (i*60 + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        rows.append(label_row)
        
        # Processed images rows
        for key, img in processed_images.items():
            # Pad to match template row width
            padded_img = np.zeros((80, template_row.shape[1]), dtype=np.uint8)
            padded_img[:, :img.shape[1]] = img
            rows.append(padded_img)
            
            # Add label
            label_row = np.zeros((30, template_row.shape[1]), dtype=np.uint8)
            cv2.putText(label_row, key, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            rows.append(label_row)
        
        # Combine all rows
        comparison = np.vstack(rows)
        
        # Save comparison
        cv2.imwrite("template_comparison.png", comparison)
        print("Saved template comparison to: template_comparison.png")
        
        # Also create individual comparisons for each card
        expected_costs = [4, 3, 4, 2]
        for i in range(1, 5):
            expected = expected_costs[i-1]
            template = templates.get(expected, None)
            
            if template is not None:
                # Find best processed image for this card
                best_img = None
                for j in range(1, 4):
                    path = f"debug_processed_{j}.png"
                    if Path(path).exists():
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (60, 80))
                            best_img = img
                            break
                
                if best_img is not None:
                    # Side by side comparison
                    comparison = np.hstack([template, best_img])
                    
                    # Add labels
                    labeled = np.zeros((comparison.shape[0] + 30, comparison.shape[1]), dtype=np.uint8)
                    labeled[30:, :] = comparison
                    cv2.putText(labeled, f"Template {expected}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                    cv2.putText(labeled, f"Card {i} Processed", (70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                    
                    cv2.imwrite(f"card_{i}_comparison.png", labeled)
                    print(f"Saved card {i} comparison to: card_{i}_comparison.png")

if __name__ == "__main__":
    create_comparison_grid()
