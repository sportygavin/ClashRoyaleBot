import argparse
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr
from tools.card_recognition_system import CardRecognitionSystem


def main():
    parser = argparse.ArgumentParser(description='Analyze elixir digit recognition and show what the system sees.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()

    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')
    
    print("Analyzing elixir digit recognition...")
    print("This will show you what the system sees vs what it recognizes")
    print()
    
    for i in range(args.samples):
        print(f"Sample {i+1}/{args.samples}:")
        
        frame = screen_bgr()
        if frame is None:
            print("  Could not capture screen")
            continue
            
        # Extract elixir region
        region = crs.extract_current_elixir_region(frame)
        if region is None:
            print("  Could not extract elixir region")
            continue
            
        # Save the raw region
        cv2.imwrite(f'elixir_raw_{i}.png', region)
        print(f"  Raw elixir region saved: elixir_raw_{i}.png")
        
        # Show preprocessing variants
        variants = crs._digit_preprocess_variants(region)
        for j, variant in enumerate(variants):
            cv2.imwrite(f'elixir_variant_{i}_{j}.png', variant)
            print(f"  Preprocessing variant {j} saved: elixir_variant_{i}_{j}.png")
        
        # Try recognition
        elixir_val, conf = crs.recognize_current_elixir(frame)
        print(f"  Recognized: {elixir_val} (confidence: {conf:.2f})")
        
        # Show template matches
        print("  Template matches:")
        for label, template in crs.digit_templates.items():
            if template is not None:
                # Simple template matching
                res = cv2.matchTemplate(variants[0], template, cv2.TM_CCOEFF_NORMED)
                score = float(cv2.minMaxLoc(res)[1])
                print(f"    {label}: {score:.3f}")
        
        print()
        
        import time
        time.sleep(1)
    
    print("Analysis complete!")
    print("Check the generated images to see what the system is seeing.")
    print("Compare elixir_raw_*.png with templates/digits/*.png")


if __name__ == '__main__':
    main()
