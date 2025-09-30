import argparse
import cv2
import numpy as np
import pyautogui
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport


def main():
    parser = argparse.ArgumentParser(description='Debug opponent detection with detailed change analysis.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()
    
    calib = load_calibration(args.calib)
    viewport = default_viewport(calib)
    vx, vy, vw, vh = viewport
    
    # Get opponent region from calibration
    if 'opponent_region_roi' in calib:
        roi = calib['opponent_region_roi']
        opp_x_start = int(vx + roi['x_r'] * vw)
        opp_y_start = int(vy + roi['y_r'] * vh)
        opp_x_end = int(vx + (roi['x_r'] + roi['w_r']) * vw)
        opp_y_end = int(vy + (roi['y_r'] + roi['h_r']) * vh)
    else:
        # Default region (top half of viewport)
        opp_x_start = vx
        opp_y_start = vy
        opp_x_end = vx + vw
        opp_y_end = vy + vh // 2
    
    print("Opponent Detection Debug")
    print(f"Viewport: {viewport}")
    print(f"Opponent region: ({opp_x_start}, {opp_y_start}) to ({opp_x_end}, {opp_y_end})")
    print(f"Region size: {opp_x_end - opp_x_start} x {opp_y_end - opp_y_start}")
    print(f"Change threshold: {args.threshold}")
    print()
    
    prev_region = None
    frame_count = 0
    start_time = time.time()
    
    while time.time() - start_time < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Extract opponent region
        current_region = frame[opp_y_start:opp_y_end, opp_x_start:opp_x_end]
        
        if prev_region is not None:
            # Calculate difference
            diff = cv2.absdiff(prev_region, current_region)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Count changed pixels
            changed_pixels = np.sum(gray_diff > 30)  # Threshold for "changed"
            total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
            change_ratio = changed_pixels / total_pixels
            
            print(f"Frame {frame_count}: Change ratio = {change_ratio:.4f} ({changed_pixels}/{total_pixels} pixels)")
            
            if change_ratio > args.threshold:
                print(f"  *** DETECTED CHANGE! ***")
                
                # Save debug images
                cv2.imwrite(f'debug_prev_{frame_count}.png', prev_region)
                cv2.imwrite(f'debug_curr_{frame_count}.png', current_region)
                cv2.imwrite(f'debug_diff_{frame_count}.png', diff)
                
                # Create full screen with rectangle
                full_screen = frame.copy()
                cv2.rectangle(full_screen, 
                             (opp_x_start, opp_y_start), 
                             (opp_x_end, opp_y_end), 
                             (0, 255, 0), 3)
                cv2.imwrite(f'debug_full_{frame_count}.png', full_screen)
                
                print(f"  Saved debug images: debug_prev_{frame_count}.png, debug_curr_{frame_count}.png, debug_diff_{frame_count}.png")
        
        prev_region = current_region.copy()
        frame_count += 1
        time.sleep(0.5)  # Check every 0.5 seconds
    
    print(f"\nDebug complete. Checked {frame_count} frames.")
    print("If no changes were detected, try:")
    print("1. Lower threshold: --threshold 0.01")
    print("2. Check if opponent region is correct")
    print("3. Make sure you're playing cards in the opponent region")


if __name__ == '__main__':
    main()
