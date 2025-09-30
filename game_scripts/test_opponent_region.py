import argparse
import cv2
import numpy as np
import pyautogui
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport


def main():
    parser = argparse.ArgumentParser(description='Test opponent detection region and show what area it monitors.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=10)
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
    
    print("Opponent Region Test")
    print(f"Viewport: {viewport}")
    print(f"Opponent region: ({opp_x_start}, {opp_y_start}) to ({opp_x_end}, {opp_y_end})")
    print(f"Region size: {opp_x_end - opp_x_start} x {opp_y_end - opp_y_start}")
    print()
    
    import time
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Create full screen image with opponent region rectangle
        full_screen = frame.copy()
        cv2.rectangle(full_screen, 
                     (opp_x_start, opp_y_start), 
                     (opp_x_end, opp_y_end), 
                     (0, 255, 0), 3)  # Green rectangle
        
        # Save full screen with rectangle
        cv2.imwrite(f'opponent_region_test_{frame_count}.png', full_screen)
        
        # Extract and save opponent region
        opponent_region = frame[opp_y_start:opp_y_end, opp_x_start:opp_x_end]
        cv2.imwrite(f'opponent_region_cropped_{frame_count}.png', opponent_region)
        
        print(f"Frame {frame_count}: Saved full screen and cropped opponent region")
        frame_count += 1
        time.sleep(1.0)  # Wait 1 second between captures
    
    print(f"\nCaptured {frame_count} frames.")
    print("Check these files:")
    print("- opponent_region_test_*.png (full screen with green rectangle)")
    print("- opponent_region_cropped_*.png (just the opponent region)")
    print()
    print("If the green rectangle is not over the opponent's side, run:")
    print("python3 game_scripts/adjust_opponent_region.py")


if __name__ == '__main__':
    main()
