import argparse
import cv2
import numpy as np
import pyautogui
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import load_calibration, default_viewport


def click_callback(event, x, y, flags, param):
    """Handle mouse clicks to set opponent region position."""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        
        # Calculate position relative to viewport
        viewport = param['viewport']
        vx, vy, vw, vh = viewport
        
        # Since we're showing only the viewport, click coordinates are already relative
        rel_x = x / vw
        rel_y = y / vh
        
        print(f"Relative position: x_r={rel_x:.6f}, y_r={rel_y:.6f}")
        
        # Create opponent region around the click point
        # Opponent region should cover the top area where opponent cards appear
        region_width = 0.8   # 80% of viewport width
        region_height = 0.4  # 40% of viewport height (top area)
        
        roi = {
            "x_r": rel_x - region_width/2,
            "y_r": rel_y - region_height/2,
            "w_r": region_width,
            "h_r": region_height
        }
        
        print(f"Opponent Region ROI: {roi}")
        
        # Update calibration file
        calib_path = 'cv_out/calibration_manual_fixed.json'
        calib = load_calibration(calib_path)
        calib['opponent_region_roi'] = roi
        
        with open(calib_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"Updated {calib_path} with new opponent region position")
        
        # Draw a rectangle on the image to show the selected area
        img = param['image'].copy()
        cv2.rectangle(img, 
                     (int(roi['x_r'] * vw), int(roi['y_r'] * vh)),
                     (int((roi['x_r'] + roi['w_r']) * vw), int((roi['y_r'] + roi['h_r']) * vh)),
                     (0, 255, 0), 2)
        
        cv2.imshow('Opponent Region Selector', img)
        print("Green rectangle shows the selected opponent region. Press 'q' to quit.")


def main():
    parser = argparse.ArgumentParser(description='Manually adjust opponent detection region.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--width', type=float, default=0.8, help='Region width (0.0-1.0)')
    parser.add_argument('--height', type=float, default=0.4, help='Region height (0.0-1.0)')
    args = parser.parse_args()
    
    print("Opponent Region Selector")
    print("1. Make sure Clash Royale is visible on screen")
    print("2. Click once on the center of where opponent cards appear")
    print("3. Press 'q' to quit")
    print()
    
    # Load calibration
    calib_path = args.calib
    calib = load_calibration(calib_path)
    viewport = default_viewport(calib)
    
    # Take screenshot and crop to viewport
    vx, vy, vw, vh = viewport
    screenshot = pyautogui.screenshot(region=(vx, vy, vw, vh))
    viewport_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Show image
    cv2.namedWindow('Opponent Region Selector', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Opponent Region Selector', click_callback, 
                        {'image': viewport_img, 'viewport': viewport})
    
    cv2.imshow('Opponent Region Selector', viewport_img)
    
    print("Click on the center of where opponent cards appear...")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()
