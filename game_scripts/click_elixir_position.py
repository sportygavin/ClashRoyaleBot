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
    """Handle mouse clicks to set elixir position."""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        
        # Calculate position relative to viewport
        viewport = param['viewport']
        vx, vy, vw, vh = viewport
        
        # Since we're showing only the viewport, click coordinates are already relative
        rel_x = x / vw
        rel_y = y / vh
        
        print(f"Relative position: x_r={rel_x:.6f}, y_r={rel_y:.6f}")
        
        # Create a rectangle around the click point (wider, less tall)
        roi = {
            "x_r": rel_x - 0.04,  # 8% width total
            "y_r": rel_y - 0.015,  # 3% height total
            "w_r": 0.08,   # 8% width
            "h_r": 0.03    # 3% height
        }
        
        print(f"Elixir ROI: {roi}")
        
        # Update calibration file
        calib_path = 'cv_out/calibration_manual_fixed.json'
        calib = load_calibration(calib_path)
        calib['current_elixir_roi'] = roi
        
        with open(calib_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"Updated {calib_path} with new elixir position")
        
        # Draw a rectangle on the image to show the selected area
        img = param['image'].copy()
        cv2.rectangle(img, 
                     (int(roi['x_r'] * vw), int(roi['y_r'] * vh)),
                     (int((roi['x_r'] + roi['w_r']) * vw), int((roi['y_r'] + roi['h_r']) * vh)),
                     (0, 255, 0), 2)
        
        cv2.imshow('Elixir Position Selector', img)
        print("Green rectangle shows the selected elixir region. Press 'q' to quit.")


def main():
    print("Elixir Position Selector")
    print("1. Make sure Clash Royale is visible on screen")
    print("2. Click once on the elixir number")
    print("3. Press 'q' to quit")
    print()
    
    # Load calibration
    calib_path = 'cv_out/calibration_manual_fixed.json'
    calib = load_calibration(calib_path)
    viewport = default_viewport(calib)
    
    # Take screenshot and crop to viewport
    vx, vy, vw, vh = viewport
    screenshot = pyautogui.screenshot(region=(vx, vy, vw, vh))
    viewport_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Show image
    cv2.namedWindow('Elixir Position Selector', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Elixir Position Selector', click_callback, 
                        {'image': viewport_img, 'viewport': viewport})
    
    cv2.imshow('Elixir Position Selector', viewport_img)
    
    print("Click on the elixir number to set its position...")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()
