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
    """Handle mouse clicks to set elixir bar position."""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        
        # Calculate position relative to viewport
        viewport = param['viewport']
        vx, vy, vw, vh = viewport
        
        # Since we're showing only the viewport, click coordinates are already relative
        rel_x = x / vw
        rel_y = y / vh
        
        print(f"Relative position: x_r={rel_x:.6f}, y_r={rel_y:.6f}")
        
        # Create a rectangle around the click point for the elixir bar
        # Bar should be much wider to capture the entire elixir bar
        bar_width = 0.8   # 80% of viewport width
        bar_height = 0.04 # 4% of viewport height
        
        roi = {
            "x_r": rel_x - bar_width/2,
            "y_r": rel_y - bar_height/2,
            "w_r": bar_width,
            "h_r": bar_height
        }
        
        print(f"Elixir Bar ROI: {roi}")
        
        # Update calibration file
        calib_path = 'cv_out/calibration_manual_fixed.json'
        calib = load_calibration(calib_path)
        calib['elixir_bar_roi'] = roi
        
        with open(calib_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"Updated {calib_path} with new elixir bar position")
        
        # Draw a rectangle on the image to show the selected area
        img = param['image'].copy()
        cv2.rectangle(img, 
                     (int(roi['x_r'] * vw), int(roi['y_r'] * vh)),
                     (int((roi['x_r'] + roi['w_r']) * vw), int((roi['y_r'] + roi['h_r']) * vh)),
                     (0, 255, 0), 2)
        
        cv2.imshow('Elixir Bar Position Selector', img)
        print("Green rectangle shows the selected elixir bar region. Press 'q' to quit.")


def main():
    print("Elixir Bar Position Selector")
    print("1. Make sure Clash Royale is visible on screen")
    print("2. Click once on the center of the elixir bar")
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
    cv2.namedWindow('Elixir Bar Position Selector', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Elixir Bar Position Selector', click_callback, 
                        {'image': viewport_img, 'viewport': viewport})
    
    cv2.imshow('Elixir Bar Position Selector', viewport_img)
    
    print("Click on the center of the elixir bar to set its position...")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()
