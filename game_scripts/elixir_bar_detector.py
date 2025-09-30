import argparse
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport


def detect_elixir_by_bar(screenshot, calibration):
    """Detect elixir by counting filled segments in the elixir bar."""
    
    # Get viewport
    viewport = default_viewport(calibration)
    vx, vy, vw, vh = viewport
    
    # Get elixir bar region from calibration - use FULL SCREEN coordinates
    if 'elixir_bar_roi' in calibration:
        bar_roi = calibration['elixir_bar_roi']
        # Convert viewport-relative coordinates to full screen coordinates
        bar_x_start = int(vx + bar_roi['x_r'] * vw)
        bar_y_start = int(vy + bar_roi['y_r'] * vh)
        bar_x_end = int(vx + (bar_roi['x_r'] + bar_roi['w_r']) * vw)
        bar_y_end = int(vy + (bar_roi['y_r'] + bar_roi['h_r']) * vh)
    else:
        # Fallback to default position if not calibrated
        bar_x_start = int(vx + 0.20 * vw)
        bar_y_start = int(vy + 0.96 * vh)
        bar_x_end = int(vx + 0.80 * vw)
        bar_y_end = int(vy + 0.99 * vh)
    
    # Extract bar region from full screenshot
    bar_region = screenshot[bar_y_start:bar_y_end, bar_x_start:bar_x_end]
    
    if bar_region.size == 0:
        return None, 0.0
    
    # Save debug image of the bar region
    cv2.imwrite('elixir_bar_debug.png', bar_region)
    
    # Create full screen image with elixir bar rectangle
    full_screen = screenshot.copy()
    # Draw rectangle on full screen (coordinates are already in screen space)
    cv2.rectangle(full_screen, 
                 (bar_x_start, bar_y_start), 
                 (bar_x_end, bar_y_end), 
                 (0, 255, 0), 3)  # Green rectangle
    cv2.imwrite('elixir_bar_full_screen.png', full_screen)
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for filled (pink/magenta) segments
    # Pink/magenta in HSV: Hue around 300-330, high saturation, high value
    lower_pink = np.array([140, 50, 50])   # Lower bound for pink/magenta
    upper_pink = np.array([180, 255, 255])  # Upper bound for pink/magenta
    
    # Create mask for filled segments
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Save mask for debugging
    cv2.imwrite('elixir_bar_mask.png', mask)
    
    # Find contours of filled segments
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (segments should be roughly rectangular)
    bar_height = bar_y_end - bar_y_start
    bar_width = bar_x_end - bar_x_start
    expected_segment_width = bar_width // 10  # 10 segments total
    expected_segment_area = expected_segment_width * bar_height * 0.5  # At least 50% of expected area
    
    filled_segments = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > expected_segment_area:
            filled_segments += 1
    
    # Alternative method: count by horizontal position
    # Divide bar into 10 equal segments and check each one
    segment_width = bar_width // 10
    filled_count = 0
    
    for i in range(10):
        segment_x_start = i * segment_width
        segment_x_end = (i + 1) * segment_width
        segment = mask[:, segment_x_start:segment_x_end]
        
        # Count non-zero pixels in this segment
        filled_pixels = np.count_nonzero(segment)
        total_pixels = segment.size
        
        # If more than 30% of segment is filled, count it
        if filled_pixels > total_pixels * 0.3:
            filled_count += 1
    
    # Return the count with confidence based on how clear the detection was
    confidence = min(1.0, filled_count / 10.0 + 0.5)  # Higher confidence for clearer detections
    
    return filled_count, confidence


def main():
    parser = argparse.ArgumentParser(description='Test elixir detection by counting bar segments.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=15)
    parser.add_argument('--hz', type=float, default=3.0)
    args = parser.parse_args()
    
    calibration = load_calibration(args.calib)
    
    print("Testing elixir bar detection...")
    print("This will count filled segments instead of reading digits")
    print()
    
    import time
    interval = 1.0 / args.hz
    start = time.time()
    
    while time.time() - start < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(interval)
            continue
            
        elixir_count, confidence = detect_elixir_by_bar(frame, calibration)
        print(f"Elixir: {elixir_count} (confidence: {confidence:.2f})")
        
        time.sleep(interval)
    
    print("Done! Check elixir_bar_debug.png and elixir_bar_mask.png for visual analysis.")


if __name__ == '__main__':
    main()
