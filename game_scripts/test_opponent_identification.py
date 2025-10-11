import argparse
import cv2
import numpy as np
import sys
import os
import time
from typing import Optional, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport
from tools.card_recognition_system import CardRecognitionSystem


class OpponentCardIdentifier:
    def __init__(self, calibration_path: str):
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        self.crs = CardRecognitionSystem(calibration_path, 'database/clash_royale_cards.json')
        
        # Get opponent region from calibration
        vx, vy, vw, vh = self.viewport
        if 'opponent_region_roi' in self.calib:
            roi = self.calib['opponent_region_roi']
            self.opponent_region = {
                'x': int(vx + roi['x_r'] * vw),
                'y': int(vy + roi['y_r'] * vh),
                'w': int(roi['w_r'] * vw),
                'h': int(roi['h_r'] * vh)
            }
        else:
            self.opponent_region = {
                'x': vx,
                'y': vy,
                'w': vw,
                'h': vh // 2
            }
        
        # Previous frame for change detection
        self.prev_frame = None
        self.change_threshold = 0.1
        
    def detect_and_identify_opponent_card(self, current_frame: np.ndarray) -> Optional[Dict]:
        """Detect opponent card and try to identify it."""
        # Extract opponent region
        x, y, w, h = self.opponent_region['x'], self.opponent_region['y'], self.opponent_region['w'], self.opponent_region['h']
        opponent_region = current_frame[y:y+h, x:x+w]
        
        if opponent_region.size == 0:
            return None
        
        # Check for changes
        if self.prev_frame is not None:
            # Calculate difference
            diff = cv2.absdiff(self.prev_frame, opponent_region)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Count changed pixels
            changed_pixels = np.sum(gray_diff > 30)
            total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
            change_ratio = changed_pixels / total_pixels
            
            if change_ratio > self.change_threshold:
                print(f"Change detected! Ratio: {change_ratio:.4f}")
                
                # Try to identify the card
                card_name = self.crs.recognize_card(opponent_region)
                
                return {
                    'change_ratio': change_ratio,
                    'card_name': card_name,
                    'timestamp': time.time(),
                    'region': self.opponent_region
                }
        
        # Update previous frame
        self.prev_frame = opponent_region.copy()
        return None
    
    def test_opponent_identification(self, duration: int = 60):
        """Test opponent card identification."""
        print("Opponent Card Identifier Test")
        print(f"Monitoring region: {self.opponent_region}")
        print(f"Change threshold: {self.change_threshold}")
        print()
        print("Instructions:")
        print("1. Make sure Clash Royale is visible")
        print("2. Play some cards in the opponent region")
        print("3. Watch for detection messages")
        print()
        
        start_time = time.time()
        frame_count = 0
        detections = 0
        
        while time.time() - start_time < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            detection = self.detect_and_identify_opponent_card(frame)
            
            if detection:
                detections += 1
                print(f"Frame {frame_count}: OPPONENT CARD DETECTED!")
                print(f"  Change ratio: {detection['change_ratio']:.4f}")
                print(f"  Identified card: {detection['card_name']}")
                
                # Save debug images
                timestamp = int(time.time())
                cv2.imwrite(f'opponent_test_{frame_count}_{timestamp}.png', frame)
                
                # Extract and save opponent region
                x, y, w, h = self.opponent_region['x'], self.opponent_region['y'], self.opponent_region['w'], self.opponent_region['h']
                opponent_region = frame[y:y+h, x:x+w]
                cv2.imwrite(f'opponent_region_test_{frame_count}_{timestamp}.png', opponent_region)
                
                print(f"  Saved images: opponent_test_{frame_count}_{timestamp}.png")
                print(f"  Saved region: opponent_region_test_{frame_count}_{timestamp}.png")
                print()
            
            frame_count += 1
            time.sleep(0.5)  # Check every 0.5 seconds
        
        print(f"Test complete!")
        print(f"Total frames: {frame_count}")
        print(f"Detections: {detections}")
        print(f"Detection rate: {detections/frame_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Test opponent card identification.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()
    
    identifier = OpponentCardIdentifier(args.calib)
    identifier.change_threshold = args.threshold
    
    identifier.test_opponent_identification(args.duration)


if __name__ == '__main__':
    main()
