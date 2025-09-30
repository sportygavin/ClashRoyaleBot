import argparse
import cv2
import numpy as np
import sys
import os
import time
from typing import List, Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport, get_card_center_xy
from tools.card_recognition_system import CardRecognitionSystem


class HandChangeDetector:
    def __init__(self, calibration_path: str):
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        self.crs = CardRecognitionSystem(calibration_path, 'database/clash_royale_cards.json')
        
        # Track previous hand state
        self.prev_hand = None
        self.prev_hand_names = None
        
    def get_current_hand(self) -> List[Optional[str]]:
        """Get current hand cards."""
        frame = screen_bgr()
        if frame is None:
            return [None] * 4
        
        hand = []
        for i in range(4):
            center_x, center_y = get_card_center_xy(self.calib, i)
            if center_x is None or center_y is None:
                hand.append(None)
                continue
                
            # Extract card region
            card_w = int(self.calib['cards']['width_r'] * self.viewport[2])
            card_h = int((self.calib['card_row']['bottom_r'] - self.calib['card_row']['top_r']) * self.viewport[3])
            
            x1 = int(center_x - card_w // 2)
            y1 = int(center_y - card_h // 2)
            x2 = int(center_x + card_w // 2)
            y2 = int(center_y + card_h // 2)
            
            # Ensure within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                card_region = frame[y1:y2, x1:x2]
                card_name = self.crs.recognize_card(card_region)
                hand.append(card_name)
            else:
                hand.append(None)
        
        return hand
    
    def detect_hand_changes(self) -> List[Dict]:
        """Detect what cards changed in our hand."""
        current_hand = self.get_current_hand()
        changes = []
        
        if self.prev_hand is not None:
            for i in range(4):
                prev_card = self.prev_hand[i]
                curr_card = current_hand[i]
                
                if prev_card != curr_card:
                    change_type = "unknown"
                    if prev_card is not None and curr_card is None:
                        change_type = "card_played"  # Card disappeared (we played it)
                    elif prev_card is None and curr_card is not None:
                        change_type = "card_drawn"  # Card appeared (we drew it)
                    elif prev_card is not None and curr_card is not None:
                        change_type = "card_changed"  # Different card
                    
                    changes.append({
                        'slot': i,
                        'prev_card': prev_card,
                        'curr_card': curr_card,
                        'change_type': change_type,
                        'timestamp': time.time()
                    })
        
        self.prev_hand = current_hand.copy()
        return changes
    
    def monitor_hand_changes(self, duration: int = 60):
        """Monitor hand changes for specified duration."""
        print("Hand Change Detector")
        print("Monitoring our hand for card changes...")
        print("This detects when WE play cards (cards disappear from hand)")
        print()
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            changes = self.detect_hand_changes()
            
            if changes:
                print(f"Frame {frame_count}: Hand changes detected!")
                for change in changes:
                    print(f"  Slot {change['slot']}: {change['prev_card']} -> {change['curr_card']} ({change['change_type']})")
                
                # Save screenshot when changes detected
                frame = screen_bgr()
                if frame is not None:
                    cv2.imwrite(f'hand_change_{frame_count}_{int(time.time())}.png', frame)
                    print(f"  Saved screenshot: hand_change_{frame_count}_{int(time.time())}.png")
            
            frame_count += 1
            time.sleep(0.5)  # Check every 0.5 seconds
        
        print(f"\nMonitoring complete. Checked {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description='Detect changes in our hand (when we play cards).')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    
    detector = HandChangeDetector(args.calib)
    detector.monitor_hand_changes(args.duration)


if __name__ == '__main__':
    main()
