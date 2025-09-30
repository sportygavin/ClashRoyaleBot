import argparse
import time
import cv2
import numpy as np
import sys
import os
from typing import Optional, Tuple, Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport
from tools.card_recognition_system import CardRecognitionSystem


class OpponentDetector:
    def __init__(self, calibration_path: str):
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        self.crs = CardRecognitionSystem(calibration_path, 'database/clash_royale_cards.json')
        
        # Previous frame for change detection
        self.prev_frame = None
        self.prev_opponent_region = None
        
        # Opponent region from calibration file
        vx, vy, vw, vh = self.viewport
        
        if 'opponent_region_roi' in self.calib:
            # Use calibrated opponent region
            roi = self.calib['opponent_region_roi']
            self.opponent_region = {
                'x': int(vx + roi['x_r'] * vw),
                'y': int(vy + roi['y_r'] * vh),
                'w': int(roi['w_r'] * vw),
                'h': int(roi['h_r'] * vh)
            }
        else:
            # Fallback to top half of viewport
            self.opponent_region = {
                'x': vx,
                'y': vy,
                'w': vw,
                'h': vh // 2
            }
        
        # Change detection threshold
        self.change_threshold = 0.1  # 10% of pixels changed
        
    def detect_opponent_card(self, current_frame: np.ndarray) -> Optional[Dict]:
        """Detect if opponent played a card and identify it."""
        if current_frame is None:
            return None
            
        # Extract opponent region (top half)
        opp_region = current_frame[
            self.opponent_region['y']:self.opponent_region['y'] + self.opponent_region['h'],
            self.opponent_region['x']:self.opponent_region['x'] + self.opponent_region['w']
        ]
        
        if self.prev_opponent_region is None:
            self.prev_opponent_region = opp_region.copy()
            return None
            
        # Calculate difference between frames
        diff = cv2.absdiff(opp_region, self.prev_opponent_region)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Count changed pixels
        changed_pixels = np.count_nonzero(gray_diff > 30)  # Threshold for significant change
        total_pixels = gray_diff.size
        change_ratio = changed_pixels / total_pixels
        
        # Save debug images
        cv2.imwrite('opponent_current.png', opp_region)
        cv2.imwrite('opponent_previous.png', self.prev_opponent_region)
        cv2.imwrite('opponent_diff.png', diff)
        
        print(f"Opponent region change: {change_ratio:.3f} ({changed_pixels}/{total_pixels} pixels)")
        
        if change_ratio > self.change_threshold:
            print(f"OPPONENT PLAYED A CARD! Change ratio: {change_ratio:.3f}")
            
            # Try to identify the card using template matching
            card_info = self._identify_opponent_card(opp_region)
            
            # Update previous frame
            self.prev_opponent_region = opp_region.copy()
            
            return {
                'change_ratio': change_ratio,
                'card_info': card_info,
                'timestamp': time.time()
            }
        
        # Update previous frame
        self.prev_opponent_region = opp_region.copy()
        return None
        
    def _identify_opponent_card(self, opponent_region: np.ndarray) -> Optional[Dict]:
        """Try to identify what card the opponent played."""
        # This is a simplified version - in reality, we'd need to:
        # 1. Detect troop spawns in the opponent region
        # 2. Match against our card templates
        # 3. Handle different card types (troops, spells, buildings)
        
        # For now, let's just detect if there's a significant change
        # and try basic template matching
        
        best_match = None
        best_score = 0.0
        
        # Try matching against our card templates
        for card_name, template in self.crs.card_templates.items():
            if template is None:
                continue
                
            # Resize template to match opponent region scale
            h, w = opponent_region.shape[:2]
            template_resized = cv2.resize(template, (w//4, h//4))
            
            # Template matching
            result = cv2.matchTemplate(opponent_region, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score and max_val > 0.3:  # Threshold for match
                best_score = max_val
                best_match = card_name
                
        if best_match:
            print(f"Identified opponent card: {best_match} (confidence: {best_score:.2f})")
            return {
                'name': best_match,
                'confidence': best_score,
                'card_info': self.crs.database.get(best_match, {})
            }
        else:
            print("Could not identify opponent card")
            return None
            
    def get_counter_strategy(self, opponent_card: Dict) -> str:
        """Determine counter strategy based on opponent's card."""
        card_info = opponent_card.get('card_info', {})
        card_name = opponent_card.get('name', '').lower()
        card_type = card_info.get('type', '').lower()
        
        # Simple counter logic
        if card_type == 'troop':
            # If opponent plays troop, defend on same side
            return 'defend_same_side'
        elif card_type == 'spell':
            # If opponent plays spell, counter-attack on opposite side
            return 'counter_opposite_side'
        elif card_type == 'building':
            # If opponent plays building, attack opposite side
            return 'attack_opposite_side'
        else:
            # Default: defend same side
            return 'defend_same_side'


def main():
    parser = argparse.ArgumentParser(description='Detect opponent card plays and suggest counter strategies.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=0.1, help='Change detection threshold')
    args = parser.parse_args()
    
    detector = OpponentDetector(args.calib)
    detector.change_threshold = args.threshold
    
    print("Opponent Detection System Started!")
    print(f"Monitoring opponent region: {detector.opponent_region}")
    print(f"Change threshold: {args.threshold}")
    print("Waiting for opponent to play cards...")
    print()
    
    start_time = time.time()
    opponent_plays = []
    
    while time.time() - start_time < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(0.1)
            continue
            
        opponent_card = detector.detect_opponent_card(frame)
        
        if opponent_card:
            opponent_plays.append(opponent_card)
            
            # Get counter strategy
            if opponent_card['card_info']:
                strategy = detector.get_counter_strategy(opponent_card['card_info'])
                print(f"Counter strategy: {strategy}")
                
                # Suggest what to play
                if strategy == 'defend_same_side':
                    print("→ Play defensive cards on same side")
                elif strategy == 'counter_opposite_side':
                    print("→ Play counter-attack cards on opposite side")
                elif strategy == 'attack_opposite_side':
                    print("→ Play attack cards on opposite side")
            
            print("-" * 50)
        
        time.sleep(0.1)  # Check 10 times per second
    
    print(f"\nDetection complete! Found {len(opponent_plays)} opponent plays:")
    for i, play in enumerate(opponent_plays, 1):
        card_info = play.get('card_info', {})
        if card_info:
            print(f"{i}. {card_info.get('name', 'Unknown')} (confidence: {card_info.get('confidence', 0):.2f})")
        else:
            print(f"{i}. Unknown card (change ratio: {play['change_ratio']:.3f})")


if __name__ == '__main__':
    main()
