import argparse
import cv2
import numpy as np
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pyautogui

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport
from game_scripts.yolo_opponent_detector import YOLOv8OpponentDetector
from tools.card_recognition_system import CardRecognitionSystem


class ReactiveClashRoyaleBot:
    def __init__(self, calibration_path: str, model_path: str):
        # Initialize YOLOv8 opponent detector
        self.opponent_detector = YOLOv8OpponentDetector(calibration_path, model_path)
        
        # Initialize card recognition for our hand
        self.card_system = CardRecognitionSystem(calibration_file=calibration_path)
        
        # Load calibration
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        
        # Game state tracking
        self.last_opponent_cards = []
        self.last_detection_time = 0
        self.reaction_cooldown = 2.0  # seconds between reactions
        
        # Strategy parameters
        self.min_elixir_for_reaction = 3
        self.confidence_threshold = 0.5
        
        # Card counter strategies
        self.counter_strategies = {
            "Princess": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",  # Place near towers
                "priority": "high"
            },
            "Knight": {
                "counters": ["Archer", "Musketeer", "Goblin"],
                "placement": "defensive",
                "priority": "medium"
            },
            "Goblin Gang": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",
                "priority": "high"
            },
            "Ice Spirit": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",
                "priority": "low"
            },
            "Goblin Barrel": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",
                "priority": "high"
            },
            "Spear Goblin": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",
                "priority": "medium"
            },
            "Goblin": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",
                "priority": "medium"
            }
        }
        
        print("Reactive Clash Royale Bot Initialized!")
        print(f"Monitoring opponent region: {self.opponent_detector.opponent_region}")
        print(f"Counter strategies loaded for {len(self.counter_strategies)} card types")
    
    def get_current_hand(self) -> List[Dict]:
        """Get current cards in hand."""
        try:
            screenshot = screen_bgr()
            if screenshot is None:
                return []
            
            # Check if screenshot is valid
            if not hasattr(screenshot, 'shape') or len(screenshot.shape) != 3:
                print(f"Invalid screenshot format: {type(screenshot)}")
                return []
                
            cards = self.card_system.extract_cards_from_screen(screenshot)
            recognized_cards = []
            
            for i, card_img in enumerate(cards):
                if card_img is not None and hasattr(card_img, 'shape'):
                    card_name, confidence = self.card_system.recognize_card_by_template(card_img)
                    if card_name and confidence > 0.3:
                        card_info = self.card_system.database.get(card_name, {})
                        recognized_cards.append({
                            'name': card_name,
                            'confidence': confidence,
                            'elixir_cost': card_info.get('elixir_cost', 0),
                            'position': i
                        })
            
            return recognized_cards
        except Exception as e:
            print(f"Error getting hand: {e}")
            return []
    
    def get_current_elixir(self) -> int:
        """Get current elixir count."""
        try:
            screenshot = screen_bgr()
            if screenshot is None:
                return 0
            elixir = self.card_system.recognize_current_elixir(screenshot)
            # Handle tuple return (elixir, confidence)
            if isinstance(elixir, tuple):
                elixir = elixir[0]
            return int(elixir) if elixir is not None else 0
        except Exception as e:
            print(f"Error getting elixir: {e}")
            return 0
    
    def detect_opponent_cards(self) -> List[Dict]:
        """Detect opponent cards using YOLOv8."""
        try:
            frame = screen_bgr()
            if frame is None:
                return []
            
            detections = self.opponent_detector.detect_opponent_cards(frame)
            return detections
        except Exception as e:
            print(f"Error detecting opponent cards: {e}")
            return []
    
    def should_react_to_opponent(self, opponent_cards: List[Dict]) -> bool:
        """Determine if we should react to opponent's play."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_detection_time < self.reaction_cooldown:
            return False
        
        # Check if opponent played new cards
        if not opponent_cards:
            return False
        
        # Check if we have enough elixir
        current_elixir = self.get_current_elixir()
        if current_elixir < self.min_elixir_for_reaction:
            return False
        
        # Check if opponent played high-priority cards
        for detection in opponent_cards:
            card_name = detection['class_name']
            confidence = detection['confidence']
            
            if confidence >= self.confidence_threshold:
                strategy = self.counter_strategies.get(card_name)
                if strategy and strategy['priority'] in ['high', 'medium']:
                    return True
        
        return False
    
    def choose_counter_card(self, opponent_cards: List[Dict], hand: List[Dict]) -> Optional[Dict]:
        """Choose the best counter card from hand."""
        if not hand:
            return None
        
        # Find highest priority opponent card
        highest_priority = None
        highest_priority_level = 0
        
        for detection in opponent_cards:
            card_name = detection['class_name']
            confidence = detection['confidence']
            
            if confidence >= self.confidence_threshold:
                strategy = self.counter_strategies.get(card_name)
                if strategy:
                    priority_level = {'high': 3, 'medium': 2, 'low': 1}.get(strategy['priority'], 0)
                    if priority_level > highest_priority_level:
                        highest_priority_level = priority_level
                        highest_priority = card_name
        
        if not highest_priority:
            return None
        
        # Find best counter in hand
        strategy = self.counter_strategies[highest_priority]
        counters = strategy['counters']
        
        # Look for exact counter matches first
        for card in hand:
            if card['name'] in counters:
                return card
        
        # Look for any defensive card if no exact counter
        for card in hand:
            if card['name'] in ['Archer', 'Musketeer', 'Knight']:
                return card
        
        return None
    
    def place_card(self, card: Dict, placement_type: str = "defensive"):
        """Place a card on the board."""
        try:
            # Get card position in hand
            card_positions = self.calib.get('card_centers', [])
            if card['position'] >= len(card_positions):
                print(f"Invalid card position: {card['position']}")
                return False
            
            # Get placement coordinates based on strategy
            if placement_type == "defensive":
                # Place near our towers
                vx, vy, vw, vh = self.viewport
                placement_x = vx + vw * 0.3  # Left side of our area
                placement_y = vy + vh * 0.7  # Lower part of our area
            else:
                # Default placement
                vx, vy, vw, vh = self.viewport
                placement_x = vx + vw * 0.5
                placement_y = vy + vh * 0.6
            
            # Get card center position
            card_center = card_positions[card['position']]
            card_x = int(self.viewport[0] + card_center['x_r'] * self.viewport[2])
            card_y = int(self.viewport[1] + card_center['y_r'] * self.viewport[3])
            
            # Drag card to placement position
            pyautogui.moveTo(card_x, card_y)
            pyautogui.dragTo(placement_x, placement_y, duration=0.3)
            
            print(f"Placed {card['name']} at ({placement_x:.0f}, {placement_y:.0f})")
            return True
            
        except Exception as e:
            print(f"Error placing card: {e}")
            return False
    
    def run_reactive_bot(self, duration: int = 300):
        """Run the reactive bot that responds to opponent plays."""
        print("Reactive Clash Royale Bot Starting!")
        print(f"Duration: {duration} seconds")
        print(f"Reaction cooldown: {self.reaction_cooldown} seconds")
        print(f"Min elixir for reaction: {self.min_elixir_for_reaction}")
        print()
        
        start_time = time.time()
        frame_count = 0
        reactions = 0
        
        while time.time() - start_time < duration:
            try:
                # Detect opponent cards
                opponent_cards = self.detect_opponent_cards()
                
                # Check if we should react
                if self.should_react_to_opponent(opponent_cards):
                    print(f"\nFrame {frame_count}: Opponent detected!")
                    
                    # Log detected cards
                    for detection in opponent_cards:
                        print(f"  - {detection['class_name']} (conf: {detection['confidence']:.3f})")
                    
                    # Get current hand
                    hand = self.get_current_hand()
                    if hand:
                        print(f"  Current hand: {[card['name'] for card in hand]}")
                        
                        # Choose counter card
                        counter_card = self.choose_counter_card(opponent_cards, hand)
                        
                        if counter_card:
                            print(f"  Countering with: {counter_card['name']}")
                            
                            # Place the counter card
                            if self.place_card(counter_card, "defensive"):
                                reactions += 1
                                self.last_detection_time = time.time()
                                print(f"  ✓ Reaction #{reactions} successful!")
                            else:
                                print(f"  ✗ Failed to place counter card")
                        else:
                            print(f"  No suitable counter card in hand")
                    else:
                        print(f"  Could not detect hand cards")
                
                frame_count += 1
                
                # Status update every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"Status: {elapsed:.1f}s elapsed, {reactions} reactions, {frame_count} frames")
                
                time.sleep(0.5)  # Check every 0.5 seconds
                
            except KeyboardInterrupt:
                print("\nBot stopped by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)
        
        print(f"\nReactive bot completed!")
        print(f"Total reactions: {reactions}")
        print(f"Total frames: {frame_count}")
        print(f"Reaction rate: {reactions/frame_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Reactive Clash Royale Bot with YOLOv8')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--cooldown', type=float, default=2.0, help='Reaction cooldown in seconds')
    parser.add_argument('--min-elixir', type=int, default=3, help='Minimum elixir for reactions')
    args = parser.parse_args()
    
    bot = ReactiveClashRoyaleBot(args.calib, args.model)
    bot.confidence_threshold = args.conf
    bot.reaction_cooldown = args.cooldown
    bot.min_elixir_for_reaction = args.min_elixir
    
    bot.run_reactive_bot(args.duration)


if __name__ == '__main__':
    main()
