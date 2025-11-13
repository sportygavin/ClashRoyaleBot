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

from game_scripts.strategy_utils import (
    screen_bgr, load_calibration, default_viewport, 
    get_card_center_xy, drag_card_to, stable_elixir
)
from src.vision.game_vision import ClashRoyaleVision
from core import GameState


class ReactiveClashRoyaleBot:
    def __init__(self, calibration_path: str, model_path: str = None):
        # Initialize ClashRoyaleVision (uses fixed coordinate system)
        self.vision = ClashRoyaleVision(calibration_path=calibration_path)
        
        # Load calibration for card placement
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
        print(f"Using ClashRoyaleVision with fixed coordinate system")
        print(f"YOLO available: {self.vision.yolo_available}")
        print(f"Counter strategies loaded for {len(self.counter_strategies)} card types")
    
    def get_current_hand(self) -> List[Dict]:
        """Get current cards in hand using ClashRoyaleVision."""
        try:
            screen = self.vision.capture_screen()
            if screen is None:
                return []
            
            # Extract game info (uses YOLO or fallback)
            game_info = self.vision.extract_game_info(screen)
            if game_info is None:
                return []
            
            # Convert Card objects to dict format
            recognized_cards = []
            for i, card in enumerate(game_info.player_cards):
                recognized_cards.append({
                    'name': card.name,
                    'confidence': 1.0,  # YOLO confidence if available
                    'elixir_cost': card.cost,
                    'position': i,
                    'is_available': card.is_available,
                    'card_obj': card  # Keep reference to original Card object
                })
            
            return recognized_cards
        except Exception as e:
            print(f"Error getting hand: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_current_elixir(self) -> int:
        """Get current elixir count."""
        try:
            screen = self.vision.capture_screen()
            if screen is None:
                return 0
            game_info = self.vision.extract_game_info(screen)
            if game_info:
                return int(game_info.current_elixir) if game_info.current_elixir else 0
            return 0
        except Exception as e:
            print(f"Error getting elixir: {e}")
            return 0
    
    def detect_opponent_cards(self) -> List[Dict]:
        """Detect opponent cards using ClashRoyaleVision."""
        try:
            screen = self.vision.capture_screen()
            if screen is None:
                return []
            
            # Use ClashRoyaleVision to detect opponent cards
            opponent_cards = self.vision.detect_opponent_cards(screen)
            
            # Convert Card objects to dict format
            detections = []
            for card in opponent_cards:
                detections.append({
                    'class_name': card.name,
                    'confidence': 1.0,  # YOLO confidence if available
                    'position': card.position,
                    'cost': card.cost
                })
            
            return detections
        except Exception as e:
            print(f"Error detecting opponent cards: {e}")
            import traceback
            traceback.print_exc()
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
        """Place a card on the board using correct coordinates (same as right_loop.py)."""
        try:
            # Get card position using get_card_center_xy (same as right_loop.py)
            card_index = card['position']
            if card_index >= 4:
                print(f"Invalid card position: {card_index}")
                return False
            
            card_xy = get_card_center_xy(self.calib, self.viewport, card_index)
            
            # Get placement coordinates based on strategy
            vx, vy, vw, vh = self.viewport
            if placement_type == "defensive":
                # Place near our towers (left side, lower part)
                target_x = vx + int(0.3 * vw)
                target_y = vy + int(0.7 * vh)
            else:
                # Default placement (center)
                target_x = vx + int(0.5 * vw)
                target_y = vy + int(0.6 * vh)
            
            target_xy = (target_x, target_y)
            
            # Use drag_card_to from strategy_utils (same as right_loop.py)
            drag_card_to(card_xy, target_xy, duration=0.3, pre_delay=0.15)
            
            print(f"Placed {card['name']} from {card_xy} to {target_xy}")
            return True
            
        except Exception as e:
            print(f"Error placing card: {e}")
            import traceback
            traceback.print_exc()
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
    parser = argparse.ArgumentParser(description='Reactive Clash Royale Bot with ClashRoyaleVision (Fixed Coordinates)')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json',
                       help='Calibration file path')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--cooldown', type=float, default=2.0, help='Reaction cooldown in seconds')
    parser.add_argument('--min-elixir', type=int, default=3, help='Minimum elixir for reactions')
    args = parser.parse_args()
    
    bot = ReactiveClashRoyaleBot(args.calib)
    bot.confidence_threshold = args.conf
    bot.reaction_cooldown = args.cooldown
    bot.min_elixir_for_reaction = args.min_elixir
    
    bot.run_reactive_bot(args.duration)


if __name__ == '__main__':
    main()
