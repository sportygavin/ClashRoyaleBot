import argparse
import cv2
import numpy as np
import sys
import os
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pyautogui

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport
from game_scripts.yolo_opponent_detector import YOLOv8OpponentDetector
from tools.card_recognition_system import CardRecognitionSystem


def ratios_to_abs(viewport_r: dict, screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
    """Convert relative viewport ratios to absolute pixel coordinates."""
    x = int(viewport_r['x_r'] * screen_w)
    y = int(viewport_r['y_r'] * screen_h)
    w = int(viewport_r['w_r'] * screen_w)
    h = int(viewport_r['h_r'] * screen_h)
    return x, y, w, h


def get_card_center_xy(calib: dict, viewport_px: Tuple[int, int, int, int], card_index: int) -> Tuple[int, int]:
    """Get card center coordinates for mouse placement."""
    vx, vy, vw, vh = viewport_px

    centers_x_r = calib['cards']['centers_x_r']
    cx = int(vx + centers_x_r[card_index] * vw)

    row_top_r = calib['card_row']['top_r']
    row_bottom_r = calib['card_row']['bottom_r']
    row_top_y = vy + int(row_top_r * vh)
    row_bottom_y = vy + int(row_bottom_r * vh)
    row_h = max(row_bottom_y - row_top_y, 1)

    top_offset_r = calib['cards'].get('top_offset_r', 0.1)
    bottom_offset_r = calib['cards'].get('bottom_offset_r', 0.1)
    card_top = row_top_y + int(top_offset_r * row_h)
    card_bottom = row_bottom_y - int(bottom_offset_r * row_h)
    cy = (card_top + card_bottom) // 2

    return cx, cy


def choose_defensive_target(viewport_px: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Choose defensive placement target (closer to bridge for better positioning)."""
    vx, vy, vw, vh = viewport_px
    
    # Place closer to bridge (higher up on board)
    target_x_r = random.uniform(0.3, 0.4)   # Center-left side
    target_y_r = random.uniform(0.5, 0.6)   # Higher up, closer to bridge
    
    tx = vx + int(target_x_r * vw)
    ty = vy + int(target_y_r * vh)
    return tx, ty


def drag_card_to(card_xy: Tuple[int, int], target_xy: Tuple[int, int], duration: float = 0.25, pre_delay: float = 0.15):
    """Drag card from source to target position."""
    sx, sy = card_xy
    tx, ty = target_xy
    time.sleep(pre_delay)
    pyautogui.moveTo(sx, sy, duration=0.08)
    pyautogui.dragTo(tx, ty, duration=duration, button='left')


class CompleteReactiveBot:
    def __init__(self, calibration_path: str, model_path: str):
        # Initialize YOLOv8 opponent detector
        self.opponent_detector = YOLOv8OpponentDetector(calibration_path, model_path)
        
        # Initialize card recognition for our hand (using existing working system)
        self.card_system = CardRecognitionSystem(calibration_file=calibration_path)
        
        # Load calibration
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        
        # Game state tracking
        self.last_detection_time = 0
        self.reaction_cooldown = 2.0  # seconds between reactions
        
        # Strategy parameters
        self.min_elixir_for_reaction = 3
        self.confidence_threshold = 0.5
        
        # Smart reaction tracking to avoid spamming
        self.recent_reactions = []  # Track recent reactions
        self.max_reactions_per_opponent = 2  # Max reactions per opponent card
        self.reaction_window = 5.0  # Time window for tracking reactions (seconds)
        
        # Card counter strategies (using exact card names from our database)
        self.counter_strategies = {
            "Princess": {
                "counters": ["Archers", "Musketeer"],
                "placement": "defensive",  # Place near towers
                "priority": "high"
            },
            "Knight": {
                "counters": ["Archers", "Musketeer", "muskateer", "Goblins"],
                "placement": "defensive",
                "priority": "medium"
            },
            "Goblin Gang": {
                "counters": ["Archers", "Musketeer"],
                "placement": "defensive",
                "priority": "high"
            },
            "Ice Spirit": {
                "counters": ["Archers", "Musketeer"],
                "placement": "defensive",
                "priority": "low"
            },
            "Goblin Barrel": {
                "counters": ["Archers", "Musketeer"],
                "placement": "defensive",
                "priority": "high"
            },
            "Spear Goblin": {
                "counters": ["Archers", "Musketeer"],
                "placement": "defensive",
                "priority": "medium"
            },
            "Goblin": {
                "counters": ["Archers", "Musketeer"],
                "placement": "defensive",
                "priority": "medium"
            }
        }
        
        print("Complete Reactive Clash Royale Bot Initialized!")
        print(f"Monitoring opponent region: {self.opponent_detector.opponent_region}")
        print(f"Counter strategies loaded for {len(self.counter_strategies)} card types")
    
    def should_react_to_opponent_card(self, opponent_card: str, current_time: float) -> bool:
        """Check if we should react to this opponent card to avoid spamming."""
        # Clean old reactions outside the window
        self.recent_reactions = [
            r for r in self.recent_reactions 
            if current_time - r['time'] <= self.reaction_window
        ]
        
        # Count recent reactions to this specific opponent card
        recent_count = sum(1 for r in self.recent_reactions if r['opponent_card'] == opponent_card)
        
        if recent_count >= self.max_reactions_per_opponent:
            print(f"  ⏸️  Skipping reaction to {opponent_card} (already reacted {recent_count} times recently)")
            return False
        
        return True
    
    def record_reaction(self, opponent_card: str, our_card: str, current_time: float):
        """Record a reaction for spam prevention."""
        self.recent_reactions.append({
            'opponent_card': opponent_card,
            'our_card': our_card,
            'time': current_time
        })
    
    def get_current_hand(self, screenshot=None):
        """Get current cards in hand using the existing working system."""
        try:
            if screenshot is None:
                screenshot = screen_bgr()
            
            if screenshot is None:
                return None
                
            # Use the existing analyze_hand method
            hand_info = self.card_system.analyze_hand(screenshot)
            return hand_info
        except Exception as e:
            print(f"Error getting hand: {e}")
            return None
    
    def get_current_elixir(self, screenshot=None):
        """Get current elixir count using the existing working system."""
        try:
            if screenshot is None:
                screenshot = screen_bgr()
            
            if screenshot is None:
                return 0
                
            # Use the existing elixir detection
            elixir_result = self.card_system.recognize_current_elixir(screenshot)
            
            # Handle tuple return (elixir, confidence)
            if isinstance(elixir_result, tuple):
                elixir = elixir_result[0]
            else:
                elixir = elixir_result
                
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
    
    def should_react_to_opponent(self, opponent_cards: List[Dict], current_elixir: int) -> bool:
        """Determine if we should react to opponent's play."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_detection_time < self.reaction_cooldown:
            return False
        
        # Check if opponent played new cards
        if not opponent_cards:
            return False
        
        # Check if we have enough elixir
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
    
    def choose_counter_card(self, opponent_cards: List[Dict], hand_info: Dict) -> Optional[Dict]:
        """Choose the best counter card from hand."""
        if not hand_info:
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
        
        print(f"  🎯 Looking for counters to {highest_priority}: {counters}")
        
        # Look for exact counter matches first
        for card_key, card_data in hand_info.items():
            if isinstance(card_data, dict):
                card_info = card_data.get('card_info', {})
                if card_info is None:
                    continue
                    
                card_name = card_info.get('name', '')
                print(f"    Checking card: {card_name}")
                
                if card_name in counters or card_name.lower() in [c.lower() for c in counters]:
                    print(f"    ✅ Found exact counter: {card_name}")
                    position = card_data.get('card_number', 1) - 1  # Convert to 0-based
                    print(f"    📍 Card position: {card_data.get('card_number', 1)} -> {position}")
                    return {
                        'name': card_name,
                        'position': position,
                        'elixir_cost': card_info.get('elixir_cost', 0),
                        'card_data': card_data
                    }
        
        # Look for any defensive card if no exact counter
        print(f"  🔍 No exact counter found, looking for defensive cards...")
        for card_key, card_data in hand_info.items():
            if isinstance(card_data, dict):
                card_info = card_data.get('card_info', {})
                if card_info is None:
                    continue
                    
                card_name = card_info.get('name', '')
                print(f"    Checking defensive card: {card_name}")
                
                if card_name in ['Archers', 'Musketeer', 'muskateer', 'Knight'] or card_name.lower() in ['archers', 'musketeer', 'muskateer', 'knight']:
                    print(f"    ✅ Found defensive card: {card_name}")
                    return {
                        'name': card_name,
                        'position': card_data.get('card_number', 1) - 1,  # Convert to 0-based
                        'elixir_cost': card_info.get('elixir_cost', 0),
                        'card_data': card_data
                    }
        
        print(f"  ❌ No suitable counter found")
        return None
    
    def place_card(self, card: Dict, placement_type: str = "defensive"):
        """Place a card on the board using the existing mouse placement system."""
        try:
            # Get screen dimensions
            screen_w, screen_h = pyautogui.size()
            
            # Convert viewport to absolute coordinates
            viewport_r = self.calib.get('viewport', {
                'x_r': 0.0, 'y_r': 0.0, 'w_r': 1.0, 'h_r': 1.0
            })
            viewport_px = ratios_to_abs(viewport_r, screen_w, screen_h)
            
            # Get card center coordinates
            card_xy = get_card_center_xy(self.calib, viewport_px, card['position'])
            
            # Choose target based on placement strategy
            if placement_type == "defensive":
                target_xy = choose_defensive_target(viewport_px)
            else:
                # Default placement
                vx, vy, vw, vh = viewport_px
                target_x = vx + int(0.5 * vw)
                target_y = vy + int(0.6 * vh)
                target_xy = (target_x, target_y)
            
            print(f"🎯 Placing {card['name']} from {card_xy} to {target_xy}")
            
            # Drag card to target position
            drag_card_to(card_xy, target_xy, duration=0.25, pre_delay=0.15)
            
            print(f"✅ Successfully placed {card['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error placing card: {e}")
            return False
    
    def run_reactive_bot(self, duration: int = 300):
        """Run the reactive bot that responds to opponent plays."""
        print("Complete Reactive Clash Royale Bot Starting!")
        print(f"Duration: {duration} seconds")
        print(f"Reaction cooldown: {self.reaction_cooldown} seconds")
        print(f"Min elixir for reaction: {self.min_elixir_for_reaction}")
        print()
        
        start_time = time.time()
        frame_count = 0
        reactions = 0
        
        while time.time() - start_time < duration:
            try:
                # Capture screenshot once for both detections
                screenshot = screen_bgr()
                if screenshot is None:
                    time.sleep(0.1)
                    continue
                
                # Detect opponent cards
                opponent_cards = self.detect_opponent_cards()
                
                # Get current elixir
                current_elixir = self.get_current_elixir(screenshot)
                
                # Check if we should react
                if self.should_react_to_opponent(opponent_cards, current_elixir):
                    current_time = time.time()
                    
                    # Get the primary opponent card for spam prevention
                    primary_opponent = opponent_cards[0]['class_name'] if opponent_cards else "Unknown"
                    
                    # Check spam prevention
                    if not self.should_react_to_opponent_card(primary_opponent, current_time):
                        continue
                    
                    print(f"\n🎯 Frame {frame_count}: Opponent detected!")
                    print(f"💰 Current elixir: {current_elixir}")
                    
                    # Log detected cards
                    for detection in opponent_cards:
                        print(f"  👁️  {detection['class_name']} (conf: {detection['confidence']:.3f})")
                    
                    # Get current hand
                    hand_info = self.get_current_hand(screenshot)
                    if hand_info:
                        print(f"  🃏 Current hand detected")
                        
                        # Choose counter card
                        counter_card = self.choose_counter_card(opponent_cards, hand_info)
                        
                        if counter_card:
                            print(f"  🤖 Countering with: {counter_card['name']} (cost: {counter_card['elixir_cost']})")
                            
                            # Check if we can afford the counter
                            if current_elixir >= counter_card['elixir_cost']:
                                # Place the counter card
                                if self.place_card(counter_card, "defensive"):
                                    reactions += 1
                                    self.last_detection_time = current_time
                                    # Record this reaction for spam prevention
                                    self.record_reaction(primary_opponent, counter_card['name'], current_time)
                                    print(f"  ✅ Reaction #{reactions} successful!")
                                else:
                                    print(f"  ❌ Failed to place counter card")
                            else:
                                print(f"  💸 Not enough elixir for {counter_card['name']} (need {counter_card['elixir_cost']}, have {current_elixir})")
                        else:
                            print(f"  ❌ No suitable counter card in hand")
                    else:
                        print(f"  ❌ Could not detect hand cards")
                
                frame_count += 1
                
                # Status update every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"📊 Status: {elapsed:.1f}s elapsed, {reactions} reactions, {frame_count} frames")
                
                time.sleep(0.5)  # Check every 0.5 seconds
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(1)
        
        print(f"\n🏁 Complete reactive bot finished!")
        print(f"📈 Total reactions: {reactions}")
        print(f"📊 Total frames: {frame_count}")
        print(f"⚡ Reaction rate: {reactions/frame_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Complete Reactive Clash Royale Bot with YOLOv8')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--cooldown', type=float, default=2.0, help='Reaction cooldown in seconds')
    parser.add_argument('--min-elixir', type=int, default=3, help='Minimum elixir for reactions')
    args = parser.parse_args()
    
    bot = CompleteReactiveBot(args.calib, args.model)
    bot.confidence_threshold = args.conf
    bot.reaction_cooldown = args.cooldown
    bot.min_elixir_for_reaction = args.min_elixir
    
    bot.run_reactive_bot(args.duration)


if __name__ == '__main__':
    main()
