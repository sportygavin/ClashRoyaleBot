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


class SimpleReactiveBot:
    def __init__(self, calibration_path: str, model_path: str):
        # Initialize YOLOv8 opponent detector
        self.opponent_detector = YOLOv8OpponentDetector(calibration_path, model_path)
        
        # Load calibration
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        
        # Game state tracking
        self.last_detection_time = 0
        self.reaction_cooldown = 2.0  # seconds between reactions
        
        # Strategy parameters
        self.confidence_threshold = 0.5
        
        # Card counter strategies
        self.counter_strategies = {
            "Princess": {
                "counters": ["Archer", "Musketeer"],
                "placement": "defensive",
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
        
        print("Simple Reactive Clash Royale Bot Initialized!")
        print(f"Monitoring opponent region: {self.opponent_detector.opponent_region}")
        print(f"Counter strategies loaded for {len(self.counter_strategies)} card types")
    
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
        
        # Check if opponent played high-priority cards
        for detection in opponent_cards:
            card_name = detection['class_name']
            confidence = detection['confidence']
            
            if confidence >= self.confidence_threshold:
                strategy = self.counter_strategies.get(card_name)
                if strategy and strategy['priority'] in ['high', 'medium']:
                    return True
        
        return False
    
    def simulate_card_placement(self, opponent_cards: List[Dict]):
        """Simulate placing a counter card (without actually placing)."""
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
        
        if highest_priority:
            strategy = self.counter_strategies[highest_priority]
            counters = strategy['counters']
            
            # Simulate choosing a counter (just print for now)
            print(f"  🤖 Would counter {highest_priority} with: {counters[0] if counters else 'Any defensive card'}")
            print(f"  📍 Placement strategy: {strategy['placement']}")
            
            # Simulate placement coordinates
            vx, vy, vw, vh = self.viewport
            if strategy['placement'] == "defensive":
                placement_x = vx + vw * 0.3  # Left side of our area
                placement_y = vy + vh * 0.7  # Lower part of our area
            else:
                placement_x = vx + vw * 0.5
                placement_y = vy + vh * 0.6
            
            print(f"  🎯 Would place at: ({placement_x:.0f}, {placement_y:.0f})")
            return True
        
        return False
    
    def run_reactive_bot(self, duration: int = 300):
        """Run the reactive bot that responds to opponent plays."""
        print("Simple Reactive Clash Royale Bot Starting!")
        print(f"Duration: {duration} seconds")
        print(f"Reaction cooldown: {self.reaction_cooldown} seconds")
        print(f"Confidence threshold: {self.confidence_threshold}")
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
                    print(f"\n🎯 Frame {frame_count}: Opponent detected!")
                    
                    # Log detected cards
                    for detection in opponent_cards:
                        print(f"  👁️  {detection['class_name']} (conf: {detection['confidence']:.3f})")
                    
                    # Simulate counter strategy
                    if self.simulate_card_placement(opponent_cards):
                        reactions += 1
                        self.last_detection_time = time.time()
                        print(f"  ✅ Reaction #{reactions} simulated!")
                    else:
                        print(f"  ❌ No suitable counter strategy")
                
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
        
        print(f"\n🏁 Simple reactive bot completed!")
        print(f"📈 Total reactions: {reactions}")
        print(f"📊 Total frames: {frame_count}")
        print(f"⚡ Reaction rate: {reactions/frame_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Simple Reactive Clash Royale Bot with YOLOv8')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--model', required=True, help='Path to trained YOLOv8 model')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--cooldown', type=float, default=2.0, help='Reaction cooldown in seconds')
    args = parser.parse_args()
    
    bot = SimpleReactiveBot(args.calib, args.model)
    bot.confidence_threshold = args.conf
    bot.reaction_cooldown = args.cooldown
    
    bot.run_reactive_bot(args.duration)


if __name__ == '__main__':
    main()
