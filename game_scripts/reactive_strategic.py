import argparse
import time
import random
import sys
import os
from typing import List, Tuple, Optional, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import (
    load_calibration,
    default_viewport,
    get_card_center_xy,
    drag_card_to,
    screen_bgr,
    stable_elixir,
)
from tools.card_recognition_system import CardRecognitionSystem
from game_scripts.opponent_detector import OpponentDetector


class ReactiveStrategicBot:
    def __init__(self, calibration_path: str):
        self.calib = load_calibration(calibration_path)
        self.vp = default_viewport(self.calib)
        self.crs = CardRecognitionSystem(calibration_path, 'database/clash_royale_cards.json')
        self.opponent_detector = OpponentDetector(calibration_path)
        
        # Game state tracking
        self.current_side = None
        self.last_opponent_play = None
        self.last_opponent_side = None
        self.defensive_mode = False
        self.counter_attack_mode = False
        
    def get_position_from_strategy(self, strategy: str, card_name: str = "", force_side: str = None) -> Tuple[int, int]:
        """Convert placement strategy to actual coordinates with reactive logic."""
        vx, vy, vw, vh = self.vp
        
        # Determine side based on opponent's last play
        if force_side:
            side = force_side
        elif self.last_opponent_side and self.defensive_mode:
            # Defend on same side as opponent
            side = self.last_opponent_side
        elif self.last_opponent_side and self.counter_attack_mode:
            # Counter-attack on opposite side
            side = 'left' if self.last_opponent_side == 'right' else 'right'
        elif self.current_side:
            # Continue current push
            side = self.current_side
        else:
            # Random side
            side = random.choice(['left', 'right'])
            
        # Update current side
        self.current_side = side
        
        if strategy == "bridge_front":
            if side == 'left':
                x = vx + int(0.25 * vw)
            else:
                x = vx + int(0.75 * vw)
            
            if card_name.lower() == 'giant':
                y = vy + int(0.48 * vh)
            elif card_name.lower() in ['golem', 'pekka', 'mega_knight']:
                y = vy + int(0.52 * vh)
            else:
                y = vy + int(0.62 * vh)
            return x, y
            
        elif strategy == "behind_tank":
            if side == 'left':
                x = vx + int(0.25 * vw)
            else:
                x = vx + int(0.75 * vw)
            
            if card_name.lower() == 'pekka':
                y = vy + int(0.58 * vh)
            else:
                y = vy + int(0.68 * vh)
            return x, y
            
        elif strategy == "enemy_side":
            # Spells target opponent towers
            if side == 'left':
                x = vx + int(0.25 * vw)
            else:
                x = vx + int(0.75 * vw)
            y = vy + int(0.35 * vh)
            return x, y
            
        elif strategy == "defensive_building":
            if side == 'left':
                x = vx + int(0.25 * vw)
            else:
                x = vx + int(0.75 * vw)
            y = vy + int(0.85 * vh)
            return x, y
            
        elif strategy == "near_towers":
            if side == 'left':
                x = vx + int(0.25 * vw)
            else:
                x = vx + int(0.75 * vw)
            
            if card_name.lower() == 'knight':
                y = vy + int(0.85 * vh)
            else:
                y = vy + int(0.90 * vh)
            return x, y
            
        else:
            x = vx + int(0.5 * vw)
            y = vy + int(0.8 * vh)
            return x, y
    
    def get_card_priority(self, card_info: dict, opponent_card: Optional[Dict] = None) -> int:
        """Get priority for card selection based on opponent's play."""
        elixir_cost = card_info.get('elixir_cost', 0)
        card_type = card_info.get('type', '').lower()
        card_name = card_info.get('name', '').lower()
        
        # Adjust priority based on opponent's card
        if opponent_card:
            opponent_info = opponent_card.get('card_info', {})
            opponent_type = opponent_info.get('type', '').lower()
            
            # Prioritize counters
            if opponent_type == 'troop' and card_type == 'troop':
                # Defensive troops get higher priority
                if card_name in ['knight', 'valkyrie', 'mini_pekka']:
                    return 1  # High priority for defensive troops
            elif opponent_type == 'spell' and card_type == 'troop':
                # Counter-attack troops get higher priority
                return 2
            elif opponent_type == 'building' and card_type == 'troop':
                # Attack troops get higher priority
                return 2
        
        # Default priority based on elixir cost
        if elixir_cost <= 2:
            return 1
        elif elixir_cost in [3, 4]:
            return 2
        elif elixir_cost >= 5:
            return 3
        return 4
    
    def update_strategy_mode(self, opponent_card: Optional[Dict]):
        """Update strategy mode based on opponent's play."""
        if opponent_card:
            opponent_info = opponent_card.get('card_info', {})
            opponent_type = opponent_info.get('type', '').lower()
            
            # Determine opponent's side (simplified - assume left/right based on position)
            # In reality, we'd need more sophisticated side detection
            self.last_opponent_side = random.choice(['left', 'right'])
            
            if opponent_type == 'troop':
                self.defensive_mode = True
                self.counter_attack_mode = False
                print("🛡️ DEFENSIVE MODE: Opponent played troop")
            elif opponent_type == 'spell':
                self.defensive_mode = False
                self.counter_attack_mode = True
                print("⚔️ COUNTER-ATTACK MODE: Opponent played spell")
            elif opponent_type == 'building':
                self.defensive_mode = False
                self.counter_attack_mode = True
                print("🏰 ATTACK MODE: Opponent played building")
        else:
            # Reset modes after some time
            if time.time() - (self.last_opponent_play or 0) > 10:
                self.defensive_mode = False
                self.counter_attack_mode = False
    
    def run(self, duration: int, min_gap: float, show: bool, min_elixir: int):
        """Run the reactive strategic bot."""
        start = time.time()
        last_t = 0.0
        
        print("🤖 Reactive Strategic Bot Started!")
        print("Features:")
        print("- Detects opponent card plays")
        print("- Adapts strategy based on opponent")
        print("- Defends same side or counter-attacks opposite side")
        print()
        
        while time.time() - start < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.2)
                continue
            
            # Detect opponent plays
            opponent_card = self.opponent_detector.detect_opponent_card(frame)
            if opponent_card:
                self.last_opponent_play = time.time()
                self.update_strategy_mode(opponent_card)
            
            # Get our hand and elixir
            hand_info = self.crs.analyze_hand(frame)
            cur_elixir, econf = stable_elixir(self.crs, samples=3, delay_s=0.04)
            
            if cur_elixir is None or cur_elixir < min_elixir:
                if show:
                    print(f"Waiting for elixir... (current: {cur_elixir}, min: {min_elixir})")
                time.sleep(0.5)
                continue
            
            # Get affordable cards
            affordable_cards: List[Tuple[int, dict, int]] = []
            if isinstance(hand_info, dict):
                for key in sorted(hand_info.keys()):
                    card = hand_info[key]
                    idx = card.get('card_number')
                    card_info = card.get('card_info')
                    
                    if card_info is None:
                        continue
                        
                    cost = card_info.get('elixir_cost')
                    if isinstance(idx, int) and isinstance(cost, int) and cur_elixir >= cost:
                        priority = self.get_card_priority(card_info, opponent_card)
                        affordable_cards.append((idx - 1, card_info, priority))
            
            affordable_cards.sort(key=lambda x: x[2])
            
            if show:
                mode_info = ""
                if self.defensive_mode:
                    mode_info = " [DEFENSIVE]"
                elif self.counter_attack_mode:
                    mode_info = " [COUNTER-ATTACK]"
                print(f"Elixir: {cur_elixir} | Affordable: {len(affordable_cards)} cards{mode_info}")
            
            now = time.time()
            if affordable_cards and (now - last_t) >= min_gap:
                card_idx, card_info, priority = affordable_cards[0]
                card_cost = card_info.get('elixir_cost', 0)
                card_name = card_info.get('name', f'Card {card_idx+1}')
                placement_strategy = card_info.get('placement_strategy', 'near_towers')
                
                card_xy = get_card_center_xy(self.calib, self.vp, card_idx)
                target_xy = self.get_position_from_strategy(placement_strategy, card_name)
                
                if show:
                    side_info = f" (side: {self.current_side})" if self.current_side else ""
                    print(f"Playing {card_name} (cost: {card_cost}, strategy: {placement_strategy}{side_info}) -> {target_xy}")
                
                drag_card_to(card_xy, target_xy)
                last_t = now
            else:
                time.sleep(0.2)
        
        print("Reactive strategic bot finished!")


def main():
    parser = argparse.ArgumentParser(description='Reactive strategic bot that responds to opponent plays.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--min-gap', type=float, default=1.2)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--min-elixir', type=int, default=3)
    args = parser.parse_args()
    
    bot = ReactiveStrategicBot(args.calib)
    bot.run(args.duration, args.min_gap, args.show, args.min_elixir)


if __name__ == '__main__':
    main()
