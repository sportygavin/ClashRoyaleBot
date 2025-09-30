import argparse
import time
import random
import sys
import os

from typing import List, Tuple, Optional

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


def get_position_from_strategy(strategy: str, viewport_px, card_name: str = "", current_side: str = None) -> Tuple[int, int]:
    """Convert placement strategy to actual coordinates."""
    vx, vy, vw, vh = viewport_px
    
    # Determine side - use current_side if available, otherwise choose randomly
    if current_side is None:
        side = random.choice(['left', 'right'])
    else:
        side = current_side
    
    if strategy == "bridge_front":
        # Tanks and win conditions - play at bridge (front)
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left bridge
        else:
            x = vx + int(0.75 * vw)  # Right bridge
        
        # Giant goes even further forward (closest to bridge)
        if card_name.lower() == 'giant':
            y = vy + int(0.48 * vh)  # Giant goes highest up (closest to bridge)
        elif card_name.lower() in ['golem', 'pekka', 'mega_knight']:
            y = vy + int(0.52 * vh)  # Other tanks close to giant
        else:
            y = vy + int(0.62 * vh)  # Standard bridge position
        return x, y
        
    elif strategy == "behind_tank":
        # Support troops - play behind tanks on same side
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left side
        else:
            x = vx + int(0.75 * vw)  # Right side
        
        # PEKKA goes closer to Giant, other support troops higher up
        if card_name.lower() == 'pekka':
            y = vy + int(0.58 * vh)  # PEKKA very close to Giant
        else:
            y = vy + int(0.68 * vh)  # Other support troops higher up
        return x, y
        
    elif strategy == "enemy_side":
        # Spells - play directly on opponent towers
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left opponent tower
        else:
            x = vx + int(0.75 * vw)  # Right opponent tower
        y = vy + int(0.35 * vh)  # Opponent tower area
        return x, y
        
    elif strategy == "defensive_building":
        # Buildings - play defensively near towers
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left tower area
        else:
            x = vx + int(0.75 * vw)  # Right tower area
        y = vy + int(0.85 * vh)  # Near our towers
        return x, y
        
    elif strategy == "near_towers":
        # Defensive troops - play near our towers or behind them
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left tower area
        else:
            x = vx + int(0.75 * vw)  # Right tower area
        
        # Knight positioning is good, others can go further back
        if card_name.lower() == 'knight':
            y = vy + int(0.85 * vh)  # Knight position (good as is)
        else:
            y = vy + int(0.90 * vh)  # Further back behind towers
        return x, y
        
    else:
        # Default - center back
        x = vx + int(0.5 * vw)
        y = vy + int(0.8 * vh)
        return x, y


def get_card_priority(card_info: dict) -> int:
    """Get priority for card selection based on elixir cost and type."""
    elixir_cost = card_info.get('elixir_cost', 0)
    card_type = card_info.get('type', '').lower()
    placement_strategy = card_info.get('placement_strategy', 'near_towers')
    
    # High priority: Low cost cards for cycling
    if elixir_cost <= 2:
        return 1
    
    # Medium priority: Medium cost cards
    elif elixir_cost in [3, 4]:
        return 2
    
    # Low priority: High cost cards (only when we have lots of elixir)
    elif elixir_cost >= 5:
        return 3
    
    return 4


def main():
    parser = argparse.ArgumentParser(description='Strategic card placement using database placement strategies.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--min-gap', type=float, default=1.2)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--min-elixir', type=int, default=3, help='Minimum elixir to start playing')
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    vp = default_viewport(calib)
    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')

    start = time.time()
    last_t = 0.0
    current_side = None  # Track current side for consistency
    
    print("Database Strategic Bot Started!")
    print("Using placement strategies from card database:")
    print("- bridge_front: Tanks and win conditions")
    print("- behind_tank: Support troops")
    print("- enemy_side: Spells")
    print("- defensive_building: Buildings")
    print("- near_towers: Defensive troops")
    print()
    
    while time.time() - start < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(0.2)
            continue
            
        hand_info = crs.analyze_hand(frame)
        cur_elixir, econf = stable_elixir(crs, samples=5, delay_s=0.04)

        # Only start playing when we have minimum elixir
        if cur_elixir is None or cur_elixir < args.min_elixir:
            if args.show:
                print(f"Waiting for elixir... (current: {cur_elixir}, min: {args.min_elixir})")
            time.sleep(0.5)
            continue

        # Get affordable cards with their priorities
        affordable_cards: List[Tuple[int, dict, int]] = []
        if isinstance(hand_info, dict):
            for key in sorted(hand_info.keys()):
                card = hand_info[key]
                idx = card.get('card_number')
                card_info = card.get('card_info')
                
                # Skip if card_info is None or missing elixir_cost
                if card_info is None:
                    continue
                    
                cost = card_info.get('elixir_cost')
                if isinstance(idx, int) and isinstance(cost, int) and cur_elixir >= cost:
                    priority = get_card_priority(card_info)
                    affordable_cards.append((idx - 1, card_info, priority))

        # Sort by priority (lower priority number = higher priority)
        affordable_cards.sort(key=lambda x: x[2])

        if args.show:
            print(f"Elixir: {cur_elixir} (conf: {econf:.2f}) | Affordable: {len(affordable_cards)} cards")

        now = time.time()
        if affordable_cards and (now - last_t) >= args.min_gap:
            # Play the highest priority affordable card
            card_idx, card_info, priority = affordable_cards[0]
            card_cost = card_info.get('elixir_cost', 0)
            card_name = card_info.get('name', f'Card {card_idx+1}')
            placement_strategy = card_info.get('placement_strategy', 'near_towers')
            
            # Determine side for this card
            if placement_strategy in ['bridge_front', 'behind_tank']:
                # For tanks and support, choose side if not set, or use current side
                if current_side is None:
                    current_side = random.choice(['left', 'right'])
            elif placement_strategy == 'enemy_side':
                # For spells, use current side if available, otherwise choose randomly
                if current_side is None:
                    current_side = random.choice(['left', 'right'])
            
            card_xy = get_card_center_xy(calib, vp, card_idx)
            target_xy = get_position_from_strategy(placement_strategy, vp, card_name, current_side)
            
            if args.show:
                side_info = f" (side: {current_side})" if current_side else ""
                print(f"Playing {card_name} (cost: {card_cost}, strategy: {placement_strategy}{side_info}) -> {target_xy}")
            
            drag_card_to(card_xy, target_xy)
            last_t = now
        else:
            time.sleep(0.2)

    print("Database strategic bot finished!")


if __name__ == '__main__':
    main()
