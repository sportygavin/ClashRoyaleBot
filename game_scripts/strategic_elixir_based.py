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


def get_card_strategy_position(card_info: dict, viewport_px) -> Tuple[int, int]:
    """Determine where to place a card based on its type and cost."""
    vx, vy, vw, vh = viewport_px
    
    elixir_cost = card_info.get('elixir_cost', 0)
    card_type = card_info.get('type', '').lower()
    rarity = card_info.get('rarity', '').lower()
    
    # High-cost tanks (5+ elixir) - place at bridge (front)
    if elixir_cost >= 5 and card_type in ['troop']:
        # Randomly choose left or right bridge
        side = random.choice(['left', 'right'])
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left bridge
        else:
            x = vx + int(0.75 * vw)  # Right bridge
        y = vy + int(0.62 * vh)  # Front position
        return x, y
    
    # Medium-cost cards (3-4 elixir) - place behind tanks
    elif elixir_cost in [3, 4]:
        # Place behind the bridge positions
        side = random.choice(['left', 'right'])
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left side
        else:
            x = vx + int(0.75 * vw)  # Right side
        y = vy + int(0.75 * vh)  # Behind tanks
        return x, y
    
    # Low-cost cards (1-2 elixir) - place near our towers
    elif elixir_cost <= 2:
        # Place near our towers (back of our side)
        side = random.choice(['left', 'right'])
        if side == 'left':
            x = vx + int(0.25 * vw)  # Left tower area
        else:
            x = vx + int(0.75 * vw)  # Right tower area
        y = vy + int(0.85 * vh)  # Near our towers
        return x, y
    
    # Default position - center back
    else:
        x = vx + int(0.5 * vw)
        y = vy + int(0.8 * vh)
        return x, y


def get_card_priority(card_info: dict) -> int:
    """Get priority for card selection (lower = higher priority)."""
    elixir_cost = card_info.get('elixir_cost', 0)
    card_type = card_info.get('type', '').lower()
    
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
    parser = argparse.ArgumentParser(description='Strategic elixir-based card placement with positioning strategy.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--min-gap', type=float, default=1.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--min-elixir', type=int, default=3, help='Minimum elixir to start playing')
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    vp = default_viewport(calib)
    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')

    start = time.time()
    last_t = 0.0
    
    print("Strategic Elixir-Based Bot Started!")
    print("Strategy:")
    print("- High-cost tanks (5+ elixir): Bridge positions (front)")
    print("- Medium-cost cards (3-4 elixir): Behind tanks")
    print("- Low-cost cards (1-2 elixir): Near our towers")
    print("- Only plays cards when affordable")
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
            
            card_xy = get_card_center_xy(calib, vp, card_idx)
            target_xy = get_card_strategy_position(card_info, vp)
            
            if args.show:
                print(f"Playing {card_name} (cost: {card_cost}, priority: {priority}) -> {target_xy}")
            
            drag_card_to(card_xy, target_xy)
            last_t = now
        else:
            time.sleep(0.2)

    print("Strategic bot finished!")


if __name__ == '__main__':
    main()
