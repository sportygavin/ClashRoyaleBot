import argparse
import time
import sys
import os

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


def center_target(viewport_px):
    vx, vy, vw, vh = viewport_px
    x = vx + int(0.5 * vw)
    y = vy + int(0.62 * vh)
    return x, y


def main():
    parser = argparse.ArgumentParser(description='Repeatedly place cards near center-front.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--gap', type=float, default=1.0)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    vp = default_viewport(calib)
    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')

    start = time.time()
    while time.time() - start < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(0.2)
            continue
            
        hand = crs.analyze_hand(frame)
        cur_elixir, econf = stable_elixir(crs, samples=3, delay_s=0.04)
        
        idx = int(time.time() * 2) % 4
        card_xy = get_card_center_xy(calib, vp, idx)
        target_xy = center_target(vp)
        
        # Check if we have enough elixir to play this card
        if isinstance(hand, dict) and cur_elixir is not None:
            card_info = hand.get(f'card_{idx+1}', {})
            card_cost = (card_info.get('card_info') or {}).get('elixir_cost')
            
            if isinstance(card_cost, int) and cur_elixir >= card_cost:
                if args.show:
                    print(f"Playing card {idx+1} (cost: {card_cost}, elixir: {cur_elixir}) -> {target_xy}")
                drag_card_to(card_xy, target_xy)
            elif args.show:
                print(f"Skipping card {idx+1} (cost: {card_cost}, elixir: {cur_elixir}) - not enough elixir")
        else:
            if args.show:
                print(f"Playing card {idx+1} -> {target_xy} (no elixir check)")
            drag_card_to(card_xy, target_xy)
            
        time.sleep(args.gap)


if __name__ == '__main__':
    main()


