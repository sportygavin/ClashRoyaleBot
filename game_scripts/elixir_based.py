import argparse
import time
import random

from typing import List, Tuple, Optional

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
    parser = argparse.ArgumentParser(description='Play cards only when affordable based on current elixir.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--min-gap', type=float, default=1.2)
    parser.add_argument('--prefer-cheap', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    vp = default_viewport(calib)
    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')

    start = time.time()
    last_t = 0.0
    while time.time() - start < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(0.2)
            continue
        hand_info = crs.analyze_hand(frame)
        cur_elixir, econf = stable_elixir(crs, samples=5, delay_s=0.04)

        affordable: List[Tuple[int, int]] = []
        if isinstance(hand_info, dict):
            for key in sorted(hand_info.keys()):
                card = hand_info[key]
                idx = card.get('card_number')
                cost = (card.get('card_info') or {}).get('elixir_cost')
                if isinstance(idx, int) and isinstance(cost, int) and cur_elixir is not None:
                    if cur_elixir >= cost:
                        affordable.append((idx - 1, cost))

        if args.prefer_cheap and affordable:
            affordable.sort(key=lambda x: x[1])

        if args.show:
            print(f"elixir={cur_elixir} conf={econf:.2f} affordable={affordable}")

        now = time.time()
        if affordable and (now - last_t) >= args.min_gap:
            idx, _ = affordable[0]
            card_xy = get_card_center_xy(calib, vp, idx)
            target_xy = center_target(vp)
            drag_card_to(card_xy, target_xy)
            last_t = now
        else:
            time.sleep(0.2)


if __name__ == '__main__':
    main()


