import argparse
import time

from game_scripts.strategy_utils import (
    load_calibration,
    default_viewport,
    get_card_center_xy,
    drag_card_to,
    screen_bgr,
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
        hand = crs.analyze_hand(screen_bgr())
        idx = int(time.time() * 2) % 4
        card_xy = get_card_center_xy(calib, vp, idx)
        target_xy = center_target(vp)
        if args.show:
            print(f"Playing card {idx+1} -> {target_xy}")
        drag_card_to(card_xy, target_xy)
        time.sleep(args.gap)


if __name__ == '__main__':
    main()


