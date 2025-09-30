import argparse
import time

from game_scripts.strategy_utils import (
    load_calibration,
    default_viewport,
    get_card_center_xy,
    drag_card_to,
)


def right_lane_target(viewport_px):
    vx, vy, vw, vh = viewport_px
    x = vx + int(0.72 * vw)
    y = vy + int(0.62 * vh)
    return x, y


def main():
    parser = argparse.ArgumentParser(description='Repeatedly place cards on right lane front.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--gap', type=float, default=1.0)
    parser.add_argument('--card', type=int, default=1, choices=[1,2,3,4])
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    vp = default_viewport(calib)

    start = time.time()
    while time.time() - start < args.duration:
        idx = args.card - 1
        card_xy = get_card_center_xy(calib, vp, idx)
        target_xy = right_lane_target(vp)
        drag_card_to(card_xy, target_xy)
        time.sleep(args.gap)


if __name__ == '__main__':
    main()


