import json
import time
import argparse
from typing import Tuple

import pyautogui


def load_calibration(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def ratios_to_abs(viewport_r: dict, screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
    x = int(viewport_r['x_r'] * screen_w)
    y = int(viewport_r['y_r'] * screen_h)
    w = int(viewport_r['w_r'] * screen_w)
    h = int(viewport_r['h_r'] * screen_h)
    return x, y, w, h


def get_card_center_xy(calib: dict, viewport_px: Tuple[int, int, int, int], card_index: int) -> Tuple[int, int]:
    vx, vy, vw, vh = viewport_px

    centers_x_r = calib['cards']['centers_x_r']
    if not (0 <= card_index < len(centers_x_r)):
        raise ValueError('card_index out of range for centers_x_r')

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


def get_target_xy(viewport_px: Tuple[int, int, int, int], target_x_r: float, target_y_r: float) -> Tuple[int, int]:
    vx, vy, vw, vh = viewport_px
    tx = vx + int(target_x_r * vw)
    ty = vy + int(target_y_r * vh)
    return tx, ty


def drag_card_to(card_xy: Tuple[int, int], target_xy: Tuple[int, int], duration: float = 0.25, pre_delay: float = 0.25):
    sx, sy = card_xy
    tx, ty = target_xy
    time.sleep(pre_delay)
    pyautogui.moveTo(sx, sy, duration=0.1)
    pyautogui.dragTo(tx, ty, duration=duration, button='left')


def main():
    parser = argparse.ArgumentParser(description='Pick a card from hand (1-4) and place on board.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json', help='Calibration JSON path')
    parser.add_argument('--card', type=int, required=True, choices=[1, 2, 3, 4], help='Card index in hand (1-4)')
    parser.add_argument('--target-x', type=float, default=0.5, help='Target X as ratio of viewport width [0-1]')
    parser.add_argument('--target-y', type=float, default=0.65, help='Target Y as ratio of viewport height [0-1]')
    parser.add_argument('--duration', type=float, default=0.25, help='Drag duration in seconds')
    parser.add_argument('--delay', type=float, default=0.25, help='Pre-drag delay in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Print coordinates but do not move mouse')
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    screen_w, screen_h = pyautogui.size()

    viewport_r = calib.get('viewport') or {
        'x_r': 0.0,
        'y_r': 0.0,
        'w_r': 1.0,
        'h_r': 1.0,
    }
    viewport_px = ratios_to_abs(viewport_r, screen_w, screen_h)

    card_xy = get_card_center_xy(calib, viewport_px, args.card - 1)
    target_xy = get_target_xy(viewport_px, args.target_x, args.target_y)

    print(f"Viewport(px): {viewport_px}")
    print(f"Card {args.card} center(px): {card_xy}")
    print(f"Target(px): {target_xy}")

    if args.dry_run:
        return

    drag_card_to(card_xy, target_xy, duration=args.duration, pre_delay=args.delay)


if __name__ == '__main__':
    main()


