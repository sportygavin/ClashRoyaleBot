import argparse
import json
import random
import time
from typing import Dict, List, Optional, Tuple

import os
import sys
import numpy as np
import cv2
import pyautogui

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.card_recognition_system import CardRecognitionSystem


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


def choose_target(viewport_px: Tuple[int, int, int, int]) -> Tuple[int, int]:
    vx, vy, vw, vh = viewport_px
    # Simple heuristic: play near center-front on our side
    target_x_r = random.uniform(0.42, 0.58)
    target_y_r = random.uniform(0.55, 0.70)
    tx = vx + int(target_x_r * vw)
    ty = vy + int(target_y_r * vh)
    return tx, ty


def drag_card_to(card_xy: Tuple[int, int], target_xy: Tuple[int, int], duration: float = 0.25, pre_delay: float = 0.15):
    sx, sy = card_xy
    tx, ty = target_xy
    time.sleep(pre_delay)
    pyautogui.moveTo(sx, sy, duration=0.08)
    pyautogui.dragTo(tx, ty, duration=duration, button='left')


def main():
    parser = argparse.ArgumentParser(description='Auto player: play affordable cards over a 3-minute window.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json', help='Calibration JSON path')
    parser.add_argument('--duration', type=int, default=180, help='Run time in seconds')
    parser.add_argument('--min-gap', type=float, default=1.2, help='Minimum seconds between plays')
    parser.add_argument('--safe-elixir', type=int, default=0, help='Keep at least this elixir before playing')
    parser.add_argument('--prefer-cheap', action='store_true', help='Prefer cheaper cards first')
    parser.add_argument('--show', action='store_true', help='Show simple prints each cycle')
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    crs = CardRecognitionSystem(args.calib, "database/clash_royale_cards.json")

    screen_w, screen_h = pyautogui.size()
    viewport_r = calib.get('viewport') or {'x_r': 0.0, 'y_r': 0.0, 'w_r': 1.0, 'h_r': 1.0}
    viewport_px = ratios_to_abs(viewport_r, screen_w, screen_h)

    start = time.time()
    last_play_t = 0.0

    def read_stable_elixir(samples: int = 3, delay_s: float = 0.06) -> Tuple[Optional[int], float]:
        vals: List[Optional[int]] = []
        confs: List[float] = []
        for _ in range(max(1, samples)):
            scr_local = pyautogui.screenshot()
            if scr_local is None:
                continue
            frame_local = cv2.cvtColor(np.array(scr_local), cv2.COLOR_RGB2BGR)
            v, c = crs.recognize_current_elixir(frame_local)
            vals.append(v)
            confs.append(c)
            time.sleep(delay_s)
        # Pick most common non-None; break ties by avg confidence
        best_v: Optional[int] = None
        best_score: float = -1.0
        unique = set(x for x in vals if x is not None)
        if unique:
            for value in unique:
                idxs = [i for i, x in enumerate(vals) if x == value]
                avg_conf = sum(confs[i] for i in idxs) / max(len(idxs), 1)
                score = len(idxs) + avg_conf  # prioritize count, then confidence
                if score > best_score:
                    best_score = score
                    best_v = value
            # Report combined confidence
            idxs = [i for i, x in enumerate(vals) if x == best_v]
            return best_v, sum(confs[i] for i in idxs) / max(len(idxs), 1)
        # Fallback to single best confidence if all None
        if confs:
            max_i = int(np.argmax(confs))
            return vals[max_i], confs[max_i]
        return None, 0.0

    while time.time() - start < args.duration:
        # Capture one frame internally via CRS
        scr = pyautogui.screenshot()
        frame_bgr = None
        if scr is not None:
            frame_bgr = cv2.cvtColor(np.array(scr), cv2.COLOR_RGB2BGR)
        if frame_bgr is None or getattr(frame_bgr, 'size', 0) == 0:
            time.sleep(0.2)
            continue

        # Recognize current elixir (stabilized over a few samples)
        cur_elixir, elixir_conf = read_stable_elixir(samples=3, delay_s=0.05)
        # Recognize hand
        hand_info = crs.analyze_hand(frame_bgr)

        # Gather affordable cards
        affordable: List[Tuple[int, int]] = []  # (card_index, cost)
        if isinstance(hand_info, dict):
            for key in sorted(hand_info.keys()):
                card = hand_info[key]
                if not isinstance(card, dict):
                    continue
                idx = card.get('card_number')
                card_info = card.get('card_info') or {}
                cost = card_info.get('elixir_cost')
                if isinstance(idx, int) and isinstance(cost, int):
                    # Play only when current elixir is at least the card cost (+ optional buffer)
                    if cur_elixir is not None and (cur_elixir - args.safe_elixir) >= cost:
                        # Convert to zero-based index for placement helper
                        affordable.append((idx - 1, cost))

        if args.prefer_cheap:
            affordable.sort(key=lambda x: x[1])
        else:
            random.shuffle(affordable)

        if args.show:
            print(f"elixir={cur_elixir} conf={elixir_conf:.2f} affordable={affordable}")

        now = time.time()
        if affordable and (now - last_play_t) >= args.min_gap:
            idx, cost = affordable[0]
            card_xy = get_card_center_xy(calib, viewport_px, idx)
            target_xy = choose_target(viewport_px)
            drag_card_to(card_xy, target_xy)
            last_play_t = now
        else:
            time.sleep(0.2)


if __name__ == '__main__':
    main()


