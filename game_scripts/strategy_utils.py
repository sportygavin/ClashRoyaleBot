import json
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui

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


def screen_bgr() -> Optional[np.ndarray]:
    scr = pyautogui.screenshot()
    if scr is None:
        return None
    return cv2.cvtColor(np.array(scr), cv2.COLOR_RGB2BGR)


def drag_card_to(card_xy: Tuple[int, int], target_xy: Tuple[int, int], duration: float = 0.25, pre_delay: float = 0.15):
    sx, sy = card_xy
    tx, ty = target_xy
    time.sleep(pre_delay)
    pyautogui.moveTo(sx, sy, duration=0.08)
    pyautogui.dragTo(tx, ty, duration=duration, button='left')


def stable_elixir(crs: CardRecognitionSystem, samples: int = 5, delay_s: float = 0.05) -> Tuple[Optional[int], float]:
    vals: List[Optional[int]] = []
    confs: List[float] = []
    for _ in range(max(1, samples)):
        bgr = screen_bgr()
        if bgr is None or getattr(bgr, 'size', 0) == 0:
            continue
        v, c = crs.recognize_current_elixir(bgr)
        vals.append(v)
        confs.append(c)
        time.sleep(delay_s)
    best_v: Optional[int] = None
    best_score = -1.0
    unique = set(x for x in vals if x is not None)
    if unique:
        for value in unique:
            idxs = [i for i, x in enumerate(vals) if x == value]
            avg_conf = sum(confs[i] for i in idxs) / max(len(idxs), 1)
            score = len(idxs) + avg_conf
            if score > best_score:
                best_score = score
                best_v = value
        idxs = [i for i, x in enumerate(vals) if x == best_v]
        return best_v, sum(confs[i] for i in idxs) / max(len(idxs), 1)
    if confs:
        import numpy as _np
        max_i = int(_np.argmax(confs))
        return vals[max_i], confs[max_i]
    return None, 0.0


def default_viewport(calib: dict) -> Tuple[int, int, int, int]:
    screen_w, screen_h = pyautogui.size()
    viewport_r = calib.get('viewport') or {'x_r': 0.0, 'y_r': 0.0, 'w_r': 1.0, 'h_r': 1.0}
    return ratios_to_abs(viewport_r, screen_w, screen_h)


