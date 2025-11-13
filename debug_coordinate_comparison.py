#!/usr/bin/env python3
"""
Debug script to compare coordinates between right_loop.py approach and YOLO detection.
This will show exactly what coordinates each system is using.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pyautogui
import cv2
import numpy as np
from game_scripts.strategy_utils import load_calibration, default_viewport, get_card_center_xy, screen_bgr
from src.vision.game_vision import ClashRoyaleVision

print("=" * 70)
print("COORDINATE COMPARISON: right_loop.py vs YOLO Detection")
print("=" * 70)
print()

# 1. Load calibration (same as right_loop.py)
calib_path = 'cv_out/calibration_manual_fixed.json'
calib = load_calibration(calib_path)
vp = default_viewport(calib)
vx, vy, vw, vh = vp

print("1. CALIBRATION & VIEWPORT")
print(f"   Calibration file: {calib_path}")
print(f"   Screen size (pyautogui.size()): {pyautogui.size()}")
print(f"   Viewport (from default_viewport): {vp}")
print(f"   vx={vx}, vy={vy}, vw={vw}, vh={vh}")
print()

# 2. Get card centers using right_loop.py method
print("2. RIGHT_LOOP.PY METHOD (get_card_center_xy)")
card_centers_right_loop = []
for i in range(4):
    cx, cy = get_card_center_xy(calib, vp, i)
    card_centers_right_loop.append((cx, cy))
    print(f"   Card {i+1}: ({cx}, {cy})")
print()

# 3. Capture screen and check dimensions
screen = screen_bgr()
if screen is None:
    print("ERROR: Could not capture screen")
    sys.exit(1)

screen_h, screen_w = screen.shape[:2]
print("3. SCREEN CAPTURE")
print(f"   Screen capture dimensions: {screen_w}x{screen_h}")
print(f"   pyautogui.size(): {pyautogui.size()}")
if screen_w != pyautogui.size()[0] or screen_h != pyautogui.size()[1]:
    print("   ⚠️  WARNING: Screen capture dimensions don't match pyautogui.size()!")
print()

# 4. Check CardRecognitionSystem approach (what right_loop uses internally)
print("4. CARDRECOGNITIONSYSTEM APPROACH (internal)")
from tools.card_recognition_system import CardRecognitionSystem
crs = CardRecognitionSystem(calib_path, 'database/clash_royale_cards.json')

# CardRecognitionSystem uses screenshot dimensions, not pyautogui.size()
viewport_crs = calib['viewport']
vp_x_crs = int(viewport_crs['x_r'] * screen_w)
vp_y_crs = int(viewport_crs['y_r'] * screen_h)
vp_w_crs = int(viewport_crs['w_r'] * screen_w)
vp_h_crs = int(viewport_crs['h_r'] * screen_h)

print(f"   Viewport (using screenshot dimensions): ({vp_x_crs}, {vp_y_crs}) {vp_w_crs}x{vp_h_crs}")
print(f"   Viewport (using pyautogui.size()): ({vx}, {vy}) {vw}x{vh}")

if (vp_x_crs, vp_y_crs, vp_w_crs, vp_h_crs) != (vx, vy, vw, vh):
    print("   ⚠️  WARNING: Viewport calculation differs!")
    print(f"      Difference: x={vp_x_crs-vx}, y={vp_y_crs-vy}, w={vp_w_crs-vw}, h={vp_h_crs-vh}")
print()

# 5. Check YOLO detection approach
print("5. YOLO DETECTION APPROACH")
vision = ClashRoyaleVision(calibration_path=calib_path)
print(f"   YOLO viewport: {vision.viewport}")
if vision.viewport != vp:
    print("   ⚠️  WARNING: YOLO viewport doesn't match right_loop viewport!")
    print(f"      YOLO: {vision.viewport}")
    print(f"      right_loop: {vp}")
print()

# Extract card region as YOLO does
if vision.viewport:
    vx_yolo, vy_yolo, vw_yolo, vh_yolo = vision.viewport
    viewport_region = screen[vy_yolo:vy_yolo+vh_yolo, vx_yolo:vx_yolo+vw_yolo]
    
    if vision.calibration and 'card_row' in vision.calibration:
        card_row = vision.calibration['card_row']
        vh_actual = viewport_region.shape[0]
        card_region_y1_vp = int(card_row.get('top_r', 0.85) * vh_actual)
        card_region_y2_vp = int(card_row.get('bottom_r', 1.0) * vh_actual)
        print(f"   Card region (relative to viewport): y={card_region_y1_vp}-{card_region_y2_vp}")
        print(f"   Card region (full screen): y={vy_yolo+card_region_y1_vp}-{vy_yolo+card_region_y2_vp}")
print()

# 6. Test actual YOLO detection
print("6. ACTUAL YOLO DETECTIONS")
if vision.yolo_available:
    game_info = vision.extract_game_info(screen)
    if game_info:
        player_cards = game_info.player_cards
        print(f"   Detected {len(player_cards)} player cards:")
        for card in player_cards:
            print(f"     {card.name}: position={card.position}, cost={card.cost}")
        
        # Compare with expected positions
        print()
        print("   COMPARISON:")
        for i, (expected_x, expected_y) in enumerate(card_centers_right_loop):
            print(f"     Card {i+1} expected (right_loop): ({expected_x}, {expected_y})")
            # Find closest YOLO detection
            if player_cards:
                closest = min(player_cards, key=lambda c: 
                    abs(c.position[0] - expected_x) + abs(c.position[1] - expected_y))
                dist = abs(closest.position[0] - expected_x) + abs(closest.position[1] - expected_y)
                print(f"     Closest YOLO detection: {closest.name} at {closest.position} (distance: {dist} pixels)")
    else:
        print("   No game info extracted")
else:
    print("   YOLO not available")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("Expected card positions (right_loop.py):")
for i, (x, y) in enumerate(card_centers_right_loop):
    print(f"  Card {i+1}: ({x}, {y})")
print()
print("If YOLO detections don't match these positions, there's a coordinate mismatch!")

