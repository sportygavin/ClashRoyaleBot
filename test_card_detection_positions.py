#!/usr/bin/env python3
"""
Test program to visualize card detection positions on screen.
Shows where YOLO is detecting cards and verifies coordinate accuracy.
"""

import argparse
import cv2
import numpy as np
import pyautogui
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_scripts.strategy_utils import load_calibration, default_viewport, screen_bgr
from src.vision.game_vision import ClashRoyaleVision


def draw_viewport_rectangle(img: np.ndarray, viewport: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 3):
    """Draw viewport rectangle on full screen image"""
    vx, vy, vw, vh = viewport
    cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), color, thickness)
    # Add label
    label = f"Viewport: ({vx}, {vy}) {vw}x{vh}"
    cv2.putText(img, label, (vx, vy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_card_region(img: np.ndarray, viewport: Tuple[int, int, int, int], calibration: dict, color: Tuple[int, int, int] = (255, 0, 255), thickness: int = 2):
    """Draw card hand region rectangle"""
    vx, vy, vw, vh = viewport
    
    if 'card_row' in calibration:
        card_row = calibration['card_row']
        row_top_r = card_row.get('top_r', 0.85)
        row_bottom_r = card_row.get('bottom_r', 1.0)
        
        row_top_y = vy + int(row_top_r * vh)
        row_bottom_y = vy + int(row_bottom_r * vh)
        
        cv2.rectangle(img, (vx, row_top_y), (vx + vw, row_bottom_y), color, thickness)
        label = f"Card Row: y={row_top_y}-{row_bottom_y}"
        cv2.putText(img, label, (vx, row_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw card centers
    if 'cards' in calibration and 'centers_x_r' in calibration['cards']:
        centers_x_r = calibration['cards']['centers_x_r']
        row_top_r = calibration['card_row'].get('top_r', 0.85)
        row_bottom_r = calibration['card_row'].get('bottom_r', 1.0)
        row_top_y = vy + int(row_top_r * vh)
        row_bottom_y = vy + int(row_bottom_r * vh)
        cy = (row_top_y + row_bottom_y) // 2
        
        for i, center_x_r in enumerate(centers_x_r):
            cx = vx + int(center_x_r * vw)
            # Draw crosshair
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), 2)
            cv2.line(img, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 2)
            cv2.line(img, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 2)
            label = f"C{i+1}: ({cx}, {cy})"
            cv2.putText(img, label, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img


def draw_opponent_region(img: np.ndarray, viewport: Tuple[int, int, int, int], calibration: dict, color: Tuple[int, int, int] = (255, 255, 0), thickness: int = 2):
    """Draw opponent region rectangle"""
    vx, vy, vw, vh = viewport
    
    if 'opponent_region_roi' in calibration:
        roi = calibration['opponent_region_roi']
        opp_x = vx + int(roi['x_r'] * vw)
        opp_y = vy + int(roi['y_r'] * vh)
        opp_w = int(roi['w_r'] * vw)
        opp_h = int(roi['h_r'] * vh)
        
        cv2.rectangle(img, (opp_x, opp_y), (opp_x + opp_w, opp_y + opp_h), color, thickness)
        label = f"Opponent Region: ({opp_x}, {opp_y}) {opp_w}x{opp_h}"
        cv2.putText(img, label, (opp_x, opp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def draw_detections(img: np.ndarray, cards: List, label_prefix: str = "", color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    """Draw YOLO detection bounding boxes and labels"""
    for card in cards:
        pos = card.position
        name = card.name
        cost = card.cost
        is_avail = card.is_available
        
        # Get bounding box (we need to estimate size since Card object doesn't store bbox)
        # For now, draw a circle at the position
        cx, cy = pos
        
        # Draw circle at detection center
        cv2.circle(img, (cx, cy), 30, color, thickness)
        
        # Draw label
        status = "✓" if is_avail else "✗"
        label = f"{label_prefix}{status} {name} ({cost}⚡)"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw background for text
        cv2.rectangle(img, 
                     (cx - label_size // 2 - 5, cy - 35),
                     (cx + label_size // 2 + 5, cy - 15),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, label, (cx - label_size // 2, cy - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw coordinates
        coord_label = f"({cx}, {cy})"
        cv2.putText(img, coord_label, (cx - 30, cy + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Test and visualize card detection positions')
    parser.add_argument('--calib', default='cv_out/calibration.json',
                       help='Path to calibration JSON file')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds')
    parser.add_argument('--output-dir', default='detection_test_output',
                       help='Output directory for annotated images')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all frames (not just frames with detections)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Card Detection Position Test")
    print("=" * 70)
    print()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}")
    print()
    
    # Load calibration
    print("🔧 Loading calibration...")
    try:
        calib = load_calibration(args.calib)
        viewport = default_viewport(calib)
        vx, vy, vw, vh = viewport
        print(f"✅ Calibration loaded: {args.calib}")
        print(f"   Viewport: ({vx}, {vy}) size {vw}x{vh}")
        screen_w, screen_h = pyautogui.size()
        print(f"   Screen: {screen_w}x{screen_h}")
        print()
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        return
    
    # Initialize vision system
    print("🔧 Initializing vision system...")
    try:
        vision = ClashRoyaleVision(calibration_path=args.calib)
        if not vision.yolo_available:
            print("❌ YOLO model not available!")
            print("   Make sure the model exists at: yolo_training/clash_royale_cards/weights/best.pt")
            return
        print("✅ YOLO model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error initializing vision: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Statistics
    stats = {
        'frames_processed': 0,
        'frames_with_player_cards': 0,
        'frames_with_opponent_cards': 0,
        'total_player_detections': 0,
        'total_opponent_detections': 0,
    }
    
    print("🎮 Starting detection test...")
    print(f"   Duration: {args.duration} seconds")
    print("   Make sure Clash Royale is running and visible!")
    print()
    print("Press Ctrl+C to stop early")
    print("-" * 70)
    print()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Capture full screen
            screen = screen_bgr()
            if screen is None:
                print("⚠️  Failed to capture screen, retrying...")
                time.sleep(0.5)
                continue
            
            stats['frames_processed'] += 1
            frame_count += 1
            
            # Create annotated image
            annotated = screen.copy()
            
            # Draw viewport rectangle (yellow)
            annotated = draw_viewport_rectangle(annotated, viewport, color=(0, 255, 255))
            
            # Draw card region (magenta)
            annotated = draw_card_region(annotated, viewport, calib, color=(255, 0, 255))
            
            # Draw opponent region (cyan)
            annotated = draw_opponent_region(annotated, viewport, calib, color=(255, 255, 0))
            
            # Detect game state
            game_state = vision.detect_game_state(screen)
            
            if game_state.value != "in_game":
                if frame_count % 10 == 0:
                    print(f"⏳ Frame {frame_count}: {game_state.value} (waiting for game...)")
                if args.save_all:
                    timestamp = int(time.time())
                    image_path = output_path / f"frame_{frame_count:04d}_{game_state.value}_{timestamp}.png"
                    cv2.imwrite(str(image_path), annotated)
                time.sleep(0.5)
                continue
            
            # Extract game info (uses YOLO)
            game_info = vision.extract_game_info(screen)
            
            if game_info is None:
                time.sleep(0.5)
                continue
            
            # Get detections
            player_cards = game_info.player_cards
            opponent_cards = game_info.opponent_cards
            
            # Update statistics
            if len(player_cards) > 0:
                stats['frames_with_player_cards'] += 1
                stats['total_player_detections'] += len(player_cards)
            
            if len(opponent_cards) > 0:
                stats['frames_with_opponent_cards'] += 1
                stats['total_opponent_detections'] += len(opponent_cards)
            
            # Draw detections
            if player_cards:
                annotated = draw_detections(annotated, player_cards, label_prefix="P:", color=(0, 255, 0))
            
            if opponent_cards:
                annotated = draw_detections(annotated, opponent_cards, label_prefix="O:", color=(0, 0, 255))
            
            # Print detection results
            print(f"📊 Frame {frame_count} (in-game):")
            print(f"   Player cards: {len(player_cards)}")
            for card in player_cards:
                status = "✓" if card.is_available else "✗"
                print(f"     {status} {card.name} ({card.cost}⚡) at {card.position}")
            
            print(f"   Opponent cards: {len(opponent_cards)}")
            for card in opponent_cards:
                print(f"     • {card.name} ({card.cost}⚡) at {card.position}")
            
            # Save annotated image
            should_save = args.save_all or len(player_cards) > 0 or len(opponent_cards) > 0
            if should_save:
                timestamp = int(time.time())
                image_path = output_path / f"detection_{frame_count:04d}_{timestamp}.png"
                cv2.imwrite(str(image_path), annotated)
                print(f"   💾 Saved: {image_path.name}")
            
            print()
            
            # Small delay
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    # Print statistics
    print()
    print("=" * 70)
    print("📈 Detection Statistics")
    print("=" * 70)
    print(f"Total frames processed: {stats['frames_processed']}")
    print()
    print("Player Cards:")
    print(f"  Frames with detections: {stats['frames_with_player_cards']}")
    print(f"  Total detections: {stats['total_player_detections']}")
    if stats['frames_with_player_cards'] > 0:
        avg = stats['total_player_detections'] / stats['frames_with_player_cards']
        print(f"  Average per detection frame: {avg:.2f}")
    print()
    print("Opponent Cards:")
    print(f"  Frames with detections: {stats['frames_with_opponent_cards']}")
    print(f"  Total detections: {stats['total_opponent_detections']}")
    if stats['frames_with_opponent_cards'] > 0:
        avg = stats['total_opponent_detections'] / stats['frames_with_opponent_cards']
        print(f"  Average per detection frame: {avg:.2f}")
    print()
    print(f"📁 Annotated images saved to: {output_path.absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    main()

