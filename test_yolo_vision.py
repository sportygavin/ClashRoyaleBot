#!/usr/bin/env python3
"""
Test YOLO card detection in game setting

This script:
- Captures game screens
- Runs YOLO detection on player cards and opponent cards
- Visualizes detections with bounding boxes
- Saves annotated images
- Prints detection statistics
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vision.game_vision import ClashRoyaleVision
from core import GameState

def draw_detections(image, cards, color=(0, 255, 0), label_prefix="", scale_factor=2.0):
    """Draw bounding boxes and labels for detected cards
    
    Args:
        image: Screenshot image (in screenshot coordinate space)
        cards: List of Card objects with positions in pyautogui coordinate space
        color: Color for drawing
        label_prefix: Prefix for labels
        scale_factor: Factor to convert pyautogui coords to screenshot coords (default 2.0 for Retina)
    """
    annotated = image.copy()
    
    for card in cards:
        # Convert position from pyautogui space to screenshot space
        pos_py = card.position
        pos_screenshot = (int(pos_py[0] * scale_factor), int(pos_py[1] * scale_factor))
        
        name = card.name
        cost = card.cost
        is_available = card.is_available
        
        # Draw a circle at card position (in screenshot space)
        cv2.circle(annotated, pos_screenshot, int(25 * scale_factor), color, int(2 * scale_factor))
        
        # Draw a thicker circle if card is unavailable
        if not is_available:
            cv2.circle(annotated, pos_screenshot, int(25 * scale_factor), (128, 128, 128), int(1 * scale_factor))
        
        # Draw label with background
        label = f"{label_prefix}{name} ({cost})"
        if not is_available:
            label += " [X]"
        
        # Calculate text position (above the circle, in screenshot space)
        text_x = pos_screenshot[0]
        text_y = pos_screenshot[1] - int(35 * scale_factor)
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            annotated,
            (text_x - text_width // 2 - 5, text_y - text_height - 5),
            (text_x + text_width // 2 + 5, text_y + baseline + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            label,
            (text_x - text_width // 2, text_y),
            font,
            font_scale,
            color,
            thickness
        )
    
    return annotated

def test_yolo_detection(duration=60, save_images=True, output_dir="yolo_test_output"):
    """Test YOLO detection in game setting"""
    print("=" * 60)
    print("YOLO Card Detection Test")
    print("=" * 60)
    print()
    
    # Create output directory
    if save_images:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"📁 Output directory: {output_path.absolute()}")
        print()
    
    # Initialize vision system
    print("🔧 Initializing vision system...")
    calibration_path = "cv_out/calibration_manual_fixed.json"
    vision = ClashRoyaleVision(calibration_path=calibration_path)
    
    if not vision.yolo_available:
        print("❌ YOLO model not available!")
        print("   Make sure the model exists at: yolo_training/clash_royale_cards/weights/best.pt")
        return
    
    print("✅ YOLO model loaded successfully")
    print(f"   Card classes: {len(vision.card_classes)}")
    print()
    
    # Statistics
    stats = {
        'frames_processed': 0,
        'in_game_frames': 0,
        'player_cards_detected': 0,
        'opponent_cards_detected': 0,
        'total_player_detections': 0,
        'total_opponent_detections': 0,
        'card_counts': {}
    }
    
    print("🎮 Starting detection test...")
    print(f"   Duration: {duration} seconds")
    print("   Make sure Clash Royale is running and visible!")
    print()
    print("Press Ctrl+C to stop early")
    print("-" * 60)
    print()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Capture screen
            screen = vision.capture_screen()
            
            if screen is None:
                print("⚠️  Failed to capture screen, retrying...")
                time.sleep(0.5)
                continue
            
            stats['frames_processed'] += 1
            frame_count += 1
            
            # Detect game state
            game_state = vision.detect_game_state(screen)
            
            if game_state != GameState.IN_GAME:
                if frame_count % 10 == 0:  # Print every 10 frames
                    print(f"⏳ Frame {frame_count}: {game_state.value} (waiting for game to start...)")
                time.sleep(0.5)
                continue
            
            stats['in_game_frames'] += 1
            
            # Extract game info (this uses YOLO)
            game_info = vision.extract_game_info(screen)
            
            if game_info is None:
                time.sleep(0.5)
                continue
            
            # Get detections
            player_cards = game_info.player_cards
            opponent_cards = game_info.opponent_cards
            
            # Update statistics
            stats['total_player_detections'] += len(player_cards)
            stats['total_opponent_detections'] += len(opponent_cards)
            
            if len(player_cards) > 0:
                stats['player_cards_detected'] += 1
            if len(opponent_cards) > 0:
                stats['opponent_cards_detected'] += 1
            
            # Count card types
            for card in player_cards + opponent_cards:
                card_name = card.name
                stats['card_counts'][card_name] = stats['card_counts'].get(card_name, 0) + 1
            
            # Print detection results
            print(f"📊 Frame {frame_count} (in-game):")
            print(f"   Player cards: {len(player_cards)}")
            for card in player_cards:
                status = "✓" if card.is_available else "✗"
                print(f"     {status} {card.name} ({card.cost} elixir) at {card.position}")
            
            print(f"   Opponent cards: {len(opponent_cards)}")
            for card in opponent_cards:
                print(f"     • {card.name} ({card.cost} elixir) at {card.position}")
            
            # Save annotated images
            if save_images and (len(player_cards) > 0 or len(opponent_cards) > 0):
                annotated = screen.copy()
                
                # Calculate scale factor: screenshot dimensions / pyautogui dimensions
                import pyautogui
                screen_h, screen_w = screen.shape[:2]
                pyautogui_w, pyautogui_h = pyautogui.size()
                scale_factor_x = screen_w / pyautogui_w
                scale_factor_y = screen_h / pyautogui_h
                scale_factor = (scale_factor_x + scale_factor_y) / 2  # Average scale factor
                
                # Draw player cards (green)
                if player_cards:
                    annotated = draw_detections(annotated, player_cards, color=(0, 255, 0), 
                                               label_prefix="P:", scale_factor=scale_factor)
                
                # Draw opponent cards (red)
                if opponent_cards:
                    annotated = draw_detections(annotated, opponent_cards, color=(0, 0, 255), 
                                               label_prefix="O:", scale_factor=scale_factor)
                
                # Save image
                timestamp = int(time.time())
                image_path = output_path / f"yolo_detection_{frame_count}_{timestamp}.png"
                cv2.imwrite(str(image_path), annotated)
                print(f"   💾 Saved: {image_path.name}")
            
            print()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    # Print statistics
    print()
    print("=" * 60)
    print("📈 Detection Statistics")
    print("=" * 60)
    print(f"Total frames processed: {stats['frames_processed']}")
    print(f"In-game frames: {stats['in_game_frames']}")
    print()
    print("Player Cards:")
    print(f"  Frames with detections: {stats['player_cards_detected']}")
    print(f"  Total detections: {stats['total_player_detections']}")
    if stats['in_game_frames'] > 0:
        avg_player = stats['total_player_detections'] / stats['in_game_frames']
        print(f"  Average per frame: {avg_player:.2f}")
    print()
    print("Opponent Cards:")
    print(f"  Frames with detections: {stats['opponent_cards_detected']}")
    print(f"  Total detections: {stats['total_opponent_detections']}")
    if stats['in_game_frames'] > 0:
        avg_opponent = stats['total_opponent_detections'] / stats['in_game_frames']
        print(f"  Average per frame: {avg_opponent:.2f}")
    print()
    
    if stats['card_counts']:
        print("Card Type Distribution:")
        for card_name, count in sorted(stats['card_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {card_name}: {count}")
    
    print()
    if save_images:
        print(f"📁 Annotated images saved to: {output_path.absolute()}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Test YOLO card detection in game setting')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds (default: 60)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving annotated images')
    parser.add_argument('--output-dir', default='yolo_test_output', help='Output directory for images (default: yolo_test_output)')
    
    args = parser.parse_args()
    
    test_yolo_detection(
        duration=args.duration,
        save_images=not args.no_save,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
