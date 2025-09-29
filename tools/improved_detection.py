"""
Improved game state detection for Clash Royale
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision
from core import GameState

def improved_detect_game_state(screen):
    """Improved game state detection"""
    if screen is None:
        return GameState.MENU
    
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print(f"Screen size: {width}x{height}")
    
    # Multiple detection methods
    detections = []
    
    # Method 1: Look for elixir bar (top center)
    elixir_region = gray[30:80, width//2-160:width//2+160]
    elixir_brightness = np.mean(elixir_region)
    print(f"Elixir region brightness: {elixir_brightness:.1f}")
    detections.append(("elixir", elixir_brightness > 80))
    
    # Method 2: Look for arena (center should be darker in game)
    arena_region = gray[height//3:height*2//3, width//4:width*3//4]
    arena_brightness = np.mean(arena_region)
    print(f"Arena region brightness: {arena_brightness:.1f}")
    detections.append(("arena", arena_brightness < 70))
    
    # Method 3: Look for card slots (bottom area)
    card_region = gray[height-150:height-50, width//4:width*3//4]
    card_brightness = np.mean(card_region)
    print(f"Card region brightness: {card_brightness:.1f}")
    detections.append(("cards", card_brightness > 60))
    
    # Method 4: Look for towers (should be visible in game)
    left_tower = gray[height//2-50:height//2+50, 100:200]
    right_tower = gray[height//2-50:height//2+50, width-200:width-100]
    tower_brightness = (np.mean(left_tower) + np.mean(right_tower)) / 2
    print(f"Tower region brightness: {tower_brightness:.1f}")
    detections.append(("towers", tower_brightness > 50))
    
    # Method 5: Look for time display (top center)
    time_region = gray[50:100, width//2-50:width//2+50]
    time_brightness = np.mean(time_region)
    print(f"Time region brightness: {time_brightness:.1f}")
    detections.append(("time", time_brightness > 100))
    
    # Count positive detections
    positive_detections = sum(1 for _, detected in detections if detected)
    print(f"Positive detections: {positive_detections}/5")
    
    # Determine game state based on detections
    if positive_detections >= 3:
        return GameState.IN_GAME
    elif positive_detections >= 1:
        return GameState.MATCHMAKING
    else:
        return GameState.MENU

def test_improved_detection():
    """Test the improved detection"""
    print("🔍 Testing Improved Game State Detection")
    print("="*50)
    
    try:
        vision = ClashRoyaleVision()
        screen = vision.capture_screen()
        
        if screen is None:
            print("❌ Failed to capture screen")
            return
        
        print("✓ Screen captured successfully")
        
        # Test original detection
        original_state = vision.detect_game_state(screen)
        print(f"Original detection: {original_state}")
        
        # Test improved detection
        improved_state = improved_detect_game_state(screen)
        print(f"Improved detection: {improved_state}")
        
        # Save screenshot
        cv2.imwrite("improved_debug.png", screen)
        print("📸 Screenshot saved as improved_debug.png")
        
        return improved_state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    print("🔍 Improved Clash Royale Detection Test")
    print("="*50)
    print("\nMake sure:")
    print("1. BlueStacks is running")
    print("2. Clash Royale is open")
    print("3. You're in a match (or menu)")
    print("4. Press Enter when ready...")
    
    input()
    
    state = test_improved_detection()
    
    if state == GameState.IN_GAME:
        print("\n🎉 Successfully detected IN_GAME state!")
    elif state == GameState.MATCHMAKING:
        print("\n⏳ Detected MATCHMAKING state")
    else:
        print("\n📱 Detected MENU state")

if __name__ == "__main__":
    main()
