"""
Debug script to analyze what the bot sees in Clash Royale
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

def analyze_screen():
    """Analyze what the bot sees on screen"""
    print("🔍 Analyzing Clash Royale screen...")
    
    try:
        vision = ClashRoyaleVision()
        
        # Capture screen
        screen = vision.capture_screen()
        if screen is None:
            print("❌ Failed to capture screen")
            return
        
        print(f"✓ Screen captured: {screen.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Check elixir bar region
        elixir_region = gray[30:80, 800:1120]
        elixir_mean = np.mean(elixir_region)
        print(f"📊 Elixir region brightness: {elixir_mean:.1f}")
        
        # Check different regions for game elements
        regions = {
            "Top Left": gray[0:100, 0:200],
            "Top Center": gray[0:100, 400:800],
            "Top Right": gray[0:100, 1200:1600],
            "Center": gray[400:600, 400:800],
            "Bottom": gray[800:1000, 400:800]
        }
        
        print("\n📊 Region Analysis:")
        for name, region in regions.items():
            brightness = np.mean(region)
            print(f"  {name}: {brightness:.1f}")
        
        # Detect game state
        game_state = vision.detect_game_state(screen)
        print(f"\n🎮 Detected game state: {game_state}")
        
        # Try to extract game info
        if game_state == GameState.IN_GAME:
            print("🎯 Attempting to extract game info...")
            game_info = vision.extract_game_info(screen)
            if game_info:
                print(f"✓ Game info extracted:")
                print(f"  - Player elixir: {game_info.player_elixir}")
                print(f"  - Cards: {len(game_info.player_cards)}")
                print(f"  - Time remaining: {game_info.time_remaining}")
            else:
                print("❌ Failed to extract game info")
        else:
            print("⚠️  Not in game state, skipping game info extraction")
        
        # Save screenshot for analysis
        screenshot_path = "debug_screenshot.png"
        cv2.imwrite(screenshot_path, screen)
        print(f"\n📸 Screenshot saved to: {screenshot_path}")
        
        return screen
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def main():
    """Main function"""
    print("🔍 Clash Royale Screen Analysis")
    print("="*40)
    print("\nMake sure:")
    print("1. BlueStacks is running")
    print("2. Clash Royale is open")
    print("3. You're in a match (or menu)")
    print("4. Press Enter when ready...")
    
    input()
    
    screen = analyze_screen()
    
    if screen is not None:
        print("\n✅ Analysis complete!")
        print("Check the debug_screenshot.png file to see what the bot captured.")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main()
