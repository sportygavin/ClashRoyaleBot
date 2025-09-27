"""
Debug Game State Detection

This script helps debug why the bot isn't detecting IN_GAME state
"""

import sys
import cv2
import numpy as np
import pyautogui
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision
from core import GameState

def debug_game_state():
    """Debug game state detection"""
    print("🔍 Debugging Game State Detection")
    print("="*50)
    
    # Initialize vision system
    vision = ClashRoyaleVision()
    
    print("📸 Capturing screen...")
    screen = vision.capture_screen()
    
    if screen is None:
        print("❌ Failed to capture screen!")
        return
    
    print(f"✅ Screen captured: {screen.shape[1]}x{screen.shape[0]}")
    
    # Save full screen
    cv2.imwrite("debug_screen.png", screen)
    print("💾 Saved: debug_screen.png")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print(f"📐 Screen dimensions: {width}x{height}")
    
    # Analyze different regions
    print("\n🔍 Analyzing screen regions...")
    
    # Elixir region (top 4-8% of screen)
    elixir_region = gray[
        int(height * 0.04):int(height * 0.08),
        int(width * 0.25):int(width * 0.75)
    ]
    elixir_brightness = np.mean(elixir_region)
    print(f"💧 Elixir region brightness: {elixir_brightness:.1f}")
    
    # Arena region (middle 60% of screen height)
    arena_region = gray[
        int(height * 0.20):int(height * 0.80),
        int(width * 0.10):int(width * 0.90)
    ]
    arena_brightness = np.mean(arena_region)
    print(f"🏟️  Arena region brightness: {arena_brightness:.1f}")
    
    # Card region (bottom 15% of screen)
    card_region = gray[
        int(height * 0.85):int(height * 1.0),
        int(width * 0.10):int(width * 0.90)
    ]
    card_brightness = np.mean(card_region)
    print(f"🃏 Card region brightness: {card_brightness:.1f}")
    
    # Test detection logic
    print("\n🧪 Testing detection logic...")
    
    # Current thresholds
    elixir_threshold = 60
    arena_threshold = 120
    card_threshold = 30
    
    print(f"📊 Current thresholds:")
    print(f"   Elixir > {elixir_threshold}: {elixir_brightness > elixir_threshold}")
    print(f"   Arena < {arena_threshold}: {arena_brightness < arena_threshold}")
    print(f"   Card > {card_threshold}: {card_brightness > card_threshold}")
    
    # Overall detection
    in_game = (
        elixir_brightness > elixir_threshold and
        arena_brightness < arena_threshold and
        card_brightness > card_threshold
    )
    
    print(f"\n🎮 In-game detection: {in_game}")
    
    # Create visual debug image
    debug_image = screen.copy()
    
    # Draw regions
    cv2.rectangle(debug_image, 
                 (int(width * 0.25), int(height * 0.04)), 
                 (int(width * 0.75), int(height * 0.08)), 
                 (0, 255, 0), 2)
    cv2.putText(debug_image, f"Elixir: {elixir_brightness:.1f}", 
               (int(width * 0.25), int(height * 0.04) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.rectangle(debug_image, 
                 (int(width * 0.10), int(height * 0.20)), 
                 (int(width * 0.90), int(height * 0.80)), 
                 (255, 0, 0), 2)
    cv2.putText(debug_image, f"Arena: {arena_brightness:.1f}", 
               (int(width * 0.10), int(height * 0.20) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.rectangle(debug_image, 
                 (int(width * 0.10), int(height * 0.85)), 
                 (int(width * 0.90), int(height * 1.0)), 
                 (0, 0, 255), 2)
    cv2.putText(debug_image, f"Cards: {card_brightness:.1f}", 
               (int(width * 0.10), int(height * 0.85) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite("debug_regions.png", debug_image)
    print("💾 Saved: debug_regions.png")
    
    # Suggest new thresholds
    print("\n💡 Suggested new thresholds:")
    print(f"   Elixir threshold: {max(30, elixir_brightness - 20)}")
    print(f"   Arena threshold: {min(200, arena_brightness + 50)}")
    print(f"   Card threshold: {max(20, card_brightness - 10)}")
    
    return {
        "elixir_brightness": elixir_brightness,
        "arena_brightness": arena_brightness,
        "card_brightness": card_brightness,
        "in_game": in_game
    }

def main():
    """Main debug function"""
    print("🔍 Clash Royale Game State Debug Tool")
    print("="*60)
    print("\nThis tool will help debug why the bot isn't detecting")
    print("the IN_GAME state correctly.")
    
    print("\n📋 Instructions:")
    print("1. Make sure Clash Royale is running in BlueStacks")
    print("2. Start a match (training or real match)")
    print("3. Wait for the match to begin")
    print("4. Run this debug tool")
    
    print("\nPress Enter when ready...")
    input()
    
    # Run debug
    result = debug_game_state()
    
    if result:
        print("\n📁 Check the generated images:")
        print("   - debug_screen.png (full screen)")
        print("   - debug_regions.png (with region overlays)")
        
        print("\n💡 Next steps:")
        print("1. Review the brightness values")
        print("2. Check if the regions are correct")
        print("3. Adjust thresholds if needed")
        print("4. Test again with: python3 test_card_detection.py")

if __name__ == "__main__":
    main()
