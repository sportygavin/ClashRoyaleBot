"""
Test the improved screen capture system
"""

import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision

def test_improved_capture():
    """Test the improved screen capture"""
    print("🔍 Testing Improved Screen Capture")
    print("="*40)
    
    try:
        vision = ClashRoyaleVision()
        screen = vision.capture_screen()
        
        if screen is None:
            print("❌ Failed to capture screen")
            return False
        
        height, width = screen.shape[:2]
        print(f"✅ Captured screen: {width}x{height}")
        print(f"📐 Aspect ratio: {width/height:.2f}")
        print(f"🎯 Game area: {vision.game_area}")
        
        # Save the screenshot
        cv2.imwrite("improved_capture.png", screen)
        print("📸 Saved: improved_capture.png")
        
        # Test game state detection
        game_state = vision.detect_game_state(screen)
        print(f"🎮 Game state: {game_state}")
        
        # Test card detection
        if game_state.value == "in_game":
            game_info = vision.extract_game_info(screen)
            if game_info and game_info.player_cards:
                print(f"🃏 Detected {len(game_info.player_cards)} cards")
                for i, card in enumerate(game_info.player_cards):
                    print(f"  Card {i+1}: {card.name} at {card.position} (Available: {card.is_available})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🔍 Testing Improved Screen Capture")
    print("="*40)
    print("\nThis will test the new screen capture that:")
    print("- Maintains proper aspect ratio (16:9)")
    print("- Only crops left and right sides")
    print("- Keeps full height")
    print("- Uses dynamic card positioning")
    print("\nPress Enter to continue...")
    
    input()
    
    success = test_improved_capture()
    
    if success:
        print("\n✅ Improved screen capture is working!")
        print("Check improved_capture.png to see the result")
    else:
        print("\n❌ Screen capture needs more work")

if __name__ == "__main__":
    main()
