"""
Force IN_GAME detection for testing purposes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision
from core import GameState

class ForceInGameVision(ClashRoyaleVision):
    """Vision system that forces IN_GAME detection for testing"""
    
    def detect_game_state(self, screen):
        """Force IN_GAME detection for testing"""
        if screen is None:
            return GameState.MENU
        
        # Always return IN_GAME for testing
        return GameState.IN_GAME

def test_forced_detection():
    """Test with forced IN_GAME detection"""
    print("🧪 Testing with forced IN_GAME detection")
    print("="*50)
    
    try:
        vision = ForceInGameVision()
        screen = vision.capture_screen()
        
        if screen is None:
            print("❌ Failed to capture screen")
            return False
        
        print("✓ Screen captured successfully")
        
        # Test detection
        game_state = vision.detect_game_state(screen)
        print(f"🎮 Detected game state: {game_state}")
        
        if game_state == GameState.IN_GAME:
            print("✅ IN_GAME state detected!")
            
            # Try to extract game info
            game_info = vision.extract_game_info(screen)
            if game_info:
                print("✅ Game info extracted successfully!")
                print(f"  - Player elixir: {game_info.player_elixir}")
                print(f"  - Cards: {len(game_info.player_cards)}")
                print(f"  - Time remaining: {game_info.time_remaining}")
                
                # Show available cards
                available_cards = [card for card in game_info.player_cards if card.is_available]
                if available_cards:
                    print(f"  - Available cards: {len(available_cards)}")
                    for card in available_cards:
                        print(f"    * {card.name} (Cost: {card.cost})")
                else:
                    print("  - No available cards (normal in menu)")
                
                return True
            else:
                print("❌ Failed to extract game info")
                return False
        else:
            print("❌ Not in IN_GAME state")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🧪 Force IN_GAME Detection Test")
    print("="*40)
    print("\nThis will force the bot to think it's in a match")
    print("even if you're in the menu. Press Enter to continue...")
    
    input()
    
    success = test_forced_detection()
    
    if success:
        print("\n🎉 Bot is working! The issue was just game state detection.")
        print("To use the bot properly, start a Clash Royale match and run:")
        print("python3 verbose_main.py")
    else:
        print("\n❌ Bot has issues beyond game state detection")

if __name__ == "__main__":
    main()
