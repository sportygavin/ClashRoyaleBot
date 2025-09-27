"""
Simple test to verify if the bot is working now
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision
from core import GameState

def simple_test():
    """Simple test of the current detection"""
    print("🧪 Simple Detection Test")
    print("="*30)
    
    try:
        vision = ClashRoyaleVision()
        screen = vision.capture_screen()
        
        if screen is None:
            print("❌ Failed to capture screen")
            return False
        
        print("✓ Screen captured successfully")
        
        # Test detection
        game_state = vision.detect_game_state(screen)
        print(f"🎮 Detected game state: {game_state}")
        
        if game_state == GameState.IN_GAME:
            print("✅ IN_GAME detected! Bot should work now.")
            
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
                    print("  - No available cards (normal if not in active match)")
                
                return True
            else:
                print("❌ Failed to extract game info")
                return False
        else:
            print(f"⚠️  Not in IN_GAME state: {game_state}")
            print("💡 Make sure you're in an actual Clash Royale match (not menu or matchmaking)")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🧪 Testing Current Bot Detection")
    print("="*40)
    print("\nMake sure you're in a Clash Royale MATCH (not menu or matchmaking)")
    print("Press Enter to continue...")
    
    input()
    
    success = simple_test()
    
    if success:
        print("\n🎉 Bot detection is working!")
        print("You can now run: python3 macbook_bot.py")
    else:
        print("\n❌ Bot detection needs more work")
        print("Try starting an actual Clash Royale match")

if __name__ == "__main__":
    main()
