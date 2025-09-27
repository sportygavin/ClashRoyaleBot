"""
Quick test script to verify bot can detect BlueStacks and Clash Royale
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision
from src.automation.game_automation import ClashRoyaleAutomation
from core import GameState

def test_bluestacks_detection():
    """Test if bot can detect BlueStacks window"""
    print("🔍 Testing BlueStacks detection...")
    
    try:
        vision = ClashRoyaleVision()
        automation = ClashRoyaleAutomation()
        
        print("✓ Vision and automation systems initialized")
        
        # Test screen capture
        print("📸 Testing screen capture...")
        screen = vision.capture_screen()
        
        if screen is not None:
            print(f"✓ Screen captured successfully: {screen.shape}")
        else:
            print("❌ Failed to capture screen")
            return False
        
        # Test game state detection
        print("🎮 Testing game state detection...")
        game_state = vision.detect_game_state(screen)
        print(f"✓ Game state detected: {game_state}")
        
        if game_state == GameState.IN_GAME:
            print("🎉 Clash Royale detected in game!")
            return True
        elif game_state == GameState.MENU:
            print("📱 Clash Royale detected in menu")
            print("💡 Make sure Clash Royale is open and visible")
            return True
        else:
            print(f"⚠️  Game state: {game_state}")
            print("💡 Make sure Clash Royale is open in BlueStacks")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_card_detection():
    """Test if bot can detect cards"""
    print("\n🃏 Testing card detection...")
    
    try:
        vision = ClashRoyaleVision()
        screen = vision.capture_screen()
        
        if screen is None:
            print("❌ Cannot capture screen")
            return False
        
        game_info = vision.extract_game_info(screen)
        
        if game_info and game_info.player_cards:
            print(f"✓ Detected {len(game_info.player_cards)} cards")
            for i, card in enumerate(game_info.player_cards):
                print(f"  Card {i+1}: {card.name} (Cost: {card.cost}, Available: {card.is_available})")
            return True
        else:
            print("❌ No cards detected")
            print("💡 Make sure you're in a Clash Royale match")
            return False
            
    except Exception as e:
        print(f"❌ Error during card detection: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 Clash Royale Bot - BlueStacks Test")
    print("="*50)
    
    print("\n📋 Instructions:")
    print("1. Make sure BlueStacks is running")
    print("2. Open Clash Royale in BlueStacks")
    print("3. Go to a match (or stay in menu)")
    print("4. Press Enter when ready...")
    
    input()
    
    # Test 1: Basic detection
    if not test_bluestacks_detection():
        print("\n❌ Basic detection failed")
        return False
    
    # Test 2: Card detection (if in game)
    print("\n" + "="*30)
    if test_card_detection():
        print("\n🎉 All tests passed! Bot is ready to run.")
        print("\nNext step: Run 'python3 main.py' to start the bot!")
    else:
        print("\n⚠️  Card detection failed, but basic detection works.")
        print("💡 Try going into a Clash Royale match and run this test again.")
    
    return True

if __name__ == "__main__":
    main()
