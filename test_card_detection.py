"""
Card Detection Test Script

This script tests if the bot can properly detect:
1. Cards in your hand
2. Elixir costs for each card
3. Card availability (ready to play)
"""

import sys
import cv2
import numpy as np
import pyautogui
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vision.game_vision import ClashRoyaleVision
from core import GameState
from config import GAME_CONFIG

def test_card_detection():
    """Test card detection capabilities"""
    print("🃏 Clash Royale Card Detection Test")
    print("="*50)
    
    # Initialize vision system
    vision = ClashRoyaleVision()
    
    print("📸 Capturing screen...")
    screen = vision.capture_screen()
    
    if screen is None:
        print("❌ Failed to capture screen!")
        return False
    
    print(f"✅ Screen captured: {screen.shape[1]}x{screen.shape[0]}")
    
    # Save full screen for reference
    cv2.imwrite("test_screen.png", screen)
    print("💾 Saved: test_screen.png")
    
    # Test game state detection
    print("\n🎮 Testing game state detection...")
    game_state = vision.detect_game_state(screen)
    print(f"📍 Detected game state: {game_state}")
    
    if game_state != GameState.IN_GAME:
        print("⚠️  Not in game! Please start a match and try again.")
        print("💡 The bot needs to be in an active match to detect cards.")
        return False
    
    # Test card extraction
    print("\n🃏 Testing card extraction...")
    cards = vision._extract_player_cards(screen)
    
    print(f"✅ Found {len(cards)} cards:")
    print("-" * 40)
    
    for i, card in enumerate(cards):
        print(f"Card {i+1}:")
        print(f"  📍 Position: {card.position}")
        print(f"  💰 Cost: {card.cost} elixir")
        print(f"  🎯 Available: {'Yes' if card.is_available else 'No'}")
        print(f"  ⏱️  Cooldown: {card.cooldown_remaining}s")
        print(f"  🏷️  Name: {card.name}")
        print()
    
    # Test elixir detection
    print("💧 Testing elixir detection...")
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    elixir_count = vision._extract_elixir(gray)
    print(f"💰 Current elixir: {elixir_count}")
    
    # Save individual card regions for analysis
    print("\n📸 Saving individual card regions...")
    height, width = screen.shape[:2]
    
    for i, card in enumerate(cards):
        # Extract card region
        card_x, card_y = card.position
        card_region = screen[
            max(0, card_y-60):min(height, card_y+60),
            max(0, card_x-60):min(width, card_x+60)
        ]
        
        # Save card image
        cv2.imwrite(f"card_{i+1}.png", card_region)
        print(f"💾 Saved: card_{i+1}.png")
    
    # Create a visual summary
    print("\n🎨 Creating visual summary...")
    summary_image = screen.copy()
    
    # Draw card positions
    for i, card in enumerate(cards):
        card_x, card_y = card.position
        
        # Draw rectangle around card
        color = (0, 255, 0) if card.is_available else (0, 0, 255)
        cv2.rectangle(summary_image, 
                     (card_x-60, card_y-60), 
                     (card_x+60, card_y+60), 
                     color, 2)
        
        # Draw card number and cost
        cv2.putText(summary_image, f"C{i+1}: {card.cost}", 
                   (card_x-50, card_y-70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw elixir bar
    elixir_x, elixir_y = GAME_CONFIG["elixir_bar"]["position"]
    cv2.circle(summary_image, (elixir_x, elixir_y), 10, (255, 0, 0), 2)
    cv2.putText(summary_image, f"Elixir: {elixir_count}", 
               (elixir_x+20, elixir_y+5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imwrite("card_detection_summary.png", summary_image)
    print("💾 Saved: card_detection_summary.png")
    
    return True

def test_card_recognition():
    """Test if we can recognize specific card types"""
    print("\n🔍 Testing card recognition...")
    
    # This is a placeholder for actual card recognition
    # In a real implementation, you'd use template matching or ML
    print("💡 Card recognition would require:")
    print("   - Template images for each card")
    print("   - Template matching or ML model")
    print("   - Database of card costs and stats")
    
    return True

def main():
    """Main test function"""
    print("🧪 Clash Royale Bot Card Detection Test")
    print("="*60)
    print("\nThis test will verify that the bot can:")
    print("✅ Detect cards in your hand")
    print("✅ Read elixir costs")
    print("✅ Check card availability")
    print("✅ Detect current elixir count")
    
    print("\n📋 Instructions:")
    print("1. Make sure Clash Royale is running in BlueStacks")
    print("2. Start a match (training or real match)")
    print("3. Wait for the match to begin")
    print("4. Run this test")
    
    print("\nPress Enter when ready...")
    input()
    
    # Run tests
    success = test_card_detection()
    
    if success:
        print("\n🎉 Card detection test completed!")
        print("📁 Check the generated images:")
        print("   - test_screen.png (full screen)")
        print("   - card_1.png to card_4.png (individual cards)")
        print("   - card_detection_summary.png (visual summary)")
        
        # Test card recognition
        test_card_recognition()
        
        print("\n✅ Next steps:")
        print("1. Review the generated images")
        print("2. Check if card positions are correct")
        print("3. Verify elixir costs are accurate")
        print("4. Run the main bot: python3 macbook_bot.py")
    else:
        print("\n❌ Test failed! Please check:")
        print("1. Is Clash Royale running in BlueStacks?")
        print("2. Are you in an active match?")
        print("3. Is the game window visible?")

if __name__ == "__main__":
    main()
