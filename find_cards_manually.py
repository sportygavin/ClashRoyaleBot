"""
Manual Card Position Finder

This tool helps you manually identify where the cards are on screen
"""

import sys
import cv2
import numpy as np
import pyautogui
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def find_cards_manually():
    """Help user manually find card positions"""
    print("🃏 Manual Card Position Finder")
    print("="*50)
    
    print("📸 Capturing screen...")
    screenshot = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    height, width = screen.shape[:2]
    print(f"✅ Screen captured: {width}x{height}")
    
    # Save full screen
    cv2.imwrite("manual_card_finder.png", screen)
    print("💾 Saved: manual_card_finder.png")
    
    print("\n📋 Instructions:")
    print("1. Open manual_card_finder.png")
    print("2. Look at the image and find where your 4 cards are")
    print("3. Estimate the Y position (height) where the cards are")
    print("4. Enter your observations below")
    
    print(f"\n📐 Screen dimensions: {width}x{height}")
    print("💡 Card positions are typically:")
    print(f"   - Near the bottom: Y = {height - 100} to {height - 50}")
    print(f"   - Middle-bottom: Y = {height//2 + 200} to {height//2 + 400}")
    print(f"   - Bottom 10%: Y = {int(height * 0.9)} to {height}")
    
    print("\n🔍 Common Clash Royale card positions:")
    print("   - Cards are usually at the bottom of the screen")
    print("   - They're arranged horizontally in a row")
    print("   - Each card is roughly 100-150 pixels wide")
    print("   - Cards are typically 50-100 pixels from the bottom edge")
    
    # Try some common positions
    print("\n🧪 Testing common card positions...")
    
    test_positions = [
        ("Bottom edge", height - 50),
        ("Bottom 5%", int(height * 0.95)),
        ("Bottom 10%", int(height * 0.90)),
        ("Bottom 15%", int(height * 0.85)),
        ("Bottom 20%", int(height * 0.80)),
    ]
    
    test_image = screen.copy()
    
    for label, test_y in test_positions:
        # Draw horizontal line at this Y position
        cv2.line(test_image, (0, test_y), (width, test_y), (0, 255, 0), 2)
        cv2.putText(test_image, f"{label}: Y={test_y}", 
                   (10, test_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite("card_position_tests.png", test_image)
    print("💾 Saved: card_position_tests.png")
    
    print("\n📁 Check these images:")
    print("   - manual_card_finder.png (full screen)")
    print("   - card_position_tests.png (with test lines)")
    
    print("\n💡 Which horizontal line is closest to your cards?")
    for label, test_y in test_positions:
        print(f"   {label}: Y = {test_y}")
    
    return test_positions

def create_corrected_positions():
    """Create corrected card positions based on user input"""
    print("\n🔧 Creating Corrected Card Positions")
    print("="*50)
    
    print("Based on the test lines, which Y position looks closest to your cards?")
    print("Enter the Y coordinate (or press Enter to use bottom 10%):")
    
    try:
        user_input = input("Y position: ").strip()
        if user_input:
            card_y = int(user_input)
        else:
            # Default to bottom 10%
            screenshot = pyautogui.screenshot()
            height = screenshot.size[1]
            card_y = int(height * 0.90)
            print(f"Using default: Y = {card_y}")
        
        # Calculate card positions
        screenshot = pyautogui.screenshot()
        width = screenshot.size[0]
        card_spacing = width // 5
        
        card_positions = []
        for i in range(4):
            card_x = card_spacing + (i * card_spacing)
            card_positions.append((card_x, card_y))
        
        print(f"\n✅ Corrected card positions:")
        for i, pos in enumerate(card_positions):
            print(f"   Card {i+1}: ({pos[0]}, {pos[1]})")
        
        # Create config update
        config_update = f'''
# Corrected card positions
def get_corrected_card_positions():
    """Get corrected card positions based on manual testing"""
    # Capture screen to get current dimensions
    screenshot = pyautogui.screenshot()
    width, height = screenshot.size
    
    # Card Y position (manually determined)
    card_y = {card_y}
    
    # Card spacing
    card_spacing = width // 5
    
    # Calculate positions
    positions = []
    for i in range(4):
        card_x = card_spacing + (i * card_spacing)
        positions.append((card_x, card_y))
    
    return positions
'''
        
        print("\n📝 Add this function to your game_vision.py:")
        print("="*50)
        print(config_update)
        
        # Save to file
        with open("corrected_card_positions.py", "w") as f:
            f.write(config_update)
        
        print("✅ Configuration saved to: corrected_card_positions.py")
        
        return card_positions
        
    except ValueError:
        print("❌ Invalid input! Please enter a number.")
        return None
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return None

def main():
    """Main function"""
    print("🃏 Clash Royale Manual Card Position Finder")
    print("="*60)
    print("\nThis tool helps you manually identify where")
    print("the cards are positioned on your screen.")
    
    print("\n📋 Instructions:")
    print("1. Make sure Clash Royale is running in BlueStacks")
    print("2. Start a match (training or real match)")
    print("3. Wait for the match to begin")
    print("4. Run this tool")
    
    print("\nPress Enter when ready...")
    input()
    
    # Find cards manually
    test_positions = find_cards_manually()
    
    if test_positions:
        print("\n🎯 Next step:")
        print("1. Look at the generated images")
        print("2. Identify which horizontal line is closest to your cards")
        print("3. Run this tool again and enter the correct Y position")
        
        # Ask for corrected positions
        corrected = create_corrected_positions()
        
        if corrected:
            print("\n🎉 Card positions corrected!")
            print("💡 Next steps:")
            print("1. Update game_vision.py with the corrected positions")
            print("2. Test again with: python3 test_card_detection.py")

if __name__ == "__main__":
    main()
