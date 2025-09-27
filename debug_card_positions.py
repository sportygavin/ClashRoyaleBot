"""
Debug Card Positions

This script helps debug why card detection is capturing the wrong areas
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

def debug_card_positions():
    """Debug card position detection"""
    print("🃏 Debugging Card Positions")
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
    cv2.imwrite("debug_full_screen.png", screen)
    print("💾 Saved: debug_full_screen.png")
    
    height, width = screen.shape[:2]
    print(f"📐 Screen dimensions: {width}x{height}")
    
    # Calculate card positions (same as in the vision system)
    print("\n🧮 Calculating card positions...")
    
    card_y = int(height * 0.92)  # 92% down from top
    card_spacing = width // 5  # Divide width into 5 sections
    
    print(f"📍 Card Y position: {card_y}")
    print(f"📏 Card spacing: {card_spacing}")
    
    card_positions = []
    for i in range(4):
        card_x = card_spacing + (i * card_spacing)
        pos = (card_x, card_y)
        card_positions.append(pos)
        print(f"🃏 Card {i+1}: ({card_x}, {card_y})")
    
    # Create debug image with all regions marked
    debug_image = screen.copy()
    
    # Draw card regions
    for i, pos in enumerate(card_positions):
        card_x, card_y = pos
        
        # Draw rectangle around card area
        cv2.rectangle(debug_image, 
                     (card_x-60, card_y-60), 
                     (card_x+60, card_y+60), 
                     (0, 255, 0), 2)
        
        # Draw card number
        cv2.putText(debug_image, f"Card {i+1}", 
                   (card_x-50, card_y-70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(debug_image, (card_x, card_y), 5, (0, 0, 255), -1)
    
    # Draw other important areas
    # Elixir bar area
    elixir_y = int(height * 0.05)
    cv2.rectangle(debug_image, 
                 (int(width * 0.25), int(height * 0.04)), 
                 (int(width * 0.75), int(height * 0.08)), 
                 (255, 0, 0), 2)
    cv2.putText(debug_image, "Elixir Bar", 
               (int(width * 0.25), int(height * 0.04) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Arena area
    cv2.rectangle(debug_image, 
                 (int(width * 0.10), int(height * 0.20)), 
                 (int(width * 0.90), int(height * 0.80)), 
                 (0, 0, 255), 2)
    cv2.putText(debug_image, "Arena", 
               (int(width * 0.10), int(height * 0.20) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite("debug_card_positions.png", debug_image)
    print("💾 Saved: debug_card_positions.png")
    
    # Extract and save individual card regions
    print("\n📸 Extracting individual card regions...")
    for i, pos in enumerate(card_positions):
        card_x, card_y = pos
        
        # Extract card region
        card_region = screen[
            max(0, card_y-60):min(height, card_y+60),
            max(0, card_x-60):min(width, card_x+60)
        ]
        
        # Save card image
        cv2.imwrite(f"debug_card_{i+1}.png", card_region)
        print(f"💾 Saved: debug_card_{i+1}.png")
    
    # Try different card positions
    print("\n🔍 Trying alternative card positions...")
    
    # Alternative 1: Bottom 10% of screen
    alt_y1 = int(height * 0.90)
    print(f"📍 Alternative Y1 (90%): {alt_y1}")
    
    # Alternative 2: Bottom 5% of screen  
    alt_y2 = int(height * 0.95)
    print(f"📍 Alternative Y2 (95%): {alt_y2}")
    
    # Alternative 3: Fixed offset from bottom
    alt_y3 = height - 100
    print(f"📍 Alternative Y3 (height-100): {alt_y3}")
    
    # Create comparison image
    comparison_image = screen.copy()
    
    # Draw all alternatives
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    labels = ["92%", "90%", "95%"]
    
    for alt_y, color, label in zip([card_y, alt_y1, alt_y2], colors, labels):
        for i in range(4):
            card_x = card_spacing + (i * card_spacing)
            cv2.rectangle(comparison_image, 
                         (card_x-60, alt_y-60), 
                         (card_x+60, alt_y+60), 
                         color, 2)
            cv2.putText(comparison_image, f"{label}", 
                       (card_x-50, alt_y-70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite("debug_card_comparison.png", comparison_image)
    print("💾 Saved: debug_card_comparison.png")
    
    return card_positions

def main():
    """Main debug function"""
    print("🃏 Clash Royale Card Position Debug Tool")
    print("="*60)
    print("\nThis tool will help debug why card detection")
    print("is capturing the wrong screen areas.")
    
    print("\n📋 Instructions:")
    print("1. Make sure Clash Royale is running in BlueStacks")
    print("2. Start a match (training or real match)")
    print("3. Wait for the match to begin")
    print("4. Run this debug tool")
    
    print("\nPress Enter when ready...")
    input()
    
    # Run debug
    positions = debug_card_positions()
    
    if positions:
        print("\n📁 Check the generated images:")
        print("   - debug_full_screen.png (full screen)")
        print("   - debug_card_positions.png (with position overlays)")
        print("   - debug_card_comparison.png (alternative positions)")
        print("   - debug_card_1.png to debug_card_4.png (individual cards)")
        
        print("\n💡 Next steps:")
        print("1. Review debug_card_positions.png")
        print("2. Check if the green rectangles are over the actual cards")
        print("3. If not, try the alternative positions in debug_card_comparison.png")
        print("4. Adjust the card position calculation in game_vision.py")

if __name__ == "__main__":
    main()
