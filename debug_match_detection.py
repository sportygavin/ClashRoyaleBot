"""
Real-time game state detection debugger
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

def debug_detection_in_real_time():
    """Debug detection while you're in a match"""
    print("🔍 Real-time Detection Debugger")
    print("="*50)
    print("\nMake sure you're IN A CLASH ROYALE MATCH")
    print("This will analyze what the bot sees and why it's not detecting IN_GAME")
    print("Press Enter when ready...")
    
    input()
    
    try:
        vision = ClashRoyaleVision()
        
        for sample in range(3):  # Take 3 samples
            print(f"\n--- Sample {sample+1}/3 ---")
            
            screen = vision.capture_screen()
            if screen is None:
                print("❌ Failed to capture screen")
                continue
            
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            print(f"📸 Screen: {width}x{height}")
            
            # Test the current detection method
            game_state = vision.detect_game_state(screen)
            print(f"🎮 Current detection: {game_state}")
            
            # Debug each detection region
            print("\n🔍 Debugging detection regions:")
            
            # Elixir region
            elixir_region = gray[
                int(height * 0.04):int(height * 0.08),  # 78:157
                int(width * 0.26):int(width * 0.37)     # 786:1119
            ]
            elixir_brightness = np.mean(elixir_region)
            print(f"  Elixir region: {elixir_brightness:.1f} (need >85)")
            
            # Arena region
            arena_region = gray[
                int(height * 0.20):int(height * 0.61),  # 393:1198
                int(width * 0.13):int(width * 0.40)     # 393:1209
            ]
            arena_brightness = np.mean(arena_region)
            print(f"  Arena region: {arena_brightness:.1f} (need <75)")
            
            # Card region
            card_region = gray[
                int(height * 0.92):int(height * 1.0),   # 1807:1964
                int(width * 0.07):int(width * 0.46)     # 212:1391
            ]
            card_brightness = np.mean(card_region)
            print(f"  Card region: {card_brightness:.1f} (need >40)")
            
            # Check if all conditions are met
            elixir_ok = elixir_brightness > 85
            arena_ok = arena_brightness < 75
            card_ok = card_brightness > 40
            
            print(f"\n📊 Detection results:")
            print(f"  Elixir: {'✅' if elixir_ok else '❌'} ({elixir_brightness:.1f})")
            print(f"  Arena: {'✅' if arena_ok else '❌'} ({arena_brightness:.1f})")
            print(f"  Cards: {'✅' if card_ok else '❌'} ({card_brightness:.1f})")
            
            all_ok = elixir_ok and arena_ok and card_ok
            print(f"  Overall: {'✅ IN_GAME' if all_ok else '❌ NOT IN_GAME'}")
            
            # Try alternative detection methods
            print(f"\n🔍 Alternative detection methods:")
            
            # Method 1: Look for time display (should be bright in game)
            time_region = gray[50:100, width//2-100:width//2+100]
            time_brightness = np.mean(time_region)
            print(f"  Time region: {time_brightness:.1f}")
            
            # Method 2: Look for towers (should be visible)
            left_tower = gray[height//2-100:height//2+100, 50:150]
            right_tower = gray[height//2-100:height//2+100, width-150:width-50]
            tower_brightness = (np.mean(left_tower) + np.mean(right_tower)) / 2
            print(f"  Tower region: {tower_brightness:.1f}")
            
            # Method 3: Look for elixir droplets (should have bright spots)
            elixir_region_2 = gray[40:70, width//2-200:width//2+200]
            elixir_max = np.max(elixir_region_2)
            print(f"  Elixir max brightness: {elixir_max:.1f}")
            
            # Save screenshot for analysis
            cv2.imwrite(f"debug_match_{sample+1}.png", screen)
            print(f"📸 Saved: debug_match_{sample+1}.png")
            
            time.sleep(2)  # Wait 2 seconds between samples
        
        print("\n" + "="*50)
        print("📊 Analysis complete!")
        print("Check the debug_match_*.png files to see what the bot captured.")
        print("Look at the detection results above to see why it's not detecting IN_GAME.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def create_improved_detection():
    """Create improved detection based on analysis"""
    print("\n🔧 Creating Improved Detection")
    print("="*40)
    
    # This would analyze the debug results and create better detection
    # For now, let's create a more flexible detection
    
    improved_code = '''
def improved_detect_game_state(self, screen):
    """Improved game state detection based on real analysis"""
    if screen is None:
        return GameState.MENU
    
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Multiple detection methods with different thresholds
    detections = []
    
    # Method 1: Elixir bar detection (multiple regions)
    elixir_regions = [
        gray[int(height*0.04):int(height*0.08), int(width*0.26):int(width*0.37)],
        gray[50:100, width//2-200:width//2+200],
        gray[40:80, width//2-150:width//2+150]
    ]
    
    elixir_scores = [np.mean(region) for region in elixir_regions]
    max_elixir = max(elixir_scores)
    detections.append(("elixir", max_elixir > 80))  # Lowered threshold
    
    # Method 2: Arena detection (should be darker in game)
    arena_regions = [
        gray[int(height*0.20):int(height*0.61), int(width*0.13):int(width*0.40)],
        gray[height//3:height*2//3, width//4:width*3//4],
        gray[300:600, 400:1200]
    ]
    
    arena_scores = [np.mean(region) for region in arena_regions]
    min_arena = min(arena_scores)
    detections.append(("arena", min_arena < 80))  # Raised threshold
    
    # Method 3: Card area detection
    card_regions = [
        gray[int(height*0.92):int(height*1.0), int(width*0.07):int(width*0.46)],
        gray[height-150:height-50, width//4:width*3//4],
        gray[1800:1900, 200:1400]
    ]
    
    card_scores = [np.mean(region) for region in card_regions]
    max_cards = max(card_scores)
    detections.append(("cards", max_cards > 30))  # Lowered threshold
    
    # Method 4: Time display detection
    time_regions = [
        gray[50:100, width//2-100:width//2+100],
        gray[40:90, width//2-50:width//2+50],
        gray[60:110, width//2-75:width//2+75]
    ]
    
    time_scores = [np.mean(region) for region in time_regions]
    max_time = max(time_scores)
    detections.append(("time", max_time > 80))  # Lowered threshold
    
    # Method 5: Tower detection
    left_tower = gray[height//2-100:height//2+100, 50:150]
    right_tower = gray[height//2-100:height//2+100, width-150:width-50]
    tower_brightness = (np.mean(left_tower) + np.mean(right_tower)) / 2
    detections.append(("towers", tower_brightness > 30))  # Lowered threshold
    
    # Count positive detections
    positive_count = sum(1 for _, condition in detections if condition)
    
    print(f"Detection: {positive_count}/5")
    for name, condition in detections:
        print(f"  {name}: {'✓' if condition else '✗'}")
    
    # Determine state
    if positive_count >= 3:
        return GameState.IN_GAME
    elif positive_count >= 1:
        return GameState.MATCHMAKING
    else:
        return GameState.MENU
'''
    
    print("✅ Improved detection code created")
    print("This uses multiple detection methods with adjusted thresholds")
    return improved_code

def main():
    """Main function"""
    debug_detection_in_real_time()
    create_improved_detection()

if __name__ == "__main__":
    main()
