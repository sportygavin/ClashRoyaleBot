"""
Real-time detection debugging for Clash Royale
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
    print("🔍 Real-time Clash Royale Detection Debug")
    print("="*50)
    print("\nMake sure you're IN A CLASH ROYALE MATCH")
    print("Press Enter when ready...")
    
    input()
    
    try:
        vision = ClashRoyaleVision()
        
        for i in range(5):  # Take 5 samples
            print(f"\n--- Sample {i+1}/5 ---")
            
            screen = vision.capture_screen()
            if screen is None:
                print("❌ Failed to capture screen")
                continue
            
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            print(f"Screen: {width}x{height}")
            
            # Test multiple detection methods
            detections = {}
            
            # Method 1: Original elixir detection
            elixir_region = gray[30:80, 800:1120]
            elixir_brightness = np.mean(elixir_region)
            detections['elixir'] = elixir_brightness
            print(f"Elixir region (800:1120, 30:80): {elixir_brightness:.1f}")
            
            # Method 2: Center elixir detection
            center_elixir = gray[30:80, width//2-160:width//2+160]
            center_elixir_brightness = np.mean(center_elixir)
            detections['center_elixir'] = center_elixir_brightness
            print(f"Center elixir region: {center_elixir_brightness:.1f}")
            
            # Method 3: Arena detection
            arena_region = gray[200:600, 400:1200]
            arena_brightness = np.mean(arena_region)
            detections['arena'] = arena_brightness
            print(f"Arena region (400:1200, 200:600): {arena_brightness:.1f}")
            
            # Method 4: Card area detection
            card_region = gray[900:1000, 200:1400]
            card_brightness = np.mean(card_region)
            detections['cards'] = card_brightness
            print(f"Card region (200:1400, 900:1000): {card_brightness:.1f}")
            
            # Method 5: Look for specific patterns
            # Check for the elixir bar pattern (should have some bright spots)
            elixir_region_2 = gray[40:70, width//2-200:width//2+200]
            elixir_max = np.max(elixir_region_2)
            elixir_std = np.std(elixir_region_2)
            detections['elixir_pattern'] = (elixir_max, elixir_std)
            print(f"Elixir pattern - Max: {elixir_max:.1f}, Std: {elixir_std:.1f}")
            
            # Method 6: Look for towers
            left_tower = gray[height//2-100:height//2+100, 50:150]
            right_tower = gray[height//2-100:height//2+100, width-150:width-50]
            tower_brightness = (np.mean(left_tower) + np.mean(right_tower)) / 2
            detections['towers'] = tower_brightness
            print(f"Tower brightness: {tower_brightness:.1f}")
            
            # Method 7: Look for time display
            time_region = gray[50:100, width//2-100:width//2+100]
            time_brightness = np.mean(time_region)
            detections['time'] = time_brightness
            print(f"Time region brightness: {time_brightness:.1f}")
            
            # Test current detection
            current_state = vision.detect_game_state(screen)
            print(f"Current detection: {current_state}")
            
            # Save screenshot for analysis
            cv2.imwrite(f"debug_sample_{i+1}.png", screen)
            print(f"Screenshot saved: debug_sample_{i+1}.png")
            
            time.sleep(1)  # Wait 1 second between samples
        
        print("\n" + "="*50)
        print("📊 Analysis complete!")
        print("Check the debug_sample_*.png files to see what the bot captured.")
        print("Look for patterns that indicate you're in a match.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def create_adaptive_detection():
    """Create detection based on actual screen analysis"""
    print("\n🔧 Creating adaptive detection...")
    
    # This would analyze the debug samples and create better detection
    # For now, let's create a more flexible detection
    
    detection_code = '''
def adaptive_detect_game_state(self, screen):
    """Adaptive game state detection based on actual screen analysis"""
    if screen is None:
        return GameState.MENU
    
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Multiple detection criteria
    criteria = []
    
    # 1. Elixir bar detection (multiple regions)
    elixir_regions = [
        gray[30:80, 800:1120],  # Original
        gray[40:70, width//2-200:width//2+200],  # Center
        gray[50:80, width//2-100:width//2+100]   # Narrow center
    ]
    
    elixir_scores = [np.mean(region) for region in elixir_regions]
    max_elixir = max(elixir_scores)
    criteria.append(("elixir", max_elixir > 85))
    
    # 2. Arena detection (should be darker in game)
    arena_regions = [
        gray[200:600, 400:1200],
        gray[height//3:height*2//3, width//4:width*3//4],
        gray[300:500, 500:1100]
    ]
    
    arena_scores = [np.mean(region) for region in arena_regions]
    min_arena = min(arena_scores)
    criteria.append(("arena", min_arena < 70))
    
    # 3. Card area detection
    card_regions = [
        gray[900:1000, 200:1400],
        gray[height-150:height-50, width//4:width*3//4],
        gray[850:950, 300:1300]
    ]
    
    card_scores = [np.mean(region) for region in card_regions]
    max_cards = max(card_scores)
    criteria.append(("cards", max_cards > 60))
    
    # 4. Tower detection
    left_tower = gray[height//2-100:height//2+100, 50:150]
    right_tower = gray[height//2-100:height//2+100, width-150:width-50]
    tower_brightness = (np.mean(left_tower) + np.mean(right_tower)) / 2
    criteria.append(("towers", tower_brightness > 40))
    
    # 5. Time display detection
    time_regions = [
        gray[50:100, width//2-100:width//2+100],
        gray[40:90, width//2-50:width//2+50],
        gray[60:110, width//2-75:width//2+75]
    ]
    
    time_scores = [np.mean(region) for region in time_regions]
    max_time = max(time_scores)
    criteria.append(("time", max_time > 90))
    
    # Count positive criteria
    positive_count = sum(1 for _, condition in criteria if condition)
    
    print(f"Detection criteria: {positive_count}/5")
    for name, condition in criteria:
        print(f"  {name}: {'✓' if condition else '✗'}")
    
    # Determine state
    if positive_count >= 3:
        return GameState.IN_GAME
    elif positive_count >= 1:
        return GameState.MATCHMAKING
    else:
        return GameState.MENU
'''
    
    print("✅ Adaptive detection code created")
    return detection_code

def main():
    """Main function"""
    debug_detection_in_real_time()
    create_adaptive_detection()

if __name__ == "__main__":
    main()
