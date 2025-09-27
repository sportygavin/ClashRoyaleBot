"""
Ultra-robust game state detection for Clash Royale
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

def ultra_robust_detect_game_state(screen):
    """Ultra-robust game state detection using multiple methods"""
    if screen is None:
        return GameState.MENU
    
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print(f"🔍 Ultra-robust detection for {width}x{height}")
    
    # Method 1: Look for elixir bar (multiple regions and thresholds)
    elixir_regions = [
        gray[50:100, width//2-200:width//2+200],  # Center elixir
        gray[40:80, width//2-150:width//2+150],   # Narrow center
        gray[60:110, width//2-100:width//2+100],  # Small center
        gray[30:70, width//2-300:width//2+300],   # Wide center
    ]
    
    elixir_scores = [np.mean(region) for region in elixir_regions]
    max_elixir = max(elixir_scores)
    min_elixir = min(elixir_scores)
    elixir_std = np.std(elixir_scores)
    
    print(f"  Elixir: max={max_elixir:.1f}, min={min_elixir:.1f}, std={elixir_std:.1f}")
    
    # Method 2: Look for arena (should be darker in game)
    arena_regions = [
        gray[height//3:height*2//3, width//4:width*3//4],  # Center arena
        gray[height//4:height*3//4, width//3:width*2//3],  # Large arena
        gray[200:600, 400:1200],  # Fixed arena
        gray[300:500, 500:1100],  # Small arena
    ]
    
    arena_scores = [np.mean(region) for region in arena_regions]
    min_arena = min(arena_scores)
    max_arena = max(arena_scores)
    arena_std = np.std(arena_scores)
    
    print(f"  Arena: min={min_arena:.1f}, max={max_arena:.1f}, std={arena_std:.1f}")
    
    # Method 3: Look for card area (should be visible)
    card_regions = [
        gray[height-150:height-50, width//4:width*3//4],  # Bottom cards
        gray[height-200:height-100, width//3:width*2//3],  # Large card area
        gray[1800:1900, 200:1400],  # Fixed card area
        gray[1700:1850, 300:1300],  # Small card area
    ]
    
    card_scores = [np.mean(region) for region in card_regions]
    max_cards = max(card_scores)
    min_cards = min(card_scores)
    card_std = np.std(card_scores)
    
    print(f"  Cards: max={max_cards:.1f}, min={min_cards:.1f}, std={card_std:.1f}")
    
    # Method 4: Look for time display (should be bright)
    time_regions = [
        gray[50:100, width//2-100:width//2+100],  # Center time
        gray[40:90, width//2-50:width//2+50],     # Small time
        gray[60:110, width//2-75:width//2+75],    # Medium time
    ]
    
    time_scores = [np.mean(region) for region in time_regions]
    max_time = max(time_scores)
    
    print(f"  Time: max={max_time:.1f}")
    
    # Method 5: Look for towers (should be visible)
    left_tower = gray[height//2-100:height//2+100, 50:150]
    right_tower = gray[height//2-100:height//2+100, width-150:width-50]
    tower_brightness = (np.mean(left_tower) + np.mean(right_tower)) / 2
    
    print(f"  Towers: {tower_brightness:.1f}")
    
    # Method 6: Look for elixir droplets (should have bright spots)
    elixir_droplet_region = gray[40:70, width//2-200:width//2+200]
    elixir_max = np.max(elixir_droplet_region)
    elixir_std_droplet = np.std(elixir_droplet_region)
    
    print(f"  Elixir droplets: max={elixir_max:.1f}, std={elixir_std_droplet:.1f}")
    
    # Scoring system - more flexible
    score = 0
    max_score = 0
    
    # Elixir scoring (multiple criteria)
    if max_elixir > 50:  # Very low threshold
        score += 1
    if max_elixir > 70:  # Medium threshold
        score += 1
    if elixir_std > 10:  # Some variation
        score += 1
    if elixir_max > 100:  # Bright spots
        score += 1
    max_score += 4
    
    # Arena scoring (multiple criteria)
    if min_arena < 150:  # Not too bright
        score += 1
    if min_arena < 100:  # Darker
        score += 1
    if arena_std > 20:  # Some variation
        score += 1
    if max_arena - min_arena > 30:  # Good contrast
        score += 1
    max_score += 4
    
    # Card scoring
    if max_cards > 20:  # Very low threshold
        score += 1
    if max_cards > 40:  # Medium threshold
        score += 1
    if card_std > 5:  # Some variation
        score += 1
    max_score += 3
    
    # Time scoring
    if max_time > 60:  # Low threshold
        score += 1
    if max_time > 80:  # Medium threshold
        score += 1
    max_score += 2
    
    # Tower scoring
    if tower_brightness > 20:  # Very low threshold
        score += 1
    if tower_brightness > 40:  # Medium threshold
        score += 1
    max_score += 2
    
    # Elixir droplet scoring
    if elixir_max > 80:  # Bright spots
        score += 1
    if elixir_std_droplet > 15:  # Good variation
        score += 1
    max_score += 2
    
    print(f"\n📊 Score: {score}/{max_score} ({score/max_score*100:.1f}%)")
    
    # Determine state based on score
    if score >= max_score * 0.6:  # 60% or more
        return GameState.IN_GAME
    elif score >= max_score * 0.3:  # 30% or more
        return GameState.MATCHMAKING
    else:
        return GameState.MENU

def test_ultra_robust_detection():
    """Test the ultra-robust detection"""
    print("🔍 Testing Ultra-Robust Detection")
    print("="*50)
    
    try:
        vision = ClashRoyaleVision()
        screen = vision.capture_screen()
        
        if screen is None:
            print("❌ Failed to capture screen")
            return None
        
        print("✓ Screen captured successfully")
        
        # Test original detection
        original_state = vision.detect_game_state(screen)
        print(f"Original detection: {original_state}")
        
        # Test ultra-robust detection
        robust_state = ultra_robust_detect_game_state(screen)
        print(f"Ultra-robust detection: {robust_state}")
        
        # Save screenshot
        cv2.imwrite("ultra_robust_debug.png", screen)
        print("📸 Screenshot saved as ultra_robust_debug.png")
        
        return robust_state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    print("🔍 Ultra-Robust Clash Royale Detection")
    print("="*50)
    print("\nThis uses multiple detection methods with flexible scoring")
    print("Make sure you're in a Clash Royale match")
    print("Press Enter to continue...")
    
    input()
    
    state = test_ultra_robust_detection()
    
    if state == GameState.IN_GAME:
        print("\n🎉 Successfully detected IN_GAME state!")
        print("The bot should now work properly!")
    elif state == GameState.MATCHMAKING:
        print("\n⏳ Detected MATCHMAKING state")
    else:
        print("\n📱 Detected MENU state")
        print("💡 Make sure you're actually in a Clash Royale match")

if __name__ == "__main__":
    main()
