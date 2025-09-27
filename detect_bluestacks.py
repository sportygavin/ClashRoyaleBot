"""
Improved screen capture system for BlueStacks on MacBook
"""

import sys
import cv2
import numpy as np
import pyautogui
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def detect_screen_resolution():
    """Detect actual screen resolution"""
    screenshot = pyautogui.screenshot()
    height, width = screenshot.size
    print(f"📱 Detected screen resolution: {width}x{height}")
    return width, height

def find_bluestacks_in_screen(screen):
    """Find BlueStacks window in the captured screen"""
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print(f"🔍 Analyzing screen: {width}x{height}")
    
    # Look for BlueStacks-like patterns
    # BlueStacks typically has a darker area (the Android screen)
    # surrounded by lighter UI elements
    
    # Check different regions
    regions = []
    
    # Full screen regions
    regions.append(("Full Screen", 0, 0, width, height))
    
    # Quarter screen regions
    regions.append(("Top Left", 0, 0, width//2, height//2))
    regions.append(("Top Right", width//2, 0, width//2, height//2))
    regions.append(("Bottom Left", 0, height//2, width//2, height//2))
    regions.append(("Bottom Right", width//2, height//2, width//2, height//2))
    
    # Center regions
    regions.append(("Center", width//4, height//4, width//2, height//2))
    regions.append(("Center Large", width//8, height//8, width*3//4, height*3//4))
    
    # Common BlueStacks window sizes
    regions.append(("1280x720", 0, 0, 1280, 720))
    regions.append(("1920x1080", 0, 0, 1920, 1080))
    
    best_region = None
    best_score = 0
    
    for name, x, y, w, h in regions:
        # Make sure region is within screen bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w <= 0 or h <= 0:
            continue
            
        region = gray[y:y+h, x:x+w]
        if region.size == 0:
            continue
            
        # Calculate score based on characteristics that indicate BlueStacks
        brightness = np.mean(region)
        std_dev = np.std(region)
        
        # BlueStacks typically has:
        # - Moderate brightness (not too dark, not too bright)
        # - Good contrast (higher std dev)
        # - Reasonable size
        score = 0
        
        if 50 < brightness < 150:  # Good brightness range
            score += 1
        if std_dev > 30:  # Good contrast
            score += 1
        if w > 800 and h > 600:  # Reasonable size
            score += 1
        if w > 1200 and h > 800:  # Good size
            score += 1
            
        print(f"  {name}: {w}x{h} at ({x},{y}) - brightness: {brightness:.1f}, std: {std_dev:.1f}, score: {score}")
        
        if score > best_score:
            best_score = score
            best_region = (name, x, y, w, h, brightness, std_dev)
    
    return best_region

def create_adaptive_capture():
    """Create adaptive screen capture based on actual setup"""
    print("🔧 Creating Adaptive Screen Capture")
    print("="*40)
    
    # Detect screen resolution
    screen_width, screen_height = detect_screen_resolution()
    
    # Take a test screenshot
    print("📸 Taking test screenshot...")
    screenshot = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Find best BlueStacks region
    print("🔍 Looking for BlueStacks window...")
    best_region = find_bluestacks_in_screen(screen)
    
    if best_region:
        name, x, y, w, h, brightness, std_dev = best_region
        print(f"✅ Best region found: {name}")
        print(f"   Position: ({x}, {y})")
        print(f"   Size: {w}x{h}")
        print(f"   Brightness: {brightness:.1f}")
        print(f"   Contrast: {std_dev:.1f}")
        
        # Crop to the best region
        bluestacks_screen = screen[y:y+h, x:x+w]
        
        # Save the cropped image
        cv2.imwrite("detected_bluestacks.png", bluestacks_screen)
        print("📸 Saved: detected_bluestacks.png")
        
        # Update config with detected values
        config_update = f'''
# Detected BlueStacks configuration
GAME_CONFIG["android_emulator"]["game_area"] = {{
    "x": {x},
    "y": {y}, 
    "width": {w},
    "height": {h}
}}
GAME_CONFIG["android_emulator"]["screen_resolution"] = ({w}, {h})
'''
        
        print("\n📝 Add this to your config.py:")
        print(config_update)
        
        return bluestacks_screen
    else:
        print("❌ Could not find BlueStacks window")
        print("💡 Make sure BlueStacks is running and visible")
        return None

def main():
    """Main function"""
    print("🔍 BlueStacks Detection and Configuration")
    print("="*50)
    print("\nThis will detect your BlueStacks window and configure the bot")
    print("Press Enter to continue...")
    
    input()
    
    bluestacks_screen = create_adaptive_capture()
    
    if bluestacks_screen is not None:
        print("\n🎉 BlueStacks detected successfully!")
        print("The bot should now work much better with your setup.")
    else:
        print("\n❌ Could not detect BlueStacks")
        print("Try running BlueStacks in full screen mode")

if __name__ == "__main__":
    main()
