"""
Improved screen capture that targets BlueStacks window specifically
"""

import sys
import cv2
import numpy as np
import pyautogui
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def find_bluestacks_window():
    """Find BlueStacks window on screen"""
    try:
        import subprocess
        
        # Use AppleScript to find BlueStacks window
        script = '''
        tell application "System Events"
            set windowList to every window of every process whose name contains "BlueStacks"
            if windowList is not {} then
                set windowBounds to bounds of item 1 of windowList
                return (item 1 of windowBounds) & "," & (item 2 of windowBounds) & "," & (item 3 of windowBounds) & "," & (item 4 of windowBounds)
            else
                return "not found"
            end if
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 and "not found" not in result.stdout:
            coords = result.stdout.strip().split(',')
            if len(coords) == 4:
                x, y, width, height = map(int, coords)
                return {"x": x, "y": y, "width": width, "height": height}
        
        return None
        
    except Exception as e:
        print(f"Error finding BlueStacks window: {e}")
        return None

def capture_bluestacks_screen():
    """Capture screen specifically from BlueStacks window"""
    print("🔍 Looking for BlueStacks window...")
    
    # Try to find BlueStacks window
    window_bounds = find_bluestacks_window()
    
    if window_bounds:
        print(f"✓ Found BlueStacks window: {window_bounds}")
        
        # Capture only the BlueStacks window
        screenshot = pyautogui.screenshot(region=(
            window_bounds["x"],
            window_bounds["y"],
            window_bounds["width"],
            window_bounds["height"]
        ))
        
        # Convert to OpenCV format
        screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        print(f"✓ Captured BlueStacks window: {screen.shape}")
        return screen
    else:
        print("❌ BlueStacks window not found")
        print("📋 Make sure BlueStacks is running and visible")
        return None

def capture_full_screen():
    """Capture full screen (current method)"""
    print("📸 Capturing full screen...")
    
    screenshot = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print(f"✓ Captured full screen: {screen.shape}")
    return screen

def compare_methods():
    """Compare full screen vs BlueStacks-specific capture"""
    print("🔍 Comparing Screen Capture Methods")
    print("="*50)
    
    # Method 1: Full screen
    print("\n1️⃣ Full Screen Method:")
    full_screen = capture_full_screen()
    if full_screen is not None:
        cv2.imwrite("full_screen.png", full_screen)
        print("📸 Saved: full_screen.png")
    
    # Method 2: BlueStacks window
    print("\n2️⃣ BlueStacks Window Method:")
    bluestacks_screen = capture_bluestacks_screen()
    if bluestacks_screen is not None:
        cv2.imwrite("bluestacks_window.png", bluestacks_screen)
        print("📸 Saved: bluestacks_window.png")
    
    # Method 3: Try to find BlueStacks in full screen
    if full_screen is not None:
        print("\n3️⃣ Looking for BlueStacks in full screen...")
        
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(full_screen, cv2.COLOR_BGR2GRAY)
        
        # Look for BlueStacks-like patterns
        # This is a simplified approach - in practice you'd use more sophisticated detection
        height, width = gray.shape
        
        # Check different regions for potential BlueStacks window
        regions_to_check = [
            (0, 0, width//2, height//2),  # Top left
            (width//2, 0, width//2, height//2),  # Top right
            (0, height//2, width//2, height//2),  # Bottom left
            (width//2, height//2, width//2, height//2),  # Bottom right
            (width//4, height//4, width//2, height//2),  # Center
        ]
        
        for i, (x, y, w, h) in enumerate(regions_to_check):
            region = gray[y:y+h, x:x+w]
            brightness = np.mean(region)
            print(f"  Region {i+1} ({x},{y},{w},{h}): brightness = {brightness:.1f}")
            
            # Save region for inspection
            region_bgr = full_screen[y:y+h, x:x+w]
            cv2.imwrite(f"region_{i+1}.png", region_bgr)
    
    print("\n📊 Analysis complete!")
    print("Check the saved images to see what each method captured.")

def main():
    """Main function"""
    print("🔍 Screen Capture Method Analysis")
    print("="*40)
    print("\nThis will show you exactly what the bot sees")
    print("Press Enter to continue...")
    
    input()
    
    compare_methods()

if __name__ == "__main__":
    main()
