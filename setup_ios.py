"""
iOS Simulator Setup Script for Clash Royale Bot

This script helps set up the iOS Simulator environment for the bot.
"""

import subprocess
import sys
import time
from pathlib import Path

def check_xcode_installation():
    """Check if Xcode is installed"""
    print("Checking Xcode installation...")
    
    # Check if Xcode app exists
    xcode_path = Path("/Applications/Xcode.app")
    if xcode_path.exists():
        print("✓ Xcode is installed")
        return True
    else:
        print("❌ Xcode not found")
        print("Please install Xcode from Mac App Store:")
        print("1. Open Mac App Store")
        print("2. Search 'Xcode'")
        print("3. Click 'Get' (it's free)")
        print("4. Wait for download (~15GB)")
        return False

def check_ios_simulator():
    """Check if iOS Simulator is available"""
    print("Checking iOS Simulator...")
    
    try:
        # Try to list available simulators
        result = subprocess.run(
            ["xcrun", "simctl", "list", "devices"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ iOS Simulator is available")
            return True
        else:
            print("❌ iOS Simulator not available")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ iOS Simulator check timed out")
        return False
    except FileNotFoundError:
        print("❌ Xcode command line tools not found")
        return False

def launch_ios_simulator():
    """Launch iOS Simulator"""
    print("Launching iOS Simulator...")
    
    try:
        # Launch iOS Simulator
        subprocess.Popen([
            "open", "-a", "Simulator"
        ])
        
        print("✓ iOS Simulator launched")
        print("Please wait for it to fully load...")
        time.sleep(5)
        return True
        
    except Exception as e:
        print(f"❌ Failed to launch iOS Simulator: {e}")
        return False

def install_clash_royale_instructions():
    """Provide instructions for installing Clash Royale"""
    print("\n" + "="*60)
    print("📱 INSTALL CLASH ROYALE IN iOS SIMULATOR")
    print("="*60)
    print()
    print("1. In the iOS Simulator, tap the Safari app")
    print("2. Go to the App Store (search 'App Store' if needed)")
    print("3. Search for 'Clash Royale'")
    print("4. Tap 'Get' to install")
    print("5. Wait for installation to complete")
    print()
    print("Note: You may need to sign in with an Apple ID")
    print("Use a test account, not your main account!")
    print()

def test_bot_configuration():
    """Test bot configuration for iOS Simulator"""
    print("Testing bot configuration...")
    
    try:
        # Import and test configuration
        from config import GAME_CONFIG
        
        platform = GAME_CONFIG["platform"]
        if platform == "ios_simulator":
            print("✓ Bot configured for iOS Simulator")
            return True
        else:
            print(f"❌ Bot configured for {platform}, not iOS Simulator")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def run_basic_tests():
    """Run basic bot tests"""
    print("Running basic bot tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_bot.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Basic tests passed")
            return True
        else:
            print("❌ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🍎 iOS Simulator Setup for Clash Royale Bot")
    print("="*50)
    
    # Step 1: Check Xcode
    if not check_xcode_installation():
        print("\nPlease install Xcode first, then run this script again.")
        return False
    
    # Step 2: Check iOS Simulator
    if not check_ios_simulator():
        print("\nPlease ensure Xcode is fully installed with iOS Simulator.")
        return False
    
    # Step 3: Test bot configuration
    if not test_bot_configuration():
        print("\nBot configuration issue. Please check config.py")
        return False
    
    # Step 4: Run basic tests
    if not run_basic_tests():
        print("\nBasic tests failed. Please check the bot code.")
        return False
    
    # Step 5: Launch iOS Simulator
    if not launch_ios_simulator():
        print("\nFailed to launch iOS Simulator.")
        return False
    
    # Step 6: Installation instructions
    install_clash_royale_instructions()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Install Clash Royale in iOS Simulator (see instructions above)")
    print("2. Run the bot: python main.py")
    print("3. The bot will automatically detect the iOS Simulator")
    print()
    print("Troubleshooting:")
    print("- If bot doesn't detect game, check iOS Simulator window title")
    print("- Make sure Clash Royale is visible and not minimized")
    print("- Adjust screen resolution in config.py if needed")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
