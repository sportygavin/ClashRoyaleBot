"""
Android Emulator Setup Script for Clash Royale Bot

This script helps set up Android emulators (BlueStacks, MuMuPlayer, Nox) for the bot.
"""

import subprocess
import sys
import time
from pathlib import Path

def check_bluestacks_installation():
    """Check if BlueStacks is installed"""
    print("Checking BlueStacks installation...")
    
    # Check common BlueStacks installation paths
    bluestacks_paths = [
        "/Applications/BlueStacks.app",
        "/Applications/BlueStacks 5.app",
        "/Applications/BlueStacks X.app"
    ]
    
    for path in bluestacks_paths:
        if Path(path).exists():
            print(f"✓ BlueStacks found at: {path}")
            return True
    
    print("❌ BlueStacks not found")
    return False

def check_mumuplayer_installation():
    """Check if MuMuPlayer is installed"""
    print("Checking MuMuPlayer installation...")
    
    mumu_paths = [
        "/Applications/MuMuPlayer.app",
        "/Applications/MuMuPlayer Pro.app"
    ]
    
    for path in mumu_paths:
        if Path(path).exists():
            print(f"✓ MuMuPlayer found at: {path}")
            return True
    
    print("❌ MuMuPlayer not found")
    return False

def check_nox_installation():
    """Check if Nox Player is installed"""
    print("Checking Nox Player installation...")
    
    nox_paths = [
        "/Applications/NoxPlayer.app",
        "/Applications/Nox App Player.app"
    ]
    
    for path in nox_paths:
        if Path(path).exists():
            print(f"✓ Nox Player found at: {path}")
            return True
    
    print("❌ Nox Player not found")
    return False

def check_adb_installation():
    """Check if ADB (Android Debug Bridge) is available"""
    print("Checking ADB installation...")
    
    try:
        result = subprocess.run(
            ["adb", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ ADB is available")
            return True
        else:
            print("❌ ADB not working properly")
            return False
            
    except FileNotFoundError:
        print("❌ ADB not found")
        return False
    except subprocess.TimeoutExpired:
        print("❌ ADB check timed out")
        return False

def install_adb():
    """Install ADB via Homebrew"""
    print("Installing ADB...")
    
    try:
        # Check if Homebrew is installed
        subprocess.run(["brew", "--version"], check=True, capture_output=True)
        
        # Install android-platform-tools (includes ADB)
        subprocess.run([
            "brew", "install", "android-platform-tools"
        ], check=True)
        
        print("✓ ADB installed successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to install ADB")
        return False
    except FileNotFoundError:
        print("❌ Homebrew not found. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False

def launch_bluestacks():
    """Launch BlueStacks"""
    print("Launching BlueStacks...")
    
    try:
        subprocess.Popen([
            "open", "-a", "BlueStacks"
        ])
        
        print("✓ BlueStacks launched")
        print("Please wait for it to fully load...")
        time.sleep(10)
        return True
        
    except Exception as e:
        print(f"❌ Failed to launch BlueStacks: {e}")
        return False

def test_bot_configuration():
    """Test bot configuration for Android emulator"""
    print("Testing bot configuration...")
    
    try:
        from config import GAME_CONFIG
        
        platform = GAME_CONFIG["platform"]
        if platform == "android_emulator":
            print("✓ Bot configured for Android emulator")
            return True
        else:
            print(f"❌ Bot configured for {platform}, not Android emulator")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def install_clash_royale_instructions():
    """Provide instructions for installing Clash Royale"""
    print("\n" + "="*60)
    print("📱 INSTALL CLASH ROYALE IN ANDROID EMULATOR")
    print("="*60)
    print()
    print("1. In the Android emulator, open Google Play Store")
    print("2. Search for 'Clash Royale'")
    print("3. Tap 'Install' to download and install")
    print("4. Wait for installation to complete")
    print("5. Open Clash Royale and complete the tutorial")
    print("6. Sign in with a SuperCell ID (use a test account)")
    print()
    print("Important:")
    print("- Use a test SuperCell ID, not your main account")
    print("- Complete the tutorial manually before running the bot")
    print("- Set game language to English")
    print()

def main():
    """Main setup function"""
    print("🤖 Android Emulator Setup for Clash Royale Bot")
    print("="*50)
    
    # Check for emulators
    emulators_found = []
    
    if check_bluestacks_installation():
        emulators_found.append("BlueStacks")
    
    if check_mumuplayer_installation():
        emulators_found.append("MuMuPlayer")
    
    if check_nox_installation():
        emulators_found.append("Nox Player")
    
    if not emulators_found:
        print("\n❌ No Android emulators found!")
        print("\nPlease install one of these emulators:")
        print("1. BlueStacks: https://www.bluestacks.com/")
        print("2. MuMuPlayer Pro: https://mumu.163.com/")
        print("3. Nox Player: https://www.bignox.com/")
        print("\nRecommended: BlueStacks (best MacBook support)")
        return False
    
    print(f"\n✓ Found emulators: {', '.join(emulators_found)}")
    
    # Check ADB
    if not check_adb_installation():
        print("\nInstalling ADB...")
        if not install_adb():
            print("\nPlease install ADB manually:")
            print("brew install android-platform-tools")
            return False
    
    # Test bot configuration
    if not test_bot_configuration():
        print("\nBot configuration issue. Please check config.py")
        return False
    
    # Launch BlueStacks (if available)
    if "BlueStacks" in emulators_found:
        if not launch_bluestacks():
            print("\nFailed to launch BlueStacks.")
            return False
    
    # Installation instructions
    install_clash_royale_instructions()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Install Clash Royale in the Android emulator")
    print("2. Complete the tutorial manually")
    print("3. Run the bot: python3 main.py")
    print()
    print("Troubleshooting:")
    print("- If bot doesn't detect game, check emulator window title")
    print("- Make sure Clash Royale is visible and not minimized")
    print("- Adjust screen resolution in config.py if needed")
    print("- Enable ADB debugging in emulator settings")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
