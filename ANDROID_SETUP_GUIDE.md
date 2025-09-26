# 🤖 Android Emulator Setup Guide for MacBook

## **Why Android Emulator is Better**

✅ **Reliable App Store access** - No connection issues like iOS Simulator  
✅ **Better bot compatibility** - Most Clash Royale bots are built for Android  
✅ **Mature tooling** - ADB, automation tools work perfectly  
✅ **Multiple options** - BlueStacks, MuMuPlayer, Nox Player  
✅ **MacBook optimized** - All major emulators support Apple Silicon  

---

## **Step 1: Choose Your Android Emulator**

### **🥇 Recommended: BlueStacks 5**
- **Best MacBook support** (Intel + Apple Silicon)
- **Easy setup** and configuration
- **Good performance** on MacBook
- **Download:** https://www.bluestacks.com/

### **🥈 Alternative: MuMuPlayer Pro**
- **Good performance** and stability
- **Free** with premium features
- **Download:** https://mumu.163.com/

### **🥉 Alternative: Nox Player**
- **Lightweight** and fast
- **Good for older MacBooks**
- **Download:** https://www.bignox.com/

---

## **Step 2: Install BlueStacks (Recommended)**

### **Download & Install:**
1. **Go to:** https://www.bluestacks.com/
2. **Click "Download BlueStacks 5"**
3. **Run the installer** (.dmg file)
4. **Follow installation wizard**
5. **Wait for setup** (5-10 minutes)

### **Initial Setup:**
1. **Launch BlueStacks**
2. **Sign in with Google account** (use test account)
3. **Complete Android setup**
4. **Wait for Google Play Store** to load

---

## **Step 3: Install Clash Royale**

### **In BlueStacks:**
1. **Open Google Play Store**
2. **Search "Clash Royale"**
3. **Tap "Install"**
4. **Wait for download** (2-3 minutes)
5. **Open Clash Royale**
6. **Complete tutorial manually**
7. **Sign in with SuperCell ID** (use test account)

### **Important Notes:**
- ✅ **Use test accounts** (not your main accounts)
- ✅ **Complete tutorial manually** before running bot
- ✅ **Set language to English**
- ✅ **Enable all permissions**

---

## **Step 4: Install ADB (Android Debug Bridge)**

ADB is needed for bot automation:

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ADB
brew install android-platform-tools
```

---

## **Step 5: Configure Bot for Android**

The bot is already configured for Android emulator. Just update the window title if needed:

```python
# In config.py
"android_emulator": {
    "window_title": "BlueStacks",  # Change if using different emulator
    "screen_resolution": (1920, 1080),
    "emulator_type": "bluestacks"
}
```

---

## **Step 6: Test the Setup**

```bash
# Run Android setup script
python3 setup_android.py

# If setup passes, run the bot
python3 main.py
```

---

## **🔧 Troubleshooting**

### **BlueStacks Won't Launch:**
- **Check system requirements** (8GB RAM minimum)
- **Enable virtualization** in BIOS/UEFI
- **Try different BlueStacks version**

### **Bot Doesn't Detect Game:**
- **Check window title** in config.py
- **Make sure Clash Royale is visible**
- **Adjust screen resolution** if needed

### **ADB Connection Issues:**
- **Enable ADB debugging** in BlueStacks settings
- **Check ADB port** (usually 5555)
- **Restart BlueStacks** and try again

### **Performance Issues:**
- **Close other apps** to free memory
- **Reduce BlueStacks resolution**
- **Use performance mode** in BlueStacks

---

## **⚡ Quick Commands**

```bash
# Check if BlueStacks is running
ps aux | grep -i bluestacks

# Check ADB devices
adb devices

# Launch BlueStacks
open -a BlueStacks

# Run bot setup
python3 setup_android.py

# Run the bot
python3 main.py
```

---

## **🎯 What Happens Next**

Once everything is set up:

1. **Bot detects BlueStacks** automatically
2. **Finds Clash Royale** in the emulator
3. **Starts playing matches** automatically
4. **Learns from each game** using ML
5. **Improves strategy** over time
6. **Records match data** for analysis

---

## **📊 Expected Performance**

- **Setup time:** 30-45 minutes (first time)
- **Bot reaction time:** 100-200ms
- **Learning improvement:** 5-10% win rate increase per 50 matches
- **Memory usage:** ~4-6GB (BlueStacks + bot)

---

## **🚀 Ready to Start?**

1. **Download BlueStacks** from https://www.bluestacks.com/
2. **Install and setup** (follow steps above)
3. **Install Clash Royale** in BlueStacks
4. **Run `python3 setup_android.py`**
5. **Run `python3 main.py`** to start the bot!

**This approach will be much more reliable than iOS Simulator!** 🤖📱
