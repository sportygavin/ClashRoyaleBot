# 🍎 iOS Simulator Setup Guide

## **Step 1: Install Xcode** ⬇️

You need to install the full Xcode app (not just command line tools):

### **Installation Steps:**
1. **Open Mac App Store** (click the App Store icon in your dock)
2. **Search "Xcode"** in the search bar
3. **Click "Get"** (it's free but large ~15GB)
4. **Wait for download** (this will take 30-60 minutes depending on your internet)

### **What Xcode Includes:**
- iOS Simulator
- Development tools
- Device simulators
- Everything you need for iOS automation

---

## **Step 2: Install Clash Royale** 📱

Once Xcode is installed:

### **Launch iOS Simulator:**
1. **Open Xcode** (from Applications folder)
2. **Go to:** `Xcode > Open Developer Tool > Simulator`
3. **Choose:** iPhone 15 Pro (or latest iPhone model)

### **Install Clash Royale:**
1. **In Simulator, tap Safari**
2. **Go to App Store** (search "App Store" if needed)
3. **Search "Clash Royale"**
4. **Tap "Get" to install**
5. **Wait for installation**

**Note:** You may need to sign in with an Apple ID. Use a test account, not your main account!

---

## **Step 3: Test the Bot** 🤖

Once everything is installed:

```bash
# Run the setup script
python3 setup_ios.py

# If setup passes, run the bot
python3 main.py
```

---

## **Step 4: Configure Bot Settings** ⚙️

The bot is already configured for iOS Simulator, but you can adjust:

### **Screen Resolution** (if needed):
```python
# In config.py
"ios_simulator": {
    "simulator_name": "iPhone 15 Pro",
    "screen_resolution": (1179, 2556),  # Adjust if needed
    "game_area": {"x": 0, "y": 0, "width": 1179, "height": 2556}
}
```

### **Card Positions** (if needed):
```python
# In config.py
"card_slots": {
    "count": 4,
    "positions": [
        (400, 900), (600, 900), (800, 900), (1000, 900)
    ]
}
```

---

## **Troubleshooting** 🔧

### **Bot doesn't detect game:**
- Make sure iOS Simulator window title contains "Simulator"
- Ensure Clash Royale is visible and not minimized
- Check screen resolution matches your simulator

### **Card placement not working:**
- Adjust card positions in config.py
- Make sure simulator is in portrait mode
- Check that Clash Royale is in the correct screen area

### **Performance issues:**
- Close other apps to free up memory
- Use a smaller simulator (iPhone SE instead of Pro Max)
- Reduce bot reaction time in config.py

---

## **Quick Commands** ⚡

```bash
# Check if Xcode is installed
ls /Applications/ | grep -i xcode

# Launch iOS Simulator
open -a Simulator

# Run bot setup
python3 setup_ios.py

# Run the bot
python3 main.py

# Run tests
python3 tests/test_bot.py
```

---

## **What Happens Next** 🚀

Once everything is set up:

1. **Bot will detect iOS Simulator**
2. **Automatically find Clash Royale**
3. **Start playing matches**
4. **Learn from each game**
5. **Improve over time**

The bot will:
- ✅ Capture screen from iOS Simulator
- ✅ Detect game states and cards
- ✅ Make strategic decisions
- ✅ Place cards automatically
- ✅ Record match data
- ✅ Learn from victories/defeats

---

## **Need Help?** 💬

If you run into issues:
1. Check the troubleshooting section above
2. Run `python3 setup_ios.py` to verify setup
3. Check the logs in the `logs/` directory
4. Make sure iOS Simulator is running and Clash Royale is installed

**You're all set! Once Xcode downloads, you'll have a fully functional Clash Royale bot running on iOS Simulator!** 🎉
