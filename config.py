"""
Clash Royale Bot - Main Configuration
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, SCREENSHOTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Game configuration
GAME_CONFIG = {
    # Platform options: "android_emulator", "ios_simulator", "macos_native", "iphone_physical"
    "platform": "android_emulator",  # Using Android emulator for MacBook
    
    # For Android emulator (BlueStacks/MuMuPlayer/Nox)
    "android_emulator": {
        "window_title": "BlueStacks",  # Change to "MuMuPlayer" or "Nox" if using those
        "screen_resolution": (3024, 1964),  # Updated for your MacBook Retina display
        "game_area": {"x": 0, "y": 0, "width": 3024, "height": 1964},
        "emulator_type": "bluestacks",  # Options: "bluestacks", "mumuplayer", "nox"
        "adb_port": 5555  # ADB port for automation
    },
    
    # For macOS native (if Clash Royale has desktop version)
    "macos_native": {
        "window_title": "Clash Royale",
        "screen_resolution": (1920, 1080),
        "game_area": {"x": 0, "y": 0, "width": 1920, "height": 1080}
    },
    
    # For iOS Simulator
    "ios_simulator": {
        "simulator_name": "iPhone 15 Pro",
        "screen_resolution": (1179, 2556),  # iPhone 15 Pro resolution
        "game_area": {"x": 0, "y": 0, "width": 1179, "height": 2556}
    },
    
    # For physical iPhone (via Xcode)
    "iphone_physical": {
        "device_name": "iPhone",
        "screen_resolution": (1179, 2556),
        "game_area": {"x": 0, "y": 0, "width": 1179, "height": 2556}
    },
    
    # Universal card positions (adjusted for 3024x1964 resolution)
    "card_slots": {
        "count": 4,
        "positions": [
            (630, 1470), (945, 1470), (1260, 1470), (1575, 1470)  # Scaled for 3024x1964
        ]
    },
    "elixir_bar": {
        "position": (1512, 78),  # Scaled for 3024x1964
        "max_elixir": 10
    }
}

# AI/ML Configuration
ML_CONFIG = {
    "model_type": "reinforcement_learning",
    "learning_rate": 0.001,
    "batch_size": 32,
    "memory_size": 10000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 0.995
}

# Bot behavior settings
BOT_CONFIG = {
    "reaction_time_ms": 100,
    "max_thinking_time_ms": 2000,
    "enable_learning": True,
    "save_replays": True,
    "log_level": "INFO"
}
