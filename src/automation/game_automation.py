"""
Action Executor for Clash Royale Bot

This module handles:
- Card placement automation
- Touch input simulation
- Game interaction
"""

import pyautogui
import time
from typing import Tuple, Optional
from core import ActionExecutor
from config import GAME_CONFIG

class ClashRoyaleAutomation(ActionExecutor):
    """Handles automated game interactions"""
    
    def __init__(self):
        # Get platform-specific configuration
        platform = GAME_CONFIG["platform"]
        platform_config = GAME_CONFIG.get(platform, GAME_CONFIG["ios_simulator"])
        
        self.game_area = platform_config["game_area"]
        self.card_positions = GAME_CONFIG["card_slots"]["positions"]
        self.platform = platform
        
        # Configure pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def place_card(self, card_index: int, position: Tuple[int, int]) -> bool:
        """Place a card at specified position"""
        try:
            if card_index < 0 or card_index >= len(self.card_positions):
                return False
            
            # Get card position
            card_pos = self.card_positions[card_index]
            
            # Click and drag from card to target position
            pyautogui.moveTo(card_pos[0], card_pos[1])
            pyautogui.mouseDown()
            time.sleep(0.1)  # Small delay for drag start
            
            pyautogui.moveTo(position[0], position[1])
            time.sleep(0.1)  # Small delay for drag end
            
            pyautogui.mouseUp()
            
            return True
            
        except Exception as e:
            print(f"Error placing card: {e}")
            return False
    
    def click_position(self, position: Tuple[int, int]) -> bool:
        """Click at specified screen position"""
        try:
            pyautogui.click(position[0], position[1])
            return True
            
        except Exception as e:
            print(f"Error clicking position: {e}")
            return False
    
    def drag_card(self, card_index: int, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> bool:
        """Drag a card from start to end position"""
        try:
            if card_index < 0 or card_index >= len(self.card_positions):
                return False
            
            # Start drag from card
            card_pos = self.card_positions[card_index]
            pyautogui.moveTo(card_pos[0], card_pos[1])
            pyautogui.mouseDown()
            time.sleep(0.1)
            
            # Drag to start position
            pyautogui.moveTo(start_pos[0], start_pos[1])
            time.sleep(0.1)
            
            # Continue to end position
            pyautogui.moveTo(end_pos[0], end_pos[1])
            time.sleep(0.1)
            
            pyautogui.mouseUp()
            return True
            
        except Exception as e:
            print(f"Error dragging card: {e}")
            return False
    
    def simulate_human_delay(self, min_ms: int = 50, max_ms: int = 200):
        """Add human-like delay between actions"""
        import random
        delay = random.uniform(min_ms / 1000.0, max_ms / 1000.0)
        time.sleep(delay)
