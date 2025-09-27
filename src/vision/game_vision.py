"""
Computer Vision System for Clash Royale Bot

This module handles:
- Screen capture and processing
- Game state detection
- Card recognition
- Elixir detection
- Troop position tracking
"""

import cv2
import numpy as np
import pyautogui
from typing import Dict, List, Tuple, Optional
from PIL import Image
import time
from core import ComputerVisionSystem, GameState, GameInfo, Card
from config import GAME_CONFIG

class ClashRoyaleVision(ComputerVisionSystem):
    """Computer vision system for Clash Royale"""
    
    def __init__(self):
        # Get platform-specific configuration
        platform = GAME_CONFIG["platform"]
        platform_config = GAME_CONFIG.get(platform, GAME_CONFIG["ios_simulator"])
        
        self.game_area = platform_config["game_area"]
        self.card_positions = GAME_CONFIG["card_slots"]["positions"]
        self.elixir_position = GAME_CONFIG["elixir_bar"]["position"]
        self.platform = platform
        
        # Template matching templates (will be loaded from files)
        self.card_templates = {}
        self.ui_templates = {}
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load image templates for recognition"""
        # This will load card images, UI elements, etc.
        # For now, we'll use placeholder templates
        pass
    
    def capture_screen(self) -> np.ndarray:
        """Capture the game screen with proper aspect ratio"""
        try:
            # First, capture the full screen to detect actual dimensions
            full_screenshot = pyautogui.screenshot()
            full_height, full_width = full_screenshot.size
            
            # Calculate proper game area that maintains aspect ratio
            # BlueStacks typically uses 16:9 aspect ratio
            target_aspect_ratio = 16 / 9
            
            # Calculate the maximum width we can use while maintaining aspect ratio
            max_width = int(full_height * target_aspect_ratio)
            
            # If the calculated width is larger than screen width, use screen width
            if max_width > full_width:
                max_width = full_width
                # Recalculate height to maintain aspect ratio
                target_height = int(full_width / target_aspect_ratio)
            else:
                target_height = full_height
            
            # Center the game area on the screen
            x_offset = (full_width - max_width) // 2
            y_offset = (full_height - target_height) // 2
            
            # Capture the centered game area
            screenshot = pyautogui.screenshot(region=(
                x_offset,
                y_offset,
                max_width,
                target_height
            ))
            
            # Convert to OpenCV format
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Update game area for other methods
            self.game_area = {
                "x": x_offset,
                "y": y_offset,
                "width": max_width,
                "height": target_height
            }
            
            return screen
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def detect_game_state(self, screen: np.ndarray) -> GameState:
        """Detect current game state from screen"""
        if screen is None:
            return GameState.MENU
        
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Check for in-game elements
        if self._detect_elixir_bar(gray):
            return GameState.IN_GAME
        
        # Check for matchmaking elements
        if self._detect_matchmaking(gray):
            return GameState.MATCHMAKING
        
        # Check for match end screen
        if self._detect_match_end(gray):
            return GameState.MATCH_END
        
        return GameState.MENU
    
    def _detect_elixir_bar(self, gray_screen: np.ndarray) -> bool:
        """Detect if elixir bar is visible (indicates in-game)"""
        height, width = gray_screen.shape
        
        # Scale regions based on actual captured screen size
        # These are relative percentages that work for any screen size
        elixir_region = gray_screen[
            int(height * 0.04):int(height * 0.08),  # Top 4-8% of screen
            int(width * 0.25):int(width * 0.75)     # Middle 50% of screen width
        ]
        
        # More flexible detection - look for UI elements that indicate in-game
        elixir_brightness = np.mean(elixir_region)
        
        # Also check for other in-game indicators
        # Look for the arena (middle area should be darker in game)
        arena_region = gray_screen[
            int(height * 0.20):int(height * 0.80),  # Middle 60% of screen height
            int(width * 0.10):int(width * 0.90)     # Middle 80% of screen width
        ]
        arena_brightness = np.mean(arena_region)
        
        # Look for card slots at bottom
        card_region = gray_screen[
            int(height * 0.85):int(height * 1.0),   # Bottom 15% of screen
            int(width * 0.10):int(width * 0.90)     # Middle 80% of screen width
        ]
        card_brightness = np.mean(card_region)
        
        # In-game if we have reasonable brightness in UI areas
        # and darker arena (indicating game field)
        # Adjusted thresholds based on actual screen analysis
        in_game_indicators = (
            elixir_brightness > 30 and  # Elixir bar visible (lowered from 60)
            arena_brightness < 120 and  # Arena is darker
            card_brightness > 20        # Card area visible (lowered from 30)
        )
        
        return in_game_indicators
    
    def _detect_matchmaking(self, gray_screen: np.ndarray) -> bool:
        """Detect matchmaking screen"""
        # Look for matchmaking UI elements
        # This would use template matching in a real implementation
        return False
    
    def _detect_match_end(self, gray_screen: np.ndarray) -> bool:
        """Detect match end screen"""
        # Look for victory/defeat screen elements
        # This would use template matching in a real implementation
        return False
    
    def extract_game_info(self, screen: np.ndarray) -> GameInfo:
        """Extract detailed game information from screen"""
        if screen is None:
            return None
        
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Extract various game elements
        player_elixir = self._extract_elixir(gray)
        player_cards = self._extract_player_cards(screen)
        player_towers = self._extract_tower_health(gray, is_player=True)
        opponent_towers = self._extract_tower_health(gray, is_player=False)
        arena_troops = self._extract_arena_troops(screen)
        time_remaining = self._extract_match_time(gray)
        
        return GameInfo(
            state=GameState.IN_GAME,
            player_elixir=player_elixir,
            opponent_elixir=10.0,  # Placeholder - would need opponent detection
            player_towers=player_towers,
            opponent_towers=opponent_towers,
            player_cards=player_cards,
            opponent_cards=[],  # Placeholder - opponent cards harder to detect
            time_remaining=time_remaining,
            arena_troops=arena_troops
        )
    
    def _extract_elixir(self, gray_screen: np.ndarray) -> float:
        """Extract current elixir amount"""
        # Focus on elixir bar area
        elixir_region = gray_screen[30:80, 800:1120]
        
        # Simple elixir detection (placeholder)
        # Real implementation would use OCR or template matching
        return 10.0  # Placeholder
    
    def _extract_player_cards(self, screen: np.ndarray) -> List[Card]:
        """Extract player's available cards"""
        cards = []
        height, width = screen.shape[:2]
        
        # Calculate card positions relative to the actual screen size
        # 4 cards evenly distributed across the bottom 15% of the screen
        card_y = int(height * 0.92)  # 92% down from top
        card_spacing = width // 5  # Divide width into 5 sections (4 cards + margins)
        
        for i in range(4):
            # Calculate x position for each card
            card_x = card_spacing + (i * card_spacing)
            pos = (card_x, card_y)
            
            # Extract card region
            card_region = screen[pos[1]-50:pos[1]+50, pos[0]-50:pos[0]+50]
            
            # Detect if card is available (not grayed out)
            is_available = self._is_card_available(card_region)
            
            # Get card name and cost (placeholder)
            card_name = f"card_{i+1}"
            card_cost = 3  # Placeholder
            
            cards.append(Card(
                name=card_name,
                cost=card_cost,
                position=pos,
                is_available=is_available,
                cooldown_remaining=0.0
            ))
        
        return cards
    
    def _is_card_available(self, card_region: np.ndarray) -> bool:
        """Check if card is available to play"""
        # Simple brightness check - available cards are brighter
        brightness = np.mean(card_region)
        return brightness > 100  # Placeholder threshold
    
    def _extract_tower_health(self, gray_screen: np.ndarray, is_player: bool) -> Dict[str, int]:
        """Extract tower health"""
        # Placeholder implementation
        return {
            "king_tower": 2400,
            "left_tower": 1400,
            "right_tower": 1400
        }
    
    def _extract_arena_troops(self, screen: np.ndarray) -> List[Dict]:
        """Extract troops currently on arena"""
        # This would use object detection/recognition
        # For now, return empty list
        return []
    
    def _extract_match_time(self, gray_screen: np.ndarray) -> int:
        """Extract remaining match time"""
        # Look for timer in top center
        # Placeholder implementation
        return 180  # 3 minutes
    
    def get_match_result(self) -> Dict:
        """Get match result when game ends"""
        # This would detect victory/defeat screen
        return {
            "result": "victory",  # or "defeat"
            "trophies_gained": 30,
            "match_duration": 180
        }
