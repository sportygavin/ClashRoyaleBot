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
import os
import json
from pathlib import Path
from core import ComputerVisionSystem, GameState, GameInfo, Card
from config import GAME_CONFIG

# Try to import ultralytics for YOLO, but handle gracefully if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO detection will be disabled.")

class ClashRoyaleVision(ComputerVisionSystem):
    """Computer vision system for Clash Royale"""
    
    def __init__(self, calibration_path: str = "cv_out/calibration_manual_fixed.json"):
        # Get platform-specific configuration
        platform = GAME_CONFIG["platform"]
        platform_config = GAME_CONFIG.get(platform, GAME_CONFIG["ios_simulator"])
        
        self.game_area = platform_config["game_area"]
        self.card_positions = GAME_CONFIG["card_slots"]["positions"]
        self.elixir_position = GAME_CONFIG["elixir_bar"]["position"]
        self.platform = platform
        
        # Load calibration data
        self.calibration_path = calibration_path
        self.calibration = None
        self.viewport = None
        self.actual_screen_size = None  # Store actual screenshot dimensions (for Retina displays)
        self._load_calibration()
        
        # Template matching templates (will be loaded from files)
        self.card_templates = {}
        self.ui_templates = {}
        
        # YOLO model and configuration
        self.yolo_model = None
        self.yolo_available = False
        self.card_classes = {}  # Map class ID to card name
        self.card_database = {}  # Card database for costs
        
        # Load YOLO model if available
        self._load_yolo_model()
        
        # Load card database
        self._load_card_database()
        
        # Load templates
        self._load_templates()
    
    def _load_calibration(self):
        """Load calibration data for viewport extraction - uses ACTUAL screenshot dimensions, not pyautogui.size()"""
        # Resolve path relative to project root if needed (same as strategy_utils.load_calibration)
        if not os.path.isabs(self.calibration_path):
            # Get project root (parent of src directory)
            project_root = Path(__file__).parent.parent.parent
            calib_path = project_root / self.calibration_path
        else:
            calib_path = Path(self.calibration_path)
        
        if not calib_path.exists():
            print(f"Warning: Calibration file not found at {calib_path}")
            print(f"  (resolved from: {self.calibration_path})")
            print("Using default screen capture method")
            return
        
        try:
            with open(calib_path, 'r') as f:
                self.calibration = json.load(f)
            
            # CRITICAL: Use actual screenshot dimensions, not pyautogui.size()
            # On Retina displays, screenshot is 2x larger than pyautogui.size()
            # This matches CardRecognitionSystem.extract_cards_from_screen() approach
            screenshot = pyautogui.screenshot()
            if screenshot:
                actual_screen_w = screenshot.width
                actual_screen_h = screenshot.height
            else:
                # Fallback to pyautogui.size() if screenshot fails
                actual_screen_w, actual_screen_h = pyautogui.size()
            
            viewport_r = self.calibration.get('viewport') or {'x_r': 0.0, 'y_r': 0.0, 'w_r': 1.0, 'h_r': 1.0}
            # Use actual screenshot dimensions (same as CardRecognitionSystem)
            vx = int(viewport_r['x_r'] * actual_screen_w)
            vy = int(viewport_r['y_r'] * actual_screen_h)
            vw = int(viewport_r['w_r'] * actual_screen_w)
            vh = int(viewport_r['h_r'] * actual_screen_h)
            self.viewport = (vx, vy, vw, vh)
            self.actual_screen_size = (actual_screen_w, actual_screen_h)
            print(f"✅ Calibration loaded from {calib_path}")
            print(f"   Viewport: ({vx}, {vy}) size {vw}x{vh} (actual screen: {actual_screen_w}x{actual_screen_h})")
            pyautogui_size = pyautogui.size()
            if actual_screen_w != pyautogui_size[0] or actual_screen_h != pyautogui_size[1]:
                print(f"   Note: pyautogui.size() = {pyautogui_size[0]}x{pyautogui_size[1]} (Retina scaling detected)")
        except Exception as e:
            print(f"Error loading calibration: {e}")
            import traceback
            traceback.print_exc()
            self.calibration = None
            self.viewport = None
            self.actual_screen_size = None
    
    def _load_yolo_model(self):
        """Load YOLO model for card detection"""
        if not YOLO_AVAILABLE:
            print("YOLO not available - using fallback detection methods")
            return
        
        model_path = Path('yolo_training/clash_royale_cards/weights/best.pt')
        
        if not model_path.exists():
            print(f"Warning: YOLO model not found at {model_path}")
            print("Card detection will use fallback methods")
            return
        
        try:
            self.yolo_model = YOLO(str(model_path))
            self.yolo_available = True
            
            # Define card class mapping based on dataset.yaml
            # These match the classes in yolo_data/dataset.yaml
            self.card_classes = {
                0: "Princess",
                1: "Knight",
                2: "Goblin Gang",
                3: "Ice Spirit",
                4: "The Log",
                5: "Goblin Barrel",
                6: "Inferno Tower",
                7: "Spear Goblin",
                8: "Goblin",
                9: "Archer"
            }
            
            print(f"✅ YOLO model loaded successfully from {model_path}")
            print(f"✅ Card classes: {len(self.card_classes)}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Card detection will use fallback methods")
            self.yolo_available = False
    
    def _load_card_database(self):
        """Load card database for elixir costs and other info"""
        # Try new database first, fallback to old one
        db_path = Path('data/cards_database.json')
        
        if not db_path.exists():
            # Fallback to old database location
            db_path = Path('database/clash_royale_cards.json')
            if not db_path.exists():
                print(f"Warning: Card database not found at {db_path}")
                return
        
        try:
            with open(db_path, 'r') as f:
                data = json.load(f)
                
                # Handle both array format (new) and dict format (old)
                if isinstance(data, list):
                    # New format: array of card objects
                    self.card_database = {card.get('key', card.get('name', '').lower()): card for card in data}
                elif isinstance(data, dict) and 'cards' in data:
                    # Old format: dict with 'cards' key
                    self.card_database = data.get('cards', {})
                else:
                    # Old format: direct dict
                    self.card_database = data
                    
            print(f"✅ Card database loaded: {len(self.card_database)} cards from {db_path}")
        except Exception as e:
            print(f"Error loading card database: {e}")
    
    def _get_card_cost(self, card_name: str) -> int:
        """Get elixir cost for a card from database"""
        if not self.card_database:
            return 3  # Default cost
        
        card_name_lower = card_name.lower()
        
        # Try exact match first (by name)
        for card_key, card_data in self.card_database.items():
            db_name = card_data.get('name', '').lower()
            if db_name == card_name_lower:
                # New format uses 'elixir', old format uses 'elixir_cost'
                return card_data.get('elixir', card_data.get('elixir_cost', 3))
        
        # Try match by key
        if card_name_lower in self.card_database:
            card_data = self.card_database[card_name_lower]
            return card_data.get('elixir', card_data.get('elixir_cost', 3))
        
        # Try partial match (e.g., "Goblin" matches "Goblins")
        for card_key, card_data in self.card_database.items():
            db_name = card_data.get('name', '').lower()
            if card_name_lower in db_name or db_name in card_name_lower:
                return card_data.get('elixir', card_data.get('elixir_cost', 3))
        
        return 3  # Default cost if not found
    
    def _load_templates(self):
        """Load image templates for recognition"""
        # This will load card images, UI elements, etc.
        # For now, we'll use placeholder templates
        pass
    
    def capture_screen(self) -> np.ndarray:
        """Capture the full game screen (like screen_bgr in strategy_utils)"""
        try:
            # Capture full screen - same as screen_bgr() in strategy_utils
            screenshot = pyautogui.screenshot()
            if screenshot is None:
                return None
            
            # Convert to OpenCV format
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Update game area to full screen dimensions
            height, width = screen.shape[:2]
            self.game_area = {
                "x": 0,
                "y": 0,
                "width": width,
                "height": height
            }
            
            return screen
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Detect opponent cards using YOLO
        opponent_cards = self.detect_opponent_cards(screen)
        
        return GameInfo(
            state=GameState.IN_GAME,
            player_elixir=player_elixir,
            opponent_elixir=10.0,  # Placeholder - would need opponent detection
            player_towers=player_towers,
            opponent_towers=opponent_towers,
            player_cards=player_cards,
            opponent_cards=opponent_cards,
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
        """Extract player's available cards using YOLO detection"""
        if not self.yolo_available or self.yolo_model is None:
            # Fallback to original method if YOLO not available
            return self._extract_player_cards_fallback(screen)
        
        cards = []
        screen_height, screen_width = screen.shape[:2]
        
        # Extract viewport from full screen first
        if self.viewport is None:
            # No viewport, use full screen
            viewport_region = screen
            vx, vy = 0, 0
        else:
            vx, vy, vw, vh = self.viewport
            # Extract viewport region from full screen
            viewport_region = screen[vy:vy+vh, vx:vx+vw]
            vh_actual, vw_actual = viewport_region.shape[:2]
        
        if viewport_region.size == 0:
            return cards
        
        # Extract card hand region from viewport (card_row is relative to viewport)
        if self.calibration and 'card_row' in self.calibration:
            card_row = self.calibration['card_row']
            # Card row coordinates are relative to viewport
            vh_actual = viewport_region.shape[0]
            card_region_y1_vp = int(card_row.get('top_r', 0.85) * vh_actual)
            card_region_y2_vp = int(card_row.get('bottom_r', 1.0) * vh_actual)
            card_region = viewport_region[card_region_y1_vp:card_region_y2_vp, :]
        else:
            # Fallback: bottom 15% of viewport
            vh_actual = viewport_region.shape[0]
            card_region_y1_vp = int(vh_actual * 0.85)
            card_region_y2_vp = vh_actual
            card_region = viewport_region[card_region_y1_vp:card_region_y2_vp, :]
        
        if card_region.size == 0:
            return cards
        
        # Run YOLO inference on card hand region
        confidence_threshold = 0.3
        try:
            results = self.yolo_model(card_region, conf=confidence_threshold, iou=0.45, verbose=False)
            
            # Process detections
            detected_cards = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get bounding box coordinates (relative to card_region)
                        # Convert tensors using tolist() which is more reliable
                        try:
                            xyxy = box.xyxy[0]
                            if hasattr(xyxy, 'cpu'):
                                xyxy = xyxy.cpu()
                            if hasattr(xyxy, 'tolist'):
                                xyxy_list = xyxy.tolist()
                            elif hasattr(xyxy, 'numpy'):
                                xyxy_list = xyxy.numpy().tolist()
                            else:
                                xyxy_list = list(xyxy)
                            x1, y1, x2, y2 = xyxy_list
                            
                            conf = box.conf[0]
                            if hasattr(conf, 'cpu'):
                                conf = conf.cpu()
                            if hasattr(conf, 'item'):
                                confidence = float(conf.item())
                            elif hasattr(conf, 'tolist'):
                                confidence = float(conf.tolist())
                            else:
                                confidence = float(conf)
                            
                            cls = box.cls[0]
                            if hasattr(cls, 'cpu'):
                                cls = cls.cpu()
                            if hasattr(cls, 'item'):
                                class_id = int(cls.item())
                            elif hasattr(cls, 'tolist'):
                                class_id = int(cls.tolist())
                            else:
                                class_id = int(cls)
                        except Exception as e:
                            print(f"Error converting tensor: {e}")
                            continue
                        
                        # Convert coordinates: card_region -> viewport -> full screen
                        # YOLO coordinates are relative to card_region
                        # Add card_region offset to get viewport coordinates
                        x1_vp = int(x1)  # card_region starts at x=0 in viewport
                        y1_vp = int(y1) + card_region_y1_vp
                        x2_vp = int(x2)
                        y2_vp = int(y2) + card_region_y1_vp
                        
                        # Convert viewport coordinates to full screen coordinates (screenshot space)
                        abs_x1 = x1_vp + vx
                        abs_y1 = y1_vp + vy
                        abs_x2 = x2_vp + vx
                        abs_y2 = y2_vp + vy
                        
                        # CRITICAL: Convert from screenshot coordinates to pyautogui coordinates
                        # On Retina displays, screenshot is 2x larger than pyautogui.size()
                        # right_loop.py uses pyautogui.size() for coordinates, so we must match that
                        pyautogui_w, pyautogui_h = pyautogui.size()
                        if self.actual_screen_size:
                            actual_w, actual_h = self.actual_screen_size
                            scale_x = pyautogui_w / actual_w
                            scale_y = pyautogui_h / actual_h
                        else:
                            scale_x = scale_y = 1.0
                        
                        # Convert to pyautogui coordinate space (same as right_loop.py uses)
                        abs_x1_py = int(abs_x1 * scale_x)
                        abs_y1_py = int(abs_y1 * scale_y)
                        abs_x2_py = int(abs_x2 * scale_x)
                        abs_y2_py = int(abs_y2 * scale_y)
                        
                        # Get card name from class ID
                        card_name = self.card_classes.get(class_id, f"Unknown_{class_id}")
                        
                        # Calculate center position in pyautogui coordinate space
                        center_x = (abs_x1_py + abs_x2_py) // 2
                        center_y = (abs_y1_py + abs_y2_py) // 2
                        pos = (center_x, center_y)
                        
                        # Debug: Print conversion details (can be removed later)
                        if len(detected_cards) < 1:  # Only print for first detection
                            print(f"DEBUG YOLO conversion: screenshot=({abs_x1}, {abs_y1})-({abs_x2}, {abs_y2}), "
                                  f"pyautogui=({abs_x1_py}, {abs_y1_py})-({abs_x2_py}, {abs_y2_py}), "
                                  f"center={pos}, scale=({scale_x:.3f}, {scale_y:.3f})")
                        
                        # Get card cost from database
                        card_cost = self._get_card_cost(card_name)
                        
                        # Check if card is available (not grayed out)
                        # Use screenshot coordinates for image extraction
                        card_bbox = screen[abs_y1:abs_y2, abs_x1:abs_x2]
                        is_available = self._is_card_available(card_bbox) if card_bbox.size > 0 else True
                        
                        detected_cards.append({
                            'name': card_name,
                            'cost': card_cost,
                            'position': pos,  # This is in pyautogui coordinate space
                            'is_available': is_available,
                            'confidence': confidence,
                            'bbox': (abs_x1, abs_y1, abs_x2, abs_y2)  # Screenshot space for image extraction
                        })
            
            # Sort by x position (left to right) and take top 4
            detected_cards.sort(key=lambda c: c['position'][0])
            detected_cards = detected_cards[:4]
            
            # Convert to Card objects
            for card_data in detected_cards:
                cards.append(Card(
                    name=card_data['name'],
                    cost=card_data['cost'],
                    position=card_data['position'],
                    is_available=card_data['is_available'],
                    cooldown_remaining=0.0
                ))
            
        except Exception as e:
            print(f"Error in YOLO card detection: {e}")
            # Fallback to original method on error
            return self._extract_player_cards_fallback(screen)
        
        # If no cards detected, fallback to original method
        if len(cards) == 0:
            return self._extract_player_cards_fallback(screen)
        
        return cards
    
    def _extract_player_cards_fallback(self, screen: np.ndarray) -> List[Card]:
        """Fallback method for extracting player cards - uses same coordinate system as right_loop.py"""
        cards = []
        
        # Use calibration-based positions (same as right_loop.py uses)
        if self.calibration and self.viewport:
            vx, vy, vw, vh = self.viewport
            if 'cards' in self.calibration and 'centers_x_r' in self.calibration['cards']:
                centers_x_r = self.calibration['cards']['centers_x_r']
                card_row = self.calibration.get('card_row', {})
                row_top_r = card_row.get('top_r', 0.85)
                row_bottom_r = card_row.get('bottom_r', 1.0)
                
                row_top_y = vy + int(row_top_r * vh)
                row_bottom_y = vy + int(row_bottom_r * vh)
                row_h = max(row_bottom_y - row_top_y, 1)
                top_offset_r = self.calibration['cards'].get('top_offset_r', 0.1)
                bottom_offset_r = self.calibration['cards'].get('bottom_offset_r', 0.1)
                card_top = row_top_y + int(top_offset_r * row_h)
                card_bottom = row_bottom_y - int(bottom_offset_r * row_h)
                cy = (card_top + card_bottom) // 2
                
                # Convert to pyautogui coordinate space
                pyautogui_w, pyautogui_h = pyautogui.size()
                if self.actual_screen_size:
                    actual_w, actual_h = self.actual_screen_size
                    scale_x = pyautogui_w / actual_w
                    scale_y = pyautogui_h / actual_h
                else:
                    scale_x = scale_y = 1.0
                
                for i in range(min(4, len(centers_x_r))):
                    cx_vp = int(centers_x_r[i] * vw)
                    cx_screenshot = vx + cx_vp
                    cx_py = int(cx_screenshot * scale_x)
                    cy_py = int(cy * scale_y)
                    pos = (cx_py, cy_py)
                    
                    # Extract card region (use screenshot coordinates)
                    card_region = screen[card_top:card_bottom, cx_screenshot-50:cx_screenshot+50]
                    
                    # Detect if card is available (not grayed out)
                    is_available = self._is_card_available(card_region) if card_region.size > 0 else True
                    
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
        
        # Fallback to old method if calibration not available
        height, width = screen.shape[:2]
        card_y = int(height * 0.92)
        card_spacing = width // 5
        
        for i in range(4):
            card_x = card_spacing + (i * card_spacing)
            pos = (card_x, card_y)
            card_region = screen[pos[1]-50:pos[1]+50, pos[0]-50:pos[0]+50]
            is_available = self._is_card_available(card_region)
            cards.append(Card(
                name=f"card_{i+1}",
                cost=3,
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
    
    def detect_opponent_cards(self, screen: np.ndarray) -> List[Card]:
        """Detect opponent cards in the arena using YOLO"""
        if not self.yolo_available or self.yolo_model is None:
            return []
        
        cards = []
        screen_height, screen_width = screen.shape[:2]
        
        # Extract viewport from full screen first
        if self.viewport is None:
            viewport_region = screen
            vx, vy = 0, 0
            vh_actual, vw_actual = screen_height, screen_width
        else:
            vx, vy, vw, vh = self.viewport
            viewport_region = screen[vy:vy+vh, vx:vx+vw]
            vh_actual, vw_actual = viewport_region.shape[:2]
        
        if viewport_region.size == 0:
            return cards
        
        # Extract opponent region from viewport (opponent_region_roi is relative to viewport)
        if self.calibration and 'opponent_region_roi' in self.calibration:
            opp_roi = self.calibration['opponent_region_roi']
            # Opponent region ROI is relative to viewport
            opponent_region_x1_vp = int(opp_roi.get('x_r', 0.78) * vw_actual)
            opponent_region_y1_vp = int(opp_roi.get('y_r', 0.22) * vh_actual)
            opponent_region_x2_vp = int((opp_roi.get('x_r', 0.78) + opp_roi.get('w_r', 1.68)) * vw_actual)
            opponent_region_y2_vp = int((opp_roi.get('y_r', 0.22) + opp_roi.get('h_r', 0.69)) * vh_actual)
            # Clamp to viewport bounds
            opponent_region_x1_vp = max(0, min(opponent_region_x1_vp, vw_actual))
            opponent_region_y1_vp = max(0, min(opponent_region_y1_vp, vh_actual))
            opponent_region_x2_vp = max(0, min(opponent_region_x2_vp, vw_actual))
            opponent_region_y2_vp = max(0, min(opponent_region_y2_vp, vh_actual))
            opponent_region = viewport_region[opponent_region_y1_vp:opponent_region_y2_vp, opponent_region_x1_vp:opponent_region_x2_vp]
        else:
            # Fallback: top half of viewport
            opponent_region_y1_vp = int(vh_actual * 0.10)
            opponent_region_y2_vp = int(vh_actual * 0.50)
            opponent_region_x1_vp = 0
            opponent_region_x2_vp = vw_actual
            opponent_region = viewport_region[opponent_region_y1_vp:opponent_region_y2_vp, :]
        
        if opponent_region.size == 0:
            return cards
        
        # Run YOLO inference on opponent region
        confidence_threshold = 0.3
        try:
            results = self.yolo_model(opponent_region, conf=confidence_threshold, iou=0.45, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get bounding box coordinates (relative to opponent_region)
                        # Convert tensors using tolist() which is more reliable
                        try:
                            xyxy = box.xyxy[0]
                            if hasattr(xyxy, 'cpu'):
                                xyxy = xyxy.cpu()
                            if hasattr(xyxy, 'tolist'):
                                xyxy_list = xyxy.tolist()
                            elif hasattr(xyxy, 'numpy'):
                                xyxy_list = xyxy.numpy().tolist()
                            else:
                                xyxy_list = list(xyxy)
                            x1, y1, x2, y2 = xyxy_list
                            
                            conf = box.conf[0]
                            if hasattr(conf, 'cpu'):
                                conf = conf.cpu()
                            if hasattr(conf, 'item'):
                                confidence = float(conf.item())
                            elif hasattr(conf, 'tolist'):
                                confidence = float(conf.tolist())
                            else:
                                confidence = float(conf)
                            
                            cls = box.cls[0]
                            if hasattr(cls, 'cpu'):
                                cls = cls.cpu()
                            if hasattr(cls, 'item'):
                                class_id = int(cls.item())
                            elif hasattr(cls, 'tolist'):
                                class_id = int(cls.tolist())
                            else:
                                class_id = int(cls)
                        except Exception as e:
                            print(f"Error converting tensor: {e}")
                            continue
                        
                        # Convert coordinates: opponent_region -> viewport -> full screen
                        # YOLO coordinates are relative to opponent_region
                        x1_vp = int(x1) + opponent_region_x1_vp
                        y1_vp = int(y1) + opponent_region_y1_vp
                        x2_vp = int(x2) + opponent_region_x1_vp
                        y2_vp = int(y2) + opponent_region_y1_vp
                        
                        # Convert viewport coordinates to full screen coordinates
                        abs_x1 = x1_vp + vx
                        abs_y1 = y1_vp + vy
                        abs_x2 = x2_vp + vx
                        abs_y2 = y2_vp + vy
                        
                        # Get card name from class ID
                        card_name = self.card_classes.get(class_id, f"Unknown_{class_id}")
                        
                        # Calculate center position
                        center_x = (abs_x1 + abs_x2) // 2
                        center_y = (abs_y1 + abs_y2) // 2
                        pos = (center_x, center_y)
                        
                        # Get card cost from database
                        card_cost = self._get_card_cost(card_name)
                        
                        cards.append(Card(
                            name=card_name,
                            cost=card_cost,
                            position=pos,
                            is_available=True,  # Opponent cards are always "available" (visible)
                            cooldown_remaining=0.0
                        ))
            
        except Exception as e:
            print(f"Error in YOLO opponent detection: {e}")
            return []
        
        return cards
    
    def _extract_arena_troops(self, screen: np.ndarray) -> List[Dict]:
        """Extract troops currently on arena"""
        # Use YOLO to detect cards in arena (both player and opponent)
        arena_troops = []
        
        if not self.yolo_available or self.yolo_model is None:
            return arena_troops
        
        screen_height, screen_width = screen.shape[:2]
        
        # Extract viewport from full screen first
        if self.viewport is None:
            viewport_region = screen
            vx, vy = 0, 0
            vh_actual, vw_actual = screen_height, screen_width
        else:
            vx, vy, vw, vh = self.viewport
            viewport_region = screen[vy:vy+vh, vx:vx+vw]
            vh_actual, vw_actual = viewport_region.shape[:2]
        
        if viewport_region.size == 0:
            return arena_troops
        
        # Extract arena region from viewport (excluding card row)
        if self.calibration and 'card_row' in self.calibration:
            card_row = self.calibration['card_row']
            arena_region_y1_vp = 0
            arena_region_y2_vp = int(card_row.get('top_r', 0.85) * vh_actual)
        else:
            arena_region_y1_vp = 0
            arena_region_y2_vp = int(0.85 * vh_actual)
        arena_region_x1_vp = 0
        arena_region_x2_vp = vw_actual
        
        arena_region = viewport_region[arena_region_y1_vp:arena_region_y2_vp, arena_region_x1_vp:arena_region_x2_vp]
        
        if arena_region.size == 0:
            return arena_troops
        
        # Run YOLO inference
        confidence_threshold = 0.3
        try:
            results = self.yolo_model(arena_region, conf=confidence_threshold, iou=0.45, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Convert tensors using tolist() which is more reliable
                        try:
                            xyxy = box.xyxy[0]
                            if hasattr(xyxy, 'cpu'):
                                xyxy = xyxy.cpu()
                            if hasattr(xyxy, 'tolist'):
                                xyxy_list = xyxy.tolist()
                            elif hasattr(xyxy, 'numpy'):
                                xyxy_list = xyxy.numpy().tolist()
                            else:
                                xyxy_list = list(xyxy)
                            x1, y1, x2, y2 = xyxy_list
                            
                            conf = box.conf[0]
                            if hasattr(conf, 'cpu'):
                                conf = conf.cpu()
                            if hasattr(conf, 'item'):
                                confidence = float(conf.item())
                            elif hasattr(conf, 'tolist'):
                                confidence = float(conf.tolist())
                            else:
                                confidence = float(conf)
                            
                            cls = box.cls[0]
                            if hasattr(cls, 'cpu'):
                                cls = cls.cpu()
                            if hasattr(cls, 'item'):
                                class_id = int(cls.item())
                            elif hasattr(cls, 'tolist'):
                                class_id = int(cls.tolist())
                            else:
                                class_id = int(cls)
                        except Exception as e:
                            print(f"Error converting tensor: {e}")
                            continue
                        
                        # Convert coordinates: arena_region -> viewport -> full screen
                        # YOLO coordinates are relative to arena_region
                        x1_vp = int(x1) + arena_region_x1_vp
                        y1_vp = int(y1) + arena_region_y1_vp
                        x2_vp = int(x2) + arena_region_x1_vp
                        y2_vp = int(y2) + arena_region_y1_vp
                        
                        # Convert viewport coordinates to full screen coordinates
                        abs_x1 = x1_vp + vx
                        abs_y1 = y1_vp + vy
                        abs_x2 = x2_vp + vx
                        abs_y2 = y2_vp + vy
                        
                        card_name = self.card_classes.get(class_id, f"Unknown_{class_id}")
                        
                        arena_troops.append({
                            'name': card_name,
                            'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                            'confidence': confidence,
                            'position': ((abs_x1 + abs_x2) // 2, (abs_y1 + abs_y2) // 2)
                        })
            
        except Exception as e:
            print(f"Error in YOLO arena detection: {e}")
            return []
        
        return arena_troops
    
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
