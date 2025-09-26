"""
Clash Royale Bot - Core System Architecture

This module defines the main components and their interactions:
1. Computer Vision System - Detects game state
2. Strategy Engine - Makes tactical decisions  
3. Action Executor - Performs card placements
4. Learning System - Improves from experience
5. Match Recorder - Captures game data
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class GameState(Enum):
    MENU = "menu"
    MATCHMAKING = "matchmaking" 
    IN_GAME = "in_game"
    MATCH_END = "match_end"

@dataclass
class Card:
    """Represents a Clash Royale card"""
    name: str
    cost: int
    position: Tuple[int, int]
    is_available: bool
    cooldown_remaining: float

@dataclass
class GameInfo:
    """Current state of the game"""
    state: GameState
    player_elixir: float
    opponent_elixir: float
    player_towers: Dict[str, int]  # tower_name -> health
    opponent_towers: Dict[str, int]
    player_cards: List[Card]
    opponent_cards: List[Card]
    time_remaining: int
    arena_troops: List[Dict]  # troops on arena

class ComputerVisionSystem(ABC):
    """Abstract base for game state detection"""
    
    @abstractmethod
    def capture_screen(self) -> np.ndarray:
        """Capture current game screen"""
        pass
    
    @abstractmethod
    def detect_game_state(self, screen: np.ndarray) -> GameState:
        """Detect current game state from screen"""
        pass
    
    @abstractmethod
    def extract_game_info(self, screen: np.ndarray) -> GameInfo:
        """Extract detailed game information"""
        pass

class StrategyEngine(ABC):
    """Abstract base for tactical decision making"""
    
    @abstractmethod
    def decide_action(self, game_info: GameInfo) -> Optional[Dict]:
        """Decide next action based on game state"""
        pass
    
    @abstractmethod
    def update_strategy(self, match_data: Dict):
        """Update strategy based on match results"""
        pass

class ActionExecutor(ABC):
    """Abstract base for performing game actions"""
    
    @abstractmethod
    def place_card(self, card_index: int, position: Tuple[int, int]) -> bool:
        """Place a card at specified position"""
        pass
    
    @abstractmethod
    def click_position(self, position: Tuple[int, int]) -> bool:
        """Click at specified screen position"""
        pass

class LearningSystem(ABC):
    """Abstract base for machine learning and improvement"""
    
    @abstractmethod
    def train_on_match(self, match_data: Dict):
        """Train model on completed match data"""
        pass
    
    @abstractmethod
    def predict_action(self, game_info: GameInfo) -> Dict:
        """Predict best action using trained model"""
        pass

class MatchRecorder(ABC):
    """Abstract base for recording match data"""
    
    @abstractmethod
    def start_recording(self, match_id: str):
        """Start recording a new match"""
        pass
    
    @abstractmethod
    def record_action(self, action: Dict, game_info: GameInfo):
        """Record an action taken"""
        pass
    
    @abstractmethod
    def end_recording(self, match_result: Dict):
        """End recording and save match data"""
        pass

class ClashRoyaleBot:
    """Main bot orchestrator"""
    
    def __init__(self):
        self.vision_system: Optional[ComputerVisionSystem] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.action_executor: Optional[ActionExecutor] = None
        self.learning_system: Optional[LearningSystem] = None
        self.match_recorder: Optional[MatchRecorder] = None
        self.is_running = False
        self.current_match_id = None
    
    def initialize_components(self):
        """Initialize all bot components"""
        # This will be implemented in specific modules
        pass
    
    def run(self):
        """Main bot execution loop"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Capture screen and detect state
                screen = self.vision_system.capture_screen()
                game_state = self.vision_system.detect_game_state(screen)
                
                if game_state == GameState.IN_GAME:
                    # Extract game information
                    game_info = self.vision_system.extract_game_info(screen)
                    
                    # Decide action
                    action = self.strategy_engine.decide_action(game_info)
                    
                    if action:
                        # Execute action
                        success = self.action_executor.place_card(
                            action['card_index'], 
                            action['position']
                        )
                        
                        # Record action
                        self.match_recorder.record_action(action, game_info)
                
                elif game_state == GameState.MATCH_END:
                    # Process match end
                    self._handle_match_end()
                
            except Exception as e:
                print(f"Error in bot loop: {e}")
                break
    
    def _handle_match_end(self):
        """Handle match completion"""
        if self.current_match_id:
            match_result = self.vision_system.get_match_result()
            self.match_recorder.end_recording(match_result)
            self.learning_system.train_on_match(match_result)
            self.current_match_id = None
    
    def stop(self):
        """Stop the bot"""
        self.is_running = False
