"""
Strategy Engine for Clash Royale Bot

This module handles:
- Basic gameplay strategies
- Card placement decisions
- Elixir management
- Defensive/offensive tactics
"""

from typing import Dict, List, Optional, Tuple
import random
from core import StrategyEngine, GameInfo, Card
from config import GAME_CONFIG

class BasicStrategyEngine(StrategyEngine):
    """Basic rule-based strategy engine"""
    
    def __init__(self):
        self.last_action_time = 0
        self.action_cooldown = 1.0  # Minimum time between actions
        
        # Strategy parameters
        self.aggressive_threshold = 8.0  # Elixir threshold for aggressive play
        self.defensive_threshold = 3.0   # Elixir threshold for defensive play
        
    def decide_action(self, game_info: GameInfo) -> Optional[Dict]:
        """Decide next action based on game state"""
        import time
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        # Analyze game state
        action = self._analyze_and_decide(game_info)
        
        if action:
            self.last_action_time = current_time
        
        return action
    
    def _analyze_and_decide(self, game_info: GameInfo) -> Optional[Dict]:
        """Analyze game state and decide action"""
        
        # Check if we have enough elixir
        available_cards = [card for card in game_info.player_cards if card.is_available]
        if not available_cards:
            return None
        
        # Choose cheapest available card
        cheapest_card = min(available_cards, key=lambda c: c.cost)
        
        if game_info.player_elixir < cheapest_card.cost:
            return None
        
        # Decide placement strategy
        if game_info.player_elixir >= self.aggressive_threshold:
            return self._aggressive_play(game_info, cheapest_card)
        elif game_info.player_elixir <= self.defensive_threshold:
            return self._defensive_play(game_info, cheapest_card)
        else:
            return self._balanced_play(game_info, cheapest_card)
    
    def _aggressive_play(self, game_info: GameInfo, card: Card) -> Optional[Dict]:
        """Aggressive play strategy"""
        # Place cards closer to opponent's towers
        target_position = self._get_aggressive_position()
        
        return {
            "action_type": "place_card",
            "card_index": self._get_card_index(card),
            "position": target_position,
            "strategy": "aggressive"
        }
    
    def _defensive_play(self, game_info: GameInfo, card: Card) -> Optional[Dict]:
        """Defensive play strategy"""
        # Place cards closer to our towers
        target_position = self._get_defensive_position()
        
        return {
            "action_type": "place_card", 
            "card_index": self._get_card_index(card),
            "position": target_position,
            "strategy": "defensive"
        }
    
    def _balanced_play(self, game_info: GameInfo, card: Card) -> Optional[Dict]:
        """Balanced play strategy"""
        # Place cards in middle area
        target_position = self._get_balanced_position()
        
        return {
            "action_type": "place_card",
            "card_index": self._get_card_index(card), 
            "position": target_position,
            "strategy": "balanced"
        }
    
    def _get_aggressive_position(self) -> Tuple[int, int]:
        """Get position for aggressive play"""
        # Place closer to opponent's side
        return (960, 400)  # Center-right area
    
    def _get_defensive_position(self) -> Tuple[int, int]:
        """Get position for defensive play"""
        # Place closer to our towers
        return (960, 600)  # Center-left area
    
    def _get_balanced_position(self) -> Tuple[int, int]:
        """Get position for balanced play"""
        # Place in center
        return (960, 500)  # Center area
    
    def _get_card_index(self, card: Card) -> int:
        """Get card index from card object"""
        for i, pos in enumerate(GAME_CONFIG["card_slots"]["positions"]):
            if card.position == pos:
                return i
        return 0
    
    def update_strategy(self, match_data: Dict):
        """Update strategy based on match results"""
        # Analyze match results and adjust strategy parameters
        if match_data.get("result") == "victory":
            # Successful strategies - could increase aggressiveness
            pass
        else:
            # Failed strategies - could adjust parameters
            pass
