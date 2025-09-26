"""
Match Recording System for Clash Royale Bot

This module handles:
- Recording match data
- Storing experiences for training
- Analyzing match performance
"""

import json
import time
from typing import Dict, List
from pathlib import Path
from core import MatchRecorder, GameInfo
from config import DATA_DIR

class ClashRoyaleRecorder(MatchRecorder):
    """Records match data for analysis and training"""
    
    def __init__(self):
        self.matches_dir = DATA_DIR / "matches"
        self.matches_dir.mkdir(exist_ok=True)
        
        self.current_match_id = None
        self.match_data = {
            "match_id": None,
            "start_time": None,
            "end_time": None,
            "actions": [],
            "experiences": [],
            "result": None,
            "trophies_gained": 0,
            "match_duration": 0
        }
    
    def start_recording(self, match_id: str):
        """Start recording a new match"""
        self.current_match_id = match_id
        self.match_data = {
            "match_id": match_id,
            "start_time": time.time(),
            "end_time": None,
            "actions": [],
            "experiences": [],
            "result": None,
            "trophies_gained": 0,
            "match_duration": 0
        }
        
        print(f"Started recording match: {match_id}")
    
    def record_action(self, action: Dict, game_info: GameInfo):
        """Record an action taken"""
        if not self.current_match_id:
            return
        
        action_record = {
            "timestamp": time.time(),
            "action": action,
            "game_state": self._game_info_to_dict(game_info)
        }
        
        self.match_data["actions"].append(action_record)
    
    def record_experience(self, state, action, reward, next_state, done):
        """Record experience for training"""
        if not self.current_match_id:
            return
        
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "timestamp": time.time()
        }
        
        self.match_data["experiences"].append(experience)
    
    def end_recording(self, match_result: Dict):
        """End recording and save match data"""
        if not self.current_match_id:
            return
        
        self.match_data["end_time"] = time.time()
        self.match_data["result"] = match_result.get("result", "unknown")
        self.match_data["trophies_gained"] = match_result.get("trophies_gained", 0)
        self.match_data["match_duration"] = match_result.get("match_duration", 0)
        
        # Save match data
        filename = f"match_{self.current_match_id}_{int(time.time())}.json"
        filepath = self.matches_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.match_data, f, indent=2)
        
        print(f"Saved match data: {filename}")
        
        # Reset for next match
        self.current_match_id = None
        self.match_data = {}
    
    def _game_info_to_dict(self, game_info: GameInfo) -> Dict:
        """Convert GameInfo to dictionary for storage"""
        return {
            "state": game_info.state.value,
            "player_elixir": game_info.player_elixir,
            "opponent_elixir": game_info.opponent_elixir,
            "player_towers": game_info.player_towers,
            "opponent_towers": game_info.opponent_towers,
            "player_cards": [
                {
                    "name": card.name,
                    "cost": card.cost,
                    "position": card.position,
                    "is_available": card.is_available,
                    "cooldown_remaining": card.cooldown_remaining
                }
                for card in game_info.player_cards
            ],
            "time_remaining": game_info.time_remaining,
            "arena_troops_count": len(game_info.arena_troops)
        }
    
    def get_match_statistics(self, match_id: str) -> Dict:
        """Get statistics for a specific match"""
        # Find match file
        match_files = list(self.matches_dir.glob(f"match_{match_id}_*.json"))
        
        if not match_files:
            return {}
        
        with open(match_files[0], 'r') as f:
            match_data = json.load(f)
        
        # Calculate statistics
        total_actions = len(match_data["actions"])
        total_experiences = len(match_data["experiences"])
        match_duration = match_data.get("match_duration", 0)
        
        # Action frequency
        action_types = {}
        for action_record in match_data["actions"]:
            action_type = action_record["action"].get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        return {
            "match_id": match_id,
            "total_actions": total_actions,
            "total_experiences": total_experiences,
            "match_duration": match_duration,
            "result": match_data.get("result", "unknown"),
            "trophies_gained": match_data.get("trophies_gained", 0),
            "action_types": action_types
        }
    
    def get_all_matches(self) -> List[Dict]:
        """Get list of all recorded matches"""
        matches = []
        
        for match_file in self.matches_dir.glob("match_*.json"):
            try:
                with open(match_file, 'r') as f:
                    match_data = json.load(f)
                
                matches.append({
                    "match_id": match_data.get("match_id"),
                    "start_time": match_data.get("start_time"),
                    "result": match_data.get("result"),
                    "trophies_gained": match_data.get("trophies_gained", 0),
                    "match_duration": match_data.get("match_duration", 0)
                })
            except Exception as e:
                print(f"Error reading match file {match_file}: {e}")
        
        return sorted(matches, key=lambda x: x.get("start_time", 0), reverse=True)
