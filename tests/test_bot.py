"""
Test Suite for Clash Royale Bot

This module contains tests for all bot components to ensure proper functionality.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import torch
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import GameState, GameInfo, Card
from src.strategy.basic_strategy import BasicStrategyEngine
from src.learning.ml_system import ClashRoyaleLearning, DQNNetwork
from src.recording.match_recorder import ClashRoyaleRecorder

class TestBasicStrategy(unittest.TestCase):
    """Test basic strategy engine"""
    
    def setUp(self):
        self.strategy = BasicStrategyEngine()
    
    def test_strategy_initialization(self):
        """Test strategy engine initializes correctly"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.aggressive_threshold, 8.0)
        self.assertEqual(self.strategy.defensive_threshold, 3.0)
    
    def test_decide_action_with_low_elixir(self):
        """Test action decision with low elixir"""
        # Create mock game info with low elixir
        game_info = GameInfo(
            state=GameState.IN_GAME,
            player_elixir=2.0,
            opponent_elixir=5.0,
            player_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            opponent_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            player_cards=[
                Card("test_card", 3, (400, 900), True, 0.0)
            ],
            opponent_cards=[],
            time_remaining=180,
            arena_troops=[]
        )
        
        action = self.strategy.decide_action(game_info)
        self.assertIsNone(action)  # Should not act with low elixir
    
    def test_decide_action_with_high_elixir(self):
        """Test action decision with high elixir"""
        game_info = GameInfo(
            state=GameState.IN_GAME,
            player_elixir=9.0,
            opponent_elixir=5.0,
            player_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            opponent_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            player_cards=[
                Card("test_card", 3, (400, 900), True, 0.0)
            ],
            opponent_cards=[],
            time_remaining=180,
            arena_troops=[]
        )
        
        action = self.strategy.decide_action(game_info)
        self.assertIsNotNone(action)
        self.assertEqual(action["strategy"], "aggressive")

class TestMLSystem(unittest.TestCase):
    """Test machine learning system"""
    
    def setUp(self):
        self.ml_system = ClashRoyaleLearning()
    
    def test_network_initialization(self):
        """Test DQN network initializes correctly"""
        network = DQNNetwork(50, 128, 20)
        self.assertIsNotNone(network)
        
        # Test forward pass
        input_tensor = np.random.randn(1, 50).astype(np.float32)
        output = network(torch.FloatTensor(input_tensor))
        self.assertEqual(output.shape, (1, 20))
    
    def test_state_to_features(self):
        """Test state conversion to features"""
        game_info = GameInfo(
            state=GameState.IN_GAME,
            player_elixir=5.0,
            opponent_elixir=7.0,
            player_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            opponent_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            player_cards=[
                Card("test_card", 3, (400, 900), True, 0.0)
            ],
            opponent_cards=[],
            time_remaining=120,
            arena_troops=[]
        )
        
        features = self.ml_system.state_to_features(game_info)
        self.assertEqual(len(features), 50)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)

class TestMatchRecorder(unittest.TestCase):
    """Test match recording system"""
    
    def setUp(self):
        self.recorder = ClashRoyaleRecorder()
    
    def test_recorder_initialization(self):
        """Test recorder initializes correctly"""
        self.assertIsNotNone(self.recorder)
        self.assertIsNone(self.recorder.current_match_id)
    
    def test_start_recording(self):
        """Test starting match recording"""
        match_id = "test_match_123"
        self.recorder.start_recording(match_id)
        
        self.assertEqual(self.recorder.current_match_id, match_id)
        self.assertIsNotNone(self.recorder.match_data["match_id"])
        self.assertIsNotNone(self.recorder.match_data["start_time"])
    
    def test_record_action(self):
        """Test recording actions"""
        match_id = "test_match_123"
        self.recorder.start_recording(match_id)
        
        game_info = GameInfo(
            state=GameState.IN_GAME,
            player_elixir=5.0,
            opponent_elixir=5.0,
            player_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            opponent_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            player_cards=[],
            opponent_cards=[],
            time_remaining=180,
            arena_troops=[]
        )
        
        action = {"action_type": "place_card", "card_index": 0, "position": (500, 500)}
        self.recorder.record_action(action, game_info)
        
        self.assertEqual(len(self.recorder.match_data["actions"]), 1)
        self.assertEqual(self.recorder.match_data["actions"][0]["action"], action)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_strategy_with_ml_prediction(self):
        """Test strategy engine with ML predictions"""
        strategy = BasicStrategyEngine()
        ml_system = ClashRoyaleLearning()
        
        game_info = GameInfo(
            state=GameState.IN_GAME,
            player_elixir=8.0,
            opponent_elixir=6.0,
            player_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            opponent_towers={"king_tower": 2400, "left_tower": 1400, "right_tower": 1400},
            player_cards=[
                Card("test_card", 3, (400, 900), True, 0.0)
            ],
            opponent_cards=[],
            time_remaining=150,
            arena_troops=[]
        )
        
        # Test basic strategy
        basic_action = strategy.decide_action(game_info)
        self.assertIsNotNone(basic_action)
        
        # Test ML prediction
        ml_action = ml_system.predict_action(game_info)
        self.assertIsNotNone(ml_action)

def run_tests():
    """Run all tests"""
    print("Running Clash Royale Bot Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBasicStrategy))
    test_suite.addTest(unittest.makeSuite(TestMLSystem))
    test_suite.addTest(unittest.makeSuite(TestMatchRecorder))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
