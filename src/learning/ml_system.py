"""
Machine Learning System for Clash Royale Bot

This module handles:
- Reinforcement learning for strategy optimization
- Neural network training
- Experience replay
- Model saving/loading
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple
from core import LearningSystem, GameInfo
from config import ML_CONFIG

class DQNNetwork(nn.Module):
    """Deep Q-Network for Clash Royale"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ExperienceReplay:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ClashRoyaleLearning(LearningSystem):
    """Machine learning system for strategy optimization"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network parameters
        self.input_size = 50  # Game state features
        self.hidden_size = 128
        self.output_size = 20  # Possible actions
        
        # Initialize networks
        self.q_network = DQNNetwork(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.target_network = DQNNetwork(self.input_size, self.hidden_size, self.output_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=ML_CONFIG["learning_rate"])
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = ExperienceReplay(ML_CONFIG["memory_size"])
        
        # Training parameters
        self.epsilon = ML_CONFIG["epsilon_start"]
        self.epsilon_end = ML_CONFIG["epsilon_end"]
        self.epsilon_decay = ML_CONFIG["epsilon_decay"]
        self.batch_size = ML_CONFIG["batch_size"]
        
        # Training state
        self.training_step = 0
        self.target_update_frequency = 1000
    
    def state_to_features(self, game_info: GameInfo) -> np.ndarray:
        """Convert game info to feature vector"""
        features = []
        
        # Elixir information
        features.append(game_info.player_elixir / 10.0)  # Normalized elixir
        features.append(game_info.opponent_elixir / 10.0)
        
        # Tower health (normalized)
        for tower_name in ["king_tower", "left_tower", "right_tower"]:
            features.append(game_info.player_towers.get(tower_name, 0) / 2400.0)
            features.append(game_info.opponent_towers.get(tower_name, 0) / 2400.0)
        
        # Card availability and costs
        for card in game_info.player_cards:
            features.append(1.0 if card.is_available else 0.0)
            features.append(card.cost / 10.0)  # Normalized cost
        
        # Time remaining
        features.append(game_info.time_remaining / 180.0)  # Normalized time
        
        # Arena troops (simplified)
        features.extend([0.0] * 20)  # Placeholder for troop positions
        
        # Pad or truncate to fixed size
        while len(features) < self.input_size:
            features.append(0.0)
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def predict_action(self, game_info: GameInfo) -> Dict:
        """Predict best action using trained model"""
        state = self.state_to_features(game_info)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return self._action_to_dict(action, game_info)
    
    def _action_to_dict(self, action_idx: int, game_info: GameInfo) -> Dict:
        """Convert action index to action dictionary"""
        # Map action index to actual game action
        available_cards = [card for card in game_info.player_cards if card.is_available]
        
        if not available_cards:
            return None
        
        # Simple mapping for now
        card_index = action_idx % len(available_cards)
        position_x = 400 + (action_idx % 4) * 200
        position_y = 400 + (action_idx // 4) * 100
        
        return {
            "action_type": "place_card",
            "card_index": card_index,
            "position": (position_x, position_y),
            "strategy": "ml_prediction"
        }
    
    def train_on_match(self, match_data: Dict):
        """Train model on completed match data"""
        # Extract experiences from match data
        experiences = match_data.get("experiences", [])
        
        for experience in experiences:
            state, action, reward, next_state, done = experience
            self.memory.push(state, action, reward, next_state, done)
        
        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
