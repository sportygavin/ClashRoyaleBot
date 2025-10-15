# Cursor AI Prompt: Advanced Clash Royale Bot Development

## Project Overview
I'm developing a Clash Royale bot using Bluestacks on Mac with computer vision. The current system can:
- Identify cards in player's hand
- Place cards from hand onto the board
- Monitor user's current elixir

**Next Development Phase Goals:**
1. Identify opponent's played cards using computer vision
2. Track opponent card positions and movements
3. Train a reinforcement learning model to make strategic decisions
4. Implement opponent elixir tracking system

## Technical Implementation Roadmap

### Phase 1: Opponent Card Detection System

**Computer Vision Architecture:**
- Implement YOLOv8 object detection model for real-time opponent card identification
- Create custom dataset of opponent card plays by capturing screenshots during battles
- Use data augmentation techniques (rotation, brightness, contrast) to improve model robustness
- Set up continuous frame analysis pipeline that processes Bluestacks screen captures

**Dataset Creation Strategy:**
```python
# Automated data collection approach
def collect_opponent_card_data():
    # Capture game screen at 30 FPS
    # Detect when opponent plays a card (elixir decrease + new unit appearance)
    # Extract card region from deployment animation
    # Auto-label based on unit appearance and behavior patterns
    # Store labeled examples for training
```

**Card Detection Implementation:**
- Train YOLOv8 model on opponent card deployment regions
- Implement bounding box tracking for deployed units
- Create card-to-unit mapping system using visual and behavioral features
- Use template matching as fallback for low-confidence detections

### Phase 2: Opponent State Tracking System

**Game State Management:**
```python
class OpponentStateTracker:
    def __init__(self):
        self.opponent_hand = [None, None, None, None]  # Inferred cards
        self.opponent_elixir = 10  # Estimated elixir
        self.deployed_units = []   # Active units with positions
        self.card_cycle = []       # Known cards in rotation
        self.last_played = None    # Most recent card
    
    def update_from_detection(self, detected_card, position, timestamp):
        # Update elixir count based on card cost
        # Track unit movement and lifecycle
        # Infer hand composition from play patterns
        # Update cycle predictions
```

**Unit Tracking System:**
- Implement multi-object tracking (MOT) for deployed units
- Use Kalman filtering for position prediction
- Track unit health, movement patterns, and target selection
- Create unit behavior models for different card types

### Phase 3: Reinforcement Learning Training System

**Deep Q-Network (DQN) Architecture:**
```python
class ClashRoyaleAgent:
    def __init__(self):
        self.state_size = 50  # Game state representation
        self.action_size = 16  # 4 cards × 4 possible positions
        self.memory = deque(maxlen=10000)
        self.model = self._build_model()
        self.target_model = self._build_model()
    
    def _build_model(self):
        # Neural network with:
        # - Input: game state (elixir, cards, unit positions, opponent state)
        # - Hidden layers: process spatial and strategic information
        # - Output: Q-values for each possible action
        
    def get_game_state(self):
        return np.array([
            self.player_elixir,
            self.opponent_elixir_estimate,
            *self.encode_hand_cards(),
            *self.encode_opponent_cards(),
            *self.encode_board_state(),
            self.tower_health_ratio,
            self.time_remaining
        ])
```

**Training Pipeline:**
- Use experience replay buffer to store game transitions
- Implement reward shaping:
  - +100 for tower damage
  - +500 for tower destruction
  - -50 for own tower damage
  - +10 for positive elixir trades
  - -5 for inefficient card plays
- Apply curriculum learning: start with simple scenarios, increase complexity
- Use self-play and opponent diversity to prevent overfitting

### Phase 4: Advanced Opponent Elixir Tracking

**Elixir Prediction Model:**
```python
class ElixirTracker:
    def __init__(self):
        self.base_elixir_rate = 1.4  # Elixir per second
        self.opponent_elixir = 10
        self.last_play_time = 0
        self.card_costs = {card_name: cost for card_name, cost in CARD_COSTS.items()}
    
    def update_opponent_elixir(self, played_card, timestamp):
        # Calculate elixir gained since last play
        elixir_gained = (timestamp - self.last_play_time) * self.base_elixir_rate
        
        # Subtract card cost and add gained elixir
        self.opponent_elixir = min(10, 
            self.opponent_elixir - self.card_costs[played_card] + elixir_gained)
        
        self.last_play_time = timestamp
    
    def predict_playable_cards(self):
        # Return cards opponent can afford
        return [card for card, cost in self.card_costs.items() 
                if cost <= self.opponent_elixir]
```

## Implementation Guidelines

### Computer Vision Setup
1. **Data Collection:**
   - Record 100+ battle replays focusing on opponent card plays
   - Extract frames showing card deployment animations
   - Use Roboflow for dataset annotation and management
   - Implement automatic data augmentation pipeline

2. **Model Training:**
   - Use transfer learning with pre-trained YOLOv8 model
   - Fine-tune on Clash Royale specific imagery
   - Implement real-time inference with OpenCV
   - Optimize for 30+ FPS processing on Mac hardware

3. **Detection Pipeline:**
   - Continuously capture Bluestacks screen region
   - Apply pre-processing (resize, normalize, enhance contrast)
   - Run YOLO inference on each frame
   - Post-process detections with confidence thresholding

### Reinforcement Learning Framework
1. **Environment Setup:**
   - Create OpenAI Gym compatible environment
   - Implement step() function for action execution
   - Define observation space with game state encoding
   - Set up reward function with domain expertise

2. **Training Configuration:**
   - Use Double DQN with target network updates
   - Implement experience replay with prioritized sampling
   - Apply epsilon-greedy exploration with decay
   - Use Adam optimizer with learning rate scheduling

3. **Model Architecture:**
   - Input layer: 50-dimensional state vector
   - Hidden layers: 256 → 512 → 256 neurons with ReLU activation
   - Output layer: Q-values for each valid action
   - Add dropout layers for regularization

### Integration and Testing
1. **Modular Architecture:**
   - Separate vision, decision-making, and execution modules
   - Use async processing for real-time performance
   - Implement robust error handling and recovery
   - Add logging and performance monitoring

2. **Testing Protocol:**
   - Start with training camp battles (simpler opponents)
   - Gradually increase difficulty as agent improves
   - Track key metrics: win rate, average damage, elixir efficiency
   - Implement A/B testing for model iterations

## Expected Challenges and Solutions

**Challenge 1: Real-time Processing**
- Solution: Use GPU acceleration and model quantization
- Implement frame skipping for non-critical moments
- Optimize inference pipeline with TensorRT or CoreML

**Challenge 2: Dataset Quality**
- Solution: Active learning to identify difficult cases
- Human-in-the-loop validation for edge cases
- Continuous model retraining with new data

**Challenge 3: Opponent Adaptation**
- Solution: Multi-agent training with diverse strategies
- Online learning to adapt to new opponent patterns
- Ensemble methods for robust decision making

## Success Metrics
- **Vision System:** >90% accuracy in opponent card detection
- **State Tracking:** <±1 elixir error in opponent elixir estimation
- **AI Performance:** >70% win rate against medium-difficulty opponents
- **Real-time:** Maintain >20 FPS processing speed

## Development Timeline
- Week 1-2: Opponent card detection system
- Week 3-4: State tracking and elixir estimation
- Week 5-8: Reinforcement learning implementation
- Week 9-10: Integration, testing, and optimization

Please implement this system incrementally, testing each component thoroughly before moving to the next phase. Focus on creating a robust foundation that can be extended and improved over time.