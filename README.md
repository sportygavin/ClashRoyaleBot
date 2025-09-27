# Clash Royale Bot 

A fully automated Clash Royale bot that learns game mechanics and strategies to compete against real players. The bot uses computer vision, machine learning, and strategic decision-making to play matches and improve over time.

## Features

- **Computer Vision**: Real-time game state detection using OpenCV
- **Strategic AI**: Rule-based and ML-powered decision making
- **Machine Learning**: Deep Q-Network (DQN) for strategy optimization
- **Match Recording**: Comprehensive data collection for analysis
- **Human-like Behavior**: Natural delays and movement patterns
- **Learning System**: Improves performance through experience

## Architecture

The bot consists of five main components:

1. **Computer Vision System** (`src/vision/`) - Detects game state, cards, elixir, troops
2. **Strategy Engine** (`src/strategy/`) - Makes tactical decisions based on game state
3. **Action Executor** (`src/automation/`) - Performs card placements and interactions
4. **Learning System** (`src/learning/`) - ML model training and optimization
5. **Match Recorder** (`src/recording/`) - Captures and analyzes match data

## Quick Start

### Prerequisites

- Python 3.8+
- Android Emulator (MEmu recommended)
- Clash Royale installed on emulator

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ClashRoyaleBot
   ```

2. **Run setup script**
   ```bash
   python setup.py
   ```

3. **Configure the bot**
   - Update `config.py` with your emulator settings
   - Ensure emulator window title matches configuration

4. **Start the bot**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
ClashRoyaleBot/
├── main.py                 # Main bot entry point
├── core.py                 # Core architecture definitions
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── setup.py               # Setup script
├── src/
│   ├── vision/            # Computer vision system
│   ├── strategy/          # Strategy engines
│   ├── automation/        # Input automation
│   ├── learning/          # Machine learning
│   └── recording/         # Match data recording
├── data/
│   ├── matches/           # Recorded match data
│   ├── models/           # Trained ML models
│   └── screenshots/      # Game screenshots
├── logs/                 # Bot logs
└── tests/               # Test files
```

## How It Works

### Game State Detection
- Captures screen using PyAutoGUI
- Uses OpenCV for image processing
- Detects cards, elixir, towers, and troops
- Recognizes different game states (menu, matchmaking, in-game, match end)

### Strategic Decision Making
- **Basic Strategy**: Rule-based decisions based on elixir and game state
- **ML Strategy**: Deep Q-Network learns optimal actions from experience
- Considers card costs, elixir management, and positioning

### Action Execution
- Simulates human-like touch inputs
- Drags cards from hand to battlefield
- Includes natural delays and movement patterns
- Prevents detection through realistic behavior

### Learning and Improvement
- Records all actions and game states
- Uses reinforcement learning (DQN) for optimization
- Learns from victories and defeats
- Continuously improves strategy over time

## Configuration

Edit `config.py` to customize:

- **Game Area**: Screen region for game detection
- **Card Positions**: Locations of cards in hand
- **ML Parameters**: Learning rate, batch size, etc.
- **Bot Behavior**: Reaction times, learning settings

## Monitoring and Analysis

The bot automatically records:
- All actions taken during matches
- Game state at each decision point
- Match outcomes and statistics
- Learning progress and model performance

View statistics in the console or analyze saved match data in `data/matches/`.

## Safety and Ethics

- **Bot Detection**: Uses human-like delays and patterns
- **Terms of Service**: Review Clash Royale's ToS before use
- **Account Safety**: Use on test accounts, not main accounts
- **Fair Play**: Designed for learning and research purposes

## Advanced Usage

### Custom Strategies
Implement custom strategy engines by extending the `StrategyEngine` class:

```python
class CustomStrategy(StrategyEngine):
    def decide_action(self, game_info: GameInfo) -> Optional[Dict]:
        # Your custom logic here
        pass
```

### Model Training
The bot automatically trains on match data. To manually train:

```python
from src.learning.ml_system import ClashRoyaleLearning

learning_system = ClashRoyaleLearning()
# Load match data and train
learning_system.train_on_match(match_data)
```

### Performance Tuning
- Adjust reaction times in `BOT_CONFIG`
- Modify ML parameters in `ML_CONFIG`
- Fine-tune vision detection thresholds

## Troubleshooting

**Bot not detecting game:**
- Check emulator window title matches config
- Verify screen resolution settings
- Ensure Clash Royale is visible and not minimized

**Poor performance:**
- Reduce reaction time for faster play
- Adjust ML learning rate
- Check computer vision detection accuracy

**Detection concerns:**
- Increase human-like delays
- Add more randomization to actions
- Use different emulator settings

## Future Enhancements

- [ ] Advanced card recognition using deep learning
- [ ] Opponent strategy analysis and counter-play
- [ ] Deck-specific strategy optimization
- [ ] Real-time strategy adaptation
- [ ] Multi-account management
- [ ] Tournament mode support
