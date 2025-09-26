"""
Main Clash Royale Bot Implementation

This is the main entry point that integrates all components:
- Computer Vision
- Strategy Engine  
- Action Automation
- Machine Learning
- Match Recording
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core import ClashRoyaleBot, GameState
from src.vision.game_vision import ClashRoyaleVision
from src.automation.game_automation import ClashRoyaleAutomation
from src.strategy.basic_strategy import BasicStrategyEngine
from src.learning.ml_system import ClashRoyaleLearning
from src.recording.match_recorder import ClashRoyaleRecorder
from config import BOT_CONFIG

class ClashRoyaleBotMain(ClashRoyaleBot):
    """Main bot implementation with all components integrated"""
    
    def __init__(self):
        super().__init__()
        self.match_count = 0
        self.victories = 0
        self.defeats = 0
        
    def initialize_components(self):
        """Initialize all bot components"""
        print("Initializing Clash Royale Bot components...")
        
        try:
            # Initialize computer vision
            self.vision_system = ClashRoyaleVision()
            print("✓ Computer Vision System initialized")
            
            # Initialize automation
            self.action_executor = ClashRoyaleAutomation()
            print("✓ Action Executor initialized")
            
            # Initialize strategy engine
            self.strategy_engine = BasicStrategyEngine()
            print("✓ Strategy Engine initialized")
            
            # Initialize learning system
            self.learning_system = ClashRoyaleLearning()
            print("✓ Learning System initialized")
            
            # Initialize match recorder
            self.match_recorder = ClashRoyaleRecorder()
            print("✓ Match Recorder initialized")
            
            print("All components initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            return False
    
    def run(self):
        """Main bot execution loop"""
        if not self.initialize_components():
            print("Failed to initialize components. Exiting.")
            return
        
        print("Starting Clash Royale Bot...")
        print("Press Ctrl+C to stop the bot")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Capture screen and detect state
                screen = self.vision_system.capture_screen()
                
                if screen is None:
                    print("Failed to capture screen. Retrying...")
                    time.sleep(1)
                    continue
                
                game_state = self.vision_system.detect_game_state(screen)
                
                if game_state == GameState.IN_GAME:
                    if not self.current_match_id:
                        # Start new match recording
                        self.match_count += 1
                        self.current_match_id = f"match_{self.match_count}_{int(time.time())}"
                        self.match_recorder.start_recording(self.current_match_id)
                        print(f"Started match: {self.current_match_id}")
                    
                    # Extract game information
                    game_info = self.vision_system.extract_game_info(screen)
                    
                    if game_info:
                        # Decide action using strategy engine
                        action = self.strategy_engine.decide_action(game_info)
                        
                        if action:
                            # Execute action
                            success = self.action_executor.place_card(
                                action['card_index'], 
                                action['position']
                            )
                            
                            if success:
                                # Record action
                                self.match_recorder.record_action(action, game_info)
                                
                                # Add human-like delay
                                self.action_executor.simulate_human_delay()
                
                elif game_state == GameState.MATCH_END:
                    # Process match end
                    self._handle_match_end()
                
                elif game_state == GameState.MATCHMAKING:
                    print("Waiting for match...")
                    time.sleep(2)
                
                else:
                    # Menu or other state
                    time.sleep(1)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def _handle_match_end(self):
        """Handle match completion"""
        if self.current_match_id:
            print(f"Match ended: {self.current_match_id}")
            
            # Get match result
            match_result = self.vision_system.get_match_result()
            
            # Update statistics
            if match_result.get("result") == "victory":
                self.victories += 1
                print("🎉 Victory!")
            else:
                self.defeats += 1
                print("😞 Defeat")
            
            # End recording
            self.match_recorder.end_recording(match_result)
            
            # Train on match data if learning is enabled
            if BOT_CONFIG["enable_learning"]:
                self.learning_system.train_on_match(match_result)
            
            # Print statistics
            self._print_statistics()
            
            self.current_match_id = None
    
    def _print_statistics(self):
        """Print current bot statistics"""
        total_matches = self.victories + self.defeats
        if total_matches > 0:
            win_rate = (self.victories / total_matches) * 100
            print(f"Statistics: {self.victories}W-{self.defeats}L ({win_rate:.1f}% win rate)")
    
    def stop(self):
        """Stop the bot"""
        self.is_running = False
        
        # Save learning model if training was enabled
        if BOT_CONFIG["enable_learning"] and self.learning_system:
            model_path = Path("models") / "clash_royale_bot.pth"
            model_path.parent.mkdir(exist_ok=True)
            self.learning_system.save_model(str(model_path))
            print(f"Saved trained model to {model_path}")
        
        print("Bot stopped.")

def main():
    """Main entry point"""
    print("=" * 50)
    print("Clash Royale Bot")
    print("=" * 50)
    
    bot = ClashRoyaleBotMain()
    bot.run()

if __name__ == "__main__":
    main()
