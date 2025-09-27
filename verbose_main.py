"""
Verbose Clash Royale Bot for debugging

This version shows detailed output of what the bot is doing.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import ClashRoyaleBot, GameState
from src.vision.game_vision import ClashRoyaleVision
from src.automation.game_automation import ClashRoyaleAutomation
from src.strategy.basic_strategy import BasicStrategyEngine
from src.learning.ml_system import ClashRoyaleLearning
from src.recording.match_recorder import ClashRoyaleRecorder
from config import BOT_CONFIG

class VerboseClashRoyaleBot(ClashRoyaleBot):
    """Verbose version of the bot with detailed logging"""
    
    def __init__(self):
        super().__init__()
        self.match_count = 0
        self.victories = 0
        self.defeats = 0
        self.action_count = 0
        
    def initialize_components(self):
        """Initialize all bot components"""
        print("🔧 Initializing Clash Royale Bot components...")
        
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
            
            print("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return False
    
    def run(self):
        """Main bot execution loop with verbose output"""
        if not self.initialize_components():
            print("❌ Failed to initialize components. Exiting.")
            return
        
        print("🚀 Starting Clash Royale Bot...")
        print("Press Ctrl+C to stop the bot")
        print("="*60)
        
        self.is_running = True
        loop_count = 0
        
        try:
            while self.is_running:
                loop_count += 1
                
                # Show status every 10 loops
                if loop_count % 10 == 0:
                    print(f"🔄 Bot running... (Loop {loop_count}, Actions: {self.action_count})")
                
                # Capture screen and detect state
                try:
                    screen = self.vision_system.capture_screen()
                    
                    if screen is None:
                        print("⚠️  Failed to capture screen. Retrying...")
                        time.sleep(1)
                        continue
                    
                    # Detect game state
                    game_state = self.vision_system.detect_game_state(screen)
                    
                    if loop_count % 20 == 0:  # Show state every 20 loops
                        print(f"🎮 Game state: {game_state}")
                    
                    if game_state == GameState.IN_GAME:
                        if not self.current_match_id:
                            # Start new match recording
                            self.match_count += 1
                            self.current_match_id = f"match_{self.match_count}_{int(time.time())}"
                            self.match_recorder.start_recording(self.current_match_id)
                            print(f"🎯 Started match: {self.current_match_id}")
                        
                        # Extract game information
                        try:
                            game_info = self.vision_system.extract_game_info(screen)
                            
                            if game_info:
                                print(f"📊 Game info - Elixir: {game_info.player_elixir:.1f}, Cards: {len(game_info.player_cards)}")
                                
                                # Show available cards
                                available_cards = [card for card in game_info.player_cards if card.is_available]
                                if available_cards:
                                    print(f"🃏 Available cards: {len(available_cards)}")
                                    for card in available_cards:
                                        print(f"   - {card.name} (Cost: {card.cost})")
                                
                                # Decide action
                                action = self.strategy_engine.decide_action(game_info)
                                
                                if action:
                                    print(f"🎯 Action decided: {action['action_type']} - Card {action['card_index']} at {action['position']}")
                                    
                                    # Execute action
                                    success = self.action_executor.place_card(
                                        action['card_index'], 
                                        action['position']
                                    )
                                    
                                    if success:
                                        self.action_count += 1
                                        print(f"✅ Action executed successfully! (Total actions: {self.action_count})")
                                        
                                        # Record action
                                        self.match_recorder.record_action(action, game_info)
                                        
                                        # Add human-like delay
                                        self.action_executor.simulate_human_delay()
                                    else:
                                        print("❌ Action execution failed")
                                else:
                                    if loop_count % 30 == 0:  # Show less frequently
                                        print("⏳ No action decided (waiting for better opportunity)")
                            
                        except Exception as e:
                            print(f"⚠️  Error processing game info: {e}")
                    
                    elif game_state == GameState.MATCH_END:
                        print("🏁 Match ended!")
                        self._handle_match_end()
                    
                    elif game_state == GameState.MATCHMAKING:
                        if loop_count % 50 == 0:  # Show less frequently
                            print("⏳ Waiting for match...")
                    
                    else:
                        if loop_count % 50 == 0:  # Show less frequently
                            print(f"📱 In {game_state.value} state...")
                    
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    time.sleep(1)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Bot stopped by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.stop()
    
    def _handle_match_end(self):
        """Handle match completion"""
        if self.current_match_id:
            print(f"🏁 Processing match end: {self.current_match_id}")
            
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
                print("🧠 Training on match data...")
                self.learning_system.train_on_match(match_result)
            
            # Print statistics
            self._print_statistics()
            
            self.current_match_id = None
    
    def _print_statistics(self):
        """Print current bot statistics"""
        total_matches = self.victories + self.defeats
        if total_matches > 0:
            win_rate = (self.victories / total_matches) * 100
            print(f"📊 Statistics: {self.victories}W-{self.defeats}L ({win_rate:.1f}% win rate)")
            print(f"🎯 Total actions: {self.action_count}")
    
    def stop(self):
        """Stop the bot"""
        self.is_running = False
        
        # Save learning model if training was enabled
        if BOT_CONFIG["enable_learning"] and self.learning_system:
            model_path = Path("models") / "clash_royale_bot.pth"
            model_path.parent.mkdir(exist_ok=True)
            self.learning_system.save_model(str(model_path))
            print(f"💾 Saved trained model to {model_path}")
        
        print("🛑 Bot stopped.")

def main():
    """Main entry point"""
    print("🤖 Verbose Clash Royale Bot")
    print("="*50)
    
    bot = VerboseClashRoyaleBot()
    bot.run()

if __name__ == "__main__":
    main()
