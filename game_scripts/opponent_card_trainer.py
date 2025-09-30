import argparse
import time
import cv2
import numpy as np
import sys
import os
import json
from typing import Optional, Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport
from tools.card_recognition_system import CardRecognitionSystem


class OpponentCardTrainer:
    def __init__(self, calibration_path: str):
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        self.crs = CardRecognitionSystem(calibration_path, 'database/clash_royale_cards.json')
        
        # Opponent region from calibration file
        vx, vy, vw, vh = self.viewport
        
        if 'opponent_region_roi' in self.calib:
            # Use calibrated opponent region
            roi = self.calib['opponent_region_roi']
            self.opponent_region = {
                'x': int(vx + roi['x_r'] * vw),
                'y': int(vy + roi['y_r'] * vh),
                'w': int(roi['w_r'] * vw),
                'h': int(roi['h_r'] * vh)
            }
        else:
            # Fallback to top half of viewport
            self.opponent_region = {
                'x': vx,
                'y': vy,
                'w': vw,
                'h': vh // 2
            }
        
        # Training data storage
        self.training_data = []
        self.change_threshold = 0.1
        
    def collect_training_data(self, duration: int):
        """Collect training data by detecting opponent plays and asking user to identify them."""
        print("🎯 Opponent Card Training System")
        print("Instructions:")
        print("1. Play Clash Royale normally")
        print("2. When opponent plays a card, I'll detect it")
        print("3. I'll show you the card image")
        print("4. Tell me what card it is")
        print("5. I'll save this as training data")
        print()
        print("Commands:")
        print("- Type the card name (e.g., 'giant', 'fireball')")
        print("- Type 'skip' to skip this detection")
        print("- Type 'quit' to stop training")
        print()
        
        start_time = time.time()
        prev_opponent_region = None
        
        while time.time() - start_time < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Extract opponent region
            opp_region = frame[
                self.opponent_region['y']:self.opponent_region['y'] + self.opponent_region['h'],
                self.opponent_region['x']:self.opponent_region['x'] + self.opponent_region['w']
            ]
            
            if prev_opponent_region is None:
                prev_opponent_region = opp_region.copy()
                continue
            
            # Detect changes
            diff = cv2.absdiff(opp_region, prev_opponent_region)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            changed_pixels = np.count_nonzero(gray_diff > 30)
            total_pixels = gray_diff.size
            change_ratio = changed_pixels / total_pixels
            
            if change_ratio > self.change_threshold:
                print(f"\n🚨 OPPONENT PLAYED A CARD! Change ratio: {change_ratio:.3f}")
                
                # Save the card image
                timestamp = int(time.time())
                card_image_path = f'opponent_card_{timestamp}.png'
                cv2.imwrite(card_image_path, opp_region)
                
                print(f"📸 Saved card image: {card_image_path}")
                print("👀 Look at the image and tell me what card this is:")
                
                # Get user input
                user_input = input("Card name (or 'skip' or 'quit'): ").strip().lower()
                
                if user_input == 'quit':
                    print("Training stopped by user.")
                    break
                elif user_input == 'skip':
                    print("Skipped this detection.")
                elif user_input:
                    # Save training data
                    training_entry = {
                        'timestamp': timestamp,
                        'card_name': user_input,
                        'image_path': card_image_path,
                        'change_ratio': change_ratio,
                        'opponent_region': self.opponent_region
                    }
                    self.training_data.append(training_entry)
                    print(f"✅ Saved training data: {user_input}")
                
                print("-" * 50)
            
            prev_opponent_region = opp_region.copy()
            time.sleep(0.1)
        
        # Save all training data
        self.save_training_data()
        print(f"\n🎉 Training complete! Collected {len(self.training_data)} examples.")
    
    def save_training_data(self):
        """Save collected training data to file."""
        training_file = 'opponent_training_data.json'
        with open(training_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"💾 Training data saved to: {training_file}")
    
    def create_opponent_templates(self):
        """Create templates from training data."""
        if not self.training_data:
            print("❌ No training data available. Run training first.")
            return
        
        print("🔧 Creating opponent card templates...")
        
        # Group by card name
        card_groups = {}
        for entry in self.training_data:
            card_name = entry['card_name']
            if card_name not in card_groups:
                card_groups[card_name] = []
            card_groups[card_name].append(entry)
        
        # Create templates directory
        os.makedirs('templates/opponent_cards', exist_ok=True)
        
        for card_name, examples in card_groups.items():
            print(f"📝 Processing {card_name} ({len(examples)} examples)...")
            
            # Load all images for this card
            images = []
            for example in examples:
                if os.path.exists(example['image_path']):
                    img = cv2.imread(example['image_path'])
                    if img is not None:
                        images.append(img)
            
            if images:
                # Create a composite template (average of all examples)
                # For now, just use the first image as template
                template = images[0]
                
                # Save template
                template_path = f'templates/opponent_cards/{card_name}.png'
                cv2.imwrite(template_path, template)
                print(f"✅ Created template: {template_path}")
        
        print("🎯 Opponent templates created!")
    
    def test_opponent_recognition(self, duration: int):
        """Test opponent card recognition using created templates."""
        print("🧪 Testing opponent card recognition...")
        
        # Load opponent templates
        opponent_templates = {}
        template_dir = 'templates/opponent_cards'
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.png'):
                    card_name = filename[:-4]
                    template_path = os.path.join(template_dir, filename)
                    template = cv2.imread(template_path)
                    if template is not None:
                        opponent_templates[card_name] = template
                        print(f"📚 Loaded opponent template: {card_name}")
        
        if not opponent_templates:
            print("❌ No opponent templates found. Create templates first.")
            return
        
        print(f"🎯 Testing with {len(opponent_templates)} templates...")
        
        start_time = time.time()
        prev_opponent_region = None
        detections = []
        
        while time.time() - start_time < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Extract opponent region
            opp_region = frame[
                self.opponent_region['y']:self.opponent_region['y'] + self.opponent_region['h'],
                self.opponent_region['x']:self.opponent_region['x'] + self.opponent_region['w']
            ]
            
            if prev_opponent_region is None:
                prev_opponent_region = opp_region.copy()
                continue
            
            # Detect changes
            diff = cv2.absdiff(opp_region, prev_opponent_region)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            changed_pixels = np.count_nonzero(gray_diff > 30)
            total_pixels = gray_diff.size
            change_ratio = changed_pixels / total_pixels
            
            if change_ratio > self.change_threshold:
                print(f"\n🚨 OPPONENT PLAYED A CARD! Change ratio: {change_ratio:.3f}")
                
                # Try to identify using opponent templates
                best_match = None
                best_score = 0.0
                
                for card_name, template in opponent_templates.items():
                    # Resize template to match opponent region
                    h, w = opp_region.shape[:2]
                    template_resized = cv2.resize(template, (w//4, h//4))
                    
                    # Template matching
                    result = cv2.matchTemplate(opp_region, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > best_score and max_val > 0.3:
                        best_score = max_val
                        best_match = card_name
                
                if best_match:
                    print(f"🎯 IDENTIFIED: {best_match} (confidence: {best_score:.2f})")
                    detections.append({
                        'card': best_match,
                        'confidence': best_score,
                        'change_ratio': change_ratio
                    })
                else:
                    print("❓ Could not identify card")
                
                print("-" * 30)
            
            prev_opponent_region = opp_region.copy()
            time.sleep(0.1)
        
        print(f"\n📊 Test Results:")
        print(f"Total detections: {len(detections)}")
        for i, detection in enumerate(detections, 1):
            print(f"{i}. {detection['card']} (conf: {detection['confidence']:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Train opponent card recognition system.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--mode', choices=['train', 'create_templates', 'test'], default='train',
                       help='Mode: train (collect data), create_templates, or test')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    args = parser.parse_args()
    
    trainer = OpponentCardTrainer(args.calib)
    
    if args.mode == 'train':
        trainer.collect_training_data(args.duration)
    elif args.mode == 'create_templates':
        trainer.create_opponent_templates()
    elif args.mode == 'test':
        trainer.test_opponent_recognition(args.duration)


if __name__ == '__main__':
    main()
