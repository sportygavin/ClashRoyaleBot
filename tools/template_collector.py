#!/usr/bin/env python3
"""
Template collector for Clash Royale card recognition.
"""

import cv2
import numpy as np
import json
import pyautogui
import time
from pathlib import Path
from datetime import datetime

class TemplateCollector:
    def __init__(self, calibration_file="cv_out/calibration_manual_fixed.json"):
        self.calib = self._load_calibration(calibration_file)
        self.templates_dir = Path("templates/cards")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Template Collector Ready!")
        print(f"✅ Templates directory: {self.templates_dir}")
    
    def _load_calibration(self, calibration_file):
        """Load calibration data."""
        with open(calibration_file) as f:
            return json.load(f)
    
    def extract_cards_from_screen(self, screenshot=None):
        """Extract cards from screen using calibration data."""
        if screenshot is None:
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Calculate viewport coordinates
        viewport = self.calib['viewport']
        vp_x = int(viewport['x_r'] * screenshot.shape[1])
        vp_y = int(viewport['y_r'] * screenshot.shape[0])
        vp_w = int(viewport['w_r'] * screenshot.shape[1])
        vp_h = int(viewport['h_r'] * screenshot.shape[0])
        
        # Extract viewport
        roi = screenshot[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
        
        # Calculate card row
        card_row = self.calib['card_row']
        row_top_y = int(card_row['top_r'] * vp_h)
        row_bottom_y = int(card_row['bottom_r'] * vp_h)
        
        # Extract individual cards
        cards = {}
        cards_config = self.calib['cards']
        
        for i in range(4):
            center_x_r = cards_config['centers_x_r'][i]
            center_x = int(center_x_r * vp_w)
            
            card_w = int(cards_config['width_r'] * vp_w)
            card_h = row_bottom_y - row_top_y
            card_top_offset = int(cards_config['top_offset_r'] * vp_h)
            card_bottom_offset = int(cards_config['bottom_offset_r'] * vp_h)
            
            # Calculate card boundaries
            card_x1 = center_x - card_w // 2
            card_y1 = row_top_y - card_top_offset
            card_x2 = center_x + card_w // 2
            card_y2 = row_bottom_y + card_bottom_offset
            
            # Extract card
            card_img = roi[card_y1:card_y2, card_x1:card_x2]
            
            if card_img.size > 0:
                cards[f"card_{i+1}"] = card_img
        
        return cards
    
    def collect_templates_interactive(self):
        """Collect templates interactively with user input."""
        print(f"\n=== Interactive Template Collection ===")
        print("This will help you collect card templates for recognition.")
        print("Make sure you have the cards you want to collect in your hand!")
        print("-" * 50)
        
        # Capture current hand
        cards = self.extract_cards_from_screen()
        
        if not cards:
            print("❌ No cards detected")
            return
        
        print(f"Detected {len(cards)} cards in your hand")
        
        # Process each card
        for card_id, card_img in cards.items():
            card_number = int(card_id.split("_")[1])
            
            # Show card image
            cv2.imshow(f"Card {card_number}", card_img)
            cv2.waitKey(1)
            
            # Get card name from user
            while True:
                card_name = input(f"\nEnter name for card {card_number} (or 'skip' to skip): ").strip()
                
                if card_name.lower() == 'skip':
                    print(f"Skipped card {card_number}")
                    break
                
                if not card_name:
                    print("Please enter a valid card name")
                    continue
                
                # Save template
                template_path = self.templates_dir / f"{card_name}.png"
                cv2.imwrite(str(template_path), card_img)
                print(f"💾 Saved template: {template_path}")
                break
            
            cv2.destroyAllWindows()
        
        print(f"\n✅ Template collection complete!")
        print(f"💾 Templates saved to: {self.templates_dir}")
    
    def collect_templates_batch(self, num_collections=10, delay_seconds=5):
        """Collect templates in batch mode."""
        print(f"\n=== Batch Template Collection ===")
        print(f"Will collect {num_collections} sets of templates")
        print(f"Delay between collections: {delay_seconds} seconds")
        print("Make sure to have different cards in your hand for each collection!")
        print("-" * 50)
        
        for i in range(num_collections):
            print(f"\n--- Collection {i+1}/{num_collections} ---")
            
            # Capture current hand
            cards = self.extract_cards_from_screen()
            
            if not cards:
                print("❌ No cards detected")
                continue
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save all cards with timestamp
            for card_id, card_img in cards.items():
                card_number = int(card_id.split("_")[1])
                template_path = self.templates_dir / f"card_{card_number}_{timestamp}.png"
                cv2.imwrite(str(template_path), card_img)
                print(f"💾 Saved: {template_path}")
            
            if i < num_collections - 1:
                print(f"Waiting {delay_seconds} seconds for next collection...")
                time.sleep(delay_seconds)
        
        print(f"\n✅ Batch collection complete!")
        print(f"💾 All templates saved to: {self.templates_dir}")
    
    def organize_templates(self):
        """Organize collected templates by card name."""
        print(f"\n=== Organizing Templates ===")
        
        # Get all template files
        template_files = list(self.templates_dir.glob("*.png"))
        
        if not template_files:
            print("❌ No template files found")
            return
        
        print(f"Found {len(template_files)} template files")
        
        # Group by card name (extract from filename)
        card_groups = {}
        
        for template_file in template_files:
            filename = template_file.stem
            
            # Extract card name (everything before the last underscore)
            if '_' in filename:
                parts = filename.split('_')
                if len(parts) >= 2:
                    card_name = '_'.join(parts[:-1])
                    if card_name not in card_groups:
                        card_groups[card_name] = []
                    card_groups[card_name].append(template_file)
        
        # Show organization
        print(f"\nTemplate Organization:")
        for card_name, files in card_groups.items():
            print(f"  {card_name}: {len(files)} templates")
            for file in files:
                print(f"    - {file.name}")
        
        # Ask user to rename templates
        print(f"\nWould you like to rename templates? (y/n): ", end="")
        if input().lower().startswith('y'):
            self._rename_templates(card_groups)
    
    def _rename_templates(self, card_groups):
        """Rename templates based on user input."""
        for card_name, files in card_groups.items():
            if len(files) == 1:
                # Single template - ask for new name
                new_name = input(f"Enter new name for '{card_name}' (or press Enter to keep): ").strip()
                if new_name:
                    old_path = files[0]
                    new_path = self.templates_dir / f"{new_name}.png"
                    old_path.rename(new_path)
                    print(f"  Renamed: {old_path.name} -> {new_path.name}")
            else:
                # Multiple templates - ask which one to keep
                print(f"\nMultiple templates for '{card_name}':")
                for i, file in enumerate(files):
                    print(f"  {i+1}. {file.name}")
                
                while True:
                    try:
                        choice = input(f"Which template to keep for '{card_name}'? (1-{len(files)}, or 'all'): ").strip()
                        if choice.lower() == 'all':
                            break
                        choice = int(choice)
                        if 1 <= choice <= len(files):
                            # Keep selected template, delete others
                            keep_file = files[choice - 1]
                            new_name = input(f"Enter new name for '{card_name}' (or press Enter to keep): ").strip()
                            if new_name:
                                new_path = self.templates_dir / f"{new_name}.png"
                                keep_file.rename(new_path)
                                print(f"  Renamed: {keep_file.name} -> {new_path.name}")
                            
                            # Delete other templates
                            for file in files:
                                if file != keep_file:
                                    file.unlink()
                                    print(f"  Deleted: {file.name}")
                            break
                        else:
                            print("Please enter a valid choice")
                    except ValueError:
                        print("Please enter a valid number")
    
    def view_templates(self):
        """View all collected templates."""
        print(f"\n=== Viewing Templates ===")
        
        template_files = list(self.templates_dir.glob("*.png"))
        
        if not template_files:
            print("❌ No template files found")
            return
        
        print(f"Found {len(template_files)} template files")
        
        for template_file in template_files:
            print(f"\nTemplate: {template_file.name}")
            
            # Load and display template
            template = cv2.imread(str(template_file))
            if template is not None:
                cv2.imshow(f"Template: {template_file.name}", template)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"❌ Could not load template: {template_file}")

def main():
    """Main function for template collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale template collector")
    parser.add_argument("--mode", choices=["interactive", "batch", "organize", "view"], default="interactive", help="Collection mode")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of collections for batch mode")
    parser.add_argument("--delay", type=int, default=5, help="Delay between batch collections")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration file")
    
    args = parser.parse_args()
    
    # Create collector
    collector = TemplateCollector(args.calib)
    
    if args.mode == "interactive":
        collector.collect_templates_interactive()
    elif args.mode == "batch":
        collector.collect_templates_batch(args.batch_size, args.delay)
    elif args.mode == "organize":
        collector.organize_templates()
    elif args.mode == "view":
        collector.view_templates()

if __name__ == "__main__":
    main()
