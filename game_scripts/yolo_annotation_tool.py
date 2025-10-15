import argparse
import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class YOLOAnnotationTool:
    def __init__(self, data_dir: str = "yolo_data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Ensure directories exist
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
        # Card classes
        self.card_classes = [
            "Giant", "Archer", "Knight", "Fireball", "Musketeer", 
            "PEKKA", "Golem", "Mega Knight", "Wizard", "Dragon",
            "Skeleton", "Bomber", "Archers", "Goblins", "Spear Goblins",
            "Skeletons", "Ice Spirit", "Fire Spirits", "Minions", "Hog Rider",
            "Valkyrie", "Musketeer", "Witch", "Bomber", "Baby Dragon",
            "Prince", "Dark Prince", "Wizard", "Mini PEKKA", "Giant Skeleton",
            "Skeleton Army", "Bomber", "Cannon", "Tesla", "Inferno Tower",
            "Bomb Tower", "Elixir Collector", "Barbarian Hut", "Tombstone",
            "Furnace", "Goblin Hut", "Inferno Dragon", "Ice Wizard", "Lumberjack",
            "Night Witch", "Bandit", "Royal Ghost", "Mega Knight", "Electro Wizard",
            "Hunter", "Executioner", "Cannon Cart", "Mega Minion", "Dart Goblin",
            "Goblin Gang", "Elite Barbarians", "Battle Ram", "Zappies", "Flying Machine",
            "Magic Archer", "Skeleton Barrel", "Goblin Giant", "Fisherman", "Firecracker",
            "Mighty Miner", "Elixir Golem", "Battle Healer", "Skeleton King", "Archer Queen",
            "Golden Knight", "Monk", "Skeleton Dragons", "Mother Witch", "Electro Spirit",
            "Electro Giant", "Cannon", "X-Bow", "Mortar", "Rocket", "Freeze",
            "Rage", "Clone", "Heal", "Mirror", "Lightning", "Poison",
            "Graveyard", "The Log", "Tornado", "Giant Snowball", "Barbarian Barrel",
            "Royal Delivery", "Earthquake", "Goblin Barrel", "Fireball", "Arrows"
        ]
        
        # Current annotation state
        self.current_image_idx = 0
        self.current_image = None
        self.current_annotations = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_class = 0
        
        # Load existing images
        self.image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        self.image_files.sort()
        
        if not self.image_files:
            print("No images found in data directory!")
            return
        
        print(f"Found {len(self.image_files)} images to annotate")
        print(f"Card classes: {len(self.card_classes)}")
        print()
        print("Controls:")
        print("- Mouse: Draw bounding box")
        print("- 0-9: Select card class (0=Giant, 1=Archer, etc.)")
        print("- 'n': Next image")
        print("- 'p': Previous image")
        print("- 's': Save annotations")
        print("- 'q': Quit")
        print()
    
    def load_image(self, idx: int):
        """Load image for annotation."""
        if 0 <= idx < len(self.image_files):
            self.current_image_idx = idx
            image_path = self.image_files[idx]
            self.current_image = cv2.imread(str(image_path))
            
            # Load existing annotations
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            self.current_annotations = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to bounding box coordinates
                            h, w = self.current_image.shape[:2]
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            self.current_annotations.append({
                                'class_id': class_id,
                                'bbox': [x1, y1, x2, y2],
                                'class_name': self.card_classes[class_id] if class_id < len(self.card_classes) else f"Unknown_{class_id}"
                            })
            
            return True
        return False
    
    def draw_annotations(self):
        """Draw current annotations on image."""
        if self.current_image is None:
            return
        
        display_image = self.current_image.copy()
        
        # Draw existing annotations
        for ann in self.current_annotations:
            x1, y1, x2, y2 = ann['bbox']
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, ann['class_name'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw current bounding box being drawn
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(display_image, self.start_point, self.end_point, (255, 0, 0), 2)
        
        # Draw info
        info_text = f"Image {self.current_image_idx + 1}/{len(self.image_files)}"
        cv2.putText(display_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        current_class_name = self.card_classes[self.current_class] if self.current_class < len(self.card_classes) else f"Unknown_{self.current_class}"
        class_text = f"Current class: {current_class_name} ({self.current_class})"
        cv2.putText(display_image, class_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_image
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.end_point = (x, y)
                self.drawing = False
                
                # Add annotation
                if self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    
                    # Ensure proper bounding box
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    self.current_annotations.append({
                        'class_id': self.current_class,
                        'bbox': [x1, y1, x2, y2],
                        'class_name': self.card_classes[self.current_class] if self.current_class < len(self.card_classes) else f"Unknown_{self.current_class}"
                    })
    
    def save_annotations(self):
        """Save current annotations to YOLO format."""
        if self.current_image is None:
            return
        
        label_path = self.labels_dir / f"{self.image_files[self.current_image_idx].stem}.txt"
        
        with open(label_path, 'w') as f:
            h, w = self.current_image.shape[:2]
            
            for ann in self.current_annotations:
                x1, y1, x2, y2 = ann['bbox']
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Saved annotations to {label_path}")
    
    def run(self):
        """Run the annotation tool."""
        if not self.image_files:
            return
        
        # Load first image
        self.load_image(0)
        
        cv2.namedWindow('YOLO Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('YOLO Annotation Tool', self.mouse_callback)
        
        while True:
            display_image = self.draw_annotations()
            cv2.imshow('YOLO Annotation Tool', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Next image
                if self.current_image_idx < len(self.image_files) - 1:
                    self.load_image(self.current_image_idx + 1)
            elif key == ord('p'):
                # Previous image
                if self.current_image_idx > 0:
                    self.load_image(self.current_image_idx - 1)
            elif key == ord('s'):
                # Save annotations
                self.save_annotations()
            elif key >= ord('0') and key <= ord('9'):
                # Select class
                self.current_class = key - ord('0')
                print(f"Selected class: {self.card_classes[self.current_class]}")
        
        cv2.destroyAllWindows()
        print("Annotation tool closed.")


def main():
    parser = argparse.ArgumentParser(description='YOLO Dataset Annotation Tool')
    parser.add_argument('--data-dir', default='yolo_data', help='Data directory path')
    args = parser.parse_args()
    
    tool = YOLOAnnotationTool(args.data_dir)
    tool.run()


if __name__ == '__main__':
    main()
