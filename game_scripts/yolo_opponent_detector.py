import argparse
import cv2
import numpy as np
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from ultralytics import YOLO
import pyautogui

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport


class YOLOv8OpponentDetector:
    def __init__(self, calibration_path: str, model_path: Optional[str] = None):
        self.calib = load_calibration(calibration_path)
        self.viewport = default_viewport(self.calib)
        
        # Get opponent region from calibration
        vx, vy, vw, vh = self.viewport
        if 'opponent_region_roi' in self.calib:
            roi = self.calib['opponent_region_roi']
            self.opponent_region = {
                'x': int(vx + roi['x_r'] * vw),
                'y': int(vy + roi['y_r'] * vh),
                'w': int(roi['w_r'] * vw),
                'h': int(roi['h_r'] * vh)
            }
        else:
            self.opponent_region = {
                'x': vx,
                'y': vy,
                'w': vw,
                'h': vh // 2
            }
        
        # Initialize YOLOv8 model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded trained model: {model_path}")
        else:
            # Use pre-trained YOLOv8 model for now
            self.model = YOLO('yolov8n.pt')
            print("Using pre-trained YOLOv8n model (will need training)")
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Data collection settings
        self.data_dir = Path("yolo_data")
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
        # Card classes - focused on common opponent cards
        self.card_classes = {
            0: "Princess",
            1: "Knight", 
            2: "Goblin Gang",
            3: "Ice Spirit",
            4: "The Log",
            5: "Goblin Barrel",
            6: "Inferno Tower",
            7: "Spear Goblin",
            8: "Goblin",
            9: "Archer"
        }
        
        # Reverse mapping for easy lookup
        self.class_to_id = {v: k for k, v in self.card_classes.items()}
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference."""
        # Extract opponent region
        x, y, w, h = self.opponent_region['x'], self.opponent_region['y'], self.opponent_region['w'], self.opponent_region['h']
        opponent_region = frame[y:y+h, x:x+w]
        
        if opponent_region.size == 0:
            return None
        
        # Resize to standard YOLO input size
        resized = cv2.resize(opponent_region, (640, 640))
        
        return resized
    
    def detect_opponent_cards(self, frame: np.ndarray) -> List[Dict]:
        """Detect opponent cards using YOLOv8."""
        processed_frame = self.preprocess_frame(frame)
        if processed_frame is None:
            return []
        
        # Run YOLO inference
        results = self.model(processed_frame, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Convert back to original frame coordinates
                    orig_x1 = int(x1 * self.opponent_region['w'] / 640 + self.opponent_region['x'])
                    orig_y1 = int(y1 * self.opponent_region['h'] / 640 + self.opponent_region['y'])
                    orig_x2 = int(x2 * self.opponent_region['w'] / 640 + self.opponent_region['x'])
                    orig_y2 = int(y2 * self.opponent_region['h'] / 640 + self.opponent_region['y'])
                    
                    detections.append({
                        'bbox': [orig_x1, orig_y1, orig_x2, orig_y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.card_classes.get(class_id, f"Unknown_{class_id}"),
                        'timestamp': time.time()
                    })
        
        return detections
    
    def collect_training_data(self, duration: int = 300):
        """Collect training data by monitoring opponent plays."""
        print("YOLOv8 Data Collection")
        print(f"Monitoring region: {self.opponent_region}")
        print(f"Data will be saved to: {self.data_dir}")
        print()
        print("Instructions:")
        print("1. Make sure Clash Royale is visible")
        print("2. Play cards in the opponent region")
        print("3. Press 'q' to quit data collection")
        print()
        
        frame_count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Extract opponent region
            x, y, w, h = self.opponent_region['x'], self.opponent_region['y'], self.opponent_region['w'], self.opponent_region['h']
            opponent_region = frame[y:y+h, x:x+w]
            
            if opponent_region.size > 0:
                # Save image for training
                image_path = self.images_dir / f"opponent_{frame_count:06d}.jpg"
                cv2.imwrite(str(image_path), opponent_region)
                
                # Create empty label file (will be annotated later)
                label_path = self.labels_dir / f"opponent_{frame_count:06d}.txt"
                with open(label_path, 'w') as f:
                    f.write("")  # Empty file for now
                
                print(f"Saved frame {frame_count}: {image_path}")
            
            frame_count += 1
            time.sleep(0.5)  # Capture every 0.5 seconds
        
        print(f"\nData collection complete!")
        print(f"Collected {frame_count} frames")
        print(f"Images saved to: {self.images_dir}")
        print(f"Labels saved to: {self.labels_dir}")
        print()
        print("Next steps:")
        print("1. Annotate the collected images with bounding boxes")
        print("2. Train the YOLOv8 model on the annotated data")
        print("3. Test the trained model")
    
    def test_detection(self, duration: int = 60):
        """Test YOLOv8 detection on live gameplay."""
        print("YOLOv8 Detection Test")
        print(f"Monitoring region: {self.opponent_region}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print()
        
        start_time = time.time()
        frame_count = 0
        detections = 0
        
        while time.time() - start_time < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Detect opponent cards
            card_detections = self.detect_opponent_cards(frame)
            
            if card_detections:
                detections += len(card_detections)
                print(f"Frame {frame_count}: {len(card_detections)} cards detected")
                
                for detection in card_detections:
                    print(f"  - {detection['class_name']} (conf: {detection['confidence']:.3f})")
                    
                    # Draw bounding box on frame
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detection['class_name']} {detection['confidence']:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save frame with detections
                timestamp = int(time.time())
                cv2.imwrite(f'yolo_detection_{frame_count}_{timestamp}.png', frame)
                print(f"  Saved detection image: yolo_detection_{frame_count}_{timestamp}.png")
            
            frame_count += 1
            time.sleep(0.5)  # Check every 0.5 seconds
        
        print(f"\nDetection test complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total detections: {detections}")
        print(f"Detection rate: {detections/frame_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Opponent Card Detection')
    parser.add_argument('--mode', choices=['collect', 'test', 'train'], default='test',
                       help='Mode: collect data, test detection, or train model')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--model', help='Path to trained YOLOv8 model')
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    args = parser.parse_args()
    
    detector = YOLOv8OpponentDetector(args.calib, args.model)
    detector.confidence_threshold = args.conf
    detector.iou_threshold = args.iou
    
    if args.mode == 'collect':
        detector.collect_training_data(args.duration)
    elif args.mode == 'test':
        detector.test_detection(args.duration)
    elif args.mode == 'train':
        print("Training mode not implemented yet. Use 'collect' to gather data first.")
        print("Then use external tools like Roboflow or YOLOv8 CLI for training.")


if __name__ == '__main__':
    main()
