import argparse
import sys
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class YOLOTrainer:
    def __init__(self, data_dir: str = "yolo_data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Card classes - focused on common opponent cards
        self.card_classes = [
            "Princess",      # 0
            "Knight",        # 1
            "Goblin Gang",   # 2
            "Ice Spirit",    # 3
            "The Log",       # 4
            "Goblin Barrel", # 5
            "Inferno Tower", # 6
            "Spear Goblin",  # 7
            "Goblin",        # 8
            "Archer"         # 9
        ]
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLOv8 training."""
        yaml_content = {
            'path': str(self.data_dir.absolute()),
            'train': 'images',
            'val': 'images',  # Use same images for validation for now
            'test': 'images',
            'nc': len(self.card_classes),
            'names': self.card_classes
        }
        
        yaml_path = self.data_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created dataset.yaml at {yaml_path}")
        return yaml_path
    
    def check_dataset(self):
        """Check if dataset is ready for training."""
        if not self.images_dir.exists():
            print(f"Images directory not found: {self.images_dir}")
            return False
        
        if not self.labels_dir.exists():
            print(f"Labels directory not found: {self.labels_dir}")
            return False
        
        # Count images and labels
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        label_files = list(self.labels_dir.glob("*.txt"))
        
        print(f"Found {len(image_files)} images")
        print(f"Found {len(label_files)} label files")
        
        if len(image_files) == 0:
            print("No images found! Run data collection first.")
            return False
        
        if len(label_files) == 0:
            print("No label files found! Run annotation tool first.")
            return False
        
        # Check for non-empty label files
        non_empty_labels = 0
        for label_file in label_files:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    non_empty_labels += 1
        
        print(f"Found {non_empty_labels} non-empty label files")
        
        if non_empty_labels < 10:
            print("Warning: Very few annotated images. Consider collecting more data.")
        
        return True
    
    def train_model(self, epochs: int = 100, batch_size: int = 16, img_size: int = 640):
        """Train YOLOv8 model on the dataset."""
        if not self.check_dataset():
            return None
        
        # Create dataset.yaml
        yaml_path = self.create_dataset_yaml()
        
        # Initialize model
        model = YOLO('yolov8n.pt')  # Use nano model for faster training
        
        print(f"Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print()
        
        # Train the model
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device='cpu',  # Use CPU for now, can change to 'cuda' if available
            project='yolo_training',
            name='clash_royale_cards',
            save=True,
            plots=True,
            val=True
        )
        
        print("Training completed!")
        print(f"Best model saved to: {results.save_dir}")
        
        return results.save_dir
    
    def validate_model(self, model_path: str):
        """Validate trained model."""
        model = YOLO(model_path)
        
        # Run validation
        results = model.val()
        
        print("Validation results:")
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Training for Clash Royale Cards')
    parser.add_argument('--mode', choices=['check', 'train', 'validate'], default='check',
                       help='Mode: check dataset, train model, or validate model')
    parser.add_argument('--data-dir', default='yolo_data', help='Data directory path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--model-path', help='Path to trained model for validation')
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.data_dir)
    
    if args.mode == 'check':
        trainer.check_dataset()
    elif args.mode == 'train':
        trainer.train_model(args.epochs, args.batch_size, args.img_size)
    elif args.mode == 'validate':
        if not args.model_path:
            print("Please provide --model-path for validation")
            return
        trainer.validate_model(args.model_path)


if __name__ == '__main__':
    main()
