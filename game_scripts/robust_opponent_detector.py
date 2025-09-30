import argparse
import cv2
import numpy as np
import sys
import os
import time
from typing import Optional, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr, load_calibration, default_viewport


class RobustOpponentDetector:
    def __init__(self, calibration_path: str):
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
        
        # Previous frames for comparison
        self.prev_frames = []
        self.max_frames = 3  # Keep last 3 frames for stability
        
        # Detection parameters
        self.change_threshold = 0.08  # 8% of pixels changed
        self.min_change_threshold = 0.02  # Minimum 2% change to consider
        self.max_change_threshold = 0.5  # Maximum 50% change (avoid noise)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better change detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def calculate_change_ratio(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate change ratio with multiple methods."""
        # Method 1: Simple difference
        diff = cv2.absdiff(frame1, frame2)
        simple_changes = np.sum(diff > 30)
        
        # Method 2: Structural similarity
        from skimage.metrics import structural_similarity as ssim
        try:
            ssim_score = ssim(frame1, frame2)
            ssim_changes = 1 - ssim_score
        except:
            ssim_changes = 0
        
        # Method 3: Edge detection changes
        edges1 = cv2.Canny(frame1, 50, 150)
        edges2 = cv2.Canny(frame2, 50, 150)
        edge_diff = cv2.absdiff(edges1, edges2)
        edge_changes = np.sum(edge_diff > 0)
        
        # Combine methods (weighted average)
        total_pixels = frame1.shape[0] * frame1.shape[1]
        simple_ratio = simple_changes / total_pixels
        edge_ratio = edge_changes / total_pixels
        
        # Weighted combination
        combined_ratio = (0.6 * simple_ratio + 0.3 * ssim_changes + 0.1 * edge_ratio)
        
        return combined_ratio
    
    def detect_opponent_card(self, current_frame: np.ndarray) -> Optional[Dict]:
        """Detect opponent card with enhanced accuracy."""
        # Extract opponent region
        x, y, w, h = self.opponent_region['x'], self.opponent_region['y'], self.opponent_region['w'], self.opponent_region['h']
        opponent_region = current_frame[y:y+h, x:x+w]
        
        if opponent_region.size == 0:
            return None
        
        # Preprocess current frame
        current_processed = self.preprocess_frame(opponent_region)
        
        # Compare with previous frames
        if len(self.prev_frames) > 0:
            # Calculate change ratios with all previous frames
            change_ratios = []
            for prev_frame in self.prev_frames:
                ratio = self.calculate_change_ratio(prev_frame, current_processed)
                change_ratios.append(ratio)
            
            # Use the maximum change ratio (most significant change)
            max_change_ratio = max(change_ratios)
            avg_change_ratio = sum(change_ratios) / len(change_ratios)
            
            # Detection logic
            if (max_change_ratio > self.change_threshold and 
                max_change_ratio > self.min_change_threshold and
                max_change_ratio < self.max_change_threshold):
                
                return {
                    'change_ratio': max_change_ratio,
                    'avg_change_ratio': avg_change_ratio,
                    'timestamp': time.time(),
                    'region': self.opponent_region
                }
        
        # Add current frame to history
        self.prev_frames.append(current_processed)
        if len(self.prev_frames) > self.max_frames:
            self.prev_frames.pop(0)
        
        return None
    
    def monitor_opponent(self, duration: int = 60):
        """Monitor opponent with enhanced detection."""
        print("Robust Opponent Detector")
        print(f"Monitoring region: {self.opponent_region}")
        print(f"Change threshold: {self.change_threshold}")
        print(f"Min threshold: {self.min_change_threshold}")
        print(f"Max threshold: {self.max_change_threshold}")
        print()
        
        start_time = time.time()
        frame_count = 0
        detections = 0
        
        while time.time() - start_time < duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            detection = self.detect_opponent_card(frame)
            
            if detection:
                detections += 1
                print(f"Frame {frame_count}: OPPONENT CARD DETECTED!")
                print(f"  Change ratio: {detection['change_ratio']:.4f}")
                print(f"  Avg ratio: {detection['avg_change_ratio']:.4f}")
                
                # Save debug images
                timestamp = int(time.time())
                cv2.imwrite(f'robust_opponent_{frame_count}_{timestamp}.png', frame)
                
                # Extract and save opponent region
                x, y, w, h = self.opponent_region['x'], self.opponent_region['y'], self.opponent_region['w'], self.opponent_region['h']
                opponent_region = frame[y:y+h, x:x+w]
                cv2.imwrite(f'robust_opponent_region_{frame_count}_{timestamp}.png', opponent_region)
                
                print(f"  Saved images: robust_opponent_{frame_count}_{timestamp}.png")
                print(f"  Saved region: robust_opponent_region_{frame_count}_{timestamp}.png")
            
            frame_count += 1
            time.sleep(0.3)  # Check every 0.3 seconds
        
        print(f"\nMonitoring complete.")
        print(f"Total frames: {frame_count}")
        print(f"Detections: {detections}")
        print(f"Detection rate: {detections/frame_count*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Robust opponent card detection with enhanced accuracy.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=0.08)
    parser.add_argument('--min-threshold', type=float, default=0.02)
    parser.add_argument('--max-threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    detector = RobustOpponentDetector(args.calib)
    detector.change_threshold = args.threshold
    detector.min_change_threshold = args.min_threshold
    detector.max_change_threshold = args.max_threshold
    
    detector.monitor_opponent(args.duration)


if __name__ == '__main__':
    main()
