import argparse
import time
import sys
import os
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr
from tools.card_recognition_system import CardRecognitionSystem


def main():
    parser = argparse.ArgumentParser(description='Debug elixir region extraction and show what it sees.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=10)
    args = parser.parse_args()

    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')
    
    print("Capturing elixir region... Press Ctrl+C to stop")
    try:
        start = time.time()
        frame_count = 0
        while time.time() - start < args.duration:
            frame = screen_bgr()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Extract and save elixir region
            region = crs.extract_current_elixir_region(frame)
            if region is not None:
                cv2.imwrite(f'debug_elixir_region_{frame_count}.png', region)
                print(f"Saved elixir region {frame_count}: {region.shape}")
                
                # Also save debug images
                debug_path = crs._save_elixir_debug(frame, f"debug_{frame_count}")
                print(f"Debug images saved: {debug_path}")
                
                # Try recognition
                elixir_val, conf = crs.recognize_current_elixir(frame)
                print(f"Frame {frame_count}: elixir={elixir_val}, conf={conf:.2f}")
                
                frame_count += 1
                time.sleep(1.0)  # Wait 1 second between captures
            else:
                print(f"Frame {frame_count}: Could not extract elixir region")
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    print(f"\nCaptured {frame_count} frames. Check debug_elixir_region_*.png files")


if __name__ == '__main__':
    main()
