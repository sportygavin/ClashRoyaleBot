import argparse
import time
from collections import Counter
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_scripts.strategy_utils import screen_bgr
from tools.card_recognition_system import CardRecognitionSystem


def main():
    parser = argparse.ArgumentParser(description='Continuously read and print current elixir with confidence.')
    parser.add_argument('--calib', default='cv_out/calibration_manual_fixed.json')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--hz', type=float, default=5.0, help='Samples per second')
    args = parser.parse_args()

    crs = CardRecognitionSystem(args.calib, 'database/clash_royale_cards.json')

    interval = 1.0 / max(0.5, args.hz)
    start = time.time()
    history = []
    while time.time() - start < args.duration:
        frame = screen_bgr()
        if frame is None:
            time.sleep(interval)
            continue
        
        # Force debug image generation
        debug_path = crs._save_elixir_debug(frame, tag=f"test_{len(history)}")
        print(f"Debug images: {debug_path}")
        
        v, c = crs.recognize_current_elixir(frame)
        history.append((v, c))
        print(f"elixir={v} conf={c:.2f}")
        time.sleep(interval)

    vals = [v for v, _ in history if v is not None]
    if vals:
        counts = Counter(vals)
        print("\nSummary (value -> count):")
        for k, cnt in counts.most_common():
            print(f"  {k}: {cnt}")
    else:
        print("\nNo valid elixir values detected.")


if __name__ == '__main__':
    main()


