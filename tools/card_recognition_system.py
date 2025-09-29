#!/usr/bin/env python3
"""
Card recognition system for Clash Royale.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyautogui
import time

class CardRecognitionSystem:
    def __init__(self, 
                 calibration_file="cv_out/calibration_manual_fixed.json",
                 database_file="database/clash_royale_cards.json",
                 templates_dir="templates/cards",
                 elixir_dx_r: float = 0.0,
                 elixir_dy_r: float = 0.0,
                 elixir_width_mult: float = 1.0):
        self.calib = self._load_calibration(calibration_file)
        self.db = self._load_database(database_file)
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        # Adjustable nudges for current elixir ROI (relative to viewport)
        self.elixir_dx_r = elixir_dx_r
        self.elixir_dy_r = elixir_dy_r
        self.elixir_width_mult = elixir_width_mult
        
        # Load card templates if they exist
        self.card_templates = self._load_card_templates()
        # Load digit templates for current elixir recognition
        self.digit_templates = self._load_digit_templates()
        
        print(f"✅ Card Recognition System Ready!")
        print(f"✅ Database: {len(self.db['cards'])} cards")
        print(f"✅ Templates: {len(self.card_templates)} loaded")
        if hasattr(self, 'digit_templates'):
            print(f"✅ Digit Templates: {len(self.digit_templates)} loaded")
    
    def _load_calibration(self, calibration_file):
        """Load calibration data."""
        with open(calibration_file) as f:
            return json.load(f)
    
    def _load_database(self, database_file):
        """Load card database."""
        with open(database_file) as f:
            return json.load(f)
    
    def _load_card_templates(self) -> Dict[str, np.ndarray]:
        """Load card templates from templates directory."""
        templates = {}
        
        if not self.templates_dir.exists():
            print(f"⚠️  Templates directory not found: {self.templates_dir}")
            return templates
        
        for template_file in self.templates_dir.glob("*.png"):
            card_name = template_file.stem
            template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
            if template is not None:
                templates[card_name] = template
                print(f"  Loaded template: {card_name}")
        
        return templates

    def _load_digit_templates(self) -> Dict[str, np.ndarray]:
        """Load digit templates from templates/digits directory (expects 0-9 and optionally 10)."""
        digits_dir = Path("templates/digits")
        templates: Dict[str, np.ndarray] = {}
        if not digits_dir.exists():
            print("⚠️  Digit templates directory not found: templates/digits")
            return templates
        for template_file in digits_dir.glob("*.png"):
            key = template_file.stem  # e.g., '0', '1', ..., '9', '10'
            img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                templates[key] = img
        if not templates:
            print("⚠️  No digit templates loaded from templates/digits")
        return templates
    
    def _auto_crop_content(self, image: np.ndarray, edge_thresh: float = 10.0, margin: int = 2) -> np.ndarray:
        """Automatically crop away uniform padding (top/bottom/left/right) using edge energy.
        Keeps a small margin to avoid over-cropping rounded borders.
        """
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Sobel to capture edges; more robust than Canny for low-contrast
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)

        # Sum energy along axes
        row_energy = mag.sum(axis=1)
        col_energy = mag.sum(axis=0)

        # Find bounds where energy exceeds a small threshold of the max
        def find_bounds(energy: np.ndarray) -> Tuple[int, int]:
            if energy.size == 0:
                return 0, 0
            max_e = float(energy.max())
            if max_e <= 1e-6:
                return 0, energy.size
            thresh = max_e * 0.02  # 2% of max energy
            indices = np.where(energy > max(thresh, edge_thresh))[0]
            if indices.size == 0:
                return 0, energy.size
            start = max(int(indices[0]) - margin, 0)
            end = min(int(indices[-1]) + 1 + margin, energy.size)
            return start, end

        r0, r1 = find_bounds(row_energy)
        c0, c1 = find_bounds(col_energy)

        cropped = image[r0:r1, c0:c1]
        return cropped if cropped.size > 0 else image

    def _preprocess_for_match(self, image: np.ndarray) -> np.ndarray:
        """Normalize contrast and suppress color to make template matching more robust."""
        if image is None or image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CLAHE for local contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(gray)
        # Light blur to reduce noise
        norm = cv2.GaussianBlur(norm, (3, 3), 0)
        return norm

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

    def _get_viewport_roi(self, screenshot: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
        """Return (roi, vp_x, vp_y, vp_w, vp_h) for convenience."""
        viewport = self.calib['viewport']
        vp_x = int(viewport['x_r'] * screenshot.shape[1])
        vp_y = int(viewport['y_r'] * screenshot.shape[0])
        vp_w = int(viewport['w_r'] * screenshot.shape[1])
        vp_h = int(viewport['h_r'] * screenshot.shape[0])
        roi = screenshot[vp_y:vp_y+vp_h, vp_x:vp_x+vp_w]
        return roi, vp_x, vp_y, vp_w, vp_h

    def _save_elixir_debug(self, screenshot: np.ndarray, tag: str = "live") -> Optional[str]:
        """Save a debug image showing the current elixir ROI on the viewport and the cropped ROI itself."""
        out_dir = Path("cv_out")
        out_dir.mkdir(parents=True, exist_ok=True)

        roi, vp_x, vp_y, vp_w, vp_h = self._get_viewport_roi(screenshot)

        # Compute ROI using same logic as extraction
        if 'current_elixir_roi' in self.calib:
            cfg = self.calib['current_elixir_roi']
            ex1 = int(cfg['x_r'] * vp_w)
            ey1 = int(cfg['y_r'] * vp_h)
            ex2 = int((cfg['x_r'] + cfg['w_r']) * vp_w)
            ey2 = int((cfg['y_r'] + cfg['h_r']) * vp_h)
            # Apply nudges and width multiplier
            dx = int(self.elixir_dx_r * vp_w)
            dy = int(self.elixir_dy_r * vp_h)
            ex1 = min(max(ex1 + dx, 0), vp_w - 1)
            ey1 = min(max(ey1 + dy, 0), vp_h - 1)
            # Apply width multiplier (keep left edge same, extend right)
            original_width = ex2 - ex1
            new_width = int(original_width * self.elixir_width_mult)
            ex2 = min(ex1 + new_width, vp_w)
            ey2 = min(max(ey2 + dy, 1), vp_h)
        else:
            card_row = self.calib['card_row']
            row_top_y = int(card_row['top_r'] * vp_h)
            row_bottom_y = int(card_row['bottom_r'] * vp_h)
            cards_cfg = self.calib['cards']
            center_x_r = cards_cfg['centers_x_r'][0]
            center_x = int(center_x_r * vp_w)
            card_w = int(cards_cfg['width_r'] * vp_w)
            box_w = max(int(card_w * 0.22), 16)
            box_h = max(int((row_bottom_y - row_top_y) * 0.28), 16)
            ex1 = max(center_x - card_w // 2 + int(card_w * 0.06), 0)
            ey1 = min(row_bottom_y + int(0.02 * vp_h), vp_h - box_h)
            # Apply nudges
            ex1 = min(max(ex1 + int(self.elixir_dx_r * vp_w), 0), vp_w - box_w)
            ey1 = min(max(ey1 + int(self.elixir_dy_r * vp_h), 0), vp_h - box_h)
            # Apply width multiplier (keep left edge same, extend right)
            new_width = int(box_w * self.elixir_width_mult)
            ex2 = min(ex1 + new_width, vp_w)
            ey2 = min(ey1 + box_h, vp_h)

        # Draw rectangle on a copy of viewport
        overlay = roi.copy()
        cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
        dbg_view_path = out_dir / f"elixir_viewport_{tag}.png"
        cv2.imwrite(str(dbg_view_path), overlay)

        # Save cropped ROI
        crop = roi[ey1:ey2, ex1:ex2]
        dbg_roi_path = out_dir / f"elixir_roi_{tag}.png"
        if crop.size > 0:
            cv2.imwrite(str(dbg_roi_path), crop)

        return str(dbg_view_path)
    
    def extract_elixir_region(self, card_img):
        """Extract elixir region from card image."""
        h, w = card_img.shape[:2]
        
        # Extract elixir region
        elixir_roi = self.calib['elixir_roi']
        x1 = int(elixir_roi['x_off_r'] * w)
        y1 = int(elixir_roi['y_off_r'] * h)
        x2 = int((elixir_roi['x_off_r'] + elixir_roi['w_r']) * w)
        y2 = int((elixir_roi['y_off_r'] + elixir_roi['h_r']) * h)
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        elixir_region = card_img[y1:y2, x1:x2]
        
        return elixir_region if elixir_region.size > 0 else None

    def extract_current_elixir_region(self, screenshot: np.ndarray) -> Optional[np.ndarray]:
        """Extract the current elixir number region at bottom-left under card 1.
        Uses calibration card row and card_1 center to derive an ROI.
        If calibration contains 'current_elixir_roi', use that directly.
        """
        roi, _, _, vp_w, vp_h = self._get_viewport_roi(screenshot)

        # If explicit calibration exists, use it
        if 'current_elixir_roi' in self.calib:
            cfg = self.calib['current_elixir_roi']
            x1 = int(cfg['x_r'] * vp_w)
            y1 = int(cfg['y_r'] * vp_h)
            x2 = int((cfg['x_r'] + cfg['w_r']) * vp_w)
            y2 = int((cfg['y_r'] + cfg['h_r']) * vp_h)
            # Apply user-provided nudges and width multiplier
            dx = int(self.elixir_dx_r * vp_w)
            dy = int(self.elixir_dy_r * vp_h)
            x1 = min(max(x1 + dx, 0), vp_w - 1)
            y1 = min(max(y1 + dy, 0), vp_h - 1)
            # Apply width multiplier (keep left edge same, extend right)
            original_width = x2 - x1
            new_width = int(original_width * self.elixir_width_mult)
            x2 = min(x1 + new_width, vp_w)
            y2 = min(max(y2 + dy, 1), vp_h)
            region = roi[y1:y2, x1:x2]
            return region if region.size > 0 else None

        # Fallback: derive from card row and card_1 center
        card_row = self.calib['card_row']
        row_top_y = int(card_row['top_r'] * vp_h)
        row_bottom_y = int(card_row['bottom_r'] * vp_h)
        cards_cfg = self.calib['cards']
        center_x_r = cards_cfg['centers_x_r'][0]
        center_x = int(center_x_r * vp_w)
        card_w = int(cards_cfg['width_r'] * vp_w)

        # Define a small box below left side of card 1
        box_w = max(int(card_w * 0.22), 16)
        box_h = max(int((row_bottom_y - row_top_y) * 0.28), 16)
        x1 = max(center_x - card_w // 2 + int(card_w * 0.06), 0)
        y1 = min(row_bottom_y + int(0.02 * vp_h), vp_h - box_h)
        # Apply user-provided nudges
        x1 = min(max(x1 + int(self.elixir_dx_r * vp_w), 0), vp_w - box_w)
        y1 = min(max(y1 + int(self.elixir_dy_r * vp_h), 0), vp_h - box_h)
        # Apply width multiplier (keep left edge same, extend right)
        new_width = int(box_w * self.elixir_width_mult)
        x2 = min(x1 + new_width, vp_w)
        y2 = min(y1 + box_h, vp_h)
        region = roi[y1:y2, x1:x2]
        return region if region.size > 0 else None

    def _preprocess_digit_roi(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        return th

    def _digit_preprocess_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate multiple preprocessing variants to improve robustness."""
        variants: List[np.ndarray] = []
        if image is None or image.size == 0:
            return variants

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = clahe.apply(gray)

        # Variant A: adaptive threshold
        a = cv2.GaussianBlur(base, (3, 3), 0)
        a = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
        variants.append(a)

        # Variant B: Otsu threshold
        b = cv2.GaussianBlur(base, (5, 5), 0)
        _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(b)

        # Variant C: Inverted adaptive
        c = cv2.GaussianBlur(base, (3, 3), 0)
        c = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
        variants.append(c)

        # Variant D: Morph opened to reduce noise
        d = a.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel, iterations=1)
        variants.append(d)

        return variants

    def recognize_current_elixir(self, screenshot: np.ndarray) -> Tuple[Optional[int], float]:
        """Recognize current elixir number using digit templates. Returns (elixir, confidence).
        Supports '10' via dedicated template or split-ROI two-digit fallback.
        """
        region = self.extract_current_elixir_region(screenshot)
        if region is None or not getattr(self, 'digit_templates', {}):
            return None, 0.0

        variants = self._digit_preprocess_variants(region)
        if not variants:
            return None, 0.0

        def match_template(proc_img: np.ndarray, tmpl_img: np.ndarray) -> float:
            th, tw = tmpl_img.shape[:2]
            if th < 5 or tw < 5:
                return 0.0
            # Ensure template fits, scale down if needed
            if th > proc_img.shape[0] or tw > proc_img.shape[1]:
                scale = min(proc_img.shape[0] / max(th, 1), proc_img.shape[1] / max(tw, 1), 1.0)
                th2 = max(int(th * scale), 5)
                tw2 = max(int(tw * scale), 5)
                tmpl_use = cv2.resize(tmpl_img, (tw2, th2), interpolation=cv2.INTER_AREA)
            else:
                tmpl_use = tmpl_img
            res = cv2.matchTemplate(proc_img, tmpl_use, cv2.TM_CCOEFF_NORMED)
            return float(cv2.minMaxLoc(res)[1])

        best_label: Optional[str] = None
        best_score: float = 0.0

        # 1) Try matching whole ROI against available templates across variants
        for proc in variants:
            # Normalize height for stability (keep aspect)
            target_h = 32
            scale = target_h / max(proc.shape[0], 1)
            proc_norm = cv2.resize(proc, (max(int(proc.shape[1] * scale), 8), target_h), interpolation=cv2.INTER_AREA)

            for label, tmpl in self.digit_templates.items():
                score = match_template(proc_norm, tmpl)
                if score > best_score:
                    best_score = score
                    best_label = label

        # 2) If no strong match, attempt split-ROI two-digit detection for '10'
        # Heuristic: try when best < 0.7 or best predicts 5/8 often confused
        if best_score < 0.7 or (best_label in {"5", "8", None}):
            # Split into left/right halves (slightly biased to left digit narrower)
            h, w = variants[0].shape[:2]
            split_x = int(w * 0.52)
            for proc in variants:
                left = proc[:, :split_x]
                right = proc[:, split_x:]

                # Match left among 1-9 (expect '1' for 10)
                left_best, left_score = None, 0.0
                for d, tmpl in self.digit_templates.items():
                    if d not in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                        continue
                    s = match_template(left, tmpl)
                    if s > left_score:
                        left_score, left_best = s, d

                # Match right among 0-9 (expect '0' for 10)
                right_best, right_score = None, 0.0
                for d, tmpl in self.digit_templates.items():
                    if d not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                        continue
                    s = match_template(right, tmpl)
                    if s > right_score:
                        right_score, right_best = s, d

                # If we found '1' and '0', propose 10 with conservative score
                if left_best == "1" and right_best == "0":
                    combined = min(left_score, right_score)
                    if combined > best_score:
                        best_label, best_score = "10", combined

        # 3) If we have a dedicated '10' template, give it a final try across variants
        if "10" in self.digit_templates:
            tmpl10 = self.digit_templates["10"]
            for proc in variants:
                score = match_template(proc, tmpl10)
                if score > best_score:
                    best_label, best_score = "10", score

        try:
            elixir_val = int(best_label) if best_label is not None else None
        except ValueError:
            elixir_val = None

        return elixir_val, float(best_score)
    
    def recognize_card_by_template(self, card_img, threshold=0.3):
        """Recognize card using template matching."""
        best_match = None
        best_score = 0.0

        # Focus the card on its central artwork and normalize
        card_cropped = self._auto_crop_content(card_img)
        card_proc = self._preprocess_for_match(card_cropped)

        # Try multiple template scales to be robust to slight size variance
        scales = [0.9, 0.95, 1.0, 1.05, 1.1]

        for card_name, template in self.card_templates.items():
            # Auto-trim padding from the template as well
            tmpl_cropped = self._auto_crop_content(template)
            tmpl_proc_base = self._preprocess_for_match(tmpl_cropped)

            for s in scales:
                h = int(tmpl_proc_base.shape[0] * s)
                w = int(tmpl_proc_base.shape[1] * s)
                if h < 8 or w < 8:
                    continue
                tmpl_proc = cv2.resize(tmpl_proc_base, (w, h), interpolation=cv2.INTER_LINEAR)

                # Template must be <= card image for cv2.matchTemplate
                if tmpl_proc.shape[0] > card_proc.shape[0] or tmpl_proc.shape[1] > card_proc.shape[1]:
                    continue

                result = cv2.matchTemplate(card_proc, tmpl_proc, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    best_match = card_name

        # Always return the best match, even if below threshold
        return best_match, best_score
    
    def recognize_card_by_features(self, card_img):
        """Recognize card using feature matching."""
        # This is a placeholder for more sophisticated feature matching
        # In practice, you'd use SIFT, ORB, or other feature detectors
        
        # For now, return None - feature matching would require more complex implementation
        return None, 0
    
    def recognize_card(self, card_img, method="template"):
        """Recognize card using specified method."""
        if method == "template":
            return self.recognize_card_by_template(card_img)
        elif method == "features":
            return self.recognize_card_by_features(card_img)
        else:
            raise ValueError(f"Unknown recognition method: {method}")
    
    def get_card_info(self, card_name):
        """Get card information from database."""
        for card_id, card_data in self.db['cards'].items():
            if card_data['name'].lower() == card_name.lower():
                return card_data
        return None
    
    def analyze_hand(self, screenshot=None, method="template"):
        """Analyze the current hand and return card information."""
        print(f"\n=== Analyzing Hand ===")
        
        # Extract cards
        cards = self.extract_cards_from_screen(screenshot)
        # Also compute current elixir from the same screen
        if screenshot is None:
            screen_bgr = pyautogui.screenshot()
            screen_bgr = cv2.cvtColor(np.array(screen_bgr), cv2.COLOR_RGB2BGR)
        else:
            screen_bgr = screenshot
        
        if not cards:
            print("❌ No cards detected")
            return None
        
        hand_analysis = {}
        
        for card_id, card_img in cards.items():
            card_number = int(card_id.split("_")[1])
            
            # Extract elixir region
            elixir_region = self.extract_elixir_region(card_img)
            
            # Recognize card
            card_name, confidence = self.recognize_card(card_img, method)
            
            # Get card info from database
            card_info = self.get_card_info(card_name) if card_name else None
            
            hand_analysis[card_id] = {
                "card_number": card_number,
                "card_name": card_name,
                "confidence": confidence,
                "card_info": card_info,
                "elixir_region": elixir_region
            }
            
            if card_name:
                confidence_indicator = "✓" if confidence >= 0.6 else "?" if confidence >= 0.3 else "⚠"
                print(f"Card {card_number}: {card_name} {confidence_indicator} (confidence: {confidence:.2f})")
                if card_info:
                    print(f"  Elixir Cost: {card_info['elixir_cost']}")
                    print(f"  Rarity: {card_info['rarity']}")
                    print(f"  Type: {card_info['type']}")
            else:
                print(f"Card {card_number}: Unknown (confidence: {confidence:.2f})")

        # Print current elixir
        cur_elixir, elx_conf = self.recognize_current_elixir(screen_bgr)
        if cur_elixir is not None:
            e_ind = "✓" if elx_conf >= 0.6 else "?" if elx_conf >= 0.3 else "⚠"
            print(f"Current Elixir: {cur_elixir} {e_ind} (confidence: {elx_conf:.2f})")
        else:
            dbg_path = self._save_elixir_debug(screen_bgr, tag="analyze")
            print(f"Current Elixir: Unknown (saved ROI debug: {dbg_path})")
        
        return hand_analysis
    
    def save_card_template(self, card_img, card_name):
        """Save card image as template for future recognition."""
        template_path = self.templates_dir / f"{card_name}.png"
        cv2.imwrite(str(template_path), card_img)
        print(f"💾 Saved template: {template_path}")
        
        # Reload templates
        self.card_templates = self._load_card_templates()
    
    def create_templates_from_hand(self, screenshot=None):
        """Create templates from current hand for future recognition."""
        print(f"\n=== Creating Templates from Hand ===")
        
        cards = self.extract_cards_from_screen(screenshot)
        
        if not cards:
            print("❌ No cards detected")
            return
        
        for card_id, card_img in cards.items():
            card_number = int(card_id.split("_")[1])
            
            # Ask user for card name
            while True:
                card_name = input(f"Enter name for card {card_number} (or 'skip'): ").strip()
                if card_name.lower() == 'skip':
                    break
                if card_name:
                    self.save_card_template(card_img, card_name)
                    break
                print("Please enter a valid card name")
    
    def live_card_monitoring(self, duration=30, method="template"):
        """Monitor cards in real-time."""
        print(f"\n=== Live Card Monitoring ===")
        print(f"Duration: {duration} seconds")
        print(f"Method: {method}")
        print("Press Ctrl+C to stop early")
        print("-" * 50)
        
        start_time = time.time()
        last_hand = None
        
        try:
            while time.time() - start_time < duration:
                # Analyze current hand
                hand_analysis = self.analyze_hand(method=method)
                
                if hand_analysis:
                    # Check if hand changed
                    current_hand = [card['card_name'] for card in hand_analysis.values()]
                    
                    if last_hand != current_hand:
                        print(f"\n🔄 Hand changed at {time.time() - start_time:.1f}s:")
                        for card_id, card_data in hand_analysis.items():
                            card_name = card_data['card_name']
                            confidence = card_data['confidence']
                            if card_name:
                                confidence_indicator = "✓" if confidence >= 0.6 else "?" if confidence >= 0.3 else "⚠"
                                print(f"  {card_id}: {card_name} {confidence_indicator} ({confidence:.2f})")
                            else:
                                print(f"  {card_id}: Unknown ({confidence:.2f})")
                        # Show current elixir on hand change
                        scr = pyautogui.screenshot()
                        scr = cv2.cvtColor(np.array(scr), cv2.COLOR_RGB2BGR)
                        elixir_val, econf = self.recognize_current_elixir(scr)
                        if elixir_val is not None:
                            e_ind = "✓" if econf >= 0.6 else "?" if econf >= 0.3 else "⚠"
                            print(f"  Elixir: {elixir_val} {e_ind} ({econf:.2f})")
                        else:
                            dbg_path = self._save_elixir_debug(scr, tag="change")
                            print(f"  Elixir: Unknown (saved ROI debug: {dbg_path})")
                        last_hand = current_hand
                
                time.sleep(1)  # Check every second
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
        
        print(f"\n✅ Monitoring complete!")

def main():
    """Test the card recognition system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale card recognition system")
    parser.add_argument("--mode", choices=["analyze", "create-templates", "monitor"], default="analyze", help="Operation mode")
    parser.add_argument("--method", choices=["template", "features"], default="template", help="Recognition method")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in seconds")
    parser.add_argument("--calib", default="cv_out/calibration_manual_fixed.json", help="Calibration file")
    parser.add_argument("--elixir-dx", type=float, default=0.0, help="Nudge current elixir ROI right (+) / left (-) as fraction of viewport width")
    parser.add_argument("--elixir-dy", type=float, default=0.0, help="Nudge current elixir ROI down (+) / up (-) as fraction of viewport height")
    parser.add_argument("--elixir-width", type=float, default=1.0, help="Multiply elixir ROI width (2.0 = double width, keeps left edge)")
    parser.add_argument("--db", default="database/clash_royale_cards.json", help="Database file")
    
    args = parser.parse_args()
    
    # Create recognition system
    recognizer = CardRecognitionSystem(args.calib, args.db, elixir_dx_r=args.elixir_dx, elixir_dy_r=args.elixir_dy, elixir_width_mult=args.elixir_width)
    
    if args.mode == "analyze":
        recognizer.analyze_hand(method=args.method)
    elif args.mode == "create-templates":
        recognizer.create_templates_from_hand()
    elif args.mode == "monitor":
        recognizer.live_card_monitoring(args.duration, args.method)

if __name__ == "__main__":
    main()
