# Stage 2: Enhanced PaddleOCR Detection + Recognition
"""
PaddleOCR for text detection and recognition.
Uses TextDetection + TextRecognition in sequence for better accuracy.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw
import json

# PaddleOCR for text detection and recognition
try:
    from paddleocr import TextDetection, TextRecognition
    PADDLEOCR_AVAILABLE = True
except ImportError:
    TextDetection = None
    TextRecognition = None
    PADDLEOCR_AVAILABLE = False
    logging.warning("paddleocr not available. Install with: pip install paddleocr")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import TextBox, LayoutSection
from utils.coordinate_utils import CoordinateConverter

logger = logging.getLogger(__name__)


class EnhancedPaddleOCRStage:
    """Stage 2: Enhanced PaddleOCR using TextDetection + TextRecognition in sequence"""
    
    def __init__(self, det_model: str = "PP-OCRv5_server_det", rec_model: str = "PP-OCRv5_server_rec"):
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("paddleocr not available")
        
        logger.info(f"Loading PaddleOCR TextDetection (model={det_model})")
        self.detector = TextDetection(model_name=det_model)
        
        logger.info(f"Loading PaddleOCR TextRecognition (model={rec_model})")
        self.recognizer = TextRecognition(model_name=rec_model)
        
        # Re-initialize logger (PaddleOCR may override it)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        logger.info("PaddleOCR loaded successfully (TextDetection + TextRecognition)")
    
    def _run_ocr_on_image(self, image_input) -> List[Tuple[List[int], str, float]]:
        """
        Run OCR using TextDetection + TextRecognition in sequence.
        1. Detect text line bounding boxes with TextDetection
        2. Crop all detected boxes
        3. Run TextRecognition in batch on all crops
        Returns list of (bbox, text, confidence).
        """
        results = []
        
        try:
            # Convert image to numpy array if needed
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    return results
            elif isinstance(image_input, np.ndarray):
                img = image_input
            else:
                img = np.array(image_input)
            
            # Ensure 3-channel RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            img_height, img_width = img.shape[:2]
            
            # Step 1: Text Detection
            det_output = self.detector.predict(input=img, batch_size=1)
            
            detected_boxes = []  # (bbox, det_score)
            crops_for_recognition = []  # cropped images
            
            for res in det_output:
                # Get detection polygons and scores
                if hasattr(res, 'keys'):
                    dt_polys = res.get('dt_polys', [])
                    dt_scores = res.get('dt_scores', [])
                else:
                    dt_polys = getattr(res, 'dt_polys', []) if hasattr(res, 'dt_polys') else []
                    dt_scores = getattr(res, 'dt_scores', []) if hasattr(res, 'dt_scores') else []
                
                if dt_polys is None or len(dt_polys) == 0:
                    continue
                
                for idx, poly in enumerate(dt_polys):
                    points = np.array(poly).astype(np.int32)
                    x0 = max(0, int(np.min(points[:, 0])))
                    y0 = max(0, int(np.min(points[:, 1])))
                    x1 = min(img_width, int(np.max(points[:, 0])))
                    y1 = min(img_height, int(np.max(points[:, 1])))
                    
                    if x1 <= x0 or y1 <= y0:
                        continue
                    
                    det_score = float(dt_scores[idx]) if idx < len(dt_scores) else 0.9
                    
                    # Crop the detected text region
                    crop = img[y0:y1, x0:x1]
                    
                    if crop.size == 0:
                        continue
                    
                    detected_boxes.append(([x0, y0, x1, y1], det_score))
                    crops_for_recognition.append(crop)
            
            if not crops_for_recognition:
                return results
            
            # Step 2: Batch Text Recognition on all crops
            try:
                rec_output = self.recognizer.predict(input=crops_for_recognition, batch_size=len(crops_for_recognition))
                
                # Parse recognition results
                rec_texts = []
                rec_scores = []
                
                for rec_res in rec_output:
                    texts = None
                    scores = None
                    
                    # TextRecResult is dict-like, use .get() to access rec_text/rec_texts
                    if hasattr(rec_res, 'get'):
                        texts = rec_res.get('rec_text', rec_res.get('rec_texts', None))
                        scores = rec_res.get('rec_score', rec_res.get('rec_scores', None))
                    
                    # Fallback: try direct attribute access
                    if texts is None:
                        for attr in ['rec_text', 'rec_texts', 'text', 'texts']:
                            if hasattr(rec_res, attr):
                                texts = getattr(rec_res, attr)
                                break
                    
                    if scores is None:
                        for attr in ['rec_score', 'rec_scores', 'score', 'scores']:
                            if hasattr(rec_res, attr):
                                scores = getattr(rec_res, attr)
                                break
                    
                    # Handle the results
                    if texts is not None:
                        if isinstance(texts, list):
                            rec_texts.extend(texts)
                        elif isinstance(texts, str):
                            rec_texts.append(texts)
                        else:
                            rec_texts.append(str(texts))
                    
                    if scores is not None:
                        if isinstance(scores, list):
                            rec_scores.extend(scores)
                        else:
                            rec_scores.append(float(scores) if scores else 0.9)
                
                # Match recognition results with detected boxes
                for i, (bbox, det_score) in enumerate(detected_boxes):
                    text = rec_texts[i] if i < len(rec_texts) else ""
                    if not isinstance(text, str):
                        text = str(text)
                    score = float(rec_scores[i]) if i < len(rec_scores) else det_score
                    results.append((bbox, text, score))
                    
            except Exception as e:
                logger.error(f"Batch recognition failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback: add boxes with empty text
                for bbox, det_score in detected_boxes:
                    results.append((bbox, "", det_score))
        
        except Exception as e:
            logger.error(f"Error in PaddleOCR: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results
    
    def detect_on_crops(self, image_path: str, sections: List[LayoutSection]) -> List[LayoutSection]:
        """
        Stage 2: Run OCR on cropped sections to detect and recognize child text boxes.
        Masks overlapping sections with different labels to avoid duplicate detection.
        """
        logger.info(f"[Stage 2] Running PaddleOCR on {len(sections)} cropped sections")
        
        image = Image.open(image_path)
        img_width, img_height = image.size
        converter = CoordinateConverter()
        
        total_children = 0
        
        for i, section in enumerate(sections):
            x0, y0, x1, y1 = section.bbox_pixels
            
            # Crop the section
            cropped = image.crop((x0, y0, x1, y1)).copy()
            crop_width, crop_height = cropped.size
            
            if crop_width < 10 or crop_height < 10:
                continue
            
            # Mask overlapping sections with different labels
            # Only mask SMALLER sections within LARGER sections (not bidirectional)
            # This ensures text in overlap zones is detected by the larger section
            draw = ImageDraw.Draw(cropped)
            current_area = converter.bbox_area(section.bbox_pixels)
            for j, other in enumerate(sections):
                if i == j:
                    continue
                if other.label == section.label:
                    continue  # Same label sections should have been merged
                
                # Only mask if other section is SMALLER than current section
                other_area = converter.bbox_area(other.bbox_pixels)
                if other_area >= current_area:
                    continue  # Don't mask larger or equal sections
                
                # Check if other section overlaps with current
                if converter.overlaps_with(section.bbox_pixels, other.bbox_pixels):
                    # Calculate the overlap region in crop-relative coordinates
                    ox0, oy0, ox1, oy1 = other.bbox_pixels
                    # Convert to section-relative coordinates
                    rel_x0 = max(0, ox0 - x0)
                    rel_y0 = max(0, oy0 - y0)
                    rel_x1 = min(crop_width, ox1 - x0)
                    rel_y1 = min(crop_height, oy1 - y0)
                    
                    if rel_x1 > rel_x0 and rel_y1 > rel_y0:
                        draw.rectangle([rel_x0, rel_y0, rel_x1, rel_y1], fill='white')
                        logger.debug(f"  Masked {other.label} area in {section.label} section")
            
            # Convert to RGB numpy array (must be 3-channel)
            cropped_rgb = cropped.convert('RGB')
            crop_array = np.array(cropped_rgb)
            
            # Run OCR on cropped region
            crop_results = self._run_ocr_on_image(crop_array)
            
            for bbox, text, conf in crop_results:
                # Convert crop-relative coordinates to full image coordinates
                full_bbox = converter.crop_to_full_coords(bbox, (x0, y0))
                
                if not converter.validate_bbox(full_bbox, min_size=3):
                    continue
                
                # Normalize relative to SECTION (not full image)
                # bbox is section-relative, normalize it to 0-1024 range based on section size
                bbox_normalized = converter.normalize_bbox(bbox, crop_width, crop_height)
                
                text_box = TextBox(
                    id=5000 + total_children + 1,  # Child IDs: 5001, 5002, ...
                    bbox_pixels=full_bbox,
                    bbox_normalized=bbox_normalized,  # Normalized relative to section
                    bbox_section_relative=bbox,  # Original crop-relative coords (pixels)
                    confidence=conf,
                    ocr_text_paddle=text,  # Original PaddleOCR text
                    ocr_text=text  # Will be refined by Qwen later
                )
                section.children.append(text_box)
                total_children += 1
        
        logger.info(f"[Stage 2A] Detected and recognized {total_children} text lines from cropped sections")
        return sections
    
    def detect_masked(self, image_path: str, sections: List[LayoutSection], output_folder: Path = None) -> List[TextBox]:
        """
        Stage 2B: Run OCR on full image with sections masked (white-filled).
        Returns orphan text boxes found outside the masked regions.
        
        Args:
            image_path: Path to the original image
            sections: List of detected sections to mask
            output_folder: Optional folder to save intermediate results (masked image, orphan boxes)
        """
        logger.info(f"[Stage 2B] Running PaddleOCR on masked full image")
        
        image = Image.open(image_path).copy()
        img_width, img_height = image.size
        draw = ImageDraw.Draw(image)
        converter = CoordinateConverter()
        
        # Mask all section regions with white
        for section in sections:
            x0, y0, x1, y1 = section.bbox_pixels
            draw.rectangle([x0, y0, x1, y1], fill='white')
        
        # Save masked image if output folder provided
        if output_folder:
            masked_image_path = output_folder / "stage2b_masked_image.png"
            image.save(str(masked_image_path), 'PNG')
            logger.info(f"[Stage 2B] Saved masked image: {masked_image_path}")
        
        # Convert to RGB numpy array (must be 3-channel)
        image_rgb = image.convert('RGB')
        masked_array = np.array(image_rgb)
        
        # Run OCR on masked image
        mask_results = self._run_ocr_on_image(masked_array)
        
        orphan_boxes = []
        for bbox, text, conf in mask_results:
            if not converter.validate_bbox(bbox, min_size=3):
                continue
            
            bbox_normalized = converter.normalize_bbox(bbox, img_width, img_height)
            
            # Orphan boxes get section-relative coords computed during merge
            text_box = TextBox(
                id=2000 + len(orphan_boxes) + 1,  # Unique int ID (offset for orphans)
                bbox_pixels=bbox,
                bbox_normalized=bbox_normalized,
                bbox_section_relative=[0, 0, 0, 0],  # Will be computed during merge
                confidence=conf,
                ocr_text_paddle=text,  # Original PaddleOCR text
                ocr_text=text  # Will be refined by Qwen later
            )
            orphan_boxes.append(text_box)
        
        # Save orphan boxes JSON if output folder provided
        if output_folder:
            orphan_json_path = output_folder / "stage2b_orphan_boxes.json"
            orphan_data = {
                "total_orphans": len(orphan_boxes),
                "orphan_boxes": [box.to_dict() for box in orphan_boxes]
            }
            with open(orphan_json_path, 'w', encoding='utf-8') as f:
                json.dump(orphan_data, f, indent=2, ensure_ascii=False)
            logger.info(f"[Stage 2B] Saved orphan boxes: {orphan_json_path}")
        
        logger.info(f"[Stage 2B] Detected and recognized {len(orphan_boxes)} orphan text lines")
        return orphan_boxes
    
    def merge_orphans(self, sections: List[LayoutSection], orphan_boxes: List[TextBox], img_width: int = None, img_height: int = None) -> List[LayoutSection]:
        """
        Stage 2C: Merge orphan boxes into sections.
        - Boxes fully contained in a section are added as children
        - Boxes NOT contained are promoted to new "Text" parent sections
        Also computes section-relative coordinates for all boxes.
        """
        logger.info(f"[Stage 2C] Merging {len(orphan_boxes)} orphan boxes")
        
        converter = CoordinateConverter()
        assigned_count = 0
        promoted_count = 0
        
        # Track the highest section ID to create unique IDs for new sections
        max_section_id = max([s.id for s in sections], default=0) if sections else 0
        
        for text_box in orphan_boxes:
            assigned = False
            for section in sections:
                if converter.is_contained(text_box.bbox_pixels, section.bbox_pixels):
                    # Compute section-relative coordinates
                    sx0, sy0 = section.bbox_pixels[0], section.bbox_pixels[1]
                    tx0, ty0, tx1, ty1 = text_box.bbox_pixels
                    text_box.bbox_section_relative = [tx0 - sx0, ty0 - sy0, tx1 - sx0, ty1 - sy0]
                    section.children.append(text_box)
                    assigned = True
                    assigned_count += 1
                    break
            
            if not assigned:
                # Promote orphan to new "Text" section
                max_section_id += 1
                bbox = text_box.bbox_pixels
                
                # Section-relative coords are [0, 0, width, height] since box is the entire section
                text_box.bbox_section_relative = [0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]]
                
                # Compute normalized bbox (if dimensions provided)
                if img_width and img_height:
                    bbox_normalized = converter.normalize_bbox(bbox, img_width, img_height)
                else:
                    bbox_normalized = text_box.bbox_normalized
                
                # Create new section
                new_section = LayoutSection(
                    id=max_section_id,
                    label="Text",
                    bbox_pixels=bbox,
                    bbox_normalized=bbox_normalized,
                    confidence=text_box.confidence,
                    children=[text_box],
                    raw_text=text_box.ocr_text_paddle
                )
                sections.append(new_section)
                promoted_count += 1
        
        # Sort children by position (top to bottom, left to right)
        for section in sections:
            section.children.sort(key=lambda c: (c.bbox_pixels[1], c.bbox_pixels[0]))
            # Combine original PaddleOCR text from children
            section.raw_text = " ".join([child.ocr_text_paddle for child in section.children])
        
        # Sort sections by position
        sections.sort(key=lambda s: (s.bbox_pixels[1], s.bbox_pixels[0]))
        
        logger.info(f"[Stage 2C] Assigned {assigned_count} orphans to existing sections, promoted {promoted_count} to new 'Text' sections")
        return sections
