# Stage 1: DocLayout-YOLO Detection
"""
DocLayout-YOLO for document layout section detection.
Detects sections like Text, Figure, Table, Caption, etc.
"""

import logging
from pathlib import Path
from typing import List

from PIL import Image

# DocLayout-YOLO for layout detection
try:
    from doclayout_yolo import YOLOv10
    DOCLAYOUT_AVAILABLE = True
except ImportError:
    DOCLAYOUT_AVAILABLE = False
    logging.warning("doclayout_yolo not available. Install with: pip install doclayout-yolo")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import LayoutSection
from utils.coordinate_utils import CoordinateConverter

logger = logging.getLogger(__name__)


class DocLayoutStage:
    """Stage 1: DocLayout-YOLO for section detection"""
    
    LABEL_MAPPING = {
        'title': 'Page-Header',
        'plain text': 'Text',
        'plain_text': 'Text',
        'text': 'Text',
        'abandon': 'Text',
        'figure': 'Figure',
        'figure_caption': 'Caption',
        'figure caption': 'Caption',
        'table': 'Table',
        'table_caption': 'Caption',
        'table caption': 'Caption',
        'table_footnote': 'Footnote',
        'table footnote': 'Footnote',
        'isolate_formula': 'Equation-Block',
        'isolate formula': 'Equation-Block',
        'formula_caption': 'Caption',
        'formula caption': 'Caption',
        'equation': 'Equation-Block',
        'header': 'Page-Header',
        'footer': 'Page-Footer',
        'page_header': 'Page-Header',
        'page_footer': 'Page-Footer',
        'section_header': 'Section-Header',
        'section header': 'Section-Header',
        'list': 'List-Group',
        'code': 'Code-Block',
    }
    
    def __init__(self, model_path: str, confidence: float = 0.2, use_gpu: bool = False):
        if not DOCLAYOUT_AVAILABLE:
            raise ImportError("doclayout_yolo not available")
        
        logger.info(f"Loading DocLayout-YOLO model: {model_path}")
        self.model = YOLOv10(model_path)
        self.confidence = confidence
        self.device = 'cuda:0' if use_gpu else 'cpu'
        logger.info("DocLayout-YOLO loaded successfully")
    
    def detect(self, image_path: str) -> List[LayoutSection]:
        """Detect layout sections in image"""
        logger.info(f"[Stage 1] Running DocLayout on: {image_path}")
        
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        det_res = self.model.predict(
            image_path,
            imgsz=1024,
            conf=self.confidence,
            device=self.device
        )
        
        sections = []
        converter = CoordinateConverter()
        
        if not det_res or len(det_res) == 0:
            logger.warning("DocLayout returned no results")
            return sections
        
        result = det_res[0]
        
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            logger.warning("DocLayout detected 0 boxes")
            return sections
        
        boxes = result.boxes
        logger.info(f"DocLayout detected {len(boxes)} boxes")
        
        for i in range(len(boxes)):
            try:
                box_xyxy = boxes.xyxy[i].cpu().numpy()
                x0, y0, x1, y1 = map(int, box_xyxy)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = result.names[cls_id] if hasattr(result, 'names') else str(cls_id)
                
                label = self.LABEL_MAPPING.get(class_name.lower(), 'Text')
                bbox_pixels = [x0, y0, x1, y1]
                
                if not converter.validate_bbox(bbox_pixels):
                    continue
                
                bbox_normalized = converter.normalize_bbox(bbox_pixels, img_width, img_height)
                
                section = LayoutSection(
                    id=len(sections) + 1,  # Unique int ID
                    label=label,
                    bbox_pixels=bbox_pixels,
                    bbox_normalized=bbox_normalized,
                    confidence=conf
                )
                sections.append(section)
                
            except Exception as e:
                logger.error(f"Error processing box {i}: {e}")
                continue
        
        logger.info(f"[Stage 1] Detected {len(sections)} sections")
        return sections
