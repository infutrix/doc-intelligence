# Data Model Schemas
"""
Dataclasses for OCR Pipeline V2.
Contains TextBox, LayoutSection, and PageResult.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class TextBox:
    """Text box detected by PaddleOCR with styling attributes from TexTAR"""
    id: int  # Unique integer identifier (unique across all sections and children)
    bbox_pixels: List[int]  # [x0, y0, x1, y1] in full image coordinates
    bbox_normalized: List[int]  # [x0, y0, x1, y1] in 0-1024 range (full image)
    bbox_section_relative: List[int]  # [x0, y0, x1, y1] in section-relative coordinates
    confidence: float
    ocr_text_paddle: str = ""  # Original text from PaddleOCR
    ocr_text: str = ""  # Refined text from Qwen3-VL (or copy of paddle if not refined)
    # TexTAR styling attributes
    font_style: str = "normal"  # normal, bold, italic, bold_italic
    decoration: str = "normal"  # normal, underline, strikeout, underline_strikeout
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LayoutSection:
    """Parent section detected by DocLayout-YOLO with child text boxes"""
    id: int  # Unique integer identifier (unique across all sections and children)
    label: str
    bbox_pixels: List[int]  # [x0, y0, x1, y1]
    bbox_normalized: List[int]  # [x0, y0, x1, y1] in 0-1024 range
    confidence: float
    children: List[TextBox] = field(default_factory=list)
    raw_text: str = ""  # Concatenated PaddleOCR text from children
    ocr_text: str = ""  # Vision LLM corrected text for the section
    children_ocr_texts: List[str] = field(default_factory=list)  # OCR text for each child
    entities: List[Dict] = field(default_factory=list)  # Key-value entities extracted
    description: str = ""  # Section description from Vision LLM
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'bbox_pixels': self.bbox_pixels,
            'bbox_normalized': self.bbox_normalized,
            'confidence': self.confidence,
            'children': [child.to_dict() for child in self.children],
            'raw_text': self.raw_text,
            'ocr_text': self.ocr_text,
            'children_ocr_texts': self.children_ocr_texts,
            'entities': self.entities,
            'description': self.description
        }


@dataclass
class PageResult:
    """Complete result for a single page"""
    image_path: str
    image_name: str
    width: int
    height: int
    sections: List[LayoutSection]
    processing_timestamp: str
    summary: str = ""  # Image-level summary from Stage 4
    
    def to_dict(self) -> Dict:
        return {
            'image_path': self.image_path,
            'image_name': self.image_name,
            'dimensions': {'width': self.width, 'height': self.height},
            'sections': [section.to_dict() for section in self.sections],
            'processing_timestamp': self.processing_timestamp,
            'summary': self.summary
        }
