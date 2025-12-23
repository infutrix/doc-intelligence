# Stages package
"""
Pipeline stages for OCR processing.
- Stage 1: DocLayout-YOLO detection
- Stage 2: PaddleOCR text detection/recognition
- Stage 3: Qwen3-VL Vision LLM OCR refinement
- Stage 4: Summary generation
"""

from .stage1_doclayout import DocLayoutStage
from .stage2_paddleocr import EnhancedPaddleOCRStage
from .stage3_vision_llm import VisionOCRStage

__all__ = ['DocLayoutStage', 'EnhancedPaddleOCRStage', 'VisionOCRStage']
