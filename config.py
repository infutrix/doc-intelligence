# OCR Pipeline V2 Configuration
"""
Configuration settings for the OCR Pipeline V2.
Contains all model paths, parameters, and logging setup.
"""

import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory (where this config file is located)
BASE_DIR = Path(__file__).parent

CONFIG = {
    # DocLayout-YOLO settings
    "doclayout_model_path": str(BASE_DIR / "models" / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"),
    "doclayout_confidence": 0.2,
    
    # PaddleOCR settings
    "paddleocr_lang": "en",
    "paddleocr_det_model": "PP-OCRv5_server_det",
    "paddleocr_rec_model": "PP-OCRv5_server_rec",
    
    # GPU settings
    "use_gpu": True,
    
    # Stage 2: PaddleOCR text detection
    "enable_stage2": True,  # Set to False to skip PaddleOCR (uses DocLayout sections only)
    
    # Stage 3: Qwen3-VL OCR settings (local Transformers inference)
    "batch_size": 5,  # Number of sections per batch
    "qwen_model_name": "Qwen/Qwen3-VL-8B-Instruct",  # Model to download/load
    "ocr_max_tokens": 8192,
}

# Bounding box normalization range (0-1024)
BBOX_NORMALIZATION_RANGE = 1024

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup and return the logger for the pipeline."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    return logging.getLogger(__name__)

# Create default logger
logger = setup_logging()
