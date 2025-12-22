# PDF Utilities
"""
PDF to image conversion utilities for OCR Pipeline.
Uses pdf2image (poppler) to convert PDF pages to PIL Images.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# pdf2image for PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Install with: pip install pdf2image")
    logger.warning("Also requires poppler: https://github.com/oschwartz10612/poppler-windows/releases")


def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF based on extension"""
    return Path(file_path).suffix.lower() == '.pdf'


def convert_pdf_to_images(
    pdf_path: str,
    output_folder: Optional[Path] = None,
    dpi: int = 300,
    fmt: str = 'PNG'
) -> List[str]:
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save converted images. If None, uses temp folder.
        dpi: Resolution for conversion (default 300)
        fmt: Output format (PNG, JPEG, etc.)
    
    Returns:
        List of paths to converted image files
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install with: pip install pdf2image\n"
            "Also install poppler: https://github.com/oschwartz10612/poppler-windows/releases"
        )
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output folder
    if output_folder is None:
        output_folder = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Converting PDF to images: {pdf_path}")
    
    # Convert PDF pages to images
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt=fmt.lower()
        )
    except Exception as e:
        logger.error(f"Failed to convert PDF: {e}")
        raise
    
    # Save images to files
    image_paths = []
    pdf_stem = pdf_path.stem
    
    for i, image in enumerate(images):
        page_num = i + 1
        image_filename = f"{pdf_stem}_page_{page_num:04d}.{fmt.lower()}"
        image_path = output_folder / image_filename
        
        image.save(str(image_path), fmt)
        image_paths.append(str(image_path))
        logger.debug(f"  Saved page {page_num}: {image_path}")
    
    logger.info(f"Converted {len(images)} pages from PDF")
    return image_paths


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF file"""
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image is required for PDF processing")
    
    from pdf2image.pdf2image import pdfinfo_from_path
    
    try:
        info = pdfinfo_from_path(str(pdf_path))
        return info.get('Pages', 0)
    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        return 0
