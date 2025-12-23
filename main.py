# OCR Pipeline V2 - Main Entry Point
"""
Main entry point for the OCR Pipeline V2.
Provides CLI interface for processing single images, PDFs, or folders.

Usage:
    python main.py --input <image_or_pdf_or_folder> --output <output_folder>
    python main.py --input ./images --output ./output
    python main.py --input ./image.png --output ./output
    python main.py --input ./document.pdf --output ./output
"""

import argparse
import sys
from pathlib import Path

from config import CONFIG, logger
from pipeline import OCRPipelineV2
from utils.pdf_utils import is_pdf, PDF2IMAGE_AVAILABLE


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline V2 - DocLayout + PaddleOCR + Qwen3-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a folder of images:
    python main.py --input ./images --output ./output
    
  Process a single image:
    python main.py --input ./image.png --output ./output
    
  Process a PDF file:
    python main.py --input ./document.pdf --output ./output
    
  Disable PaddleOCR stage:
    python main.py --input ./images --output ./output --no-stage2
    
  Set PDF DPI (default 300):
    python main.py --input ./document.pdf --output ./output --dpi 200
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image file, PDF file, or folder path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output folder path'
    )
    
    parser.add_argument(
        '--no-stage2',
        action='store_true',
        help='Disable PaddleOCR stage (use DocLayout sections only)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU for processing (default: True)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU, use CPU only'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF to image conversion (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Update config based on args
    if args.no_gpu:
        CONFIG['use_gpu'] = False
    
    # Determine if input is file or folder
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Create output folder
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize pipeline
    enable_stage2 = not args.no_stage2
    pipeline = OCRPipelineV2(enable_stage2=enable_stage2)
    
    # Process based on input type
    if input_path.is_file():
        if is_pdf(str(input_path)):
            # PDF file
            if not PDF2IMAGE_AVAILABLE:
                logger.error("pdf2image is required for PDF processing.")
                logger.error("Install with: pip install pdf2image")
                logger.error("Also install poppler: https://github.com/oschwartz10612/poppler-windows/releases")
                sys.exit(1)
            
            logger.info(f"Processing PDF: {input_path}")
            pipeline.process_pdf(str(input_path), str(output_path), dpi=args.dpi)
        else:
            # Single image
            logger.info(f"Processing single image: {input_path}")
            result = pipeline.process_page(str(input_path), output_folder=output_path)
            
            # Save result
            import json
            from utils.visualization import save_visualization
            
            json_output = output_path / "final_result.json"
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved JSON: {json_output}")
            
            vis_output = output_path / "visualization.png"
            save_visualization(result, vis_output)
            logger.info(f"✓ Saved visualization: {vis_output}")
        
    elif input_path.is_dir():
        # Folder of images (and/or PDFs)
        logger.info(f"Processing folder: {input_path}")
        pipeline.process_folder(str(input_path), str(output_path), pdf_dpi=args.dpi)
    
    else:
        logger.error(f"Invalid input path: {args.input}")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

