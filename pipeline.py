# OCR Pipeline V2 Main Orchestrator
"""
Main OCR Pipeline V2 class that orchestrates all stages:
- Stage 1: DocLayout-YOLO detection + merging
- Stage 2: PaddleOCR text detection (optional)
- Stage 3: Qwen3-VL Vision LLM OCR refinement
- Stage 4: Summary generation

Supports both image files and PDF documents.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

from config import CONFIG, logger
from data_models import TextBox, LayoutSection, PageResult
from stages import DocLayoutStage, EnhancedPaddleOCRStage, VisionOCRStage
from utils.coordinate_utils import CoordinateConverter
from utils.visualization import save_visualization, save_stage1_visualization
from utils.pdf_utils import is_pdf, convert_pdf_to_images, PDF2IMAGE_AVAILABLE


class OCRPipelineV2:
    """Main OCR Pipeline V2: DocLayout + PaddleOCR + Vision LLM (Qwen/Gemini)
    
    Pipeline flow:
    - Stage 1: DocLayout-YOLO for section detection + orphan detection + merging
    - Stage 2 (if enabled): PaddleOCR text detection on section crops
    - Stage 3: Vision LLM OCR refinement (batched, Qwen or Gemini)
    - Stage 4: Summary generation
    
    Supports:
    - Single images (PNG, JPG, etc.)
    - PDF documents (converted to images)
    - Folders containing images and/or PDFs
    """
    
    def __init__(self, enable_stage2: bool = None):
        """Initialize pipeline with config
        
        Args:
            enable_stage2: Enable PaddleOCR text detection (if None, uses CONFIG['enable_stage2'])
        """
        # Use config value if not explicitly specified
        self.enable_stage2 = enable_stage2 if enable_stage2 is not None else CONFIG.get('enable_stage2', True)
        
        # Stage 1: DocLayout
        self.doclayout = DocLayoutStage(
            model_path=CONFIG['doclayout_model_path'],
            confidence=CONFIG['doclayout_confidence'],
            use_gpu=CONFIG['use_gpu']
        )
        
        # Stage 2: PaddleOCR text detection
        if self.enable_stage2:
            self.paddleocr = EnhancedPaddleOCRStage(
                det_model=CONFIG.get('paddleocr_det_model', 'PP-OCRv5_server_det'),
                rec_model=CONFIG.get('paddleocr_rec_model', 'PP-OCRv5_server_rec')
            )
        else:
            self.paddleocr = None
            logger.info("Stage 2 disabled - using DocLayout sections only")
        
        # Stage 3: Vision LLM OCR (Qwen or Gemini)
        self.vision_ocr = VisionOCRStage(CONFIG)
        
        logger.info("OCR Pipeline V2 initialized successfully")
    
    def process_pdf(self, pdf_path: str, output_folder: str, dpi: int = 300) -> Dict:
        """Process a PDF document by converting pages to images and processing each.
        
        Args:
            pdf_path: Path to the PDF file
            output_folder: Folder to save results
            dpi: DPI for PDF to image conversion (default 300)
        
        Returns:
            Summary dictionary with all page results
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image is required for PDF processing. "
                "Install with: pip install pdf2image"
            )
        
        pdf_path = Path(pdf_path)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OCR PIPELINE V2 - PDF PROCESSING")
        logger.info(f"PDF: {pdf_path}")
        logger.info(f"Output: {output_folder}")
        logger.info(f"DPI: {dpi}")
        logger.info(f"{'='*80}\n")
        
        # Create folder for converted images
        images_folder = output_path / "pdf_pages"
        images_folder.mkdir(exist_ok=True, parents=True)
        
        # Convert PDF to images
        logger.info("Converting PDF pages to images...")
        image_paths = convert_pdf_to_images(str(pdf_path), images_folder, dpi=dpi)
        logger.info(f"Converted {len(image_paths)} pages\n")
        
        # Process each page
        all_results = []
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"\n[Page {idx}/{len(image_paths)}] {Path(image_path).name}")
            
            try:
                # Create per-page output folder
                page_output_folder = output_path / f"page_{idx:04d}"
                page_output_folder.mkdir(exist_ok=True, parents=True)
                
                # Copy page image to output folder
                shutil.copy(image_path, page_output_folder / "original.png")
                
                # Process page
                result = self.process_page(image_path, output_folder=page_output_folder)
                all_results.append(result)
                
                # Save final JSON result
                json_output = page_output_folder / "final_result.json"
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"✓ Saved JSON: {json_output}")
                
                # Save visualization
                vis_output = page_output_folder / "visualization.png"
                save_visualization(result, vis_output)
                logger.info(f"✓ Saved visualization: {vis_output}")
                
            except Exception as e:
                logger.error(f"✗ Failed page {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save PDF summary
        summary = {
            'metadata': {
                'pdf_path': str(pdf_path),
                'output_folder': str(output_folder),
                'processing_timestamp': datetime.now().isoformat(),
                'dpi': dpi,
                'total_pages': len(image_paths),
                'processed': len(all_results)
            },
            'pages': [r.to_dict() for r in all_results]
        }
        
        summary_path = output_path / "pdf_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETE: {len(all_results)}/{len(image_paths)} pages processed")
        logger.info(f"Summary: {summary_path}")
        logger.info(f"{'='*80}\n")
        
        return summary

    
    def process_page(self, image_path: str, output_folder: Path = None) -> PageResult:
        """Process a single page through all stages
        
        Args:
            image_path: Path to the image file
            output_folder: Optional folder to save intermediate results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {image_path}")
        logger.info(f"{'='*60}")
        
        image = Image.open(image_path)
        img_width, img_height = image.size
        converter = CoordinateConverter()
        
        # =====================================================================
        # STAGE 1: Section Detection + Merging (all parent sections)
        # =====================================================================
        
        # Stage 1A: DocLayout detection
        logger.info("[Stage 1A] DocLayout section detection")
        sections = self.doclayout.detect(image_path)
        logger.info(f"[Stage 1A] Detected {len(sections)} sections")
        
        # Stage 1B: Detect orphan boxes on masked image (PaddleOCR on page level)
        if self.enable_stage2 and self.paddleocr:
            logger.info("[Stage 1B] Detecting orphan boxes on masked page")
            
            # Create masked image (mask DocLayout sections)
            masked_image = image.copy()
            draw = ImageDraw.Draw(masked_image)
            for section in sections:
                x0, y0, x1, y1 = section.bbox_pixels
                draw.rectangle([x0, y0, x1, y1], fill='white')
            
            # Save masked image if output folder provided
            if output_folder:
                masked_path = output_folder / "stage1b_masked_image.png"
                masked_image.save(str(masked_path), 'PNG')
                logger.info(f"[Stage 1B] Saved masked image: {masked_path}")
            
            # Run PaddleOCR on masked image
            masked_rgb = masked_image.convert('RGB')
            masked_array = np.array(masked_rgb)
            orphan_results = self.paddleocr._run_ocr_on_image(masked_array)
            
            # Convert orphan TextBoxes to LayoutSections (they become parent sections)
            orphan_sections = []
            for bbox, text, conf in orphan_results:
                if not converter.validate_bbox(bbox, min_size=3):
                    continue
                
                bbox_normalized = converter.normalize_bbox(bbox, img_width, img_height)
                
                orphan_section = LayoutSection(
                    id=500 + len(orphan_sections) + 1,  # Orphan IDs: 501, 502, ...
                    label="Text",  # Orphans are labeled as Text
                    bbox_pixels=bbox,
                    bbox_normalized=bbox_normalized,
                    confidence=conf,
                    raw_text=text
                )
                orphan_sections.append(orphan_section)
            
            logger.info(f"[Stage 1B] Detected {len(orphan_sections)} orphan sections")
            
            # Combine DocLayout sections + orphan sections
            sections = sections + orphan_sections
            
            # Save orphan results if output folder provided
            if output_folder:
                orphan_json = output_folder / "stage1b_orphan_sections.json"
                orphan_data = {
                    "total_orphans": len(orphan_sections),
                    "orphan_sections": [s.to_dict() for s in orphan_sections]
                }
                with open(orphan_json, 'w', encoding='utf-8') as f:
                    json.dump(orphan_data, f, indent=2, ensure_ascii=False)
        
        # Stage 1C: Merge overlapping sections (all become parent sections, no children)
        logger.info(f"[Stage 1C] Merging overlapping sections ({len(sections)} input)")
        sections = self._merge_overlapping_sections(sections, img_width, img_height)
        
        # Save Stage 1 final result if output folder provided
        if output_folder:
            stage1_json = output_folder / "stage1_sections.json"
            stage1_data = {
                "total_sections": len(sections),
                "sections": [s.to_dict() for s in sections]
            }
            with open(stage1_json, 'w', encoding='utf-8') as f:
                json.dump(stage1_data, f, indent=2, ensure_ascii=False)
            logger.info(f"[Stage 1] Saved sections: {stage1_json}")
            
            # Save Stage 1 visualization
            stage1_vis = output_folder / "stage1_visualization.png"
            save_stage1_visualization(image_path, sections, stage1_vis)
            logger.info(f"[Stage 1] Saved visualization: {stage1_vis}")
        
        # =====================================================================
        # STAGE 2: PaddleOCR on sections (creates children)
        # =====================================================================
        
        stage2_ran = False
        if self.enable_stage2 and self.paddleocr:
            logger.info(f"[Stage 2] Running PaddleOCR on {len(sections)} sections to detect text lines")
            
            # Clear any existing children from Stage 1 merging (shouldn't have any now)
            for section in sections:
                section.children = []
            
            # Run PaddleOCR on each section's crop to detect children
            sections = self.paddleocr.detect_on_crops(image_path, sections)
            
            # Save Stage 2 results if output folder provided
            if output_folder:
                stage2_json = output_folder / "stage2_paddleocr.json"
                stage2_data = {
                    "total_sections": len(sections),
                    "sections": [s.to_dict() for s in sections]
                }
                with open(stage2_json, 'w', encoding='utf-8') as f:
                    json.dump(stage2_data, f, indent=2, ensure_ascii=False)
                logger.info(f"[Stage 2] Saved: {stage2_json}")
            
            stage2_ran = True
        else:
            logger.info("[Stage 2] Skipped - using DocLayout sections only")
        
        # =====================================================================
        # STAGE 3: Vision LLM OCR Refinement (Qwen or Gemini)
        # =====================================================================
        
        sections = self.vision_ocr.run_ocr(image_path, sections)
        
        # =====================================================================
        # STAGE 4: Generate Image-Level Summary
        # =====================================================================
        
        summary = self.vision_ocr.generate_summary(image_path, sections)
        
        # Create result
        result = PageResult(
            image_path=image_path,
            image_name=Path(image_path).name,
            width=img_width,
            height=img_height,
            sections=sections,
            processing_timestamp=datetime.now().isoformat(),
            summary=summary
        )
        
        logger.info(f"✓ Processed {len(sections)} sections with summary")
        return result
    
    def _merge_overlapping_sections(self, sections: List[LayoutSection], img_width: int, img_height: int) -> List[LayoutSection]:
        """
        Stage 1B: Merge overlapping DocLayout sections.
        
        - Same label + IOU >= 80%: Merge into single section (union of bboxes)
        - Different labels + any overlap: Smaller section becomes child of larger (masked)
        """
        if len(sections) <= 1:
            return sections
        
        logger.info(f"[Stage 1B] Merging overlapping sections ({len(sections)} input)")
        
        converter = CoordinateConverter()
        IOU_THRESHOLD = 0.8
        
        # Sort sections by area (largest first) - this ensures larger sections absorb smaller ones
        sections = sorted(sections, key=lambda s: converter.bbox_area(s.bbox_pixels), reverse=True)
        
        # Keep iterating until no more merges happen
        merged_anything = True
        iteration = 0
        
        while merged_anything:
            iteration += 1
            merged_anything = False
            new_sections = []
            absorbed = set()  # Track indices that have been absorbed as children
            
            logger.debug(f"[Stage 1B] Iteration {iteration}, processing {len(sections)} sections")
            
            for i in range(len(sections)):
                if i in absorbed:
                    continue
                
                current = sections[i]
                current_merged = False  # Track if current was merged this iteration
                
                for j in range(i + 1, len(sections)):
                    if j in absorbed:
                        continue
                    
                    other = sections[j]
                    iou_val = converter.iou(current.bbox_pixels, other.bbox_pixels)
                    
                    # Check if one box is contained in the other (full containment)
                    other_in_current = converter.is_contained(other.bbox_pixels, current.bbox_pixels)
                    current_in_other = converter.is_contained(current.bbox_pixels, other.bbox_pixels)
                    
                    # Check if intersection covers most of the smaller box (near-complete overlap)
                    # This catches cases where containment fails due to 1-2 pixel differences
                    intersection = converter.intersection_area(current.bbox_pixels, other.bbox_pixels)
                    smaller_area = min(converter.bbox_area(current.bbox_pixels), converter.bbox_area(other.bbox_pixels))
                    mostly_contained = (smaller_area > 0) and (intersection >= 0.8 * smaller_area)
                    
                    # Same label: merge if high IOU OR one is contained/mostly contained in other
                    if current.label == other.label and (iou_val >= IOU_THRESHOLD or other_in_current or current_in_other or mostly_contained):
                        # Merge bboxes (union) - always merge even if contained to ensure proper bounds
                        merged_bbox = converter.merge_bboxes(current.bbox_pixels, other.bbox_pixels)
                        normalized = converter.normalize_bbox(merged_bbox, img_width, img_height)
                        
                        # Merge children from both sections
                        merged_children = current.children + other.children
                        
                        # Create merged section (reuse current's ID)
                        current = LayoutSection(
                            id=current.id,
                            label=current.label,
                            bbox_pixels=merged_bbox,
                            bbox_normalized=normalized,
                            confidence=max(current.confidence, other.confidence),
                            children=merged_children
                        )
                        
                        absorbed.add(j)
                        merged_anything = True
                        current_merged = True
                        logger.debug(f"  Merged: {current.label} sections (IOU={iou_val:.2f}, contained={other_in_current or current_in_other})")
                    
                    # Different labels but VERY high IOU (nearly identical bbox): keep higher confidence
                    elif current.label != other.label and (iou_val >= 0.9 or mostly_contained):
                        # These are duplicate detections of the same region with different labels
                        # Keep the one with higher confidence, absorb the other
                        absorbed.add(j)  # other is absorbed (current already has >= area)
                        merged_anything = True
                        logger.debug(f"  Duplicate region: {other.label} (conf={other.confidence:.2f}) absorbed by {current.label} (conf={current.confidence:.2f})")
                    
                    # Different labels with partial overlap: both remain independent
                    # The overlap will be handled in Stage 2 by masking when running PaddleOCR
                
                # Add current section to new list (it may have been modified)
                new_sections.append(current)
            
            # Update sections list for next iteration
            sections = new_sections
            logger.debug(f"[Stage 1B] After iteration {iteration}: {len(sections)} sections remain")
            
            # Safety check: prevent infinite loops
            if iteration > 10:
                logger.warning("[Stage 1B] Breaking after 10 iterations to prevent infinite loop")
                break
        
        # Re-assign IDs sequentially and sort by position
        for idx, section in enumerate(sections):
            section.id = idx + 1
        
        # Sort sections by position (top to bottom, left to right)
        sections.sort(key=lambda s: (s.bbox_pixels[1], s.bbox_pixels[0]))
        
        logger.info(f"[Stage 1B] After merging: {len(sections)} sections (completed in {iteration} iterations)")
        return sections
    
    def _detect_overlapping_sections(self, sections: List[LayoutSection], img_width: int, img_height: int) -> List[LayoutSection]:
        """
        Detect overlapping DocLayout sections and treat smaller ones as children of larger ones.
        This maintains parent-child masking when PaddleOCR is disabled.
        """
        if len(sections) <= 1:
            return sections
        
        converter = CoordinateConverter()
        
        # Sort sections by area (largest first)
        sections_by_area = sorted(sections, key=lambda s: converter.bbox_area(s.bbox_pixels), reverse=True)
        
        # Track which sections become children (to be removed from main list)
        child_section_indices = set()
        
        # For each pair of sections, check if smaller is contained in or overlaps with larger
        for i, larger in enumerate(sections_by_area):
            if i in child_section_indices:
                continue
            
            for j, smaller in enumerate(sections_by_area):
                if i == j or j in child_section_indices:
                    continue
                
                # Check if smaller is fully or partially inside larger
                if converter.is_contained(smaller.bbox_pixels, larger.bbox_pixels):
                    # Smaller is fully inside larger - treat as child
                    child_text_box = TextBox(
                        id=3000 + len(larger.children) + 1,  # Unique ID for section-children
                        bbox_pixels=smaller.bbox_pixels,
                        bbox_normalized=smaller.bbox_normalized,
                        bbox_section_relative=[0, 0, 0, 0],
                        confidence=smaller.confidence,
                        ocr_text_paddle="",
                        ocr_text=""
                    )
                    larger.children.append(child_text_box)
                    child_section_indices.add(j)
                    logger.info(f"  Child relationship: '{smaller.label}' → child of '{larger.label}'")
                
                elif converter.overlaps_with(smaller.bbox_pixels, larger.bbox_pixels):
                    # Partial overlap - only treat as child if smaller area is significantly smaller
                    larger_area = converter.bbox_area(larger.bbox_pixels)
                    smaller_area = converter.bbox_area(smaller.bbox_pixels)
                    
                    if smaller_area < larger_area * 0.5:  # Smaller is at least 50% smaller
                        child_text_box = TextBox(
                            id=3000 + len(larger.children) + 1,  # Unique ID for section-children
                            bbox_pixels=smaller.bbox_pixels,
                            bbox_normalized=smaller.bbox_normalized,
                            bbox_section_relative=[0, 0, 0, 0],
                            confidence=smaller.confidence,
                            ocr_text_paddle="",
                            ocr_text=""
                        )
                        larger.children.append(child_text_box)
                        child_section_indices.add(j)
                        logger.info(f"  Overlap relationship: '{smaller.label}' → child of '{larger.label}'")
        
        # Filter out sections that became children
        remaining_sections = [s for i, s in enumerate(sections_by_area) if i not in child_section_indices]
        
        # Sort children by position
        for section in remaining_sections:
            section.children.sort(key=lambda c: (c.bbox_pixels[1], c.bbox_pixels[0]))
        
        logger.info(f"[Stage 2] Found {len(child_section_indices)} child sections, {len(remaining_sections)} parent sections remain")
        return remaining_sections
    
    def process_folder(self, input_folder: str, output_folder: str, pdf_dpi: int = 300) -> Dict:
        """Process all images and PDFs in a folder
        
        Args:
            input_folder: Path to folder containing images/PDFs
            output_folder: Path to save results
            pdf_dpi: DPI for PDF to image conversion (default 300)
        
        Returns:
            Summary dictionary with all results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"OCR PIPELINE V2 - BATCH PROCESSING")
        logger.info(f"Input: {input_folder}")
        logger.info(f"Output: {output_folder}")
        logger.info(f"{'='*80}\n")
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        # Get PDF files
        pdf_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() == '.pdf' and f.is_file()
        ])
        
        total_files = len(image_files) + len(pdf_files)
        
        if total_files == 0:
            logger.error(f"No images or PDFs found in: {input_folder}")
            return {}
        
        logger.info(f"Found {len(image_files)} images and {len(pdf_files)} PDFs\n")
        
        all_results = []
        file_idx = 0
        
        # Process images
        for image_file in image_files:
            file_idx += 1
            logger.info(f"\n[{file_idx}/{total_files}] {image_file.name}")
            
            try:
                # Create per-image output folder
                image_output_folder = output_path / image_file.stem
                image_output_folder.mkdir(exist_ok=True, parents=True)
                
                # Copy original image to output folder
                shutil.copy(str(image_file), image_output_folder / "original.png")
                
                # Process page with output folder for intermediate results
                result = self.process_page(str(image_file), output_folder=image_output_folder)
                all_results.append(result)
                
                # Save final JSON result
                json_output = image_output_folder / "final_result.json"
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"✓ Saved JSON: {json_output}")
                
                # Save visualization
                vis_output = image_output_folder / "visualization.png"
                save_visualization(result, vis_output)
                logger.info(f"✓ Saved visualization: {vis_output}")
                
            except Exception as e:
                logger.error(f"✗ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Process PDFs
        for pdf_file in pdf_files:
            file_idx += 1
            logger.info(f"\n[{file_idx}/{total_files}] {pdf_file.name} (PDF)")
            
            try:
                # Create per-PDF output folder
                pdf_output_folder = output_path / pdf_file.stem
                pdf_output_folder.mkdir(exist_ok=True, parents=True)
                
                # Process PDF (this handles page-by-page internally)
                pdf_summary = self.process_pdf(str(pdf_file), str(pdf_output_folder), dpi=pdf_dpi)
                
                # Add page results to all_results
                if 'pages' in pdf_summary:
                    for page_data in pdf_summary['pages']:
                        # Reconstruct PageResult from dict (simplified - just store dict)
                        all_results.append(page_data)
                
            except Exception as e:
                logger.error(f"✗ Failed PDF: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary
        summary = {
            'metadata': {
                'input_folder': str(input_folder),
                'output_folder': str(output_folder),
                'processing_timestamp': datetime.now().isoformat(),
                'total_images': len(image_files),
                'total_pdfs': len(pdf_files),
                'processed': len(all_results)
            },
            'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in all_results]
        }
        
        summary_path = output_path / "pipeline_v2_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETE: {len(all_results)} pages processed from {total_files} files")
        logger.info(f"Summary: {summary_path}")
        logger.info(f"{'='*80}\n")
        
        return summary

