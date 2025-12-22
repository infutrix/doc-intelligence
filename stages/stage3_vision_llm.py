# Stage 3: Vision LLM OCR (Qwen3-VL)
"""
Qwen3-VL Vision LLM for OCR refinement and entity extraction.
Processes sections in batches using structured JSON input/output.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

from PIL import Image

# Qwen3-VL via Transformers (local inference)
try:
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    process_vision_info = None
    QWEN_AVAILABLE = False
    logging.warning("Qwen3-VL not available. Install with: pip install transformers qwen-vl-utils")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import LayoutSection

logger = logging.getLogger(__name__)


class VisionOCRStage:
    """Stage 3: Qwen3-VL OCR using local Transformers inference
    
    Processes sections in batches, using structured JSON input/output.
    Model is loaded once and kept in GPU memory.
    """
    
    def __init__(self, config: Dict):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen3-VL not available. Install with: pip install transformers qwen-vl-utils torch")
        
        self.config = config
        self.batch_size = config.get("batch_size", 5)
        self.max_tokens = config.get("ocr_max_tokens", 8192)
        self.model_name = config.get("qwen_model_name", "Qwen/Qwen3-VL-8B-Instruct")
        
        logger.info(f"Loading Qwen3-VL model: {self.model_name}")
        
        # Load model with flash attention for efficiency
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        logger.info("Qwen3-VL model loaded successfully")
    

    def _create_section_crop(self, image: Image.Image, section: LayoutSection) -> Image.Image:
        """Crop section from full image"""
        x0, y0, x1, y1 = section.bbox_pixels
        return image.crop((x0, y0, x1, y1))
    
    def _build_batch_input(self, sections: List[LayoutSection]) -> List[Dict]:
        """Build structured JSON input for a batch of sections"""
        batch_input = []
        for section in sections:
            section_data = {
                "section_id": section.id,
                "label": section.label,
                "children": []
            }
            for child in section.children:
                child_data = {
                    "id": child.id,
                    "ocr": child.ocr_text_paddle,
                    "bbox_wrt_Sec_norm": child.bbox_normalized
                }
                section_data["children"].append(child_data)
            batch_input.append(section_data)
        return batch_input
    
    def _build_batch_prompt(self, batch_input: List[Dict]) -> str:
        """Build the prompt for batch OCR processing"""
        prompt = """You are an OCR refinement and entity extraction assistant. You will receive multiple document section images along with their metadata.

For each section, you need to:
1. **OCR Text**: Provide the complete, accurate OCR text with line breaks preserved. Convert to Markdown format.
2. **Entities**: Extract key-value pairs from the text (e.g., names, dates, amounts, IDs).
3. **Children Update**: For each child text box, provide the refined/corrected OCR text.
4. **Description**: Provide a brief description of what the section contains.

## Section Metadata (JSON):
```json
""" + json.dumps(batch_input, indent=2) + """
```

## Instructions:
- Each section image is provided in order (section 1, section 2, etc.)
- Use the pre-detected children OCR text to help with accuracy
- The `bbox_wrt_Sec_norm` gives position within section (0-1024 range)
- Preserve original formatting: line breaks, bullet points, numbering
- For tables, use HTML `<table>` format
- For equations, use LaTeX with \\( \\) for inline and \\[ \\] for block

## Output Format (JSON array):
```json
[
  {
    "section_id": 1,
    "ocr_text": "Full section text with line breaks",
    "entities": [{"key_1": "value_1"}, {"key_2": "value_2"}],
    "children": [{"id": 5001, "updated_ocr": "refined text"}],
    "description": "Brief description of section content"
  }
]
```

Return ONLY the JSON array, no additional text or explanation."""
        return prompt
    
    def _parse_batch_response(self, response_text: str, sections: List[LayoutSection]) -> List[Dict]:
        """Parse JSON response from Vision LLM"""
        import re
        
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                json_str = json_match.group()
                results = json.loads(json_str)
                return results
            else:
                logger.warning("Could not find JSON array in response")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return []
    
    def _call_qwen(self, images: List[Image.Image], prompt: str) -> str:
        """Call Qwen3-VL via Transformers for local inference"""
        
        # Build messages with images using Qwen3-VL format
        content = []
        
        # Add images first
        for img in images:
            content.append({
                "type": "image",
                "image": img
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        try:
            # Preparation for inference - apply_chat_template returns tokenized inputs directly
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens
                )
            
            # Decode output (only the generated part, not the input)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            logger.error(f"Qwen3-VL inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def _process_batch(self, image: Image.Image, sections: List[LayoutSection]) -> None:
        """Process a batch of sections"""
        if not sections:
            return
        
        # Create section crops
        crops = [self._create_section_crop(image, s) for s in sections]
        
        # Build input JSON
        batch_input = self._build_batch_input(sections)
        
        # Build prompt
        prompt = self._build_batch_prompt(batch_input)
        
        # Call Qwen2-VL
        response = self._call_qwen(crops, prompt)
        
        # Parse response
        results = self._parse_batch_response(response, sections)
        
        # Apply results to sections
        for result in results:
            section_id = result.get("section_id")
            
            # Find matching section
            matching = [s for s in sections if s.id == section_id]
            if not matching:
                continue
            section = matching[0]
            
            # Update section fields
            section.ocr_text = result.get("ocr_text", "")
            section.entities = result.get("entities", [])
            section.description = result.get("description", "")
            
            # Update children OCR
            children_results = result.get("children", [])
            children_ocr_texts = []
            
            for child_result in children_results:
                child_id = child_result.get("id")
                updated_ocr = child_result.get("updated_ocr", "")
                
                # Find matching child
                for child in section.children:
                    if child.id == child_id:
                        child.ocr_text = updated_ocr
                        break
                children_ocr_texts.append(updated_ocr)
            
            section.children_ocr_texts = children_ocr_texts
    
    def run_ocr(self, image_path: str, sections: List[LayoutSection]) -> List[LayoutSection]:
        """Run Vision LLM OCR on all sections in batches
        
        Args:
            image_path: Path to the image
            sections: List of layout sections to process
        """
        logger.info(f"[Stage 3] Running Qwen2-VL OCR on {len(sections)} sections (batch_size={self.batch_size})")
        
        image = Image.open(image_path)
        
        # Process in batches
        total_batches = (len(sections) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(sections))
            batch_sections = sections[start_idx:end_idx]
            
            logger.info(f"[Stage 3] Processing batch {batch_idx + 1}/{total_batches} ({len(batch_sections)} sections)")
            
            self._process_batch(image, batch_sections)
        
        logger.info(f"[Stage 3] Completed OCR for all sections")
        return sections
    
    def generate_summary(self, image_path: str, sections: List[LayoutSection]) -> str:
        """Stage 4: Generate image-level summary from all sections
        
        Combines all section descriptions and OCR text into a single comprehensive
        summary of the entire image/page.
        
        Args:
            image_path: Path to the image
            sections: List of processed sections with OCR text and descriptions
            
        Returns:
            Detailed summary of the image content
        """
        logger.info(f"[Stage 4] Generating image summary from {len(sections)} sections")
        
        if not sections:
            return ""
        
        # Build input data for summary generation
        sections_data = []
        for section in sections:
            section_info = {
                "section_id": section.id,
                "label": section.label,
                "description": section.description,
                "ocr_text": section.ocr_text,
                "entities": section.entities
            }
            sections_data.append(section_info)
        
        # Build summary prompt
        prompt = """You are a document analysis assistant. Based on the following extracted information from a document image, provide a comprehensive and detailed summary.

## Extracted Sections Data:
```json
""" + json.dumps(sections_data, indent=2, ensure_ascii=False) + """
```

## Instructions:
1. Synthesize all the information from the sections into a coherent summary
2. Identify the main topic/purpose of the document
3. Highlight key information: titles, important text, figures, tables, equations
4. Mention any notable entities (names, dates, amounts, IDs) found
5. Describe the overall structure and layout of the document
6. Keep the summary detailed but well-organized

## Output:
Provide a comprehensive summary in Markdown format. Do not include JSON, just write the summary directly."""
        
        # Load image for context
        image = Image.open(image_path)
        
        try:
            # Call Qwen2-VL with full image for additional context
            summary = self._call_qwen([image], prompt)
            logger.info(f"[Stage 4] Summary generated successfully ({len(summary)} chars)")
            return summary.strip()
        except Exception as e:
            logger.error(f"[Stage 4] Summary generation failed: {e}")
            # Fallback: create basic summary from descriptions
            fallback_summary = "## Document Summary\n\n"
            for section in sections:
                if section.description:
                    fallback_summary += f"- **{section.label}**: {section.description}\n"
            return fallback_summary
