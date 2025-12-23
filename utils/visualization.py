# Visualization Utilities
"""
Visualization functions for OCR Pipeline V2.
Generates annotated images showing segmentation and OCR results.
"""

import textwrap
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import LayoutSection, PageResult


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a font with the given size, fallback to default.
    Prioritizes fonts that support Hindi/Devanagari and other Unicode scripts.
    """
    # List of fonts to try - prioritize Unicode/multilingual fonts
    font_paths = [
        # Linux - Noto fonts (excellent Unicode support including Hindi)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Windows - Hindi/Devanagari fonts
        "C:\\Windows\\Fonts\\Nirmala.ttf",  # Nirmala UI - Windows Hindi font
        "C:\\Windows\\Fonts\\mangal.ttf",   # Mangal - Classic Hindi font
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\seguisym.ttf",  # Segoe UI Symbol
        # Mac
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        # Fallback
        "arial.ttf",
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # Final fallback
    return ImageFont.load_default()




def fit_text_in_box(draw: ImageDraw.Draw, text: str, bbox: List[int], 
                    max_font_size: int = 20, min_font_size: int = 6):
    """Find the largest font size that fits text in the bounding box"""
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0 - 4  # Padding
    box_height = y1 - y0 - 4
    
    if box_width <= 0 or box_height <= 0:
        return None, ""
    
    # Clean text
    text = text.strip() if text else ""
    if not text:
        return None, ""
    
    # Try different font sizes
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = get_font(font_size)
        
        # Calculate character width approximately
        try:
            avg_char_width = font.getlength("x")
        except:
            avg_char_width = font_size * 0.6
        
        chars_per_line = max(1, int(box_width / avg_char_width))
        
        # Wrap text
        wrapped_lines = textwrap.wrap(text, width=chars_per_line)
        if not wrapped_lines:
            wrapped_lines = [text[:chars_per_line]]
        
        # Check if it fits
        line_height = font_size + 2
        total_height = len(wrapped_lines) * line_height
        
        if total_height <= box_height:
            return font, "\n".join(wrapped_lines)
    
    # Fallback: truncate
    font = get_font(min_font_size)
    return font, text[:20] + "..."


def save_stage1_visualization(image_path: str, sections: List[LayoutSection], output_path: Path) -> None:
    """Save Stage 1 visualization showing section bboxes with labels"""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Color map for different section labels
    label_colors = {
        'Text': (0, 255, 0),       # Green
        'Figure': (255, 0, 0),      # Red
        'Table': (0, 0, 255),       # Blue
        'Caption': (255, 165, 0),   # Orange
        'Page-Header': (128, 0, 128),  # Purple
        'Page-Footer': (128, 0, 128),  # Purple
        'Equation-Block': (255, 255, 0),  # Yellow
        'Footnote': (0, 255, 255),  # Cyan
    }
    default_color = (128, 128, 128)  # Gray for unknown labels
    
    # Try to load a font
    font = get_font(16)
    
    for section in sections:
        x0, y0, x1, y1 = section.bbox_pixels
        color = label_colors.get(section.label, default_color)
        
        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # Draw label
        label_text = f"{section.id}: {section.label}"
        draw.rectangle([x0, y0 - 20, x0 + len(label_text) * 8, y0], fill=color)
        draw.text((x0 + 2, y0 - 18), label_text, fill='white', font=font)
    
    image.save(str(output_path), 'PNG')


def save_visualization(result: PageResult, output_path: Path) -> None:
    """
    Save side-by-side visualization:
    - LEFT: Original image with transparent color overlays (color-coded by TexTAR attributes), IDs in top-right
    - RIGHT: Blank white image with OCR text fitted inside corresponding boxes
    """
    # Load original image
    original = Image.open(result.image_path).convert('RGBA')
    width, height = original.size
    
    # Create overlay for left panel (transparent colors)
    overlay = Image.new('RGBA', original.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Create right panel (blank white canvas)
    right_panel = Image.new('RGB', original.size, (255, 255, 255))
    right_draw = ImageDraw.Draw(right_panel)
    
    # TexTAR attribute color mapping (RGBA with transparency)
    textar_colors = {
        # Font style colors
        ('normal', 'normal'): (100, 200, 100, 80),        # Light green
        ('bold', 'normal'): (200, 100, 100, 100),         # Red
        ('italic', 'normal'): (100, 100, 200, 100),       # Blue
        ('bold_italic', 'normal'): (180, 100, 180, 100),  # Purple
        # With underline
        ('normal', 'underline'): (100, 200, 150, 100),    # Green-cyan
        ('bold', 'underline'): (220, 100, 100, 100),      # Bright red
        ('italic', 'underline'): (100, 150, 220, 100),    # Light blue
        ('bold_italic', 'underline'): (200, 100, 200, 100),
        # With strikeout
        ('normal', 'strikeout'): (200, 200, 100, 100),    # Yellow
        ('bold', 'strikeout'): (220, 150, 100, 100),      # Orange
        ('italic', 'strikeout'): (150, 150, 200, 100),    # Gray-blue
        ('bold_italic', 'strikeout'): (200, 150, 180, 100),
        # With both
        ('normal', 'underline_strikeout'): (180, 180, 100, 100),
        ('bold', 'underline_strikeout'): (230, 130, 100, 100),
        ('italic', 'underline_strikeout'): (130, 130, 220, 100),
        ('bold_italic', 'underline_strikeout'): (210, 130, 200, 100),
    }
    default_child_color = (150, 150, 150, 80)  # Gray for unmarked
    
    # Section outline colors (solid, for borders)
    section_colors = {
        'Page-Header': (107, 107, 255),
        'Text': (78, 205, 196),
        'Figure': (69, 183, 209),
        'Table': (150, 206, 180),
        'Caption': (255, 234, 167),
        'Section-Header': (221, 160, 221),
        'Equation-Block': (152, 216, 200),
        'List-Group': (247, 220, 111),
        'Code-Block': (187, 143, 206),
        'Page-Footer': (133, 193, 233),
        'Footnote': (248, 181, 0),
    }
    default_section_color = (149, 165, 166)
    
    # Global child ID counter
    child_id = 0
    
    # Process each section
    for section_idx, section in enumerate(result.sections):
        section_bbox = section.bbox_pixels
        section_color = section_colors.get(section.label, default_section_color)
        
        # Draw section border on left panel (thicker outline)
        overlay_draw.rectangle(
            [(section_bbox[0], section_bbox[1]), (section_bbox[2], section_bbox[3])],
            outline=section_color + (200,),  # Semi-transparent outline
            width=3
        )
        
        # Draw section label on top-left
        label = f"S{section_idx + 1}: {section.label}"
        label_font = get_font(12)
        try:
            label_bbox = label_font.getbbox(label)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]
        except:
            label_w, label_h = len(label) * 7, 14
        
        label_x = section_bbox[0]
        label_y = max(0, section_bbox[1] - label_h - 4)
        
        # Label background
        overlay_draw.rectangle(
            [(label_x, label_y), (label_x + label_w + 6, label_y + label_h + 4)],
            fill=section_color + (220,)
        )
        overlay_draw.text((label_x + 3, label_y + 2), label, fill=(255, 255, 255, 255), font=label_font)
        
        # Draw section outline on right panel too
        right_draw.rectangle(
            [(section_bbox[0], section_bbox[1]), (section_bbox[2], section_bbox[3])],
            outline=section_color,
            width=2
        )
        
        # Process children (text boxes)
        for child in section.children:
            child_id += 1
            child_bbox = child.bbox_pixels
            x0, y0, x1, y1 = child_bbox
            
            # Get color based on TexTAR attributes
            attr_key = (child.font_style, child.decoration)
            fill_color = textar_colors.get(attr_key, default_child_color)
            
            # LEFT PANEL: Draw transparent colored rectangle
            overlay_draw.rectangle([(x0, y0), (x1, y1)], fill=fill_color)
            
            # Draw thin border
            border_color = (fill_color[0], fill_color[1], fill_color[2], 200)
            overlay_draw.rectangle([(x0, y0), (x1, y1)], outline=border_color, width=1)
            
            # Draw ID in top-right corner
            id_text = str(child_id)
            id_font = get_font(9)
            try:
                id_bbox = id_font.getbbox(id_text)
                id_w = id_bbox[2] - id_bbox[0]
                id_h = id_bbox[3] - id_bbox[1]
            except:
                id_w, id_h = len(id_text) * 6, 10
            
            id_x = x1 - id_w - 3
            id_y = y0 + 1
            
            # ID background
            overlay_draw.rectangle(
                [(id_x - 2, id_y), (x1, id_y + id_h + 2)],
                fill=(50, 50, 50, 180)
            )
            overlay_draw.text((id_x, id_y), id_text, fill=(255, 255, 255, 255), font=id_font)
            
            # Draw underline indicator if present
            if child.decoration in ['underline', 'underline_strikeout']:
                overlay_draw.line([(x0, y1 - 2), (x1, y1 - 2)], fill=(255, 165, 0, 200), width=2)
            
            # Draw strikeout indicator if present
            if child.decoration in ['strikeout', 'underline_strikeout']:
                mid_y = (y0 + y1) // 2
                overlay_draw.line([(x0, mid_y), (x1, mid_y)], fill=(255, 100, 100, 200), width=2)
            
            # RIGHT PANEL: Draw OCR text fitted in box
            # Use the refined OCR text, fallback to paddle text
            ocr_text = child.ocr_text if child.ocr_text else child.ocr_text_paddle
            
            # Draw box outline
            right_draw.rectangle([(x0, y0), (x1, y1)], outline=(200, 200, 200), width=1)
            
            # Fit and draw text
            if ocr_text:
                font, fitted_text = fit_text_in_box(right_draw, ocr_text, child_bbox)
                if font and fitted_text:
                    # Determine text color based on font_style
                    text_color = (0, 0, 0)  # Default black
                    if child.font_style == 'bold' or child.font_style == 'bold_italic':
                        text_color = (139, 0, 0)  # Dark red for bold
                    if child.font_style == 'italic' or child.font_style == 'bold_italic':
                        text_color = (0, 0, 139)  # Dark blue for italic
                    
                    right_draw.text((x0 + 2, y0 + 2), fitted_text, fill=text_color, font=font)
                    
                    # Draw underline
                    if child.decoration in ['underline', 'underline_strikeout']:
                        right_draw.line([(x0 + 2, y1 - 3), (x1 - 2, y1 - 3)], fill=(0, 0, 0), width=1)
                    
                    # Draw strikeout
                    if child.decoration in ['strikeout', 'underline_strikeout']:
                        mid_y = (y0 + y1) // 2
                        right_draw.line([(x0 + 2, mid_y), (x1 - 2, mid_y)], fill=(100, 100, 100), width=1)
    
    # Composite overlay onto original for left panel
    left_panel = Image.alpha_composite(original, overlay).convert('RGB')
    
    # Create side-by-side image
    combined_width = width * 2 + 20  # 20px gap
    combined = Image.new('RGB', (combined_width, height), (240, 240, 240))
    
    # Paste panels
    combined.paste(left_panel, (0, 0))
    combined.paste(right_panel, (width + 20, 0))
    
    # Add panel labels at top
    label_draw = ImageDraw.Draw(combined)
    title_font = get_font(16)
    
    # Left panel title
    label_draw.text((10, 5), "Original + Segmentation (TexTAR Colors)", fill=(50, 50, 50), font=title_font)
    # Right panel title  
    label_draw.text((width + 30, 5), "OCR Text Reconstruction", fill=(50, 50, 50), font=title_font)
    
    # Add legend at bottom
    legend_y = height - 25
    legend_font = get_font(10)
    
    legends = [
        ("Normal", (100, 200, 100)),
        ("Bold", (200, 100, 100)),
        ("Italic", (100, 100, 200)),
        ("Bold+Italic", (180, 100, 180)),
    ]
    
    legend_x = 10
    for text, color in legends:
        # Color box
        label_draw.rectangle([(legend_x, legend_y), (legend_x + 15, legend_y + 15)], fill=color)
        label_draw.text((legend_x + 20, legend_y + 2), text, fill=(50, 50, 50), font=legend_font)
        legend_x += 90
    
    # Save
    combined.save(str(output_path), 'PNG')
