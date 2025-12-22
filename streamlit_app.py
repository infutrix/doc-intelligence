# OCR Pipeline V2 - Streamlit Interactive App
"""
Streamlit application for interactive OCR visualization.
- Sidebar: Quick demo folder browser
- Main page: Upload/process new images or PDFs
- Interactive clickable sections with OCR text and entities
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Try to import streamlit-image-coordinates for click detection
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

# Try to import pipeline for processing
try:
    from pipeline import OCRPipelineV2
    from config import CONFIG
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# ============================================================================
# CONSTANTS
# ============================================================================

SECTION_COLORS = {
    'Page-Header': '#6B6BFF',
    'Text': '#4ECDC4',
    'Figure': '#45B7D1',
    'Table': '#96CEB4',
    'Caption': '#FFEAA7',
    'Section-Header': '#DDA0DD',
    'Equation-Block': '#98D8C8',
    'List-Group': '#F7DC6F',
    'Code-Block': '#BB8FCE',
    'Page-Footer': '#85C1E9',
    'Footnote': '#F8B500',
}
DEFAULT_SECTION_COLOR = '#959DA5'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_section_color(label: str) -> str:
    return SECTION_COLORS.get(label, DEFAULT_SECTION_COLOR)


def hex_to_rgba(hex_color: str, alpha: int = 100) -> tuple:
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


@st.cache_data
def create_annotated_image_cached(image_bytes: bytes, sections_json: str, selected_id: Optional[int] = None) -> bytes:
    """Create image with section bounding boxes overlaid (cached)"""
    image = Image.open(BytesIO(image_bytes))
    sections = json.loads(sections_json)
    
    img = image.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 14)
        except:
            font = ImageFont.load_default()
    
    for section in sections:
        bbox = section.get('bbox_pixels', [])
        if len(bbox) != 4:
            continue
        
        x0, y0, x1, y1 = bbox
        label = section.get('label', 'Unknown')
        section_id = section.get('id', 0)
        color_hex = get_section_color(label)
        
        if selected_id is not None and section_id == selected_id:
            fill_color = hex_to_rgba(color_hex, 150)
            border_width = 4
            border_color = hex_to_rgba('#FFFFFF', 255)
        else:
            fill_color = hex_to_rgba(color_hex, 50)
            border_width = 2
            border_color = hex_to_rgba(color_hex, 200)
        
        draw.rectangle([x0, y0, x1, y1], fill=fill_color)
        draw.rectangle([x0, y0, x1, y1], outline=border_color, width=border_width)
        
        label_text = f"{section_id}: {label}"
        try:
            label_bbox = font.getbbox(label_text)
            label_w = label_bbox[2] - label_bbox[0] + 8
            label_h = label_bbox[3] - label_bbox[1] + 6
        except:
            label_w, label_h = len(label_text) * 8 + 8, 18
        
        label_bg = hex_to_rgba(color_hex, 250)
        label_y = max(0, y0 - label_h - 2)
        draw.rectangle([x0, label_y, x0 + label_w, y0], fill=label_bg)
        draw.text((x0 + 4, label_y + 3), label_text, fill=(255, 255, 255, 255), font=font)
    
    result = Image.alpha_composite(img, overlay).convert('RGB')
    output = BytesIO()
    result.save(output, format='PNG')
    return output.getvalue()


def load_result_json(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_section_at_point(sections: List[Dict], x: int, y: int, scale: float = 1.0) -> Optional[Dict]:
    actual_x, actual_y = int(x * scale), int(y * scale)
    best_match, best_area = None, float('inf')
    
    for section in sections:
        bbox = section.get('bbox_pixels', [])
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        if x0 <= actual_x <= x1 and y0 <= actual_y <= y1:
            area = (x1 - x0) * (y1 - y0)
            if area < best_area:
                best_area, best_match = area, section
    return best_match


def get_all_page_ocr_text(sections: List[Dict]) -> str:
    return "\n\n".join([s.get('ocr_text', '') for s in sections if s.get('ocr_text')])


def get_processed_folders(base_path: str) -> List[Tuple[Path, str, int]]:
    base = Path(base_path)
    if not base.exists():
        return []
    
    results = []
    if list(base.glob("*result*.json")):
        results.append((base, "📄 Root", 0))
    
    for item in sorted(base.iterdir()):
        if item.is_dir():
            sub_json = list(item.glob("*result*.json"))
            if sub_json:
                try:
                    with open(sub_json[0], 'r') as f:
                        section_count = len(json.load(f).get('sections', []))
                except:
                    section_count = 0
                results.append((item, item.name, section_count))
    return results


def find_image_in_folder(folder_path: str, data: Dict) -> Optional[Image.Image]:
    folder = Path(folder_path)
    candidates = [folder / "original.png", folder / "original (1).png", folder / "original.jpg"]
    
    if data.get('image_path'):
        candidates.extend([Path(data['image_path']), folder / Path(data['image_path']).name])
    
    for path in candidates:
        if path and path.exists():
            try:
                return Image.open(str(path))
            except:
                continue
    
    for img_file in folder.glob("*.png"):
        if 'visualization' not in img_file.name.lower() and 'stage' not in img_file.name.lower():
            try:
                return Image.open(str(img_file))
            except:
                continue
    return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="IndiaAI Intelligent Document Processing Challenge", page_icon="📄", layout="wide", initial_sidebar_state="expanded")
    
    # CSS
    st.markdown("""
    <style>
    .entity-box { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); padding: 12px 16px; border-radius: 8px; border-left: 4px solid #667eea; margin: 8px 0; }
    .entity-key { font-weight: 700; color: #2c3e50; font-size: 0.85em; text-transform: uppercase; }
    .entity-value { font-size: 1.1em; color: #1a1a2e; margin-top: 2px; }
    .summary-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin: 10px 0; }
    .section-card { padding: 15px; border-radius: 10px; color: white; margin-bottom: 12px; }
    .ocr-box { background: #1e1e2e; color: #cdd6f4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; max-height: 250px; overflow-y: auto; }
    .upload-area { background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 12px; padding: 40px; text-align: center; margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)
    
    # Session state
    for key in ['json_path', 'folder_path', 'selected_section_id']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # ========================================================================
    # SIDEBAR: Quick Demo - Folder Browser
    # ========================================================================
    with st.sidebar:
        st.markdown("## 📂 Quick Demo")
        st.caption("Load existing processed results")
        
        base_folder = st.text_input(
            "Results Folder:",
            value=r"/workspace/Paras/Streamlit/Dhananjay/OCR_challenge/github_run",
            key="base_folder",
            label_visibility="collapsed"
        )
        
        if base_folder and Path(base_folder).exists():
            folders = get_processed_folders(base_folder)
            
            if folders:
                st.success(f"Found {len(folders)} document(s)")
                
                folder_options = [f"📄 {name} ({sec})" for _, name, sec in folders]
                selected_idx = st.radio(
                    "Documents:",
                    range(len(folders)),
                    format_func=lambda i: folder_options[i],
                    key="folder_select",
                    label_visibility="collapsed"
                )
                
                if selected_idx is not None:
                    selected_folder, _, _ = folders[selected_idx]
                    json_files = list(selected_folder.glob("*result*.json"))
                    
                    if json_files:
                        json_file = next((jf for jf in json_files if 'final' in jf.name.lower()), json_files[0])
                        
                        if st.button("🚀 Load Selected", type="primary", use_container_width=True):
                            st.session_state['json_path'] = str(json_file)
                            st.session_state['folder_path'] = str(selected_folder)
                            st.session_state['selected_section_id'] = None
                            st.rerun()
            else:
                st.warning("No results found.")
        elif base_folder:
            st.error("Folder not found.")
        
        st.markdown("---")
        
        if st.session_state['json_path']:
            st.markdown("**✅ Currently Loaded:**")
            st.caption(Path(st.session_state['folder_path'] or '').name)
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state['json_path'] = None
                st.session_state['folder_path'] = None
                st.session_state['selected_section_id'] = None
                st.rerun()
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if st.session_state['json_path']:
        # === RESULTS VIEW ===
        json_path = st.session_state['json_path']
        folder_path = st.session_state['folder_path'] or ''
        
        try:
            data = load_result_json(json_path)
        except Exception as e:
            st.error(f"Error: {e}")
            return
        
        image = find_image_in_folder(folder_path, data)
        if image is None:
            st.error("Could not find original image.")
            return
        
        sections = data.get('sections', [])
        summary = data.get('summary', '')
        img_width, img_height = image.size
        
        # Header
        doc_name = Path(folder_path).name if folder_path else "Document"
        st.markdown(f"## 📄 {doc_name}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sections", len(sections))
        col2.metric("Entities", sum(len(s.get('entities', [])) for s in sections))
        col3.metric("Size", f"{img_width}×{img_height}")
        
        st.markdown("---")
        
        # Layout: Image + Details
        col_img, col_detail = st.columns([3, 2])
        
        with col_img:
            st.markdown("### 🖼️ Document (click sections)")
            
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            sections_json = json.dumps(sections)
            annotated_bytes = create_annotated_image_cached(img_bytes, sections_json, st.session_state['selected_section_id'])
            annotated_img = Image.open(BytesIO(annotated_bytes))
            
            if CLICK_AVAILABLE:
                coords = streamlit_image_coordinates(annotated_img, key=f"img_{st.session_state['selected_section_id']}")
                
                if coords:
                    scale = img_width / min(800, img_width)
                    clicked = find_section_at_point(sections, coords['x'], coords['y'], scale)
                    if clicked and clicked.get('id') != st.session_state['selected_section_id']:
                        st.session_state['selected_section_id'] = clicked.get('id')
                        st.rerun()
            else:
                st.image(annotated_img, use_container_width=True)
        
        with col_detail:
            # Summary
            st.markdown("### 📝 Summary")
            if summary:
                truncated = summary[:180] + "..." if len(summary) > 180 else summary
                st.markdown(f'<div class="summary-box">{truncated}</div>', unsafe_allow_html=True)
                if len(summary) > 180:
                    with st.expander("📖 Full Summary"):
                        st.write(summary)
            else:
                st.info("No summary available")
            
            st.markdown("---")
            
            # Section Details
            st.markdown("### 🔍 Section Details")
            
            if st.session_state['selected_section_id'] is not None:
                selected = next((s for s in sections if s.get('id') == st.session_state['selected_section_id']), None)
                
                if selected:
                    label = selected.get('label', 'Unknown')
                    color = get_section_color(label)
                    
                    st.markdown(f'''<div class="section-card" style="background: {color};"><strong>Section {selected.get('id')}: {label}</strong><br/><span style="opacity: 0.9; font-size: 0.9em;">{selected.get('description', '')[:120]}</span></div>''', unsafe_allow_html=True)
                    
                    st.markdown("**📄 OCR Text:**")
                    ocr_text = selected.get('ocr_text', '')
                    if ocr_text:
                        st.markdown(f'<div class="ocr-box">{ocr_text}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("No OCR text")
                    
                    st.markdown("**🔑 Entities:**")
                    entities = selected.get('entities', [])
                    if entities:
                        for entity in entities:
                            if isinstance(entity, dict):
                                for key, value in entity.items():
                                    st.markdown(f'''<div class="entity-box"><div class="entity-key">🏷️ {key}</div><div class="entity-value">{value}</div></div>''', unsafe_allow_html=True)
                    else:
                        st.caption("No entities")
            else:
                st.info("👈 Click on a section in the image")
                for s in sections[:5]:
                    color = get_section_color(s.get('label', ''))
                    st.markdown(f"<span style='color:{color};'>●</span> **{s.get('id')}**: {s.get('label', '')}", unsafe_allow_html=True)
            
            # Expanders
            with st.expander("📄 Full OCR Text"):
                st.text_area("", get_all_page_ocr_text(sections), height=150, label_visibility="collapsed")
            
            with st.expander("📊 All Entities"):
                all_entities = [{'Key': k, 'Value': str(v), 'Sec': s.get('id')} for s in sections for e in s.get('entities', []) if isinstance(e, dict) for k, v in e.items()]
                if all_entities:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(all_entities), hide_index=True, use_container_width=True)
    
    else:
        # === UPLOAD/PROCESS VIEW (Main Page) ===
        st.markdown("# 📄 IndiaAI Intelligent Document Processing Challenge")
        st.markdown("Process documents with AI-powered OCR and entity extraction")
        
        st.markdown("---")
        
        # Upload area
        st.markdown("### 📤 Process New Image or PDF")
        
        if not PIPELINE_AVAILABLE:
            st.warning("⚠️ OCR Pipeline not available. Please install dependencies or use the sidebar to load existing results.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded = st.file_uploader(
                    "Drop your file here or click to browse",
                    type=['png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'],
                    key="main_upload",
                    help="Supported formats: PNG, JPG, PDF, BMP, TIFF"
                )
            
            with col2:
                st.markdown("**Supported Formats:**")
                st.markdown("- 🖼️ Images: PNG, JPG, BMP, TIFF")
                st.markdown("- 📄 Documents: PDF")
            
            if uploaded:
                st.markdown("---")
                
                # Preview
                col_preview, col_action = st.columns([2, 1])
                
                with col_preview:
                    st.markdown(f"**Selected:** {uploaded.name}")
                    st.caption(f"Size: {uploaded.size / 1024:.1f} KB")
                    
                    # Show preview for images
                    if uploaded.type.startswith('image'):
                        preview_img = Image.open(uploaded)
                        st.image(preview_img, caption="Preview", width=300)
                
                with col_action:
                    st.markdown("### Ready to Process")
                    
                    if st.button("🚀 Run OCR Pipeline", type="primary", use_container_width=True):
                        with st.spinner("🔄 Processing... This may take a few minutes."):
                            try:
                                suffix = Path(uploaded.name).suffix
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(uploaded.getvalue())
                                    tmp_path = tmp.name
                                
                                output_folder = tempfile.mkdtemp(prefix="ocr_")
                                pipeline = OCRPipelineV2()
                                
                                if suffix.lower() == '.pdf':
                                    pipeline.process_pdf(tmp_path, output_folder)
                                    json_file = Path(output_folder) / "pdf_summary.json"
                                else:
                                    result = pipeline.process_page(tmp_path, output_folder=Path(output_folder))
                                    json_file = Path(output_folder) / "final_result.json"
                                    with open(json_file, 'w') as f:
                                        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                                
                                st.session_state['json_path'] = str(json_file)
                                st.session_state['folder_path'] = output_folder
                                st.session_state['selected_section_id'] = None
                                st.success("✅ Processing complete!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                                import traceback
                                st.code(traceback.format_exc())
        
        st.markdown("---")
        
        # Features
        st.markdown("### ✨ Features")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 🔍 Section Detection")
            st.caption("DocLayout-YOLO identifies document regions")
        
        with col2:
            st.markdown("#### 📝 OCR Extraction")
            st.caption("PaddleOCR + Qwen3-VL for accurate text")
        
        with col3:
            st.markdown("#### 🔑 Entity Extraction")
            st.caption("AI extracts key-value pairs automatically")
        
        with col4:
            st.markdown("#### 📋 Page Summary")
            st.caption("AI-generated summary of document content")

        
        st.markdown("---")
        st.caption("💡 Use the sidebar to load existing processed results")
        
        if not CLICK_AVAILABLE:
            st.info("Install `streamlit-image-coordinates` for interactive click: `pip install streamlit-image-coordinates`")


if __name__ == "__main__":
    main()
