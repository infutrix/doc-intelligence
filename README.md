# Document Intelligence

<img src="resource\Document%20OCR%20Processing-2025-12-22-121736.svg" alt="Document OCR Processing" width="600">

A modular OCR pipeline using **DocLayout-YOLO**, **PaddleOCR**, and **Qwen3-VL** Vision LLM for intelligent document understanding.

---

## 🏗️ Architecture

A **three-stage "Segment-Refine-Structure" pipeline**:

1. **Segmentation**: Fine-tuned **DocLayout-YOLO** detects sections; **PaddleOCR** provides word-level coordinates. A "Mask & Discover" strategy ensures **100% data capture**.

2. **Extraction**: **Qwen3-VL-8B-Instruct** refines OCR, handles **multilingual text** (Hindi, Sanskrit, English), converts tables to **HTML**, and math to **LaTeX**.

3. **Structuring**: Generates a **hierarchical JSON schema** with entity extraction (key-value pairs) and intelligent summarization.

---

## ✨ Features

- 🔍 **4-Stage Pipeline**: DocLayout-YOLO → PaddleOCR → Qwen3-VL → Summary
- 📄 **PDF Support**: Process multi-page PDFs with configurable DPI
- 🌐 **Multilingual**: Hindi, Sanskrit, English, and more
- 📊 **Entity Extraction**: Automatic key-value pair detection
- 🖼️ **Streamlit UI**: Interactive web interface for visualization

---

## 🚀 Quick Start (Streamlit App)

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
.\venv\Scripts\activate       # Windows
```

### 2. Install Dependencies

#### :one: Install Paddle GPU (CUDA 12.6 build)
```bash
python -m pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

#### :two: Install remaining Python deps
```bash
pip install -r requirements.txt
```

### 3. Install System Dependencies

#### Poppler (for PDF support)
```bash
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
#          Extract and add bin/ folder to PATH

# Linux
sudo apt-get install poppler-utils

# Mac
brew install poppler
```

#### Hindi Fonts (for proper text rendering)
```bash
# Linux only - Windows/Mac have these pre-installed
sudo apt-get install fonts-noto fonts-noto-extra
```

### 4. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

### 5. Using the App

1. **Main Page**: Upload new images/PDFs and process them
2. **Sidebar**: Browse existing processed results
3. **Click on sections** in the image to view OCR text and entities

---

## 💻 CLI Usage

```bash
# Process image
python main.py --input image.png --output ./output

# Process PDF
python main.py --input document.pdf --output ./output --dpi 300

# Process folder
python main.py --input ./images --output ./output

# CPU only mode
python main.py --input image.png --output ./output --no-gpu
```

---

## 📁 Project Structure

```
ocrrrrchalenge/
├── main.py                 # CLI entry point
├── streamlit_app.py        # Interactive web UI
├── pipeline.py             # OCRPipelineV2 class
├── config.py               # Configuration
├── requirements.txt
├── models/                 # DocLayout-YOLO model
├── stages/
│   ├── stage1_doclayout.py
│   ├── stage2_paddleocr.py
│   └── stage3_vision_llm.py
├── utils/
│   ├── visualization.py
│   ├── pdf_utils.py
│   └── coordinate_utils.py
└── data_models/
    └── schemas.py
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
CONFIG = {
    "doclayout_model_path": "models/doclayout_yolo_docstructbench.pt",
    "doclayout_confidence": 0.2,
    "use_gpu": True,
    "enable_stage2": True,
    "batch_size": 5,
    "qwen_model_name": "Qwen/Qwen3-VL-8B-Instruct",
}
```

---


