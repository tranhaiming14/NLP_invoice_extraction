# Receipt Information Extraction Demo

A FastAPI web application that demonstrates three methods for extracting structured information (COMPANY, ADDRESS, DATE, TOTAL) from receipt images.

## Methods

1. **Method 1 — Rule-based (EasyOCR + Regex)**
   - Fast, lightweight approach
   - Uses EasyOCR for text recognition
   - Applies hand-crafted regex patterns to extract fields
   - Best for: quick demos, simple receipts

2. **Method 2 — LLM-based (EasyOCR + Gemini)**
   - Smart approach using Google's Gemini API
   - Uses EasyOCR for OCR, then sends text to Gemini LLM for structured extraction
   - Handles OCR noise and ambiguous text well
   - Requires: `GEMINI_API_KEY` environment variable
   - Best for: noisy receipts, high accuracy needed

3. **Method 3 — LayoutLM (Fine-tuned)**
   - Advanced deep-learning approach
   - Uses fine-tuned LayoutLM model trained on SROIE dataset
   - Combines text + layout understanding for token classification
   - Best for: SROIE-like receipts, highest accuracy

---

## Setup

### Prerequisites
- Python 3.11+
- A GPU (CUDA 11.8+) is recommended for faster inference, but CPU works too

### Installation

1. **Clone or navigate to the project folder:**
   ```bash
   cd d:\Desktop\Code\USTH\LNP\demo
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or:
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional, for Method 2) Set your Gemini API key:**
   ```powershell
   $env:GEMINI_API_KEY = "your-actual-api-key-here"
   ```
   Or on macOS/Linux:
   ```bash
   export GEMINI_API_KEY="your-actual-api-key-here"
   ```

### Important Notes on Model Loading

When you first run the demo:
- **EasyOCR** will download its model (~100 MB) on first use — this may take a few minutes
- **LayoutLM (Method 3)** will load the fine-tuned model from `../output/model/` — ensure the notebook training completed and saved the model there

---

## Running the Demo

1. **Start the server:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

2. **Open in browser:**
   ```
   http://localhost:8000
   ```

3. **Upload a receipt image:**
   - Click or drag-and-drop a JPEG/PNG/BMP/TIFF/WebP image
   - Select a method (Rule-based, LLM, or LayoutLM)
   - Click "Extract Information"
   - View extracted fields and switch methods to compare

---

## File Structure

```
demo/
├── app.py                     # FastAPI app with web UI
├── method1_rule.py            # Rule-based extraction (EasyOCR + regex)
├── method2_llm.py             # LLM-based extraction (EasyOCR + Gemini)
├── method3_layoutlm.py        # LayoutLM-based extraction
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## API Endpoints

- **GET** `/` — HTML frontend
- **POST** `/predict/rule` — Extract fields using Method 1 (Rule-based)
- **POST** `/predict/llm` — Extract fields using Method 2 (LLM)
- **POST** `/predict/layoutlm` — Extract fields using Method 3 (LayoutLM)

Each endpoint accepts a single file upload and returns JSON with extracted fields + debug info.

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'transformers'`
```bash
pip install -r requirements.txt
```

### Error: `ModuleNotFoundError: No module named 'torch'`
- On Windows with GPU: Install CUDA 11.8+ first, then:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Without GPU (CPU):
  ```bash
  pip install torch torchvision torchaudio
  ```

### Method 2 (LLM) returns error "LLM API call failed"
- Ensure `GEMINI_API_KEY` environment variable is set
- Check that your API key is valid at https://makersuite.google.com/app/apikey

### Method 3 (LayoutLM) returns error "Model not available"
- Ensure the notebook completed training: `../output/model/` should exist and contain:
  - `config.json`
  - `pytorch_model.bin`
  - `special_tokens_map.json`
  - `tokenizer_config.json`
  - `vocab.txt`

### Slow first inference with EasyOCR / LayoutLM
- First run downloads/loads models — this is normal
- Subsequent runs will be faster
- On first run, the download may take 2–5 minutes

---

## Performance Notes

| Method | Speed | Accuracy | GPU Needed | Dependencies |
|--------|-------|----------|-----------|--------------|
| Rule-based | Fast (< 1s) | Medium | No | EasyOCR, regex |
| LLM | Slow (3–10s) | High | No | EasyOCR, Gemini API |
| LayoutLM | Slow (3–8s) | Very High | Optional | EasyOCR, PyTorch, Transformers |

---

## License & Attribution

- **LayoutLM**: [Microsoft UNILM](https://github.com/microsoft/unilm)
- **EasyOCR**: [JaidedAI EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **SROIE Dataset**: [ICDAR 2019](https://rrc.cvc.uab.es/?ch=13)
