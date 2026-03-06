"""
Method 3 — Fine-tuned LayoutLM model for SROIE information extraction.

Uses the LayoutLM model trained in the parent notebook on the SROIE dataset.
Combines EasyOCR for text + bbox detection with LayoutLM token classification.

Install:
    pip install easyocr torch transformers
"""

import json
from pathlib import Path
from PIL import Image

import torch
import easyocr
from transformers import AutoTokenizer, LayoutLMForTokenClassification

_easyocr_reader = None

_MODEL_PATH = Path(__file__).parent.parent / "train" / "output" / "model"

_load_error: str = ""

if _MODEL_PATH.exists():
    try:
        import traceback as _tb
        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_PATH))
        _model = LayoutLMForTokenClassification.from_pretrained(str(_MODEL_PATH))
        _model.eval()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model_available = True
        _ID2LABEL = _model.config.id2label
        print(f"[method3_layoutlm] Model loaded from {_MODEL_PATH} on device {_device}")
        print(f"[method3_layoutlm] id2label: {_ID2LABEL}")
    except Exception as e:
        _load_error = _tb.format_exc()
        print(f"[method3_layoutlm] Failed to load model: {e}")
        print(_load_error)
        _model_available = False
        _tokenizer = None
        _model = None
        _ID2LABEL = {}
else:
    _load_error = f"Model path not found: {_MODEL_PATH}"
    print(f"[method3_layoutlm] {_load_error}")
    _model_available = False
    _tokenizer = None
    _model = None
    _ID2LABEL = {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_ocr_reader():
    """Lazy-initialize EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"])
    return _easyocr_reader


def _run_ocr_with_bboxes(image_path: str) -> tuple:
    """
    Use EasyOCR to extract text and bboxes, then split multi-word lines into
    individual words with proportionally distributed bboxes.
    Returns (words, bboxes, width, height) or ([], [], 0, 0) on failure.
    """
    try:
        reader = _get_ocr_reader()
        result = reader.readtext(image_path)

        img = Image.open(image_path)
        width, height = img.size
        img.close()

        print(f"[method3_layoutlm] OCR returned {len(result)} detections, image size {width}x{height}")
        words = []
        bboxes = []
        for detection in result:
            bbox_points = detection[0]
            text = detection[1].strip()
            if not text:
                continue

            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x0, y0 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            line_words = [w for w in text.split() if w]
            if not line_words:
                continue

            total_chars = sum(len(w) for w in line_words)
            if total_chars == 0:
                continue

            bbox_width = max(x2 - x0, 1)
            cur_x = x0
            for w in line_words:
                word_x2 = cur_x + int(bbox_width * len(w) / total_chars)
                words.append(w)
                bboxes.append([cur_x, y0, word_x2, y2])
                cur_x = word_x2 + 2

        print(f"[method3_layoutlm] OCR split into {len(words)} individual words")
        return words, bboxes, width, height
    except Exception as e:
        import traceback
        print(f"[method3_layoutlm] OCR failed: {e}")
        traceback.print_exc()
        return [], [], 0, 0


def _normalize_bbox(x0, y0, x2, y2, width, height):
    """Normalize bbox to [0, 1000] range, clamped and sorted."""
    if width == 0 or height == 0:
        return [0, 0, 1000, 1000]
    
    nx0 = max(0, min(int(1000 * x0 / width),  1000))
    ny0 = max(0, min(int(1000 * y0 / height), 1000))
    nx2 = max(0, min(int(1000 * x2 / width),  1000))
    ny2 = max(0, min(int(1000 * y2 / height), 1000))
    
    return [min(nx0, nx2), min(ny0, ny2), max(nx0, nx2), max(ny0, ny2)]


def _run_layoutlm_inference(words: list, bboxes: list, max_seq_len=512):
    """
    Run LayoutLM model on the words and bboxes.
    Returns (predicted_labels, token_to_word_idx) or ([], []) on failure.
    """
    if not _model_available or not words:
        return [], []
    
    try:
        # Tokenize each word and track which tokens belong to which word
        token_ids = []
        token_bboxes = []
        token_labels = []
        token_to_word_idx = []
        
        for word_idx, (word, bbox) in enumerate(zip(words, bboxes)):
            # Tokenize the word
            sub_tokens = _tokenizer.tokenize(word)
            sub_ids = _tokenizer.convert_tokens_to_ids(sub_tokens)
            
            if not sub_ids:
                continue
            
            token_ids.extend(sub_ids)
            token_bboxes.extend([bbox] * len(sub_ids))
            token_to_word_idx.extend([word_idx] * len(sub_ids))
        
        # Truncate to max_seq_len - 2 (for [CLS] and [SEP])
        max_tokens = max_seq_len - 2
        token_ids = token_ids[:max_tokens]
        token_bboxes = token_bboxes[:max_tokens]
        token_to_word_idx = token_to_word_idx[:max_tokens]
        
        # Add special tokens
        cls_id = _tokenizer.cls_token_id
        sep_id = _tokenizer.sep_token_id
        token_ids = [cls_id] + token_ids + [sep_id]
        token_bboxes = [[0, 0, 0, 0]] + token_bboxes + [[1000, 1000, 1000, 1000]]
        token_to_word_idx = [-1] + token_to_word_idx + [-1]
        
        # Attention mask
        seq_len = len(token_ids)
        attention = [1] * seq_len
        
        # Pad to max_seq_len
        pad_len = max_seq_len - seq_len
        pad_id = _tokenizer.pad_token_id
        token_ids += [pad_id] * pad_len
        token_bboxes += [[0, 0, 0, 0]] * pad_len
        token_to_word_idx += [-1] * pad_len
        attention += [0] * pad_len
        
        # Convert to tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(_device)
        bbox = torch.tensor([token_bboxes], dtype=torch.long).to(_device)
        attention_mask = torch.tensor([attention], dtype=torch.long).to(_device)
        
        # Run inference
        with torch.no_grad():
            outputs = _model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
        
        # Extract predictions for non-padded tokens
        pred_labels = [_ID2LABEL.get(int(p), "O") for p in preds[0, :seq_len]]
        
        return pred_labels, token_to_word_idx[:seq_len]
    
    except Exception as e:
        import traceback
        print(f"[method3_layoutlm] Model inference failed: {e}")
        traceback.print_exc()
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_with_layoutlm(image_path: str) -> dict:
    """
    Extract COMPANY, ADDRESS, DATE, and TOTAL using fine-tuned LayoutLM.

    Returns a dict with keys: method, company, address, date, total
    """
    if not _model_available:
        return {
            "method": "LayoutLM (Fine-tuned)",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": "LayoutLM model not available. Check output/model/ path.",
        }
    
    # Run OCR
    words, bboxes, width, height = _run_ocr_with_bboxes(image_path)
    if not words:
        return {
            "method": "LayoutLM (Fine-tuned)",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": "OCR returned no text — check image quality.",
        }
    
    print(f"[method3_layoutlm] Running inference on {len(words)} words")
    # Normalize bboxes
    norm_bboxes = [_normalize_bbox(b[0], b[1], b[2], b[3], width, height) for b in bboxes]
    
    # Run LayoutLM
    pred_labels, token_to_word = _run_layoutlm_inference(words, norm_bboxes)
    print(f"[method3_layoutlm] Inference returned {len(pred_labels)} token labels")
    if not pred_labels:
        return {
            "method": "LayoutLM (Fine-tuned)",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": "LayoutLM inference failed.",
        }
    
    # Map token predictions back to words, keeping first token label per word
    word_labels = ["O"] * len(words)
    for token_idx, (pred_label, word_idx) in enumerate(zip(pred_labels, token_to_word)):
        if word_idx >= 0 and word_idx < len(words):
            # Only keep the label from the first token of each word
            if word_labels[word_idx] == "O":
                word_labels[word_idx] = pred_label

    # Debug: print non-O predictions so we can see what the model found
    non_o = [(words[i], word_labels[i]) for i in range(len(words)) if word_labels[i] != "O"]
    print(f"[method3_layoutlm] Non-O predictions ({len(non_o)}): {non_o[:30]}")
    
    # Collect all words for each label (group consecutive runs)
    field_words: dict[str, list[str]] = {"S-COMPANY": [], "S-DATE": [], "S-ADDRESS": [], "S-TOTAL": []}
    for word, label in zip(words, word_labels):
        if label in field_words:
            field_words[label].append(word)

    return {
        "method": "LayoutLM (Fine-tuned)",
        "company":  " ".join(field_words["S-COMPANY"])  or "Not Found",
        "address":  " ".join(field_words["S-ADDRESS"])  or "Not Found",
        "date":     " ".join(field_words["S-DATE"])     or "Not Found",
        "total":    " ".join(field_words["S-TOTAL"])    or "Not Found",
        "_debug_ocr_words":        words[:50],
        "_debug_predicted_labels": word_labels[:50],
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "invoice.jpg"
    print(json.dumps(extract_with_layoutlm(path), indent=2, ensure_ascii=False))
