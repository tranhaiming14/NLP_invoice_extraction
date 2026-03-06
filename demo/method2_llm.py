"""
Method 2 — PaddleOCR + Gemini LLM extraction for SROIE receipts.

PaddleOCR handles the raw image → text step; Gemini then interprets the
noisy OCR output and returns a clean structured JSON response.

Install:
    pip install paddlepaddle paddleocr google-generativeai

Set your API key (optional):
    export GEMINI_API_KEY="your-key-here"   # Linux / macOS
    $env:GEMINI_API_KEY="your-key-here"     # Windows PowerShell
"""

import json
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
import re
from PIL import Image

# PaddleOCR may cause heavy native runtime issues on some systems.
# Import the class if available but delay initialization until it's needed.
try:
    from paddleocr import PaddleOCR
    _paddle_available = True
except Exception:
    PaddleOCR = None
    _paddle_available = False

# Lazy fallback readers
_easyocr_reader = None
try:
    import easyocr
except Exception:
    easyocr = None

try:
    import pytesseract
except Exception:
    pytesseract = None

import google.generativeai as genai

_ocr = None

# Load root .env if python-dotenv is available
if load_dotenv is not None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if not gemini_api_key:
    print("[method2_llm] WARNING: GEMINI_API_KEY environment variable not set. LLM extraction will fail.")
genai.configure(api_key=gemini_api_key)
_llm = genai.GenerativeModel("gemini-2.5-flash")

_PROMPT = """\
Extract the four key fields from the English receipt text below.
Return ONLY a valid JSON object — no markdown fences, no extra text.
Correct obvious OCR spelling errors in company names and addresses.

JSON schema (all values are strings):
{
    "company": "store or restaurant name",
    "address": "full street address",
    "date": "YYYY-MM-DD or the original format if ambiguous",
    "total": "final amount as digits and decimal point only, e.g. 12.50"
}

Receipt text:
{text}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_ocr(image_path: str) -> str:
    """Return the full OCR text from *image_path* using EasyOCR, pytesseract, or PaddleOCR."""
    global _easyocr_reader, _ocr

    if easyocr is not None:
        try:
            if _easyocr_reader is None:
                _easyocr_reader = easyocr.Reader(["en"])
            res = _easyocr_reader.readtext(image_path, detail=0, paragraph=True)
            lines = [l.strip() for l in res if l.strip()]
            return "\n".join(lines)
        except Exception:
            import traceback
            print("[method2_llm] EasyOCR failed:")
            traceback.print_exc()

    if pytesseract is not None:
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img)
        except Exception:
            import traceback
            print("[method2_llm] pytesseract failed:")
            traceback.print_exc()

    if _paddle_available:
        try:
            if _ocr is None:
                _ocr = PaddleOCR(use_angle_cls=True, lang="en")
            result = _ocr.ocr(image_path)
            if result and result[0] is not None:
                lines = [item[1][0].strip() for item in result[0] if item[1][0].strip()]
                return "\n".join(lines)
        except Exception:
            import traceback
            print("[method2_llm] PaddleOCR failed:")
            traceback.print_exc()

    print(f"[method2_llm] OCR backends available - paddle: {_paddle_available}, easyocr: {easyocr is not None}, pytesseract: {pytesseract is not None}")
    return ""


def _parse_json(raw: str) -> dict:
    """
    Robustly extract the first JSON object from an LLM response.
    Falls back gracefully if parsing fails.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip("` \n")
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"company": "", "address": "", "date": "", "total": "",
            "_parse_error": "Could not parse LLM response"}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_with_llm(image_path: str) -> dict:
    """
    Extract COMPANY, ADDRESS, DATE, and TOTAL from a receipt image using
    PaddleOCR for text recognition and Gemini for structured extraction.

    Returns a dict with keys: method, company, address, date, total
    """
    ocr_text = _run_ocr(image_path)
    
    try:
        print("[method2_llm] OCR preview:\n", ocr_text[:1000])
    except Exception:
        pass

    if not ocr_text:
        
        # If paddle runtime failed and no fallback OCR is installed, return actionable error
        if (_ocr is not None) and (easyocr is None and pytesseract is None):
            return {
                "method": "PaddleOCR + LLM (Gemini)",
                "company": "Not Found",
                "address": "Not Found",
                "date": "Not Found",
                "total": "Not Found",
                "error": (
                    "PaddleOCR failed at runtime and no fallback OCR is installed. "
                    "Install EasyOCR (pip install easyocr) or pytesseract + Tesseract OCR."
                ),
                "_debug_ocr_text_preview": "",
            }
        return {
            "method": "PaddleOCR + LLM (Gemini)",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": "OCR returned no text — check image quality",
            "_debug_ocr_text_preview": "",
        }

    # Call LLM with error handling
    try:
        raw_response = _llm.generate_content(_PROMPT.format(text=ocr_text))
        # raw_response may contain text in .text or .content depending on SDK; coerce to string
        rr_text = getattr(raw_response, 'text', None) or getattr(raw_response, 'content', None) or str(raw_response)
        parsed = _parse_json(rr_text)
        parsed["method"] = "PaddleOCR + LLM (Gemini)"
        # Add debug helpers so frontend can show what OCR / LLM saw
        parsed.setdefault('_debug_ocr_text_preview', ocr_text[:1000] + ("..." if len(ocr_text) > 1000 else ""))
        parsed.setdefault('_debug_llm_raw_preview', rr_text[:1000] + ("..." if len(rr_text) > 1000 else ""))
        return parsed
    except Exception as e:
        import traceback
        print("[method2_llm] LLM generation failed:")
        traceback.print_exc()
        return {
            "method": "PaddleOCR + LLM (Gemini)",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": f"LLM API call failed: {str(e)}. Check GEMINI_API_KEY env var.",
            "_debug_ocr_text_preview": ocr_text[:1000] + ("..." if len(ocr_text) > 1000 else ""),
        }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "invoice.jpg"
    print(json.dumps(extract_with_llm(path), indent=2, ensure_ascii=False))
