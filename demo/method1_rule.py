"""
Method 1 — PaddleOCR + Rule-based extraction for SROIE receipts.

PaddleOCR is used instead of EasyOCR because it is the dominant OCR backbone
on the SROIE leaderboard, offering word-level bounding boxes with high accuracy
on English printed receipts.

Install:
    pip install paddlepaddle paddleocr
"""

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

# Lazy-initialized PaddleOCR instance (only created when used as a fallback).
_ocr = None


_easyocr_reader = None
try:
    import easyocr
except Exception:
    easyocr = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_ocr(image_path: str) -> tuple[list[str], str]:
    """Return (lines, full_text) from OCR. Try PaddleOCR, then EasyOCR, then pytesseract.

    Always return (lines, full_text). On complete failure returns ([], "").
    """
    
    if easyocr is not None:
        try:
            global _easyocr_reader
            if _easyocr_reader is None:
                _easyocr_reader = easyocr.Reader(['en'])
            res = _easyocr_reader.readtext(image_path, detail=0, paragraph=True)
            lines = [l.strip() for l in res if l.strip()]
            return lines, "\n".join(lines)
        except Exception:
            import traceback
            print("[method1_rule] EasyOCR failed:")
            traceback.print_exc()

    
    if pytesseract is not None:
        try:
            img = Image.open(image_path)
            txt = pytesseract.image_to_string(img)
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            return lines, "\n".join(lines)
        except Exception:
            import traceback
            print("[method1_rule] pytesseract failed:")
            traceback.print_exc()

    
    if _paddle_available:
        try:
            global _ocr
            if _ocr is None:
                _ocr = PaddleOCR(use_angle_cls=True, lang="en")
            result = _ocr.ocr(image_path)
            if result and result[0] is not None:
                lines = [item[1][0].strip() for item in result[0] if item[1][0].strip()]
                return lines, "\n".join(lines)
        except Exception:
            import traceback
            print("[method1_rule] PaddleOCR failed:")
            traceback.print_exc()

    print(f"[method1_rule] OCR backends available - paddle: {_paddle_available}, easyocr: {easyocr is not None}, pytesseract: {pytesseract is not None}")
    return [], ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_rules(image_path: str) -> dict:
    """
    Extract COMPANY, ADDRESS, DATE, and TOTAL from a receipt image using
    OCR + hand-crafted rules (SROIE baseline approach).

    Parameters
    ----------
    image_path : str
        Path to the receipt image (JPEG / PNG / BMP etc.).

    Returns
    -------
    dict with keys: method, company, address, date, total
    """
    lines, full_text = _run_ocr(image_path)
    # Debug: print OCR output to server console to help diagnose missing fields
    try:
        print("[method1_rule] OCR lines:\n", "\n".join(lines[:50]))
    except Exception:
        pass

    
    if not lines and (_ocr is not None) and (easyocr is None and pytesseract is None):
        return {
            "method": "PaddleOCR + Rule-based",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": (
                "PaddleOCR failed at runtime and no fallback OCR is installed. "
                "Install EasyOCR (pip install easyocr) or pytesseract + Tesseract OCR."
            ),
            "_debug_ocr_lines": [],
            "_debug_ocr_text_preview": "",
        }

    # ── COMPANY: typically the first text line ────────────────────────────
    company = lines[0] if lines else "Not Found"

    # ── ADDRESS: scan early lines for street / location keywords ─────────
    # SROIE receipts are Malaysian; common address tokens shown below.
    address_kws = [
        "jalan", "street", "road", "avenue", "no.", "lot", "block",
        "floor", "level", "km", "taman", "kompleks", "plaza", "off",
        "lorong", "bandar", "mall", "centre", "center",
    ]
    address = "Not Found"
    for line in lines[1:12]:
        if any(kw in line.lower() for kw in address_kws):
            address = line
            break

    # ── DATE: ISO, DD/MM/YYYY, DD-MM-YYYY and DD.MM.YYYY patterns ────────
    date_match = re.search(
        r"\b(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"
        r"|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
        full_text,
    )

    # ── TOTAL: keyword followed by an amount ─────────────────────────────
    # Covers the most common SROIE total-line variants.
    total_match = re.search(
        r"(?:TOTAL|GRAND\s*TOTAL|AMOUNT\s*DUE|NET\s*TOTAL|JUMLAH\s*BESAR|JUMLAH)"
        r"[^\d\n]*([\d,]+\.?\d*)",
        full_text,
        re.IGNORECASE,
    )

    # Fallback: if keyword-based match fails, take the last numeric-looking token
    if not total_match:
        amounts = re.findall(r"[\d,]+\.?\d*", full_text)
        if amounts:
            total_val = amounts[-1]
        else:
            total_val = None
    else:
        total_val = total_match.group(1)

    return {
        "method": "PaddleOCR + Rule-based",
        "company": company,
        "address": address,
        "date": date_match.group(1) if date_match else "Not Found",
        "total": (total_val if total_val is not None else "Not Found"),
        "_debug_ocr_lines": lines[:50],
        "_debug_ocr_text_preview": (full_text[:1000] + "...") if len(full_text) > 1000 else full_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else "invoice.jpg"
    print(json.dumps(extract_with_rules(path), indent=2, ensure_ascii=False))
