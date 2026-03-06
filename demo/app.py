"""
FastAPI demo app — Receipt Information Extraction

Run:
    cd demo
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from method1_rule import extract_with_rules
from method2_llm import extract_with_llm
from method3_layoutlm import extract_with_layoutlm

app = FastAPI(title="Receipt IE Demo", version="1.0")

_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> Path:
    """Save the uploaded file to a temp location and return its path."""
    suffix = Path(upload.filename).suffix.lower() if upload.filename else ".jpg"
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.flush()
    tmp.close()  # IMPORTANT on Windows: release the handle so PIL/EasyOCR can open the file
    return Path(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# HTML frontend (embedded so the demo is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Receipt IE Demo</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f2f5; color: #1f2937; }

  header {
    background: linear-gradient(135deg, #1a56db 0%, #1e3a8a 100%);
    color: #fff; padding: 1.2rem 2rem;
    display: flex; align-items: center; gap: .8rem;
  }
  header svg { flex-shrink: 0; }
  header h1 { font-size: 1.4rem; font-weight: 700; }
  header p  { font-size: .85rem; opacity: .8; margin-top: .15rem; }

  main { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }

  .card {
    background: #fff; border-radius: 14px;
    box-shadow: 0 2px 16px rgba(0,0,0,.08);
    padding: 1.8rem; margin-bottom: 1.5rem;
  }
  .card-title { font-size: .75rem; text-transform: uppercase; letter-spacing: .07em;
                color: #6b7280; margin-bottom: 1.2rem; font-weight: 600; }

  /* Method tabs */
  .tabs { display: flex; gap: .75rem; margin-bottom: 1.5rem; }
  .tab {
    flex: 1; padding: .75rem; border: 2px solid #e5e7eb; border-radius: 10px;
    background: #fff; cursor: pointer; font-size: .9rem; font-weight: 500;
    transition: all .15s; text-align: center;
  }
  .tab:hover { border-color: #93c5fd; }
  .tab.active { border-color: #1a56db; background: #eff6ff; color: #1a56db; }
  .tab .badge {
    display: inline-block; font-size: .68rem; font-weight: 700;
    background: #dbeafe; color: #1e40af; border-radius: 999px;
    padding: .1rem .5rem; margin-left: .4rem;
  }
  .tab.active .badge { background: #1a56db; color: #fff; }

  /* Upload */
  .drop-zone {
    border: 2px dashed #d1d5db; border-radius: 12px; padding: 2.5rem 1rem;
    text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.over { border-color: #1a56db; background: #f0f7ff; }
  .drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .drop-zone .icon { color: #9ca3af; margin-bottom: .5rem; }
  .drop-zone p { color: #6b7280; font-size: .9rem; }
  .drop-zone strong { color: #1a56db; }

  #preview { margin-top: 1rem; }
  #preview img { max-height: 280px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }

  .btn-predict {
    width: 100%; margin-top: 1.2rem; padding: .9rem;
    background: #1a56db; color: #fff;
    border: none; border-radius: 10px; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background .15s;
    display: flex; align-items: center; justify-content: center; gap: .5rem;
  }
  .btn-predict:hover:not(:disabled) { background: #1e3a8a; }
  .btn-predict:disabled { background: #93c5fd; cursor: not-allowed; }

  /* Spinner */
  .spinner { display: none; align-items: center; justify-content: center; gap: .5rem;
             color: #6b7280; padding: .8rem; margin-top: .5rem; }
  .spinner svg { animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Error */
  .error-box {
    display: none; margin-top: 1rem; padding: .9rem 1rem;
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
    color: #dc2626; font-size: .9rem;
  }

  /* Results */
  #resultCard { display: none; }
  .method-tag {
    display: inline-block; font-size: .8rem; font-weight: 600;
    background: #dbeafe; color: #1e40af; border-radius: 999px;
    padding: .25rem .8rem; margin-bottom: 1rem;
  }
  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
  .field {
    background: #f9fafb; border: 1px solid #f3f4f6;
    border-radius: 10px; padding: .85rem 1rem;
  }
  .field.full { grid-column: 1 / -1; }
  .field-label {
    font-size: .7rem; text-transform: uppercase; letter-spacing: .06em;
    color: #9ca3af; margin-bottom: .3rem; font-weight: 600;
  }
  .field-value { font-size: .95rem; font-weight: 500; word-break: break-word; }
  .field-value.missing { color: #9ca3af; font-style: italic; font-weight: 400; }

  /* Compare row */
  .compare { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: .5rem; }
  .compare-btn {
    font-size: .8rem; padding: .35rem .8rem; border-radius: 999px;
    border: 1px solid #d1d5db; background: #fff; cursor: pointer;
    transition: all .15s; color: #374151;
  }
  .compare-btn:hover { border-color: #1a56db; color: #1a56db; }
</style>
</head>
<body>

<header>
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <rect x="3" y="3" width="18" height="18" rx="3"/>
    <path d="M7 8h10M7 12h10M7 16h6"/>
  </svg>
  <div>
    <h1>Receipt Information Extraction</h1>
    <p>PaddleOCR · Rule-based &amp; LLM (Gemini) · SROIE fields</p>
  </div>
</header>

<main>

  <!-- Input card -->
  <div class="card">
    <div class="card-title">1 · Choose extraction method</div>
    <div class="tabs">
      <button class="tab active" onclick="pickMethod('rule', this)">
        Method 1 — Rule-based
        <span class="badge">fast</span>
      </button>
      <button class="tab" onclick="pickMethod('llm', this)">
        Method 2 — LLM (Gemini)
        <span class="badge">smart</span>
      </button>
      <button class="tab" onclick="pickMethod('layoutlm', this)">
        Method 3 — LayoutLM
        <span class="badge">fine-tuned</span>
      </button>
    </div>

    <div class="card-title">2 · Upload receipt image</div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/*" onchange="onFileChange(this)">
      <div class="icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.5">
          <path d="M12 16V8m0 0-3 3m3-3 3 3"/>
          <path d="M20 16.7A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/>
        </svg>
      </div>
      <p><strong>Click to browse</strong> or drag &amp; drop</p>
      <p style="font-size:.8rem;margin-top:.3rem">JPEG · PNG · BMP · TIFF · WebP</p>
      <div id="preview"></div>
    </div>

    <button class="btn-predict" id="predictBtn" disabled onclick="predict()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
      Extract Information
    </button>

    <div class="spinner" id="spinner">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
                 M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>
      Processing — this may take a moment…
    </div>
    <div class="error-box" id="errorBox"></div>
  </div>

  <!-- Result card -->
  <div class="card" id="resultCard">
    <div class="card-title">3 · Extracted fields</div>
    <div id="methodTag"></div>
    <div class="fields" id="fieldsGrid"></div>
    <div style="margin-top:1rem;">
      <div class="card-title" style="margin-bottom:.5rem">Try other method</div>
      <div class="compare">
        <button class="compare-btn" onclick="pickMethodAndPredict('rule')">▶ Rule-based</button>
        <button class="compare-btn" onclick="pickMethodAndPredict('llm')">▶ LLM (Gemini)</button>
        <button class="compare-btn" onclick="pickMethodAndPredict('layoutlm')">▶ LayoutLM</button>
      </div>
    </div>
  </div>

</main>

<script>
let currentMethod = 'rule';

function pickMethod(m, el) {
  currentMethod = m;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
}

function onFileChange(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview').innerHTML =
      `<img src="${e.target.result}" alt="receipt preview" style="max-height:260px;border-radius:10px;margin-top:.8rem">`;
    document.getElementById('predictBtn').disabled = false;
  };
  reader.readAsDataURL(file);
}

// Drag & drop
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  const input = document.getElementById('fileInput');
  input.files = dt.files;
  onFileChange(input);
});

async function predict() {
  const file = document.getElementById('fileInput').files[0];
  if (!file) return;

  const btn     = document.getElementById('predictBtn');
  const spinner = document.getElementById('spinner');
  const errBox  = document.getElementById('errorBox');
  const resCard = document.getElementById('resultCard');

  btn.disabled = true;
  spinner.style.display = 'flex';
  errBox.style.display  = 'none';
  resCard.style.display = 'none';

  try {
    const fd  = new FormData();
    fd.append('file', file);
    const res = await fetch(`/predict/${currentMethod}`, { method: 'POST', body: fd });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Server error ${res.status}`);
    }
    renderResult(await res.json());
  } catch (err) {
    errBox.textContent    = '⚠ ' + err.message;
    errBox.style.display  = 'block';
  } finally {
    btn.disabled          = false;
    spinner.style.display = 'none';
  }
}

function renderResult(data) {
  // Method tag
  document.getElementById('methodTag').innerHTML =
    `<span class="method-tag">${data.method || currentMethod}</span>`;

  const fields = [
    { key: 'company', label: 'Company',  full: false },
    { key: 'total',   label: 'Total',    full: false },
    { key: 'date',    label: 'Date',     full: false },
    { key: 'address', label: 'Address',  full: true  },
  ];

  document.getElementById('fieldsGrid').innerHTML = fields.map(f => {
    const val  = (data[f.key] || '').trim();
    const disp = val && val !== 'Not Found'
      ? `<span class="field-value">${escHtml(val)}</span>`
      : `<span class="field-value missing">Not found</span>`;
    return `<div class="field${f.full ? ' full' : ''}">
      <div class="field-label">${f.label}</div>${disp}</div>`;
  }).join('');

  document.getElementById('resultCard').style.display = 'block';
  document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function pickMethodAndPredict(m) {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach((t, i) => {
    if ((i === 0 && m === 'rule') || (i === 1 && m === 'llm') || (i === 2 && m === 'layoutlm')) {
      t.classList.add('active');
    } else {
      t.classList.remove('active');
    }
  });
  currentMethod = m;
  predict();
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.post("/predict/rule", summary="Rule-based extraction")
async def predict_rule(file: UploadFile = File(...)):
    """Upload a receipt image and extract fields via PaddleOCR + rules."""
    tmp = _save_upload(file)
    try:
        result = extract_with_rules(str(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        tmp.unlink(missing_ok=True)
    return JSONResponse(result)


@app.post("/predict/llm", summary="LLM-based extraction")
async def predict_llm(file: UploadFile = File(...)):
    """Upload a receipt image and extract fields via PaddleOCR + Gemini."""
    tmp = _save_upload(file)
    try:
        result = extract_with_llm(str(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        tmp.unlink(missing_ok=True)
    return JSONResponse(result)


@app.get("/debug/layoutlm", summary="LayoutLM model state")
async def debug_layoutlm():
    """Return current LayoutLM model loading state for diagnostics."""
    from method3_layoutlm import _model_available, _ID2LABEL, _MODEL_PATH, _load_error
    return JSONResponse({
        "model_available": _model_available,
        "model_path": str(_MODEL_PATH),
        "model_path_exists": _MODEL_PATH.exists(),
        "id2label": {str(k): v for k, v in _ID2LABEL.items()},
        "load_error": _load_error,
    })


@app.post("/predict/layoutlm", summary="LayoutLM-based extraction")
async def predict_layoutlm(file: UploadFile = File(...)):
    """Upload a receipt image and extract fields via fine-tuned LayoutLM."""
    print(f"[/predict/layoutlm] Received file: {file.filename}, size hint: {file.size}")
    tmp = _save_upload(file)
    print(f"[/predict/layoutlm] Saved to temp file: {tmp}")
    try:
        result = extract_with_layoutlm(str(tmp))
        print(f"[/predict/layoutlm] Result: {result}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[/predict/layoutlm] ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return JSONResponse(result)
