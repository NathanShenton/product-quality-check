# artwork_processing.py
# Fully automatic Ingredients extraction from PDF/JPEG/PNG
# - Locates panel (vector text -> OCR boxes -> GPT bbox)
# - Strict OCR via GPT-4o (temperature=0)
# - HTML allergen bolding
# - QA similarity + bbox/page metadata

from __future__ import annotations
import io, re, json, base64
from typing import Optional, Tuple, Dict, Any, List

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from PIL import Image
import numpy as np

# Optional OCR: module works without it (will use GPT bbox fallback), but QA improves if present
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    pytesseract = None
    TESS_AVAILABLE = False

HEADER_PAT = re.compile(r"\bingredient[s]?\b", re.IGNORECASE)

# ---- QA / Utility helpers ----
def _safe_punct_scrub(s: str) -> str:
    # Non-semantic clean-ups only
    return (
        s.replace("))", ")")
         .replace(" ,", ",")
         .replace(" .", ".")
         .replace(" :", ":")
    ).strip()

def _structure_ok(s: str) -> bool:
    """
    Lightweight acceptance guard:
    - must include an 'ingredient' header
    - must be reasonably long
    - should include at least one expected token (allergen or key active); tune as you like
    """
    base = s.lower()
    has_header = ("ingredient" in base)
    long_enough = (len(s) >= 60)
    has_expected_token = any(t in base for t in ["soy", "soya", "pumpkin seed"])
    return has_header and long_enough and has_expected_token

def _similarity(a: str, b: str) -> float | None:
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(a.strip(), b.strip()))
    except Exception:
        return None

# ---------- System prompts (strict, zero-temp) ----------

INGREDIENT_OCR_SYSTEM = """
You are an exacting OCR agent. You will be given an image crop that contains a UK/EU food label INGREDIENTS statement.

Rules:
- Return the EXACT visible text. Preserve punctuation, brackets, symbols (%), capitalization and ordering.
- Do NOT infer or add text that is not clearly readable.
- If unreadable or missing, output exactly: IMAGE_UNREADABLE
- Output plain text only.
""".strip()

INGREDIENT_HTML_SYSTEM = """
You are a food-label compliance formatter. You will be given a single INGREDIENTS string.
Return the same text as HTML, but bold (<b>...</b>) UK FIC allergens only:
celery,wheat,rye,barley,oats,spelt,kamut,crustaceans,eggs,fish,lupin,milk,
molluscs,mustard,almond,hazelnut,walnut,cashew,pecan,pistachio,macadamia,
brazil nut,peanut,sesame,soy,soya,sulphur dioxide,sulphites
Rules:
- Bold ONLY the allergen tokens that appear (including common plural/suffix variants).
- Do not re-order, translate or summarize.
- Return HTML only (no commentary).
""".strip()

BBOX_FINDER_SYSTEM = """
You are a vision locator. You will be shown a full label page.
Return JSON ONLY with a single bounding box for the INGREDIENTS panel if present:
{"bbox_pct": {"x": 0-100, "y": 0-100, "w": 0-100, "h": 0-100}, "found": true/false}
Rules:
- Coordinates are percentages of the entire image.
- Prefer the main INGREDIENTS panel containing the full statement.
- If not present, return {"found": false}.
""".strip()

# ---------- Public API ----------

def process_artwork(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 350,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Fully automatic pipeline. Returns dict:
    {
      "ok": bool,
      "page_index": int,
      "bbox_pixels": [x0,y0,x1,y1] or None,
      "ingredients_text": str,
      "ingredients_html": str,
      "qa": {"similarity_to_baseline": int|None, "flags": [..]},
      "debug": {...}
    }
    """
    is_pdf = filename.lower().endswith(".pdf")
    pages: List[Image.Image] = []

    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")
        # Try vector find on page 0 first; if nothing, iterate (rare multi-page packs)
        vec = _pdf_find_ingredient_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox = vec["bbox_pixels"]
            crop_bytes = _crop_to_bytes(pages[page_idx], bbox)
            return _final_ocr_and_format(client, crop_bytes, model, page_idx, bbox, pages[page_idx])
        # No vector hit → fallback per page (OCR boxes then GPT bbox)
        for page_idx, img in enumerate(pages):
            bbox = _find_region_via_ocr(img)  # may be None
            if not bbox:
                bbox = _gpt_bbox_locator(client, img, model)  # may be None
            if bbox:
                crop_bytes = _crop_to_bytes(img, bbox)
                return _final_ocr_and_format(client, crop_bytes, model, page_idx, bbox, img)
        # No page had a detectable panel
        return _fail("Could not locate an INGREDIENTS panel in the PDF.")
    else:
        # Single image
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        bbox = _find_region_via_ocr(img)
        if not bbox:
            bbox = _gpt_bbox_locator(client, img, model)
        if not bbox:
            return _fail("Could not locate an INGREDIENTS panel in the image.")
        crop_bytes = _crop_to_bytes(img, bbox)
        return _final_ocr_and_format(client, crop_bytes, model, page_index=0, bbox=bbox, full_image=img)

# ---------- Internals ----------

def _fail(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}

def _pdf_to_page_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    imgs = []
    for page in doc:
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return imgs

def _pdf_find_ingredient_block(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        # Prefer structure with bboxes
        blocks = page.get_text("blocks")
        for b in blocks:
            x0,y0,x1,y1,txt, *_ = b
            if txt and HEADER_PAT.search(txt):
                # Grow downward generously to capture the full paragraph under header
                margin = max((y1 - y0) * 0.6, 20)
                bbox = (x0, max(0, y0 - margin), x1, y1 + margin*6)
                return {"page_index": i, "bbox_pixels": tuple(map(int, bbox))}
    return None

def _ocr_words(image: Image.Image):
    if not TESS_AVAILABLE:
        return None
    try:
        return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

def _find_region_via_ocr(full_img: Image.Image) -> Optional[Tuple[int,int,int,int]]:
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    candidates = []
    for i, word in enumerate(data["text"]):
        if not word:
            continue
        if HEADER_PAT.search(word):
            x = data["left"][i]; y = data["top"][i]
            w = data["width"][i]; h = data["height"][i]
            # Expand to a column under the header
            x0 = max(0, x - int(0.05*W))
            x1 = min(W, x + w + int(0.05*W))
            y0 = max(0, y - int(0.02*H))
            y1 = min(H, y + h + int(0.45*H))
            candidates.append((x0,y0,x1,y1))
    if candidates:
        # Choose the tallest – tends to include full statement
        return max(candidates, key=lambda b: (b[3]-b[1]))
    return None

def _encode_data_url(image_bytes: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

def _gpt_bbox_locator(client, img: Image.Image, model: str) -> Optional[Tuple[int,int,int,int]]:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    r = client.chat.completions.create(
        model=model,
        temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": BBOX_FINDER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Locate the INGREDIENTS panel and return JSON only."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    try:
        js = json.loads(r.choices[0].message.content.strip())
        if not js.get("found"):
            return None
        W, H = img.size
        pct = js["bbox_pct"]
        x = int(W * pct["x"] / 100.0); y = int(H * pct["y"] / 100.0)
        w = int(W * pct["w"] / 100.0); h = int(H * pct["h"] / 100.0)
        return (x, y, x+w, y+h)
    except Exception:
        return None

def _crop_to_bytes(img: Image.Image, bbox: Tuple[int,int,int,int]) -> bytes:
    x0,y0,x1,y1 = map(int, bbox)
    crop = img.crop((x0,y0,x1,y1))
    out = io.BytesIO(); crop.save(out, format="PNG")
    return out.getvalue()

def _gpt_exact_ocr(client, crop_bytes: bytes, model: str) -> str:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model,
        temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": INGREDIENT_OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the exact text of the Ingredients statement."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _gpt_html_allergen_bold(client, ingredient_text: str, model: str) -> str:
    r = client.chat.completions.create(
        model=model,
        temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": INGREDIENT_HTML_SYSTEM},
            {"role": "user", "content": ingredient_text}
        ]
    )
    return r.choices[0].message.content.strip()

def _qa_compare_tesseract(crop_bytes: bytes, gpt_text: str) -> Dict[str, Any]:
    flags = []
    ratio = None
    if TESS_AVAILABLE:
        try:
            baseline = pytesseract.image_to_string(Image.open(io.BytesIO(crop_bytes)))
            try:
                from rapidfuzz import fuzz
                ratio = fuzz.ratio(baseline.strip(), gpt_text.strip())
            except Exception:
                ratio = None
            if baseline and ratio is not None and ratio < 90:
                flags.append("LOW_SIMILARITY_TO_BASELINE_OCR")
        except Exception:
            pass
    if "IMAGE_UNREADABLE" in (gpt_text or "").upper():
        flags.append("IMAGE_UNREADABLE")
    return {"similarity_to_baseline": ratio, "flags": flags}

def _final_ocr_and_format(client, crop_bytes: bytes, model: str, page_index: int, bbox, full_image: Image.Image) -> Dict[str, Any]:
    # --- Pass 1: strict OCR
    gpt_text = _gpt_exact_ocr(client, crop_bytes, model)

    if gpt_text.upper() == "IMAGE_UNREADABLE":
        return {
            "ok": False,
            "error": "Detected panel unreadable.",
            "page_index": page_index,
            "bbox_pixels": list(map(int, bbox)) if bbox else None,
        }

    # --- Structure check
    structure_pass = _structure_ok(gpt_text)

    # --- Pass 2: consistency check (2nd OCR on same crop)
    gpt_text_2 = _gpt_exact_ocr(client, crop_bytes, model)
    consistency_ratio = _similarity(gpt_text_2, gpt_text) or 0.0
    consistency_ok = (gpt_text_2 == gpt_text) or (consistency_ratio >= 98.0)

    # --- Safe, non-semantic punctuation scrub
    clean_text = _safe_punct_scrub(gpt_text)

    # --- Allergen bolding on clean text
    html_out = _gpt_html_allergen_bold(client, clean_text, model)

    # --- Baseline OCR QA (Tesseract)
    qa = _qa_compare_tesseract(crop_bytes, clean_text)

    # Enrich QA with new signals + a decision
    qa.update({
        "structure_pass": structure_pass,
        "consistency_ok": consistency_ok,
        "consistency_ratio": consistency_ratio,
    })

    # Promote acceptance even if Tesseract similarity is low,
    # provided our structure + consistency are strong.
    accepted = bool(structure_pass and consistency_ok)
    if not accepted and "LOW_SIMILARITY_TO_BASELINE_OCR" in qa.get("flags", []):
        # Keep the flag, but still record our internal judgement:
        qa.setdefault("flags", [])
    qa["accepted"] = accepted

    return {
        "ok": True,
        "page_index": page_index,
        "bbox_pixels": list(map(int, bbox)) if bbox else None,
        "ingredients_text": clean_text,   # return scrubbed (non-semantic) text
        "ingredients_html": html_out,
        "qa": qa,
        "debug": {
            "image_size": full_image.size,
            "tesseract_available": TESS_AVAILABLE
        }
    }

