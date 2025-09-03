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

# Directions headers & cues
DIRECTIONS_HEADER_PAT = re.compile(
    r"\b(directions|directions for use|how to use|usage|preparation|how to prepare|"
    r"serving suggestion|instructions|method|dosage|dose)\b",
    re.IGNORECASE
)
IMPERATIVE_VERBS = re.compile(
    r"\b(take|stir|mix|add|pour|dissolve|shake|brew|steep|microwave|heat|boil|simmer|"
    r"swallow|chew|consume|drink|apply|spray)\b",
    re.IGNORECASE
)
TIME_QTY_TOKENS = re.compile(
    r"\b(\d+\s*(?:min|mins|minutes|sec|seconds|°c|°f|ml|l|cup|cups|tsp|tbsp|scoop[s]?|"
    r"capsule[s]?|tablet[s]?|drop[s]?))\b",
    re.IGNORECASE
)

# Reuse existing helpers already added earlier:
# _safe_punct_scrub, _structure_ok, _similarity
def _structure_ok_directions(s: str) -> bool:
    base = s.lower()
    has_header_or_imperative = (DIRECTIONS_HEADER_PAT.search(base) is not None) or (IMPERATIVE_VERBS.search(base) is not None)
    long_enough = len(s) >= 40
    has_time_qty = TIME_QTY_TOKENS.search(base) is not None
    return has_header_or_imperative and long_enough and has_time_qty


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

def _clean_gpt_json_block(text: str) -> str:
    """Strip ``` fences / prefixes and return the JSON object/string only."""
    import re
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"```$", "", t.strip(), flags=re.IGNORECASE)
    i = t.find("{")
    return t[i:].strip() if i != -1 else t



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

DIRECTIONS_OCR_SYSTEM = """
You are an exacting OCR agent. You will be given a crop that contains a DIRECTIONS / USAGE / PREPARATION section.
Rules:
- Return the EXACT visible text only from this section. Preserve line breaks, bullets, numbers, punctuation, symbols (°C, %, etc.).
- Do NOT add or infer missing words.
- If unreadable or not a directions section, output exactly: IMAGE_UNREADABLE
- Output plain text only.
""".strip()

DIRECTIONS_BBOX_FINDER_SYSTEM = """
You are a vision locator. You will be shown a full label page.
Return JSON ONLY with a single bounding box for the DIRECTIONS / USAGE / PREPARATION / HOW TO USE section if present:
{"bbox_pct": {"x": 0-100, "y": 0-100, "w": 0-100, "h": 0-100}, "found": true/false}
Rules:
- Coordinates are percentages of the entire image.
- Prefer the main directions/usage/preparation text panel (often near icons like a cup, spoon, clock, kettle).
- If not present, return {"found": false}.
""".strip()

DIRECTIONS_STRUCTURER_SYSTEM = """
You are a strict parser. You will receive plain text of a product's DIRECTIONS/USAGE/PREPARATION section.
Extract structured data WITHOUT guessing. If a field is not explicitly present, use null or empty list.

Return JSON ONLY with this schema:
{
  "steps": [{"order": 1, "text": "..."}, ...],
  "timings": [{"value": 2, "unit": "min"|"sec"}],
  "temperatures": [{"value": 90, "unit": "°C"|"°F"}],
  "volumes": [{"value": 200, "unit": "ml"|"L"|"cup"|"tsp"|"tbsp"}],
  "dosage": {"amount": 1, "unit": "capsule|tablet|scoop|ml|g|drops", "frequency_per_day": 1|null, "timing_notes": "with food|morning|...|null"},
  "serving_suggestion": null|string,
  "notes": null|string
}
Rules:
- Keep original wording inside step texts.
- Do NOT infer. Only extract what is explicit.
""".strip()

PICTOGRAM_TAGGER_SYSTEM = """
You are a pictogram detector. You will be shown a crop image around the DIRECTIONS area.
Return JSON ONLY with booleans and any text/numbers contained inside icons:
{
  "clock": {"present": true/false, "text": null|string},      // e.g., "2 min"
  "cup": true/false,
  "kettle": true/false,
  "spoon_stir": true/false,
  "microwave": true/false,
  "hob_pan": true/false,
  "oven": true/false,
  "shaker_bottle": true/false,
  "capsule_pill": true/false,
  "thermometer": true/false,
  "other_text_in_icons": [ "...", "..."]
}
Rules:
- If an icon clearly contains legible numbers/units (e.g., "2 min", "90°C"), put that in the 'text' or 'other_text_in_icons'.
- Do not guess.
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
            bbox_pt = vec["bbox_pixels"]              # <-- points @ 72 dpi
            scale   = render_dpi / 72.0
            bbox = tuple(int(v * scale) for v in bbox_pt)  # <-- pixels @ render_dpi
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

def process_artwork_directions(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 350,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Auto-locate and extract DIRECTIONS/USAGE/PREPARATION, including pictograms.
    Returns:
    {
      "ok": bool,
      "page_index": int,
      "bbox_pixels": [x0,y0,x1,y1] | None,
      "directions_text": str,
      "steps_html": str,
      "structured": {...},
      "pictograms": {...},
      "qa": {"similarity_to_baseline": int|None, "flags": [...],
             "structure_pass": bool, "consistency_ok": bool,
             "consistency_ratio": float, "accepted": bool},
      "debug": {...}
    }
    """
    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        # Vector → OCR → GPT-fallback
        vec = _pdf_find_directions_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            # _pdf_find_directions_block returns 72-dpi page coords → scale to render_dpi pixels
            bbox_pts = vec["bbox_pixels"]  # actually in points (72 dpi)
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
        
            # clamp + pad to avoid out-of-bounds and give GPT a little context
            bbox = _clamp_pad_bbox(bbox, pages[page_idx].size, pad_frac=0.02)
            if not bbox:
                # if scaling/clamping broke it, fall back to OCR/GPT locator
                img = pages[page_idx]
                bbox = (_find_region_via_ocr_directions(img)
                        or _gpt_bbox_locator_directions(client, img, model))
                if not bbox:
                    return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the PDF.")
                # clamp the fallback too
                bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)
                if not bbox:
                    return _fail("Directions bbox invalid after clamp.")
            img = pages[page_idx]
        
        else:
            page_idx = None; bbox = None; img = None
            for i, page_img in enumerate(pages):
                # OCR → GPT locator
                cand = (_find_region_via_ocr_directions(page_img)
                        or _gpt_bbox_locator_directions(client, page_img, model))
                if cand:
                    cand = _clamp_pad_bbox(cand, page_img.size, pad_frac=0.02)
                    if cand:
                        bbox = cand; page_idx = i; img = page_img; break
            if page_idx is None or img is None or bbox is None:
                return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the PDF.")
        
        # (Optional) sanity: if the bbox is extremely small, re-try GPT locator
        x0, y0, x1, y1 = bbox
        W, H = img.size
        if (x1 - x0) * (y1 - y0) < 0.02 * (W * H):  # <2% of page area is suspiciously small
            alt = _gpt_bbox_locator_directions(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt
        
        crop_bytes = _crop_to_bytes(img, bbox)

    else:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        bbox = (_find_region_via_ocr_directions(img)
                or _gpt_bbox_locator_directions(client, img, model))
        if not bbox:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the image.")
        crop_bytes = _crop_to_bytes(img, bbox)
        page_idx = 0

    # --- OCR pass 1
    raw = _gpt_exact_ocr_directions(client, crop_bytes, model)
    if raw.upper() == "IMAGE_UNREADABLE":
        return {
            "ok": False,
            "error": "Detected directions crop unreadable.",
            "page_index": page_idx,
            "bbox_pixels": list(map(int, bbox)) if bbox else None
        }

    # --- Structure & consistency
    structure_pass = _structure_ok_directions(raw)
    raw2 = _gpt_exact_ocr_directions(client, crop_bytes, model)
    consist_ratio = _similarity(raw2, raw) or 0.0
    consistency_ok = (raw2 == raw) or (consist_ratio >= 98.0)

    # --- Clean non-semantic punctuation
    clean_text = _safe_punct_scrub(raw)

    # --- Structure JSON & pictograms
    structured = _gpt_structure_directions(client, clean_text, model)
    # Fallback if structurer failed or returned nothing useful
    if (structured.get("error") == "STRUCTURE_PARSE_FAILED"
        or (not structured.get("steps") and not structured.get("dosage", {}).get("amount"))):
        fb = _lightweight_directions_fallback(clean_text)
        if not structured.get("steps"):
            structured["steps"] = fb["steps"]
        if (not structured.get("dosage")
            or structured["dosage"].get("amount") is None):
            structured["dosage"] = fb["dosage"]

    pictos = _gpt_pictograms(client, crop_bytes, model)

    # --- Baseline OCR QA
    qa = _qa_compare_tesseract(crop_bytes, clean_text)
    qa.update({
        "structure_pass": structure_pass,
        "consistency_ok": consistency_ok,
        "consistency_ratio": consist_ratio,
    })
    qa["accepted"] = bool(structure_pass and consistency_ok)

    # --- Steps HTML (ordered list if steps exist)
    steps_html = ""
    if isinstance(structured, dict) and structured.get("steps"):
        items = "".join(f"<li>{s.get('text','').strip()}</li>" for s in structured["steps"])
        steps_html = f"<ol>{items}</ol>"

    return {
        "ok": True,
        "page_index": page_idx,
        "bbox_pixels": list(map(int, bbox)) if bbox else None,
        "directions_text": clean_text,
        "steps_html": steps_html,
        "structured": structured,
        "pictograms": pictos,
        "qa": qa,
        "debug": {
            "image_size": img.size if is_pdf or img else None,
            "tesseract_available": TESS_AVAILABLE
        }
    }


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

def _clamp_pad_bbox(bbox, img_size, pad_frac=0.02):
    """Clamp bbox to image and add small padding (fraction of min(W,H))."""
    x0, y0, x1, y1 = map(int, bbox)
    W, H = img_size
    # clamp
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H - 1))
    y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return None

    # pad (2% by default), then clamp again
    pad = int(pad_frac * min(W, H))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return (x0, y0, x1, y1)

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

def _pdf_find_directions_block(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, txt, *_ = b
            if not txt:
                continue
            if DIRECTIONS_HEADER_PAT.search(txt):
                # Grow region downward to capture bullets/paragraph under header
                margin = max((y1 - y0) * 0.6, 20)
                bbox = (x0, max(0, y0 - margin), x1, y1 + margin * 3)  # was *6
                return {"page_index": i, "bbox_pixels": tuple(map(int, bbox))}
    return None

def _find_region_via_ocr_directions(full_img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    candidates = []
    for i, word in enumerate(data["text"]):
        if not word:
            continue
        w_lower = word.lower()
        if (DIRECTIONS_HEADER_PAT.search(w_lower)
            or IMPERATIVE_VERBS.search(w_lower)
            or TIME_QTY_TOKENS.search(w_lower)):
            x = data["left"][i]; y = data["top"][i]
            w = data["width"][i]; h = data["height"][i]
            # Expand to a block likely containing the steps & icons nearby
            x0 = max(0, x - int(0.07 * W))
            x1 = min(W, x + w + int(0.07 * W))
            y0 = max(0, y - int(0.03 * H))
            y1 = min(H, y + h + int(0.50 * H))
            candidates.append((x0, y0, x1, y1))
    if candidates:
        # Choose the largest area candidate
        return max(candidates, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    return None

def _gpt_bbox_locator_directions(client, img: Image.Image, model: str) -> Optional[Tuple[int, int, int, int]]:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": DIRECTIONS_BBOX_FINDER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Locate Directions/Usage/Preparation area and return JSON only."},
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
        return (x, y, x + w, y + h)
    except Exception:
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

def _gpt_exact_ocr_directions(client, crop_bytes: bytes, model: str) -> str:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": DIRECTIONS_OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the exact Directions/Usage/Preparation text only."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _gpt_structure_directions(client, raw_text: str, model: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": DIRECTIONS_STRUCTURER_SYSTEM},
            {"role": "user", "content": raw_text}
        ]
    )
    raw = r.choices[0].message.content.strip()
    try:
        cleaned = _clean_gpt_json_block(raw)
        return json.loads(cleaned)
    except Exception:
        return {
            "steps": [],
            "timings": [],
            "temperatures": [],
            "volumes": [],
            "dosage": {"amount": None, "unit": None, "frequency_per_day": None, "timing_notes": None},
            "serving_suggestion": None,
            "notes": None,
            "error": "STRUCTURE_PARSE_FAILED",
            "raw_model_output": raw  # helpful in Streamlit st.json
        }

_DOSAGE_RE = re.compile(
    r"\btake\s+(\d+)\s+(capsule|capsules|tablet|tablets|scoop|scoops|drop|drops)\b.*?\b(daily|per day|a day)\b",
    re.IGNORECASE
)

def _lightweight_directions_fallback(text: str) -> Dict[str, Any]:
    out = {
        "steps": [],
        "timings": [],
        "temperatures": [],
        "volumes": [],
        "dosage": {"amount": None, "unit": None, "frequency_per_day": None, "timing_notes": None},
        "serving_suggestion": None,
        "notes": None
    }

    # Try to extract dosage
    m = _DOSAGE_RE.search(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower().rstrip('s')   # normalize (capsule/capsules → capsule)
        out["dosage"] = {
            "amount": amount,
            "unit": unit,
            "frequency_per_day": 1,   # "daily" implies once per day
            "timing_notes": None
        }

    # Split into sentences and keep instruction-like ones
    bits = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    steps = []
    for b in bits:
        b2 = b.strip(" \n\r\t•-")
        if not b2:
            continue
        if re.search(r"\b(take|stir|mix|add|pour|dissolve|shake|brew|steep|drink|apply|do not exceed)\b", b2, re.IGNORECASE):
            steps.append({"order": len(steps)+1, "text": b2})
    out["steps"] = steps

    return out


def _gpt_pictograms(client, crop_bytes: bytes, model: str) -> Dict[str, Any]:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": PICTOGRAM_TAGGER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Detect icons and any numbers/units inside them."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    try:
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {"error": "PICTO_PARSE_FAILED"}

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




