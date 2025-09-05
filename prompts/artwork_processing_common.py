# artwork_processing_common.py
from __future__ import annotations

import io, re, json, base64
from typing import Optional, Tuple, Dict, Any, List

# ---------- Third-party / optional ----------
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from PIL import Image

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    pytesseract = None
    TESS_AVAILABLE = False


# ======================================================================
# Shared regex & tokens
# ======================================================================

HEADER_PAT = re.compile(r"\bingredient[s]?\b", re.IGNORECASE)

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
    r"\b(\d+\s*(?:min|mins|minutes|sec|seconds|°c|°f|℃|℉|ml|l|cup|cups|tsp|tbsp|scoop[s]?|"
    r"capsule[s]?|tablet[s]?|drop[s]?))\b",
    re.IGNORECASE
)

# --- Pack size / weights tokens ---
_U_MASS   = r"(?:mg|g|kg|oz|lb)"
_U_VOL    = r"(?:ml|cl|l)"
_U_ALL    = rf"(?:{_U_MASS}|{_U_VOL})"
_U_COUNT  = r"(?:capsule[s]?|tablet[s]?|softgel[s]?|gumm(?:y|ies)|lozenge[s]?|sachet[s]?|stick[s]?|teabag[s]?|tea\s*bag[s]?|bar[s]?|piece[s]?|serving[s]?|portion[s]?|pouch[es]?|ampoule[s]?|caps|tabs|pcs?)"
_NET_LBL  = r"(?:net\s*(?:weight|wt\.?|contents)?)"
_GROSS_LBL = r"(?:gross\s*weight|gw)"
_DRAINED_LBL = r"(?:drained\s*(?:net\s*)?weight)"
_E_MARK  = r"(?:\u212E|℮)"

# Multipack / Count / Single qty / Labeled / Compact GW/NW
MULTIPACK_RE = re.compile(rf"\b(\d+)\s*(?:x|×|\*)\s*(\d+(?:[.,]\d+)?)\s*({_U_ALL})\b", re.IGNORECASE)
COUNT_RE = re.compile(rf"\b(\d+)\s*({_U_COUNT})\b", re.IGNORECASE)
SINGLE_QTY_RE = re.compile(rf"(?:{_E_MARK}\s*)?\b(\d+(?:[.,]\d+)?)\s*({_U_ALL})\b(?:\s*{_E_MARK})?", re.IGNORECASE)
LABELED_WEIGHT_RE = re.compile(
    rf"\b(({_NET_LBL})|({_GROSS_LBL})|({_DRAINED_LBL}))\b[^\d%]*?(\d+(?:[.,]\d+)?)\s*({_U_MASS}|{_U_VOL})\b",
    re.IGNORECASE
)
COMPACT_GW_NW_RE = re.compile(r"\b(NW|GW)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(mg|g|kg|oz|lb)\b", re.IGNORECASE)


# ======================================================================
# Orientation / main-panel helper (optional)
# ======================================================================

ORIENTATION_CONTENT_FINDER_SYSTEM = """
You are a packaging preprocessor. You will see a FLAT ARTWORK sheet that may contain:
- multiple panels (front/back/sides), sometimes upside down
- trim/bleed lines, cutter guides, registration targets
- color swatches and print spec tables

Goal: identify the single MAIN consumer-facing label panel and how to orient it upright.
Return JSON ONLY:
{
  "rotation_deg": 0|90|180|270,
  "content_bbox_pct": {"x":0-100,"y":0-100,"w":0-100,"h":0-100},
  "found": true|false
}

Rules:
- Ignore dielines, crop marks, color bars, spec tables, and whitespace.
- Prefer the panel that contains brand name, product name/flavour, ingredients/nutrition/directions,
  or a large net quantity (e.g., “250 g”, “10 sachets”).
- rotation_deg rotates the whole page so the chosen panel reads upright.
- content_bbox_pct tightly frames the chosen panel after rotation.
- If nothing sensible is present, return {"found": false}.
""".strip()

def _gpt_normalize_flatpack_page(client, img: Image.Image, model: str):
    """
    Use GPT-vision to (a) choose the main consumer panel, (b) tell us the rotation,
    and (c) give a tight content bbox. Returns (pre_img, bbox_px, meta)
    or (img, None, {"found":False})
    """
    buf = io.BytesIO(); img.save(buf, format="PNG")
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, top_p=0,
            messages=[
                {"role": "system", "content": ORIENTATION_CONTENT_FINDER_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "Return JSON only."},
                    {"type": "image_url", "image_url": {"url": _encode_data_url(buf.getvalue())}}
                ]}
            ]
        )
        js = json.loads(r.choices[0].message.content.strip())
    except Exception:
        return img, None, {"found": False, "reason": "locator_failed"}

    if not js.get("found"):
        return img, None, {"found": False}

    rot = int(js.get("rotation_deg", 0)) % 360
    pct = js.get("content_bbox_pct") or {}

    rotated = img.rotate(-rot, expand=True)  # PIL CCW for +angle; invert
    RW, RH = rotated.size
    x = int(RW * float(pct.get("x", 0)) / 100.0)
    y = int(RH * float(pct.get("y", 0)) / 100.0)
    w = int(RW * float(pct.get("w", 0)) / 100.0)
    h = int(RH * float(pct.get("h", 0)) / 100.0)
    bbox = (max(0, x), max(0, y), min(RW, x + w), min(RH, y + h))
    pre = rotated.crop(bbox) if w > 0 and h > 0 else rotated
    return pre, bbox, {"found": True, "rotation_deg": rot}


# ======================================================================
# Utility helpers
# ======================================================================

def _fail(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}

def _pdf_to_page_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed; cannot render PDF pages.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    imgs: List[Image.Image] = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return imgs

def _pdf_to_page_images_adaptive(pdf_bytes: bytes, *, dpi_primary=350, dpi_fallback=600):
    """
    Render all pages at dpi_primary first. Return (pages, scale72, dpi_primary, dpi_fallback).
    For any page later flagged unreadable, call _rerender_single_page with dpi_fallback.
    """
    pages = _pdf_to_page_images(pdf_bytes, dpi=dpi_primary)
    return pages, (dpi_primary/72.0), dpi_primary, dpi_fallback

def _rerender_single_page(pdf_bytes: bytes, page_index: int, *, dpi: int):
    if fitz is None:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_index < 0 or page_index >= len(doc):
        return None
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = doc[page_index].get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img

def _safe_punct_scrub(s: str) -> str:
    return (str(s).replace("))", ")")
                  .replace(" ,", ",")
                  .replace(" .", ".")
                  .replace(" :", ":")
            ).strip()

def _structure_ok_ingredients(s: str) -> bool:
    base = (s or "").lower()
    return ("ingredient" in base) and (len(s or "") >= 50)

def _structure_ok_directions(s: str) -> bool:
    base = (s or "").lower()
    has_header_or_imperative = (DIRECTIONS_HEADER_PAT.search(base) is not None) or (IMPERATIVE_VERBS.search(base) is not None)
    long_enough = len(s or "") >= 40
    has_time_qty = TIME_QTY_TOKENS.search(base) is not None
    return has_header_or_imperative and long_enough and has_time_qty

def _similarity(a: str, b: str) -> float | None:
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio((a or "").strip(), (b or "").strip()))
    except Exception:
        return None

def _clean_gpt_json_block(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"```$", "", t.strip(), flags=re.IGNORECASE)
    i = t.find("{")
    return t[i:].strip() if i != -1 else t

def _area_pct(bbox: Tuple[int,int,int,int], size: Tuple[int,int]) -> float:
    (x0,y0,x1,y1) = bbox; (W,H) = size
    if W <= 0 or H <= 0: return 0.0
    return round(100.0 * max(0, x1-x0) * max(0, y1-y0) / (W*H), 2)

def _num(s: str) -> float:
    return float(str(s).replace(",", ".").strip())

def _norm_unit(u: str) -> str:
    u = str(u).strip().lower().replace(" ", "")
    if u in {"millilitre", "millilitres"}: return "ml"
    if u in {"litre", "litres"}: return "l"
    if u in {"gram", "grams"}: return "g"
    if u in {"kilogram", "kilograms"}: return "kg"
    if u in {"teabag", "tea bag", "teabags", "tea bags"}: return "teabags"
    if u in {"gummie", "gummy"}: return "gummies"
    if u in {"caps"}: return "capsules"
    if u in {"tabs"}: return "tablets"
    if u in {"pcs", "pc"}: return "pieces"
    return u

def _title_if_count(u: str | None) -> str | None:
    if not u: return None
    if re.fullmatch(_U_COUNT, u, flags=re.IGNORECASE):
        base = _norm_unit(u)
        return base.capitalize() if base != "teabags" else "Teabags"
    return _norm_unit(u)

def _encode_data_url(image_bytes: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

def _crop_to_bytes(img: Image.Image, bbox: Tuple[int,int,int,int]) -> bytes:
    x0,y0,x1,y1 = map(int, bbox)
    crop = img.crop((x0,y0,x1,y1))
    out = io.BytesIO(); crop.save(out, format="PNG")
    return out.getvalue()

def _ocr_words(image: Image.Image):
    if not TESS_AVAILABLE:
        return None
    try:
        return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

def _ocr_words_image_to_string(crop_bytes: bytes) -> str:
    if not TESS_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(crop_bytes)))
    except Exception:
        return ""

def _qa_compare_tesseract(crop_bytes: bytes, gpt_text: str) -> Dict[str, Any]:
    flags: List[str] = []
    ratio = None
    baseline = ""
    if TESS_AVAILABLE:
        try:
            baseline = _ocr_words_image_to_string(crop_bytes)
            try:
                from rapidfuzz import fuzz
                ratio = fuzz.ratio((baseline or "").strip(), (gpt_text or "").strip()) if baseline else None
            except Exception:
                ratio = None
            if baseline and ratio is not None and ratio < 90:
                flags.append("LOW_SIMILARITY_TO_BASELINE_OCR")
            # NEW: low text density can hint we need hi-DPI rerender
            alnum = len(re.findall(r"[A-Za-z0-9]", baseline or ""))
            if baseline and alnum < 12:
                flags.append("LOW_TEXT_DENSITY")
        except Exception:
            pass
    if "IMAGE_UNREADABLE" in (gpt_text or "").upper():
        flags.append("IMAGE_UNREADABLE")
    return {"similarity_to_baseline": ratio, "flags": flags}

def _clamp_pad_bbox(bbox, img_size, pad_frac=0.02):
    if not bbox: return None
    x0, y0, x1, y1 = map(int, bbox)
    W, H = img_size
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H - 1))
    y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max(2, int(pad_frac * min(W, H)))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return (x0, y0, x1, y1)

def _bbox_pct_to_pixels(pct: Dict[str,float], size: Tuple[int,int]) -> Tuple[int,int,int,int]:
    W,H = size
    x = int(W * float(pct.get("x",0)) / 100.0)
    y = int(H * float(pct.get("y",0)) / 100.0)
    w = int(W * float(pct.get("w",0)) / 100.0)
    h = int(H * float(pct.get("h",0)) / 100.0)
    return (x, y, x+w, y+h)


# ======================================================================
# Optional robustness helpers (opt-in in feature modules)
# ======================================================================

def _rotate_to_upright_ocr(img: Image.Image) -> Tuple[Image.Image, int, Dict[str, Any]]:
    """
    Try 0/90/180/270, choose the orientation with the most OCR word characters.
    Returns (rotated_img, rotation_deg, meta). No throw.
    """
    if not TESS_AVAILABLE:
        return img, 0, {"tess": False}

    def score(pil_img):
        try:
            txt = pytesseract.image_to_string(pil_img)
        except Exception:
            return 0
        return len(re.findall(r"[A-Za-z0-9]", txt or ""))

    best_deg, best_img, best_score = 0, img, -1
    for deg in (0, 90, 180, 270):
        candidate = img.rotate(-deg, expand=True) if deg else img
        s = score(candidate)
        if s > best_score:
            best_deg, best_img, best_score = deg, candidate, s

    return best_img, best_deg, {"tess": True, "score": best_score}

def _maybe_rerender_and_rescale(
    pdf_bytes: bytes,
    page_index: int,
    bbox_px_at_render_dpi: Optional[Tuple[int,int,int,int]],
    *,
    render_dpi: int,
    dpi_fallback: int = 600,
    min_short_side_px: int = 1200
) -> Tuple[Optional[Image.Image], Optional[Tuple[int,int,int,int]], Dict[str, Any]]:
    """
    If current page render (render_dpi) is small or text is thin, re-render this ONE page at dpi_fallback
    and rescale the bbox. Returns (hi_img, hi_bbox, meta).
    If re-render not needed/possible: (None, None, {"rerendered":False}).
    """
    if fitz is None:
        return None, None, {"rerendered": False, "reason": "no_fitz"}

    hi_img = _rerender_single_page(pdf_bytes, page_index, dpi=dpi_fallback)
    if hi_img is None:
        return None, None, {"rerendered": False, "reason": "rerender_failed"}

    if not bbox_px_at_render_dpi:
        return hi_img, None, {"rerendered": True, "scale": dpi_fallback / float(render_dpi)}

    scale = dpi_fallback / float(render_dpi)
    x0, y0, x1, y1 = map(int, bbox_px_at_render_dpi)
    hi_bbox = (int(round(x0 * scale)), int(round(y0 * scale)),
               int(round(x1 * scale)), int(round(y1 * scale)))

    # Ensure the crop isn't still tiny
    w = max(1, hi_bbox[2] - hi_bbox[0]); h = max(1, hi_bbox[3] - hi_bbox[1])
    if min(w, h) < min_short_side_px:
        hi_bbox = _clamp_pad_bbox(hi_bbox, hi_img.size, pad_frac=0.04)

    return hi_img, hi_bbox, {"rerendered": True, "scale": scale}

def _pdf_is_vector_text_heavy(pdf_bytes: bytes, sample_pages: int = 2) -> bool:
    """
    Quick sniff: if selectable text is abundant on first pages, prefer vector heuristics first.
    """
    if fitz is None:
        return False
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        if i >= sample_pages:
            break
        text = page.get_text("text") or ""
        if len(text.strip()) >= 200:
            return True
    return False


# ======================================================================
# Heuristic panel guesses (shared)
# ======================================================================

def _fallback_left_panel_bbox(img: Image.Image) -> Tuple[int,int,int,int]:
    W, H = img.size
    left = int(0.07 * W); right = int(0.45 * W)
    top = int(0.10 * H);  bottom = int(0.90 * H)
    return (left, top, right, bottom)

def _fallback_right_panel_bbox(img: Image.Image) -> Tuple[int,int,int,int]:
    W, H = img.size
    left = int(0.55 * W); right = int(0.93 * W)
    top = int(0.10 * H);  bottom = int(0.90 * H)
    return (left, top, right, bottom)

def _fallback_center_panel_bbox(img: Image.Image) -> Tuple[int,int,int,int]:
    W, H = img.size
    left = int(0.30 * W); right = int(0.70 * W)
    top = int(0.12 * H);  bottom = int(0.88 * H)
    return (left, top, right, bottom)


# ======================================================================
# Vector-text heuristics (PDF) — fast “selectable text” scans
# ======================================================================

def _pdf_find_ingredient_block(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    if fitz is None:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            x0,y0,x1,y1,txt, *_ = b
            if txt and HEADER_PAT.search(txt):
                margin = max((y1 - y0) * 0.6, 20)
                bbox = (x0, max(0, y0 - margin), x1, y1 + margin*6)
                return {"page_index": i, "bbox_pixels": tuple(map(int, bbox))}
    return None

def _pdf_find_directions_block(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    if fitz is None:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, txt, *_ = b
            if not txt:
                continue
            if DIRECTIONS_HEADER_PAT.search(txt):
                margin = max((y1 - y0) * 0.6, 20)
                bbox = (x0, max(0, y0 - margin), x1, y1 + margin * 3)
                return {"page_index": i, "bbox_pixels": tuple(map(int, bbox))}
    return None

def _pdf_find_packsize_block(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    if fitz is None:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    unit_hint = re.compile(rf"\b({_U_ALL}|{_U_COUNT})\b", re.IGNORECASE)
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        best = None
        for b in blocks:
            x0, y0, x1, y1, txt, *_ = b
            if not txt:
                continue
            if unit_hint.search(txt) or MULTIPACK_RE.search(txt) or COUNT_RE.search(txt) or SINGLE_QTY_RE.search(txt):
                margin = max((y1 - y0) * 0.8, 18)
                cand = (x0, max(0, y0 - margin), x1, y1 + margin * 1.2)
                if not best or (cand[2]-cand[0]) > (best[2]-best[0]):
                    best = cand
        if best:
            return {"page_index": i, "bbox_pixels": tuple(map(int, best))}
    return None

def _pdf_find_nutrition_block(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    if fitz is None:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    header = re.compile(
        r"(nutrition|nutritional|typical values|per\s*100\s*(?:g|ml)|per\s*(?:serving|capsule|tablet)|%?\s*(?:nrv|ri)|\bkJ\b|\bkcal\b)",
        re.IGNORECASE
    )
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        best = None
        for b in blocks:
            x0, y0, x1, y1, txt, *_ = b
            if not txt:
                continue
            if header.search(txt):
                margin = max((y1 - y0) * 0.8, 18)
                cand = (x0, max(0, y0 - margin), x1, y1 + margin * 2.0)
                if not best or (cand[2]-cand[0]) > (best[2]-best[0]):
                    best = cand
        if best:
            return {"page_index": i, "bbox_pixels": tuple(map(int, best))}
    return None
