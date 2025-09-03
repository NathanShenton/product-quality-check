# artwork_processing.py
# Fully automatic Ingredients, Directions, and Pack Size/Weight extraction from PDF/JPEG/PNG
# - Panel location: PDF vector → OCR boxes → GPT bbox (with clamp+pad & tiny-area fallback)
# - Strict OCR via GPT-4o (temperature=0)
# - Ingredients: HTML allergen bolding
# - Directions: structurer + lightweight deterministic fallback + pictogram tagging
# - Pack Size/Weight: net quantity / multipack / count units + net/gross/drained weight + ℮ mark
# - QA: second-pass consistency, optional Tesseract baseline, bbox/page metadata

from __future__ import annotations
import io, re, json, base64
from typing import Optional, Tuple, Dict, Any, List

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from PIL import Image

# Optional OCR: improves QA but not required
try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    pytesseract = None
    TESS_AVAILABLE = False

# ---------- Regex patterns ----------
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
    r"\b(\d+\s*(?:min|mins|minutes|sec|seconds|°c|°f|ml|l|cup|cups|tsp|tbsp|scoop[s]?|"
    r"capsule[s]?|tablet[s]?|drop[s]?))\b",
    re.IGNORECASE
)

# --- Pack size / weights tokens ---
# Units for mass & volume; include common typography/spacing variants
_U_MASS = r"(?:mg|g|kg|oz|lb)"
_U_VOL  = r"(?:ml|cl|l)"
_U_ALL  = rf"(?:{_U_MASS}|{_U_VOL})"
# Count units commonly present on supplements / sachets / tea etc.
_U_COUNT = r"(?:capsule[s]?|tablet[s]?|softgel[s]?|gumm(?:y|ies)|lozenge[s]?|sachet[s]?|stick[s]?|teabag[s]?|tea\s*bag[s]?|bar[s]?|piece[s]?|serving[s]?|portion[s]?|pouch[es]?|ampoule[s]?|caps|tabs|pcs?)"
# Net/gross/drained labels
_NET_LBL = r"(?:net\s*(?:weight|wt\.?|contents)?)"
_GROSS_LBL = r"(?:gross\s*weight|gw)"
_DRAINED_LBL = r"(?:drained\s*(?:net\s*)?weight)"
_E_MARK = r"(?:\u212E|℮)"  # Unicode estimated sign

# Multipack like "4 x 250 ml" or "4x250ml" or "4 × 250ml"
MULTIPACK_RE = re.compile(
    rf"\b(\d+)\s*(?:x|×|\*)\s*(\d+(?:[.,]\d+)?)\s*({_U_ALL})\b", re.IGNORECASE
)
# Count unit like "120 capsules" or "30 sachets"
COUNT_RE = re.compile(
    rf"\b(\d+)\s*({_U_COUNT})\b", re.IGNORECASE
)
# Single net quantity like "750 g" / "1 L" / "500ml" possibly followed/preceded by ℮
SINGLE_QTY_RE = re.compile(
    rf"(?:{_E_MARK}\s*)?\b(\d+(?:[.,]\d+)?)\s*({_U_ALL})\b(?:\s*{_E_MARK})?", re.IGNORECASE
)
# Labeled weights e.g., "Net weight 500 g", "Net Wt. 500g", "Gross weight: 2.1 kg", "Drained weight 200 g"
LABELED_WEIGHT_RE = re.compile(
    rf"\b(({_NET_LBL})|({_GROSS_LBL})|({_DRAINED_LBL}))\b[^\d%]*?(\d+(?:[.,]\d+)?)\s*({_U_MASS}|{_U_VOL})\b", re.IGNORECASE
)
# Compact shipping-style GW/NW lines e.g., "NW: 5.2 kg", "GW 6.0kg"
COMPACT_GW_NW_RE = re.compile(
    r"\b(NW|GW)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(mg|g|kg|oz|lb)\b", re.IGNORECASE
)

# ---------- Utility / QA helpers ----------
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
    Lightweight acceptance guard for INGREDIENTS:
    - must include an 'ingredient' header
    - must be reasonably long
    """
    base = s.lower()
    has_header = ("ingredient" in base)
    long_enough = (len(s) >= 50)
    return has_header and long_enough

def _structure_ok_directions(s: str) -> bool:
    base = s.lower()
    has_header_or_imperative = (DIRECTIONS_HEADER_PAT.search(base) is not None) or (IMPERATIVE_VERBS.search(base) is not None)
    long_enough = len(s) >= 40
    has_time_qty = TIME_QTY_TOKENS.search(base) is not None
    return has_header_or_imperative and long_enough and has_time_qty

def _similarity(a: str, b: str) -> float | None:
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(a.strip(), b.strip()))
    except Exception:
        return None

def _clean_gpt_json_block(text: str) -> str:
    """Strip ``` fences / prefixes and return the JSON object/string only."""
    t = text.strip()
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
    return float(s.replace(",", ".").strip())

def _norm_unit(u: str) -> str:
    u = u.strip().lower().replace(" ", "")
    # canonicalise common variants
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

def _title_if_count(u: Optional[str]) -> Optional[str]:
    if not u: return None
    # For count units, Title Case (Capsules/Tablets/Sachets/Teabags/Servings/Portions/Pieces/etc.)
    if re.fullmatch(_U_COUNT, u, flags=re.IGNORECASE):
        # normalise to pluralised friendly form
        base = _norm_unit(u)
        return base.capitalize() if base != "teabags" else "Teabags"
    return _norm_unit(u)  # mass/volume stay lower-case

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
  "steps": [{"order": 1, "text": "..."}],
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
  "clock": {"present": true/false, "text": null|string},
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

# --- Pack size / weights prompts ---
PACKSIZE_BBOX_FINDER_SYSTEM = """
You are a vision locator. You will be shown a full label page.
Return JSON ONLY with a single bounding box for the MAIN net quantity / pack-size statement (for example: "750 g", "1 L", "4 x 250 ml", "120 capsules").
{"bbox_pct": {"x": 0-100, "y": 0-100, "w": 0-100, "h": 0-100}, "found": true/false}
Rules:
- Coordinates are percentages of the entire image.
- Choose the primary consumer-facing net quantity (often large type, same field of vision as the name of the food). ℮ may be adjacent.
- If multiple, prefer the one that represents the pack (not per-serving).
- If not present, return {"found": false}.
""".strip()

PACKSIZE_OCR_SYSTEM = """
You are an exacting OCR agent. You will be given a crop that contains a NET QUANTITY / PACK SIZE statement (e.g., "4 × 250 ml", "750 g", "120 capsules") and possibly nearby labels like "Net weight" or "℮".
Rules:
- Return the EXACT visible text lines relevant to quantity/weight/volume/count (including ℮ if present).
- Do NOT infer or add text.
- If unreadable or missing, output exactly: IMAGE_UNREADABLE
- Output plain text only.
""".strip()

PACKSIZE_STRUCTURER_SYSTEM = """
You are a strict parser for pack size and weights. You will receive plain text lines around the pack's net quantity.
Extract WITHOUT guessing. Return JSON ONLY with:
{
  "number_of_items": int|null,
  "base_quantity": float|null,
  "unit_of_measure": "ml|l|cl|g|kg|mg|capsules|tablets|softgels|gummies|lozenges|sachets|sticks|teabags|bars|pieces|servings|portions|pouches|null",
  "net_weight": {"value": float|null, "unit": "g|kg|ml|l|cl|null"},
  "gross_weight": {"value": float|null, "unit": "g|kg|ml|l|cl|null"},
  "drained_weight": {"value": float|null, "unit": "g|kg|ml|l|cl|null"},
  "e_mark_present": true|false|null,
  "raw_candidates": [ "..." ]  // short list of lines used for the decision
}
Rules:
- For multipacks like "4 x 250 ml", set number_of_items=4, base_quantity=250, unit_of_measure="ml".
- For counts like "120 capsules", set number_of_items=1, base_quantity=120, unit_of_measure="capsules".
- If only a single net quantity like "750 g" is present, set number_of_items=1, base_quantity=750, unit_of_measure="g".
- If "Net weight"/"Net wt"/"Net contents" is present with a value+unit, place it in net_weight.
- If "Gross weight"/"GW" is present, place it in gross_weight.
- If "Drained weight" present, place it in drained_weight.
- If you see the ℮ estimated sign, set e_mark_present=true.
- Use null for anything not explicitly present.
- Do NOT infer per-serving or nutrition values.
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
    Auto INGREDIENTS pipeline (panel locate → strict OCR → allergen HTML → QA).
    Returns:
    {
      "ok": bool,
      "page_index": int,
      "bbox_pixels": [x0,y0,x1,y1] | None,
      "ingredients_text": str,
      "ingredients_html": str,
      "qa": {...},
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

        # Vector (72dpi) → scale to render_dpi → clamp+pad → tiny-area sanity → fallback
        vec = _pdf_find_ingredient_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox_pts = vec["bbox_pixels"]   # points @ 72 dpi
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
            bbox = _clamp_pad_bbox(bbox, pages[page_idx].size, pad_frac=0.02)

            img = pages[page_idx]
            if not bbox:
                # fallback to OCR/GPT locator on the same page
                bbox = (_find_region_via_ocr(img) or _gpt_bbox_locator(client, img, model))
                if bbox:
                    bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)
            if not bbox:
                # per-page fallback search
                page_idx, img, bbox = _scan_pages_for_ingredients(client, pages, model)
                if page_idx is None:
                    return _fail("Could not locate an INGREDIENTS panel in the PDF.")
        else:
            page_idx, img, bbox = _scan_pages_for_ingredients(client, pages, model)
            if page_idx is None:
                return _fail("Could not locate an INGREDIENTS panel in the PDF.")

        # Tiny-area sanity check (<2% of page area) → GPT-locator fallback on the chosen page
        if _area_pct(bbox, img.size) < 2.0:
            alt = _gpt_bbox_locator(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)
        return _final_ocr_and_format(client, crop_bytes, model, page_idx, bbox, img)

    # Single image
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return _fail("Could not open image.")
    bbox = (_find_region_via_ocr(img) or _gpt_bbox_locator(client, img, model))
    if not bbox:
        return _fail("Could not locate an INGREDIENTS panel in the image.")
    bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)
    if not bbox:
        return _fail("Ingredients bbox invalid after clamp.")

    if _area_pct(bbox, img.size) < 2.0:  # microscopic-crop sanity
        alt = _gpt_bbox_locator(client, img, model)
        if alt:
            alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
            if alt:
                bbox = alt

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
    Auto DIRECTIONS/USAGE/PREPARATION pipeline (panel locate → strict OCR → structurer+fallback → pictograms → QA).
    Returns:
    {
      "ok": bool,
      "page_index": int,
      "bbox_pixels": [x0,y0,x1,y1] | None,
      "directions_text": str,
      "steps_html": str,
      "structured": {...},
      "pictograms": {...},
      "qa": {...},
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

        vec = _pdf_find_directions_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox_pts = vec["bbox_pixels"]  # points @ 72 dpi
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
            bbox = _clamp_pad_bbox(bbox, pages[page_idx].size, pad_frac=0.02)
            img = pages[page_idx]
            if not bbox:
                # fallback on that page
                cand = (_find_region_via_ocr_directions(img) or _gpt_bbox_locator_directions(client, img, model))
                if cand:
                    bbox = _clamp_pad_bbox(cand, img.size, pad_frac=0.02)
        else:
            # per-page scan
            page_idx, img, bbox = _scan_pages_for_directions(client, pages, model)

        if page_idx is None or img is None or bbox is None:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the PDF.")

        if _area_pct(bbox, img.size) < 2.0:
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
        bbox = (_find_region_via_ocr_directions(img) or _gpt_bbox_locator_directions(client, img, model))
        if not bbox:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the image.")
        bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)
        if not bbox:
            return _fail("Directions bbox invalid after clamp.")

        if _area_pct(bbox, img.size) < 2.0:
            alt = _gpt_bbox_locator_directions(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt

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
    # fallback if structurer returned little
    if (isinstance(structured, dict) and (
        structured.get("error") == "STRUCTURE_PARSE_FAILED"
        or (not structured.get("steps") and not structured.get("dosage", {}).get("amount"))
    )):
        fb = _lightweight_directions_fallback(clean_text)
        if not structured.get("steps"):
            structured["steps"] = fb["steps"]
        if (not structured.get("dosage") or structured["dosage"].get("amount") is None):
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
            "bbox_area_pct": _area_pct(bbox, img.size),
            "tesseract_available": TESS_AVAILABLE
        }
    }

def process_artwork_packsize(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 350,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Auto PACK SIZE / WEIGHT pipeline (locate → strict OCR → structurer → regex fallback → QA).
    Returns:
    {
      "ok": bool,
      "page_index": int,
      "bbox_pixels": [x0,y0,x1,y1] | None,
      "raw_text": str,
      "parsed": {
        "number_of_items": int|null,
        "base_quantity": float|null,
        "unit_of_measure": str|null,
        "net_weight": {"value": float|null, "unit": str|null},
        "gross_weight": {"value": float|null, "unit": str|null},
        "drained_weight": {"value": float|null, "unit": str|null},
        "e_mark_present": bool|null
      },
      "qa": {...},
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

        # Try vector text scan for likely net quantity lines
        vec = _pdf_find_packsize_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox_pts = vec["bbox_pixels"]  # points @ 72 dpi
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
            bbox = _clamp_pad_bbox(bbox, pages[page_idx].size, pad_frac=0.02)
            img = pages[page_idx]
            if not bbox:
                cand = (_find_region_via_ocr_packsize(img) or _gpt_bbox_locator_packsize(client, img, model))
                if cand:
                    bbox = _clamp_pad_bbox(cand, img.size, pad_frac=0.02)
        else:
            # per-page scan
            page_idx, img, bbox = _scan_pages_for_packsize(client, pages, model)

        if page_idx is None or img is None or bbox is None:
            return _fail("Could not locate a PACK SIZE/NET QUANTITY area in the PDF.")

        if _area_pct(bbox, img.size) < 1.0:  # even smaller type is possible; try GPT re-locate
            alt = _gpt_bbox_locator_packsize(client, img, model)
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
        bbox = (_find_region_via_ocr_packsize(img) or _gpt_bbox_locator_packsize(client, img, model))
        if not bbox:
            return _fail("Could not locate a PACK SIZE/NET QUANTITY area in the image.")
        bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)
        if not bbox:
            return _fail("Pack size bbox invalid after clamp.")

        if _area_pct(bbox, img.size) < 1.0:
            alt = _gpt_bbox_locator_packsize(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)
        page_idx = 0

    # --- OCR (strict) + consistency
    raw = _gpt_exact_ocr_packsize(client, crop_bytes, model)
    if raw.upper() == "IMAGE_UNREADABLE":
        return {
            "ok": False,
            "error": "Detected pack size crop unreadable.",
            "page_index": page_idx,
            "bbox_pixels": list(map(int, bbox)) if bbox else None
        }
    raw2 = _gpt_exact_ocr_packsize(client, crop_bytes, model)
    consist_ratio = _similarity(raw2, raw) or 0.0
    consistency_ok = (raw2 == raw) or (consist_ratio >= 98.0)

    clean_text = _safe_punct_scrub(raw)

    # --- Structured parse via GPT
    parsed = _gpt_structure_packsize(client, clean_text, model)

    # --- Fallback/augment with deterministic regex if needed
    if not parsed or (isinstance(parsed, dict) and all(
        parsed.get(k) in (None, [], {}) for k in ["number_of_items", "base_quantity", "unit_of_measure", "net_weight", "gross_weight", "drained_weight"]
    )):
        parsed = _regex_parse_packsize(clean_text)
    else:
        # augment missing fields with regex results
        reg = _regex_parse_packsize(clean_text)
        parsed = _merge_packsize(parsed, reg)

    # Normalise units presentation
    parsed["unit_of_measure"] = _title_if_count(parsed.get("unit_of_measure"))
    for k in ("net_weight", "gross_weight", "drained_weight"):
        v = parsed.get(k) or {}
        if v.get("unit"):
            v["unit"] = _norm_unit(v["unit"])
        parsed[k] = v

    # --- Baseline OCR QA
    qa = _qa_compare_tesseract(crop_bytes, clean_text)
    qa.update({
        "consistency_ok": consistency_ok,
        "consistency_ratio": consist_ratio,
    })

    return {
        "ok": True,
        "page_index": page_idx,
        "bbox_pixels": list(map(int, bbox)) if bbox else None,
        "raw_text": clean_text,
        "parsed": parsed,
        "qa": qa,
        "debug": {
            "tesseract_available": TESS_AVAILABLE,
            "bbox_area_pct": _area_pct(bbox, (img.size if is_pdf or img else (0,0))),
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
    """Return bbox in PDF points (72dpi space) — scale later to render_dpi."""
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
    """Return bbox in PDF points (72dpi space) — scale later to render_dpi."""
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
    """
    Heuristic vector scan for net quantity lines:
    - look for units (g/kg/ml/l) or count units with adjacent numbers
    - expand a local margin to include nearby ℮ or labels
    """
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
                # choose widest block (often the headline quantity)
                if not best or (cand[2]-cand[0]) > (best[2]-best[0]):
                    best = cand
        if best:
            return {"page_index": i, "bbox_pixels": tuple(map(int, best))}
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
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H - 1))
    y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return None
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
            x0 = max(0, x - int(0.05*W))
            x1 = min(W, x + w + int(0.05*W))
            y0 = max(0, y - int(0.02*H))
            y1 = min(H, y + h + int(0.45*H))
            candidates.append((x0,y0,x1,y1))
    if candidates:
        return max(candidates, key=lambda b: (b[3]-b[1]))
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
            x0 = max(0, x - int(0.07 * W))
            x1 = min(W, x + w + int(0.07 * W))
            y0 = max(0, y - int(0.03 * H))
            y1 = min(H, y + h + int(0.50 * H))
            candidates.append((x0, y0, x1, y1))
    if candidates:
        return max(candidates, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    return None

def _find_region_via_ocr_packsize(full_img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Heuristic scan for lines containing quantities/units or net/gross labels.
    """
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    candidates: List[Tuple[int,int,int,int]] = []
    unit_or_label = re.compile(rf"({_U_ALL}|{_U_COUNT}|{_NET_LBL}|{_GROSS_LBL}|{_DRAINED_LBL}|{_E_MARK})", re.IGNORECASE)

    # try to group words by (block_num, par_num, line_num) to approximate a line bbox
    rows: Dict[Tuple[int,int,int], List[int]] = {}
    for i in range(len(data["text"])):
        if not data["text"][i]:
            continue
        key = (data.get("block_num", [0])[i], data.get("par_num", [0])[i], data.get("line_num", [0])[i])
        rows.setdefault(key, []).append(i)

    for idxs in rows.values():
        txt = " ".join(data["text"][i] for i in idxs if data["text"][i])
        if not txt.strip():
            continue
        if (unit_or_label.search(txt) or MULTIPACK_RE.search(txt) or COUNT_RE.search(txt) or SINGLE_QTY_RE.search(txt)
            or LABELED_WEIGHT_RE.search(txt) or COMPACT_GW_NW_RE.search(txt)):
            xs = [data["left"][i] for i in idxs]; ys = [data["top"][i] for i in idxs]
            ws = [data["width"][i] for i in idxs]; hs = [data["height"][i] for i in idxs]
            x0 = max(0, min(xs) - int(0.03 * W))
            x1 = min(W, max(xs[i] + ws[i] for i in range(len(xs))) + int(0.03 * W))
            y0 = max(0, min(ys) - int(0.02 * H))
            y1 = min(H, max(ys[i] + hs[i] for i in range(len(ys))) + int(0.06 * H))
            candidates.append((x0,y0,x1,y1))

    if candidates:
        # choose the widest candidate (headline quantity tends to be wide/large)
        return max(candidates, key=lambda b: (b[2]-b[0]) * (1.0 + 0.5*(b[3]-b[1])))
    return None

def _gpt_bbox_locator(client, img: Image.Image, model: str) -> Optional[Tuple[int,int,int,int]]:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
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

def _gpt_bbox_locator_directions(client, img: Image.Image, model: str) -> Optional[Tuple[int, int, int, int]]]:
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

def _gpt_bbox_locator_packsize(client, img: Image.Image, model: str) -> Optional[Tuple[int, int, int, int]]:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": PACKSIZE_BBOX_FINDER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Locate the MAIN net quantity / pack-size statement and return JSON only."},
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

def _crop_to_bytes(img: Image.Image, bbox: Tuple[int,int,int,int]) -> bytes:
    x0,y0,x1,y1 = map(int, bbox)
    crop = img.crop((x0,y0,x1,y1))
    out = io.BytesIO(); crop.save(out, format="PNG")
    return out.getvalue()

def _ocr_words_image_to_string(crop_bytes: bytes) -> str:
    if not TESS_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(crop_bytes)))
    except Exception:
        return ""

def _gpt_exact_ocr(client, crop_bytes: bytes, model: str) -> str:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
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
        model=model, temperature=0, top_p=0,
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
            "raw_model_output": raw
        }

def _gpt_exact_ocr_packsize(client, crop_bytes: bytes, model: str) -> str:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": PACKSIZE_OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract only the lines showing quantity/weight/volume/count (include ℮ if present)."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _gpt_structure_packsize(client, raw_text: str, model: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": PACKSIZE_STRUCTURER_SYSTEM},
            {"role": "user", "content": raw_text}
        ]
    )
    raw = r.choices[0].message.content.strip()
    try:
        cleaned = _clean_gpt_json_block(raw)
        data = json.loads(cleaned)
        # defensive normalisation on units
        if isinstance(data, dict):
            if "unit_of_measure" in data and data["unit_of_measure"]:
                data["unit_of_measure"] = _norm_unit(str(data["unit_of_measure"]))
            for k in ("net_weight", "gross_weight", "drained_weight"):
                if isinstance(data.get(k), dict) and data[k].get("unit"):
                    data[k]["unit"] = _norm_unit(str(data[k]["unit"]))
        return data
    except Exception:
        return {
            "number_of_items": None,
            "base_quantity": None,
            "unit_of_measure": None,
            "net_weight": {"value": None, "unit": None},
            "gross_weight": {"value": None, "unit": None},
            "drained_weight": {"value": None, "unit": None},
            "e_mark_present": None,
            "raw_candidates": [],
            "error": "STRUCTURE_PARSE_FAILED",
            "raw_model_output": raw
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
    m = _DOSAGE_RE.search(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower().rstrip('s')
        out["dosage"] = {"amount": amount, "unit": unit, "frequency_per_day": 1, "timing_notes": None}

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
            baseline = _ocr_words_image_to_string(crop_bytes)
            try:
                from rapidfuzz import fuzz
                ratio = fuzz.ratio(baseline.strip(), gpt_text.strip()) if baseline else None
            except Exception:
                ratio = None
            if baseline and ratio is not None and ratio < 90:
                flags.append("LOW_SIMILARITY_TO_BASELINE_OCR")
        except Exception:
            pass
    if "IMAGE_UNREADABLE" in (gpt_text or "").upper():
        flags.append("IMAGE_UNREADABLE")
    return {"similarity_to_baseline": ratio, "flags": flags}

def _scan_pages_for_ingredients(client, pages, model):
    for i, page_img in enumerate(pages):
        cand = (_find_region_via_ocr(page_img) or _gpt_bbox_locator(client, page_img, model))
        if cand:
            cand = _clamp_pad_bbox(cand, page_img.size, pad_frac=0.02)
            if cand:
                return i, page_img, cand
    return None, None, None

def _scan_pages_for_directions(client, pages, model):
    for i, page_img in enumerate(pages):
        cand = (_find_region_via_ocr_directions(page_img) or _gpt_bbox_locator_directions(client, page_img, model))
        if cand:
            cand = _clamp_pad_bbox(cand, page_img.size, pad_frac=0.02)
            if cand:
                return i, page_img, cand
    return None, None, None

def _scan_pages_for_packsize(client, pages, model):
    for i, page_img in enumerate(pages):
        cand = (_find_region_via_ocr_packsize(page_img) or _gpt_bbox_locator_packsize(client, page_img, model))
        if cand:
            cand = _clamp_pad_bbox(cand, page_img.size, pad_frac=0.02)
            if cand:
                return i, page_img, cand
    return None, None, None

def _final_ocr_and_format(client, crop_bytes: bytes, model: str, page_index: int, bbox, full_image: Image.Image) -> Dict[str, Any]:
    """
    Final stage for INGREDIENTS: exact OCR (+retry widen), consistency, scrub, allergen HTML, QA.
    Ensures the same crop bytes flow through consistency & QA if we widened the bbox.
    """
    crop_bytes_used = crop_bytes

    # --- Pass 1: strict OCR
    gpt_text = _gpt_exact_ocr(client, crop_bytes_used, model)
    if gpt_text.upper() == "IMAGE_UNREADABLE":
        # Retry once with a slightly larger crop
        bigger = _clamp_pad_bbox(bbox, full_image.size, pad_frac=0.05)  # +5%
        if bigger and bigger != bbox:
            crop_bytes_retry = _crop_to_bytes(full_image, bigger)
            gpt_text_retry = _gpt_exact_ocr(client, crop_bytes_retry, model)
            if gpt_text_retry.upper() != "IMAGE_UNREADABLE":
                bbox = bigger
                crop_bytes_used = crop_bytes_retry
                gpt_text = gpt_text_retry
            else:
                return {
                    "ok": False,
                    "error": "Detected panel unreadable.",
                    "page_index": page_index,
                    "bbox_pixels": list(map(int, bbox)) if bbox else None,
                }
        else:
            return {
                "ok": False,
                "error": "Detected panel unreadable.",
                "page_index": page_index,
                "bbox_pixels": list(map(int, bbox)) if bbox else None,
            }

    # --- Structure check (ingredients)
    structure_pass = _structure_ok(gpt_text)

    # --- Pass 2: consistency check (same crop)
    gpt_text_2 = _gpt_exact_ocr(client, crop_bytes_used, model)
    consistency_ratio = _similarity(gpt_text_2, gpt_text) or 0.0
    consistency_ok = (gpt_text_2 == gpt_text) or (consistency_ratio >= 98.0)

    # --- Safe, non-semantic punctuation scrub
    clean_text = _safe_punct_scrub(gpt_text)

    # --- Allergen bolding on clean text
    html_out = _gpt_html_allergen_bold(client, clean_text, model)

    # --- Baseline OCR QA (Tesseract) on the crop actually used
    qa = _qa_compare_tesseract(crop_bytes_used, clean_text)
    qa.update({
        "structure_pass": structure_pass,
        "consistency_ok": consistency_ok,
        "consistency_ratio": consistency_ratio,
    })
    qa["accepted"] = bool(structure_pass and consistency_ok)

    return {
        "ok": True,
        "page_index": page_index,
        "bbox_pixels": list(map(int, bbox)) if bbox else None,
        "ingredients_text": clean_text,
        "ingredients_html": html_out,
        "qa": qa,
        "debug": {
            "image_size": full_image.size,
            "bbox_area_pct": _area_pct(bbox, full_image.size),
            "tesseract_available": TESS_AVAILABLE
        }
    }

# ---------- Pack size deterministic parser & merge ----------
def _regex_parse_packsize(text: str) -> Dict[str, Any]:
    """
    Deterministic parse from OCR text for pack-size patterns.
    """
    out = {
        "number_of_items": None,
        "base_quantity": None,
        "unit_of_measure": None,
        "net_weight": {"value": None, "unit": None},
        "gross_weight": {"value": None, "unit": None},
        "drained_weight": {"value": None, "unit": None},
        "e_mark_present": True if re.search(_E_MARK, text) else False,
        "raw_candidates": []
    }
    t = text

    # 1) Multipack "N x Q U"
    m = MULTIPACK_RE.search(t)
    if m:
        n = int(m.group(1))
        qty = _num(m.group(2))
        unit = _norm_unit(m.group(3))
        out["number_of_items"] = n
        out["base_quantity"] = qty
        out["unit_of_measure"] = unit
        out["raw_candidates"].append(m.group(0))

    # 2) Count units "120 capsules"
    if out["unit_of_measure"] is None:
        m2 = COUNT_RE.search(t)
        if m2:
            out["number_of_items"] = 1
            out["base_quantity"] = float(int(m2.group(1)))
            out["unit_of_measure"] = _norm_unit(m2.group(2))
            out["raw_candidates"].append(m2.group(0))

    # 3) Single quantity "750 g" or "1 L"
    if out["unit_of_measure"] is None:
        m3 = SINGLE_QTY_RE.search(t)
        if m3:
            out["number_of_items"] = 1
            out["base_quantity"] = _num(m3.group(1))
            out["unit_of_measure"] = _norm_unit(m3.group(2))
            out["raw_candidates"].append(m3.group(0))

    # 4) Labeled weights (Net/Gross/Drained)
    for lab in LABELED_WEIGHT_RE.finditer(t):
        full = lab.group(0)
        val = _num(lab.group(5))
        unit = _norm_unit(lab.group(6))
        label_txt = lab.group(1).lower()
        if re.search(_DRAINED_LBL, label_txt, re.IGNORECASE):
            out["drained_weight"] = {"value": val, "unit": unit}
        elif re.search(_GROSS_LBL, label_txt, re.IGNORECASE):
            out["gross_weight"] = {"value": val, "unit": unit}
        else:
            out["net_weight"] = {"value": val, "unit": unit}
        out["raw_candidates"].append(full)

    # 5) Compact GW/NW
    for cg in COMPACT_GW_NW_RE.finditer(t):
        kind = cg.group(1).upper()
        val = _num(cg.group(2))
        unit = _norm_unit(cg.group(3))
        if kind == "GW":
            out["gross_weight"] = {"value": val, "unit": unit}
        elif kind == "NW":
            out["net_weight"] = {"value": val, "unit": unit}
        out["raw_candidates"].append(cg.group(0))

    return out

def _merge_packsize(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(primary or {})
    if not out:
        return fallback
    # scalar fields
    for k in ("number_of_items", "base_quantity", "unit_of_measure", "e_mark_present"):
        if out.get(k) in (None, "") and fallback.get(k) not in (None, ""):
            out[k] = fallback[k]
    # nested weights
    for k in ("net_weight", "gross_weight", "drained_weight"):
        v = out.get(k) or {}
        fv = fallback.get(k) or {}
        if (v.get("value") is None or v.get("unit") is None) and (fv.get("value") is not None or fv.get("unit") is not None):
            out[k] = {"value": fv.get("value"), "unit": fv.get("unit")}
    # raw candidates
    rc = list(out.get("raw_candidates") or [])
    for s in (fallback.get("raw_candidates") or []):
        if s not in rc:
            rc.append(s)
    out["raw_candidates"] = rc
    return out
