# artwork_processing_directions.py
from __future__ import annotations
import io, json, re
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    DIRECTIONS_HEADER_PAT, IMPERATIVE_VERBS, TIME_QTY_TOKENS,
    _fail, _pdf_to_page_images, _pdf_to_page_images_adaptive, _rerender_single_page,
    _safe_punct_scrub, _structure_ok_directions, _similarity, _clean_gpt_json_block,
    _area_pct, _encode_data_url, _crop_to_bytes, _ocr_words, _qa_compare_tesseract,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _pdf_find_directions_block, _gpt_normalize_flatpack_page
)

# ---------- Systems ----------
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

FULLPAGE_DIR_SYSTEM = """
You will receive a full label image. Extract the exact DIRECTIONS / USAGE / PREPARATION text only.
Rules:
- Copy visible characters exactly (preserve bullets, numbers, °C/°F, %, punctuation and line breaks).
- Start after a header like “Directions”, “How to use”, “Preparation”, “Usage”, “Dosage”.
- Stop before the next distinct section heading (Nutrition, Ingredients, Storage, Warnings, etc.).
- If no directions exist, output IMAGE_UNREADABLE.
Return plain text only.
""".strip()

# ---------- GPT helpers ----------
def _gpt_fullpage_directions_text(client, img: Image.Image, model: str) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role":"system","content": FULLPAGE_DIR_SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":"Return plain text only."},
                {"type":"image_url","image_url":{"url": _encode_data_url(buf.getvalue())}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _gpt_bbox_locator_directions(client, img: Image.Image, model: str):
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

# ---------- OCR heuristics ----------
def _find_region_via_ocr_directions(full_img: Image.Image):
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

# ---------- Lightweight text fallback ----------
_DOSAGE_RE = re.compile(
    r"\btake\s+(\d+)\s+(capsule|capsules|tablet|tablets|scoop|scoops|drop|drops)\b.*?\b(daily|per day|a day)\b",
    re.IGNORECASE
)
def _lightweight_directions_fallback(text: str) -> Dict[str, Any]:
    out = {
        "steps": [], "timings": [], "temperatures": [], "volumes": [],
        "dosage": {"amount": None, "unit": None, "frequency_per_day": None, "timing_notes": None},
        "serving_suggestion": None, "notes": None
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

# ---------- Shared OCR escalation ----------
def _escalate_ocr_directions(client, img: Image.Image, bbox: Tuple[int,int,int,int], model: str) -> Tuple[Tuple[int,int,int,int], bytes, str]:
    """
    Try exact OCR on bbox; widen crop progressively; if still unreadable, do full-page text.
    Returns (bbox_used, crop_bytes, text).
    """
    def _ocr(cbox):
        cb = _crop_to_bytes(img, cbox)
        txt = _gpt_exact_ocr_directions(client, cb, model)
        return txt, cb

    # First attempt
    text, crop = _ocr(bbox)
    if text.upper() != "IMAGE_UNREADABLE":
        return bbox, crop, text

    # Wider +12%
    bigger1 = _clamp_pad_bbox(bbox, img.size, pad_frac=0.12)
    if bigger1 and bigger1 != bbox:
        t1, c1 = _ocr(bigger1)
        if t1.upper() != "IMAGE_UNREADABLE":
            return bigger1, c1, t1

        # Wider +20%
        bigger2 = _clamp_pad_bbox(bigger1, img.size, pad_frac=0.20)
        if bigger2 and bigger2 != bigger1:
            t2, c2 = _ocr(bigger2)
            if t2.upper() != "IMAGE_UNREADABLE":
                return bigger2, c2, t2

    # Full page last resort
    full_text = _gpt_fullpage_directions_text(client, img, model)
    return bbox, crop, full_text

# ---------- Public API (DIRECTIONS) ----------
def process_artwork_directions(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 400,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    is_pdf = filename.lower().endswith(".pdf")

    page_idx: Optional[int] = None
    img_for_crop: Optional[Image.Image] = None
    bbox: Optional[Tuple[int,int,int,int]] = None
    used_vector_locator = False
    used_normalizer = False

    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")

        # Adaptive PDF rendering (primary + potential fallback DPI)
        pages, scale72, dpi_primary, dpi_fallback = _pdf_to_page_images_adaptive(
            file_bytes, dpi_primary=render_dpi, dpi_fallback=max(550, render_dpi + 150)
        )
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        # (1) Vector-text locator (never normalize here; bbox is in original page coords)
        vec = _pdf_find_directions_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            page_img = pages[page_idx]
            x0, y0, x1, y1 = vec["bbox_pixels"]  # points @ 72 dpi
            scale = dpi_primary / 72.0
            bbox = (int(round(x0 * scale)), int(round(y0 * scale)),
                    int(round(x1 * scale)), int(round(y1 * scale)))
            bbox = _clamp_pad_bbox(bbox, page_img.size, pad_frac=0.03)
            if bbox:
                img_for_crop = page_img
                used_vector_locator = True

        # (2) If vector failed, try OCR/GPT per page. Only now we may normalize the page image.
        if img_for_crop is None:
            for i, pg in enumerate(pages):
                # First try on raw page
                cand = (_find_region_via_ocr_directions(pg) or _gpt_bbox_locator_directions(client, pg, model))
                if not cand:
                    # Try normalized view of this page (flatpack orientation / main panel)
                    norm_pg, _, meta = _gpt_normalize_flatpack_page(client, pg, model)
                    if meta.get("found"):
                        used_normalizer = True
                    work = norm_pg or pg
                    cand = (_find_region_via_ocr_directions(work) or _gpt_bbox_locator_directions(client, work, model))
                    if cand:
                        cand = _clamp_pad_bbox(cand, work.size, pad_frac=0.03)
                        if cand:
                            page_idx, img_for_crop, bbox = i, work, cand
                            break
                else:
                    cand = _clamp_pad_bbox(cand, pg.size, pad_frac=0.03)
                    if cand:
                        page_idx, img_for_crop, bbox = i, pg, cand
                        break

        # (3) Heuristic guesses if still nothing
        if img_for_crop is None or bbox is None:
            for i, pg in enumerate(pages):
                for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                    g = _clamp_pad_bbox(guess_fn(pg), pg.size, pad_frac=0.03)
                    if g:
                        page_idx, img_for_crop, bbox = i, pg, g
                        break
                if bbox is not None:
                    break

        if page_idx is None or img_for_crop is None or bbox is None:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the PDF.")

        # Tiny-area sanity
        if _area_pct(bbox, img_for_crop.size) < 1.2:
            alt = _gpt_bbox_locator_directions(client, img_for_crop, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img_for_crop.size, pad_frac=0.03)
                if alt:
                    bbox = alt

        # First OCR pass (with escalation)
        bbox_used, crop_bytes, raw = _escalate_ocr_directions(client, img_for_crop, bbox, model)

        # If still unreadable and we used the original page (vector case), try a high-DPI rerender and reuse bbox scaled
        if raw.upper() == "IMAGE_UNREADABLE" and used_vector_locator:
            hi_img = _rerender_single_page(file_bytes, page_idx, dpi=dpi_fallback)
            if hi_img is not None:
                # scale bbox to the new page size
                ow, oh = img_for_crop.size
                nw, nh = hi_img.size
                sx, sy = (nw / max(1, ow), nh / max(1, oh))
                scaled_bbox = _clamp_pad_bbox(
                    (int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)),
                    (nw, nh), pad_frac=0.03
                )
                if scaled_bbox:
                    bbox_used, crop_bytes, raw = _escalate_ocr_directions(client, hi_img, scaled_bbox, model)
                    if raw.upper() != "IMAGE_UNREADABLE":
                        img_for_crop = hi_img
                        bbox = bbox_used

    else:
        # ----- IMAGE path -----
        try:
            base_img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")

        # Normalize flatpack orientation + isolate main panel (optional)
        norm_img, _, meta = _gpt_normalize_flatpack_page(client, base_img, model)
        if meta.get("found"):
            used_normalizer = True
        work_img = norm_img or base_img

        # Locate directions on the working image
        bbox = (_find_region_via_ocr_directions(work_img) or _gpt_bbox_locator_directions(client, work_img, model))
        bbox = _clamp_pad_bbox(bbox, work_img.size, pad_frac=0.03) if bbox else None

        if bbox is None:
            # Heuristic fallbacks
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(work_img), work_img.size, pad_frac=0.03)
                if g:
                    bbox = g
                    break

        if not bbox:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the image.")

        # Tiny-area sanity
        if _area_pct(bbox, work_img.size) < 1.2:
            alt = _gpt_bbox_locator_directions(client, work_img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, work_img.size, pad_frac=0.03)
                if alt:
                    bbox = alt

        img_for_crop = work_img
        bbox_used, crop_bytes, raw = _escalate_ocr_directions(client, img_for_crop, bbox, model)

    # ---------------- Structure & consistency ----------------
    if raw.upper() == "IMAGE_UNREADABLE":
        return {
            "ok": False,
            "error": "Detected directions crop unreadable.",
            "page_index": page_idx,
            "bbox_pixels": list(map(int, bbox)) if bbox else None
        }

    structure_pass = _structure_ok_directions(raw)

    # Consistency check on the same crop
    raw2 = _gpt_exact_ocr_directions(client, crop_bytes, model)
    consist_ratio = _similarity(raw2, raw) or 0.0
    consistency_ok = (raw2 == raw) or (consist_ratio >= 98.0)

    # Clean + structure
    clean_text = _safe_punct_scrub(raw)
    structured = _gpt_structure_directions(client, clean_text, model)

    # Lightweight text fallback if structurer struggled
    if (isinstance(structured, dict) and (
        structured.get("error") == "STRUCTURE_PARSE_FAILED"
        or (not structured.get("steps") and not structured.get("dosage", {}).get("amount"))
    )):
        fb = _lightweight_directions_fallback(clean_text)
        if not structured.get("steps"):
            structured["steps"] = fb["steps"]
        if (not structured.get("dosage") or structured["dosage"].get("amount") is None):
            structured["dosage"] = fb["dosage"]

    # Pictograms from final crop
    pictos = _gpt_pictograms(client, crop_bytes, model)

    # QA vs Tesseract baseline
    qa = _qa_compare_tesseract(crop_bytes, clean_text)
    qa.update({
        "structure_pass": structure_pass,
        "consistency_ok": consistency_ok,
        "consistency_ratio": consist_ratio,
    })
    qa["accepted"] = bool(structure_pass and consistency_ok)

    # Steps HTML
    steps_html = ""
    if isinstance(structured, dict) and structured.get("steps"):
        items = "".join(f"<li>{s.get('text','').strip()}</li>" for s in structured["steps"])
        steps_html = f"<ol>{items}</ol>"

    dbg_size = img_for_crop.size if img_for_crop else (0, 0)

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
            "image_size": dbg_size,
            "bbox_area_pct": _area_pct(bbox, dbg_size) if bbox else None,
            "tesseract_available": TESS_AVAILABLE,
            "used_vector_locator": used_vector_locator,
            "used_normalizer": used_normalizer
        }
    }
