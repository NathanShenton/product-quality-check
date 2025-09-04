# artwork_processing_directions.py
from __future__ import annotations
import io, json, re
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    DIRECTIONS_HEADER_PAT, IMPERATIVE_VERBS, TIME_QTY_TOKENS,
    _fail, _pdf_to_page_images, _safe_punct_scrub, _structure_ok_directions, _similarity,
    _area_pct, _encode_data_url, _crop_to_bytes, _ocr_words, _qa_compare_tesseract,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _pdf_find_directions_block
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


# ---------- Locators / OCR ----------
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
        from artwork_processing_common import _clean_gpt_json_block
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

# ---------- Lightweight fallback for dosage/steps ----------
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

# ---------- Public API (DIRECTIONS) -------------------------------------------
def process_artwork_directions(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 400,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        vec = _pdf_find_directions_block(file_bytes)
        page_idx: Optional[int] = None
        img = None
        bbox = None

        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox_pts = vec["bbox_pixels"]  # points @ 72 dpi
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
            img = pages[page_idx]
            bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.03)

        # (2) OCR/GPT per-page fallback
        if bbox is None:
            for i, pg in enumerate(pages):
                cand = (_find_region_via_ocr_directions(pg) or _gpt_bbox_locator_directions(client, pg, model))
                if cand:
                    cand = _clamp_pad_bbox(cand, pg.size, pad_frac=0.03)
                    if cand:
                        page_idx, img, bbox = i, pg, cand
                        break

        # (3) NEW heuristic guesses: LEFT → RIGHT → CENTER
        if bbox is None:
            for i, pg in enumerate(pages):
                for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                    g = _clamp_pad_bbox(guess_fn(pg), pg.size, pad_frac=0.03)
                    if g:
                        page_idx, img, bbox = i, pg, g
                        break
                if bbox is not None:
                    break

        if page_idx is None or img is None or bbox is None:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the PDF.")

        if _area_pct(bbox, img.size) < 1.2:
            alt = _gpt_bbox_locator_directions(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.03)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)

    else:
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        bbox = (_find_region_via_ocr_directions(img) or _gpt_bbox_locator_directions(client, img, model))
        bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.03) if bbox else None

        # NEW heuristic fallbacks
        if bbox is None:
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.03)
                if g:
                    bbox = g
                    break

        if not bbox:
            return _fail("Could not locate a DIRECTIONS/USAGE/PREPARATION panel in the image.")

        if _area_pct(bbox, img.size) < 1.2:
            alt = _gpt_bbox_locator_directions(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.03)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)
        page_idx = 0

    # --- OCR pass 1
    raw = _gpt_exact_ocr_directions(client, crop_bytes, model)
    if raw.upper() == "IMAGE_UNREADABLE":
        # 1) Retry with wider crop (+12%)
        bigger1 = _clamp_pad_bbox(bbox, img.size, pad_frac=0.12)
        if bigger1 and bigger1 != bbox:
            crop_bytes1 = _crop_to_bytes(img, bigger1)
            raw1 = _gpt_exact_ocr_directions(client, crop_bytes1, model)
            if raw1.upper() != "IMAGE_UNREADABLE":
                bbox, crop_bytes, raw = bigger1, crop_bytes1, raw1
            else:
                # 2) Retry even wider (+20%)
                bigger2 = _clamp_pad_bbox(bigger1, img.size, pad_frac=0.20)
                if bigger2 and bigger2 != bigger1:
                    crop_bytes2 = _crop_to_bytes(img, bigger2)
                    raw2 = _gpt_exact_ocr_directions(client, crop_bytes2, model)
                    if raw2.upper() != "IMAGE_UNREADABLE":
                        bbox, crop_bytes, raw = bigger2, crop_bytes2, raw2
                    else:
                        # 3) Full-page extraction as last resort
                        full_raw = _gpt_fullpage_directions_text(client, img, model)
                        if full_raw.upper() != "IMAGE_UNREADABLE":
                            # we keep bbox for pictograms if we had one; text comes from full page
                            raw = full_raw
                        else:
                            return {
                                "ok": False,
                                "error": "Detected directions crop unreadable.",
                                "page_index": page_idx,
                                "bbox_pixels": list(map(int, bbox)) if bbox else None
                            }
                else:
                    # 3) Full-page extraction as last resort
                    full_raw = _gpt_fullpage_directions_text(client, img, model)
                    if full_raw.upper() != "IMAGE_UNREADABLE":
                        raw = full_raw
                    else:
                        return {
                            "ok": False,
                            "error": "Detected directions crop unreadable.",
                            "page_index": page_idx,
                            "bbox_pixels": list(map(int, bbox)) if bbox else None
                        }
        else:
            # 3) Full-page extraction as last resort
            full_raw = _gpt_fullpage_directions_text(client, img, model)
            if full_raw.upper() != "IMAGE_UNREADABLE":
                raw = full_raw
            else:
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

    # --- Clean + structure + pictos
    clean_text = _safe_punct_scrub(raw)
    structured = _gpt_structure_directions(client, clean_text, model)
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

    # --- Steps HTML
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
