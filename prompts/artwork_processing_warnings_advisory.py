from __future__ import annotations
import io, json, re
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    _fail, _pdf_to_page_images, _encode_data_url, _crop_to_bytes, _ocr_words,
    _qa_compare_tesseract, _clamp_pad_bbox, _fallback_left_panel_bbox,
    _fallback_right_panel_bbox, _fallback_center_panel_bbox, _area_pct, _similarity,
    _safe_punct_scrub
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompts (vision-first)
# ─────────────────────────────────────────────────────────────────────────────

WARNADV_BBOX_FINDER_SYSTEM = """
You are a vision locator. You will be shown a full product label image.
Return JSON ONLY with a single bounding box for a WARNINGS / CAUTIONS / ADVISORY / SAFETY section if present:
{"bbox_pct": {"x": 0-100, "y": 0-100, "w": 0-100, "h": 0-100}, "found": true/false}
Rules:
- Coordinates are percentages of the entire image.
- Look for headings like “Warnings”, “Cautions”, “Advisory”, “Safety”, “Important”, or clusters of short imperative lines.
- If not present, return {"found": false}.
""".strip()

WARNADV_EXACT_OCR_SYSTEM = """
You are an exacting OCR agent. You will be given a crop that contains WARNINGS and/or ADVISORY text.
Rules:
- Return the EXACT visible text only from this section. Preserve line breaks, bullets, punctuation, symbols (% °C, etc.).
- Do NOT add, normalise, or infer missing words.
- If unreadable or not a warnings/advisory section, output exactly: IMAGE_UNREADABLE
- Output plain text only.
""".strip()

# Strict classifier that draws a semantic boundary without relying on a fixed taxonomy.
WARNADV_CLASSIFIER_SYSTEM = """
You are a strict classifier for product-label safety text. You will receive plain text that may include both warnings and advisories.

Definitions:
- WARNING: hard prohibitions or risk-of-harm statements. Modal verbs like “Do not”, “Must not”, “Keep out of reach”, “For external use only”, “Not suitable for…”, “Stop use”, “Discontinue”, “Avoid contact…”, “Flammable”, “May cause irritation”, “If adverse effects occur… discontinue/seek medical advice”.
- ADVISORY: guidance, soft rules, disclaimers, or contextual advice. Modal verbs like “Should”, “Recommended”, “Use with…”, “As part of…”, “Consult a healthcare professional if…”, “Food supplement is not a substitute…”, “Store in a cool dry place”, age guidance framed as advice rather than prohibition.

Instructions:
- Split the input into distinct statements (sentences, bullets, or lines).
- Classify **each** statement as either warning or advisory using semantics (prohibition/risk vs guidance).
- Preserve the **exact original wording** for each statement.
- Do **not** guess or merge multiple statements; if uncertain, place in advisory_info.
- Return JSON ONLY with this schema:
{
  "warning_info": ["exact statement 1", "…"],
  "advisory_info": ["exact statement 1", "…"]
}
""".strip()

FULLPAGE_WARNADV_SYSTEM = """
You will receive a full label image. Extract the exact WARNINGS / CAUTIONS / ADVISORY / SAFETY statements only.
Rules:
- Copy visible characters exactly (preserve bullets, numbers, punctuation, °C/°F, % and line breaks).
- Start after a heading like “Warnings”, “Cautions”, “Advisory”, “Safety”, “Important” if present.
- Stop before the next distinct section (e.g., Ingredients, Nutrition, Directions, Storage).
- If no warnings/advisory exist, output IMAGE_UNREADABLE.
Return plain text only.
""".strip()

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight heuristics used ONLY to help find the region (not to classify)
# ─────────────────────────────────────────────────────────────────────────────

# Generic cues to find the block — broad on purpose; classification is done by GPT.
_HDR_PAT = re.compile(
    r"\b(warning|warnings|caution|cautions|advisory|advisories|safety|important)\b",
    re.IGNORECASE
)
_NEG_IMPERATIVE = re.compile(
    r"\b(do\s+not|don't|must\s+not|keep\s+out\s+of\s+reach|not\s+suitable|for\s+external\s+use\s+only|avoid\s+contact|stop\s+use|discontinue|flammable|may\s+cause|irritat|choking\s+hazard)\b",
    re.IGNORECASE
)
_SOFT_ADVICE = re.compile(
    r"\b(should|recommended|advise|as\s+part\s+of|use\s+with|consult|seek\s+advice|store\s+in|if\s+pregnant|if\s+breastfeeding|varied\s+and\s+balanced\s+diet)\b",
    re.IGNORECASE
)

def _find_region_via_ocr_warnadv(full_img: Image.Image) -> Optional[Tuple[int,int,int,int]]:
    """
    Scan OCR words and pick a large crop that likely covers warnings/advisory text.
    """
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    candidates = []
    for i, word in enumerate(data["text"]):
        if not word:
            continue
        w_lower = str(word).lower()
        if _HDR_PAT.search(w_lower) or _NEG_IMPERATIVE.search(w_lower) or _SOFT_ADVICE.search(w_lower):
            x = data["left"][i]; y = data["top"][i]
            w = data["width"][i]; h = data["height"][i]
            # pad generously downward; warnings often run as short stacked lines
            x0 = max(0, x - int(0.08 * W))
            x1 = min(W, x + w + int(0.08 * W))
            y0 = max(0, y - int(0.04 * H))
            y1 = min(H, y + h + int(0.55 * H))
            candidates.append((x0, y0, x1, y1))
    if candidates:
        # pick the largest candidate region
        return max(candidates, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# GPT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpt_bbox_locator_warnadv(client, img: Image.Image, model: str):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": WARNADV_BBOX_FINDER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Locate WARNINGS/CAUTIONS/ADVISORY area and return JSON only."},
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

def _gpt_exact_ocr_warnadv(client, crop_bytes: bytes, model: str) -> str:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": WARNADV_EXACT_OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the exact WARNINGS/ADVISORY text only."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _gpt_fullpage_warnadv_text(client, img: Image.Image, model: str) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role":"system","content": FULLPAGE_WARNADV_SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":"Return plain text only."},
                {"type":"image_url","image_url":{"url": _encode_data_url(buf.getvalue())}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _gpt_classify_warnadv(client, raw_text: str, model: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": WARNADV_CLASSIFIER_SYSTEM},
            {"role": "user", "content": raw_text}
        ]
    )
    raw = r.choices[0].message.content.strip()
    # robust JSON un-fencing
    try:
        if raw.startswith("```"):
            raw = raw.split("```", maxsplit=2)[1].lstrip("json").strip()
        js = json.loads(raw)
        # ensure keys exist
        js.setdefault("warning_info", [])
        js.setdefault("advisory_info", [])
        return js
    except Exception:
        # Fallback: very light heuristic split
        warnings, advisories = [], []
        for line in re.split(r"[\n•\-•]+", raw_text):
            t = line.strip(" \t\r\n\u2022-•")
            if not t:
                continue
            if _NEG_IMPERATIVE.search(t):
                warnings.append(t)
            elif _SOFT_ADVICE.search(t) or len(t.split()) >= 3:
                advisories.append(t)
        return {"warning_info": warnings, "advisory_info": advisories, "error": "CLASSIFIER_JSON_PARSE_FAILED", "raw_model_output": raw}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_artwork_warnings_advisory(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 400,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Vision-first extraction of warnings & advisory statements into two fields:
      - warning_info   : List[str]
      - advisory_info  : List[str]
    """
    is_pdf = filename.lower().endswith(".pdf")

    # 1) Load page(s)
    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        page_idx: Optional[int] = None
        img = None
        bbox = None

        # First pass: OCR scan for likely region; else GPT bbox; else heuristic guesses
        for i, pg in enumerate(pages):
            cand = (_find_region_via_ocr_warnadv(pg) or _gpt_bbox_locator_warnadv(client, pg, model))
            if cand:
                cand = _clamp_pad_bbox(cand, pg.size, pad_frac=0.03)
                if cand:
                    page_idx, img, bbox = i, pg, cand
                    break
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
            # last resort: full-page text
            img = pages[0]
            page_idx = 0
            bbox = None

        if bbox and _area_pct(bbox, img.size) < 1.2:
            alt = _gpt_bbox_locator_warnadv(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.03)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox) if bbox else None

    else:
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        bbox = (_find_region_via_ocr_warnadv(img) or _gpt_bbox_locator_warnadv(client, img, model))
        bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.03) if bbox else None

        if bbox is None:
            # heuristic guesses if locator fails
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.03)
                if g:
                    bbox = g
                    break

        crop_bytes = _crop_to_bytes(img, bbox) if bbox else None
        page_idx = 0

    # 2) Exact OCR via GPT (crop preferred; full-page fallback)
    if crop_bytes:
        raw = _gpt_exact_ocr_warnadv(client, crop_bytes, model)
        if raw.upper() == "IMAGE_UNREADABLE":
            # retry with wider crops (+12%, then +20%), then full page
            bigger1 = _clamp_pad_bbox(bbox, img.size, pad_frac=0.12) if bbox else None
            tried_full = False
            if bigger1 and bigger1 != bbox:
                raw1 = _gpt_exact_ocr_warnadv(client, _crop_to_bytes(img, bigger1), model)
                if raw1.upper() != "IMAGE_UNREADABLE":
                    raw, bbox, crop_bytes = raw1, bigger1, _crop_to_bytes(img, bigger1)
                else:
                    bigger2 = _clamp_pad_bbox(bigger1, img.size, pad_frac=0.20)
                    if bigger2 and bigger2 != bigger1:
                        raw2 = _gpt_exact_ocr_warnadv(client, _crop_to_bytes(img, bigger2), model)
                        if raw2.upper() != "IMAGE_UNREADABLE":
                            raw, bbox, crop_bytes = raw2, bigger2, _crop_to_bytes(img, bigger2)
                        else:
                            tried_full = True
            if not crop_bytes or raw.upper() == "IMAGE_UNREADABLE" or tried_full:
                full_raw = _gpt_fullpage_warnadv_text(client, img, model)
                if full_raw.upper() != "IMAGE_UNREADABLE":
                    raw = full_raw
                else:
                    return {
                        "ok": False,
                        "error": "Detected warnings/advisory crop unreadable.",
                        "page_index": page_idx,
                        "bbox_pixels": list(map(int, bbox)) if bbox else None
                    }
    else:
        full_raw = _gpt_fullpage_warnadv_text(client, img, model)
        if full_raw.upper() == "IMAGE_UNREADABLE":
            return _fail("Could not locate WARNINGS/ADVISORY content.")
        raw = full_raw

    # 3) Consistency & QA vs second pass + Tesseract baseline
    clean_text = _safe_punct_scrub(raw)
    # second pass consistency check (if we had a crop)
    consistency_ok = True
    consist_ratio = 100.0
    if crop_bytes:
        raw2 = _gpt_exact_ocr_warnadv(client, crop_bytes, model)
        consist_ratio = _similarity(raw2, raw) or 0.0
        consistency_ok = (raw2 == raw) or (consist_ratio >= 98.0)

    qa = _qa_compare_tesseract(crop_bytes if crop_bytes else _crop_to_bytes(img, (0,0, img.size[0], img.size[1])), clean_text)

    # 4) Classification
    classified = _gpt_classify_warnadv(client, clean_text, model)
    warning_info: List[str] = [s for s in (classified.get("warning_info") or []) if str(s).strip()]
    advisory_info: List[str] = [s for s in (classified.get("advisory_info") or []) if str(s).strip()]

    # 5) Structure pass: at least one of the two lists populated
    structure_pass = bool(warning_info or advisory_info)
    qa.update({
        "structure_pass": structure_pass,
        "consistency_ok": consistency_ok,
        "consistency_ratio": consist_ratio,
    })
    qa["accepted"] = bool(structure_pass and consistency_ok)

    return {
        "ok": True,
        "page_index": (0 if not is_pdf else page_idx),
        "bbox_pixels": list(map(int, bbox)) if bbox else None,
        "warning_info": warning_info,
        "advisory_info": advisory_info,
        "qa": qa,
        "debug": {
            "image_size": (img.size if img else None),
            "bbox_area_pct": (_area_pct(bbox, img.size) if bbox else None),
            "tesseract_available": TESS_AVAILABLE,
            "classifier_error": classified.get("error") if isinstance(classified, dict) else None
        }
    }
