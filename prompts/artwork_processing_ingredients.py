# artwork_processing_ingredients.py
from __future__ import annotations
import io, json
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    HEADER_PAT,
    _fail, _pdf_to_page_images, _safe_punct_scrub, _structure_ok_ingredients, _similarity,
    _area_pct, _encode_data_url, _crop_to_bytes, _ocr_words, _qa_compare_tesseract,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _pdf_find_ingredient_block
)

# ---------- System prompts ----------
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

# ---------- Locators / OCR ----------
def _find_region_via_ocr_ingredients(full_img: Image.Image):
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

def _gpt_bbox_locator_ingredients(client, img: Image.Image, model: str):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    try:
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
        js = json.loads(r.choices[0].message.content.strip())
        if not js.get("found"):
            return None
        W, H = img.size
        pct = js["bbox_pct"]
        x = int(W * float(pct["x"]) / 100.0)
        y = int(H * float(pct["y"]) / 100.0)
        w = int(W * float(pct["w"]) / 100.0)
        h = int(H * float(pct["h"]) / 100.0)
        return (x, y, x + w, y + h)
    except Exception:
        return None

def _gpt_exact_ocr_ingredients(client, crop_bytes: bytes, model: str) -> str:
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

# ---------- Public API (INGREDIENTS) ------------------------------------------
def process_artwork(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 350,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        vec = _pdf_find_ingredient_block(file_bytes)
        page_idx: Optional[int] = None
        img = None
        bbox = None

        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox_pts = vec["bbox_pixels"]   # points @ 72 dpi
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
            img = pages[page_idx]
            bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)

        # (2) OCR/GPT per-page if needed
        if bbox is None:
            for i, pg in enumerate(pages):
                cand = (_find_region_via_ocr_ingredients(pg) or _gpt_bbox_locator_ingredients(client, pg, model))
                if cand:
                    cand = _clamp_pad_bbox(cand, pg.size, pad_frac=0.02)
                    if cand:
                        page_idx, img, bbox = i, pg, cand
                        break

        # (3) NEW heuristic fallbacks: LEFT → RIGHT → CENTER
        if bbox is None:
            for i, pg in enumerate(pages):
                for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                    g = _clamp_pad_bbox(guess_fn(pg), pg.size, pad_frac=0.02)
                    if g:
                        page_idx, img, bbox = i, pg, g
                        break
                if bbox is not None:
                    break

        if page_idx is None or img is None or bbox is None:
            return _fail("Could not locate an INGREDIENTS panel in the PDF.")

        # Tiny-area sanity
        if _area_pct(bbox, img.size) < 2.0:
            alt = _gpt_bbox_locator_ingredients(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)
        return _final_ocr_and_format(client, crop_bytes, model, page_idx, bbox, img)

    # Single image
    try:
        img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return _fail("Could not open image.")
    bbox = (_find_region_via_ocr_ingredients(img) or _gpt_bbox_locator_ingredients(client, img, model))
    bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02) if bbox else None

    # NEW heuristic fallbacks on image
    if bbox is None:
        for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
            g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.02)
            if g:
                bbox = g
                break

    if not bbox:
        return _fail("Could not locate an INGREDIENTS panel in the image.")

    if _area_pct(bbox, img.size) < 2.0:  # microscopic-crop sanity
        alt = _gpt_bbox_locator_ingredients(client, img, model)
        if alt:
            alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
            if alt:
                bbox = alt

    crop_bytes = _crop_to_bytes(img, bbox)
    return _final_ocr_and_format(client, crop_bytes, model, page_index=0, bbox=bbox, full_image=img)

# ---------- Final OCR & QA (same contract as original) ------------------------
def _final_ocr_and_format(client, crop_bytes: bytes, model: str, page_index: int, bbox, full_image: Image.Image) -> Dict[str, Any]:
    crop_bytes_used = crop_bytes
    gpt_text = _gpt_exact_ocr_ingredients(client, crop_bytes_used, model)
    if gpt_text.upper() == "IMAGE_UNREADABLE":
        bigger = _clamp_pad_bbox(bbox, full_image.size, pad_frac=0.05)
        if bigger and bigger != bbox:
            crop_bytes_retry = _crop_to_bytes(full_image, bigger)
            gpt_text_retry = _gpt_exact_ocr_ingredients(client, crop_bytes_retry, model)
            if gpt_text_retry.upper() != "IMAGE_UNREADABLE":
                bbox = bigger
                crop_bytes_used = crop_bytes_retry
                gpt_text = gpt_text_retry
            else:
                return {"ok": False, "error": "Detected panel unreadable.",
                        "page_index": page_index, "bbox_pixels": list(map(int, bbox)) if bbox else None}
        else:
            return {"ok": False, "error": "Detected panel unreadable.",
                    "page_index": page_index, "bbox_pixels": list(map(int, bbox)) if bbox else None}

    structure_pass = _structure_ok_ingredients(gpt_text)

    gpt_text_2 = _gpt_exact_ocr_ingredients(client, crop_bytes_used, model)
    consistency_ratio = _similarity(gpt_text_2, gpt_text) or 0.0
    consistency_ok = (gpt_text_2 == gpt_text) or (consistency_ratio >= 98.0)

    clean_text = _safe_punct_scrub(gpt_text)
    html_out = _gpt_html_allergen_bold(client, clean_text, model)

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
