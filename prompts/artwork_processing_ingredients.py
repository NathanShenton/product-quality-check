# artwork_processing_ingredients.py
from __future__ import annotations
import io, json
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    HEADER_PAT,
    _fail, _pdf_to_page_images, _safe_punct_scrub, _structure_ok_ingredients, _similarity,
    _area_pct, _encode_data_url, _crop_to_bytes, _ocr_words, _qa_compare_tesseract,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _pdf_find_ingredient_block
)
import re

LONG_LIST_PAT = re.compile(r"(?:,|;).*(?:,|;).*(?:,|;).*(?:,|;)", re.UNICODE)  # ≥4 separators
ALLERGEN_LEADER_PAT = re.compile(r"\b(allergens?|contains|may contain)\b", re.I)

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
You are a vision locator for the INGREDIENTS statement on a food/supplement label.
Return JSON ONLY:
{"candidates":[{"bbox_pct":{"x":0-100,"y":0-100,"w":0-100,"h":0-100}}]}
Rules:
- Propose up to 3 likely boxes (largest first). Aim to include the full paragraph after an “Ingredients:” lead-in.
- Look for long comma/semicolon-separated lists of substances. If nothing is present, return {"candidates":[]}.
""".strip()

# ---------- Locators / OCR ----------
def _find_region_via_ocr_ingredients(full_img: Image.Image):
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    candidates = []

    # bucket words by (block, para, line)
    rows = {}
    for i in range(len(data["text"])):
        if not data["text"][i]:
            continue
        key = (data.get("block_num",[0])[i], data.get("par_num",[0])[i], data.get("line_num",[0])[i])
        rows.setdefault(key, []).append(i)

    for idxs in rows.values():
        line_txt = " ".join(data["text"][i] for i in idxs if data["text"][i]).strip()
        if not line_txt:
            continue

        # route A: classic header token (“INGREDIENTS”)
        hit_header = bool(HEADER_PAT.search(line_txt))

        # route B: long comma/semicolon lists even if header is missing
        hit_list = bool(LONG_LIST_PAT.search(line_txt))

        if not (hit_header or hit_list):
            continue

        xs = [data["left"][i] for i in idxs]; ys = [data["top"][i] for i in idxs]
        ws = [data["width"][i] for i in idxs]; hs = [data["height"][i] for i in idxs]

        # generous horizontal pad; taller vertical pad (ingredients can wrap multiple lines)
        x0 = max(0, min(xs) - int(0.06 * W))
        x1 = min(W, max(xs[j] + ws[j] for j in range(len(xs))) + int(0.06 * W))
        y0 = max(0, min(ys) - int(0.03 * H))
        # if we saw a header, allow deeper vertical sweep to include the full paragraph
        ypad = 0.60 if hit_header else 0.35
        y1 = min(H, max(ys[j] + hs[j] for j in range(len(ys))) + int(ypad * H))

        # score: prefer lines with more separators; slight boost if preceded by “Ingredients”
        sep_count = line_txt.count(",") + line_txt.count(";")
        score = sep_count + (3 if hit_header else 0)
        candidates.append(((x0, y0, x1, y1), score))

    if not candidates:
        return None
    # pick best by score, tie-break on area (helps when there are multiple lists)
    return max(candidates, key=lambda t: (t[1], (t[0][2]-t[0][0])*(t[0][3]-t[0][1])))[0]

FULLPAGE_ING_SYSTEM = """
You will receive a full label image. Extract the exact INGREDIENTS statement text only.
Rules:
- Copy the visible characters exactly (punctuation, %, brackets).
- Start after the token that reads like 'Ingredients:' (case-insensitive, allow minor OCR noise like l/!).
- Stop before the next distinct section heading (e.g., Nutrition, Directions, Storage, Warnings, Recommended Use).
- If no ingredients exist, output IMAGE_UNREADABLE.
Return plain text only.
""".strip()

def _gpt_fullpage_ingredients_text(client, img: Image.Image, model: str) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role":"system","content":FULLPAGE_ING_SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":"Return plain text only."},
                {"type":"image_url","image_url":{"url":_encode_data_url(buf.getvalue())}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()


def _gpt_bbox_locator_ingredients(client, img: Image.Image, model: str):
    import json, io
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, top_p=0,
            messages=[
                {"role":"system","content":BBOX_FINDER_SYSTEM},
                {"role":"user","content":[
                    {"type":"text","text":"Return JSON only."},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]}
            ]
        )
        js = json.loads(r.choices[0].message.content.strip())
        W, H = img.size
        cands = []
        for c in (js.get("candidates") or [])[:3]:
            pct = c.get("bbox_pct") or {}
            x = int(W * float(pct.get("x",0)) / 100.0)
            y = int(H * float(pct.get("y",0)) / 100.0)
            w = int(W * float(pct.get("w",0)) / 100.0)
            h = int(H * float(pct.get("h",0)) / 100.0)
            if w>0 and h>0:
                cands.append((x,y,x+w,y+h))
        return cands[0] if cands else None
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
            bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.03)

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
        # FIRST, retry with a much wider pad (12%)
        bigger1 = _clamp_pad_bbox(bbox, full_image.size, pad_frac=0.12)
        if bigger1 and bigger1 != bbox:
            crop_bytes_retry1 = _crop_to_bytes(full_image, bigger1)
            gpt_text_retry1 = _gpt_exact_ocr_ingredients(client, crop_bytes_retry1, model)
            if gpt_text_retry1.upper() != "IMAGE_UNREADABLE":
                bbox = bigger1
                crop_bytes_used = crop_bytes_retry1
                gpt_text = gpt_text_retry1
            else:
                # SECOND, try even wider (20%) as a last crop-expansion
                bigger2 = _clamp_pad_bbox(bigger1, full_image.size, pad_frac=0.20)
                if bigger2 and bigger2 != bigger1:
                    crop_bytes_retry2 = _crop_to_bytes(full_image, bigger2)
                    gpt_text_retry2 = _gpt_exact_ocr_ingredients(client, crop_bytes_retry2, model)
                    if gpt_text_retry2.upper() != "IMAGE_UNREADABLE":
                        bbox = bigger2
                        crop_bytes_used = crop_bytes_retry2
                        gpt_text = gpt_text_retry2
                    else:
                        return {"ok": False, "error": "Detected panel unreadable.",
                                "page_index": page_index, "bbox_pixels": list(map(int, bbox)) if bbox else None}
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
