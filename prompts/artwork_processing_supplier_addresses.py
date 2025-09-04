
from __future__ import annotations
import io, json, re
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    _fail, _pdf_to_page_images, _encode_data_url, _crop_to_bytes, _ocr_words,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _area_pct, _qa_compare_tesseract, _similarity
)

# ==============================================
# Supplier Address Extraction (UK / EU)
# ==============================================

# ---- Regex & heuristics ----
UK_POSTCODE_PAT = re.compile(
    r"\\b(?:GIR\\s?0AA|[A-Z]{1,2}\\d[A-Z\\d]?\\s*\\d[ABD-HJLNP-UW-Z]{2})\\b", re.I
)

# Common company/address cues
ADDR_CUE_PAT = re.compile(
    r"\\b(company|co\\.?|ltd\\.?|limited|plc|gmbh|s\\.?r\\.?l\\.?|sa|bv|s\\.?a\\.?"
    r"|street|str\\.?|road|rd\\.?|avenue|ave\\.?|way|close|park|industrial|estate|"
    r"unit|suite|po box|postbus|postcode|zip|city|town|county|manufactured by|"
    r"manufacturer|importer|distributed by|distributor|responsible person|rp|address)\\b",
    re.I
)

UK_CUES = {
    "uk","u.k.","united kingdom","england","scotland","wales","gb","great britain",
    "northern ireland"
}

EU_MEMBER_COUNTRIES = {
    "austria","belgium","bulgaria","croatia","cyprus","czech republic","czechia","denmark",
    "estonia","finland","france","germany","greece","hungary","ireland","italy","latvia",
    "lithuania","luxembourg","malta","netherlands","poland","portugal","romania","slovakia",
    "slovenia","spain","sweden"
}

EU_LANG_CUES = {"es","it","fr","de","pt","pl","nl"}  # weak language cue tokens (not decisive)

# ---- System prompts ----

SUPPLIER_FULLPAGE_SYSTEM = """
You are a compliance assistant for food/supplement artwork.
Task: Read the FULL LABEL image and extract the SUPPLIER/RESPONSIBLE PERSON postal address blocks.
Return **ONLY** minified JSON with this exact schema (no prose):
{
  "uk": [{"text": "<exact visible text, preserve punctuation and line breaks with \\n>", "bbox_pct":{"x":0-100,"y":0-100,"w":0-100,"h":0-100}}],
  "eu": [{"text": "<exact visible text>", "bbox_pct":{"x":0-100,"y":0-100,"w":0-100,"h":0-100}}]
}
Rules:
- Include up to 2 candidates for each of "uk" and "eu" (best first). Omit a list (or use []) if not present.
- "Exact visible text" means copy the characters you can read. Do NOT invent missing lines.
- A supplier address typically includes a company name and multi-line postal address.
- Classify as UK if it clearly refers to the United Kingdom (UK, GB, England, Scotland, Wales, Northern Ireland) or has a UK postcode.
- Classify as EU if it clearly refers to an EU Member State address (e.g., Ireland, France, Germany, Netherlands, etc.).
- If the label shows a single combined "UK/EU:" address, put it in BOTH lists.
- If nothing is readable, return {"uk":[],"eu":[]}.
""".strip()

CROP_EXACT_OCR_SYSTEM = """
You are an exact OCR agent. You will receive an image crop that contains a postal address.
Rules:
- Return the EXACT visible text. Preserve punctuation and line breaks (use \\n between lines).
- Do NOT add or infer missing text.
- If unreadable, output exactly: IMAGE_UNREADABLE
- Output plain text only.
""".strip()

# ---- Helpers ----

def _gpt_fullpage_supplier_json(client, img: Image.Image, model: str) -> Dict[str, Any]:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role":"system","content":SUPPLIER_FULLPAGE_SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":"Return JSON only."},
                {"type":"image_url","image_url":{"url":_encode_data_url(buf.getvalue())}}
            ]}
        ]
    )
    try:
        js = json.loads(r.choices[0].message.content.strip())
        # normalize
        js["uk"] = js.get("uk") or []
        js["eu"] = js.get("eu") or []
        return js
    except Exception:
        return {"uk":[],"eu":[]}

def _gpt_exact_ocr_address(client, crop_bytes: bytes, model: str) -> str:
    data_url = _encode_data_url(crop_bytes)
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role":"system","content":CROP_EXACT_OCR_SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":"Extract the exact address text."},
                {"type":"image_url","image_url":{"url":data_url}}
            ]}
        ]
    )
    return r.choices[0].message.content.strip()

def _country_hint(text: str) -> str:
    t = text.lower()
    if UK_POSTCODE_PAT.search(text) or any(k in t for k in UK_CUES):
        return "UK"
    if any(c in t for c in EU_MEMBER_COUNTRIES):
        return "EU"
    # weak language cue: look for country abbreviations at end of postcode-like tokens, else unknown
    return "UNK"

def _bbox_pct_to_pixels(pct: Dict[str,float], size: Tuple[int,int]) -> Tuple[int,int,int,int]:
    W,H = size
    x = int(W * float(pct.get("x",0)) / 100.0)
    y = int(H * float(pct.get("y",0)) / 100.0)
    w = int(W * float(pct.get("w",0)) / 100.0)
    h = int(H * float(pct.get("h",0)) / 100.0)
    return (x, y, x+w, y+h)

def _rank_ocr_rows_for_addresses(full_img: Image.Image):
    """Heuristic OCR fallback: find lines with address/company cues and group nearby lines into blocks."""
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return []

    W,H = full_img.size
    # Group into line rows
    lines: Dict[Tuple[int,int,int], List[int]] = {}
    for i,txt in enumerate(data["text"]):
        if not txt:
            continue
        key = (data.get("block_num",[0])[i], data.get("par_num",[0])[i], data.get("line_num",[0])[i])
        lines.setdefault(key, []).append(i)

    cand_blocks: List[Tuple[Tuple[int,int,int,int], float, str]] = []  # (bbox, score, sample_text)

    # assemble each line box + text
    rows = []
    for idxs in lines.values():
        xs = [data["left"][i] for i in idxs]; ys = [data["top"][i] for i in idxs]
        ws = [data["width"][i] for i in idxs]; hs = [data["height"][i] for i in idxs]
        text = " ".join(data["text"][i] for i in idxs if data["text"][i]).strip()
        if not text:
            continue
        x0 = max(0, min(xs)); x1 = min(W, max(xs[j] + ws[j] for j in range(len(xs))))
        y0 = max(0, min(ys)); y1 = min(H, max(ys[j] + hs[j] for j in range(len(ys))))
        rows.append(((x0,y0,x1,y1), text))

    # group nearby rows into paragraph-ish blocks
    rows.sort(key=lambda r: (r[0][1], r[0][0]))  # by y, then x
    current: Optional[Tuple[int,int,int,int]] = None
    buff: List[str] = []
    blocks: List[Tuple[Tuple[int,int,int,int], str]] = []

    def _flush():
        nonlocal current, buff
        if current and buff:
            blocks.append((current, "\\n".join(buff)))
        current, buff = None, []

    y_gap_thresh = int(0.025 * H)  # rows within ~2.5% height
    for (bx,by,bx1,by1), txt in rows:
        if current is None:
            current = (bx,by,bx1,by1); buff=[txt]
            continue
        _, cy, _, cy1 = current
        if by - cy1 <= y_gap_thresh and abs(bx - current[0]) <= int(0.08*W):
            # merge
            nx0 = min(current[0], bx); ny0 = min(current[1], by)
            nx1 = max(current[2], bx1); ny1 = max(current[3], by1)
            current = (nx0, ny0, nx1, ny1); buff.append(txt)
        else:
            _flush()
            current = (bx,by,bx1,by1); buff=[txt]
    _flush()

    for bbox, txt in blocks:
        score = 0.0
        if ADDR_CUE_PAT.search(txt): score += 1.5
        if UK_POSTCODE_PAT.search(txt): score += 2.0
        # reward multi-line
        score += min(txt.count("\\n")+1, 4) * 0.6
        cand_blocks.append((bbox, score, txt))

    cand_blocks.sort(key=lambda t: t[1], reverse=True)
    return cand_blocks

def _pick_uk_eu_from_blocks(blocks: List[Tuple[Tuple[int,int,int,int], float, str]], img_size: Tuple[int,int]) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    uk = None; eu = None
    for bbox, _, txt in blocks:
        hint = _country_hint(txt)
        if hint == "UK" and uk is None:
            uk = (bbox, txt)
        elif hint == "EU" and eu is None:
            eu = (bbox, txt)
        if uk and eu: break
    return uk, eu

# ---- Public API ----

def process_artwork(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 400,
    model: str = "gpt-4o"
) -> Dict[str, Any]:

    is_pdf = filename.lower().endswith(".pdf")

    # Open as page images
    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")
    else:
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
            pages = [img]
        except Exception:
            return _fail("Could not open image.")

    # Try per page
    best_result: Dict[str, Any] = {"ok": False, "error": "No addresses found."}
    for page_idx, img in enumerate(pages):
        W,H = img.size

        # 1) GPT fullpage JSON classification (vision-first)
        js = _gpt_fullpage_supplier_json(client, img, model)
        uk_cands = js.get("uk") or []
        eu_cands = js.get("eu") or []

        # Convert first candidate boxes to pixels
        def _first_box(cands):
            for c in cands[:2]:
                pct = c.get("bbox_pct") or {}
                bbox = _bbox_pct_to_pixels(pct, img.size)
                # sanity clamp/pad a bit
                bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)
                text = (c.get("text") or "").strip()
                yield bbox, text

        uk_bbox_text = next(_first_box(uk_cands), (None, ""))
        eu_bbox_text = next(_first_box(eu_cands), (None, ""))

        uk_bbox, uk_text_gpt = uk_bbox_text
        eu_bbox, eu_text_gpt = eu_bbox_text

        # 2) Heuristic OCR fallback if missing
        if uk_bbox is None or eu_bbox is None:
            blocks = _rank_ocr_rows_for_addresses(img)
            uk_fallback, eu_fallback = _pick_uk_eu_from_blocks(blocks, img.size)
            if uk_bbox is None and uk_fallback:
                uk_bbox, _ = uk_fallback
                uk_bbox = _clamp_pad_bbox(uk_bbox, img.size, pad_frac=0.02)
            if eu_bbox is None and eu_fallback:
                eu_bbox, _ = eu_fallback
                eu_bbox = _clamp_pad_bbox(eu_bbox, img.size, pad_frac=0.02)

        # 3) As a last resort, guess typical panels
        if uk_bbox is None:
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.02)
                if g: uk_bbox = g; break
        if eu_bbox is None:
            for guess_fn in (_fallback_right_panel_bbox, _fallback_left_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.02)
                if g: eu_bbox = g; break

        # If still nothing meaningful, continue to next page
        if uk_bbox is None and eu_bbox is None:
            continue

        # tiny-area sanity
        if uk_bbox and _area_pct(uk_bbox, img.size) < 0.6:
            uk_bbox = _clamp_pad_bbox(uk_bbox, img.size, pad_frac=0.06)
        if eu_bbox and _area_pct(eu_bbox, img.size) < 0.6:
            eu_bbox = _clamp_pad_bbox(eu_bbox, img.size, pad_frac=0.06)

        # 4) Exact OCR via GPT on crops (with one expansion retry if unreadable)
        def _ocr_with_retries(bbox):
            if not bbox: return None, None
            crop = _crop_to_bytes(img, bbox)
            text = _gpt_exact_ocr_address(client, crop, model).strip()
            if text.upper() == "IMAGE_UNREADABLE":
                bigger = _clamp_pad_bbox(bbox, img.size, pad_frac=0.12)
                if bigger and bigger != bbox:
                    crop2 = _crop_to_bytes(img, bigger)
                    text2 = _gpt_exact_ocr_address(client, crop2, model).strip()
                    if text2.upper() != "IMAGE_UNREADABLE":
                        return bigger, text2
            return bbox, text

        uk_bbox_use, uk_text = _ocr_with_retries(uk_bbox)
        eu_bbox_use, eu_text = _ocr_with_retries(eu_bbox)

        # If GPT from fullpage provided better text for a missing crop, use it
        if (not uk_text or uk_text.upper()=="IMAGE_UNREADABLE") and uk_text_gpt:
            uk_text = uk_text_gpt
        if (not eu_text or eu_text.upper()=="IMAGE_UNREADABLE") and eu_text_gpt:
            eu_text = eu_text_gpt

        # Final classification sanity (in case GPT mis-labelled)
        if uk_text:
            if _country_hint(uk_text) == "EU" and _country_hint(eu_text or "") != "EU":
                # swap if looks wrong
                uk_text, eu_text = eu_text, uk_text
                uk_bbox_use, eu_bbox_use = eu_bbox_use, uk_bbox_use

        # If both empty, try next page
        if not (uk_text or eu_text):
            continue

        # QA consistency pass (light): re-run OCR once and compare similarity
        qa = {}
        if uk_text:
            crop = _crop_to_bytes(img, uk_bbox_use) if uk_bbox_use else None
            uk_text_2 = _gpt_exact_ocr_address(client, crop, model) if crop else uk_text
            qa_uk = _qa_compare_tesseract(crop, uk_text) if crop else {}
            qa_uk.update({
                "consistency_ratio": _similarity(uk_text, uk_text_2) or 0.0,
            })
            qa["uk"] = qa_uk
        if eu_text:
            crop = _crop_to_bytes(img, eu_bbox_use) if eu_bbox_use else None
            eu_text_2 = _gpt_exact_ocr_address(client, crop, model) if crop else eu_text
            qa_eu = _qa_compare_tesseract(crop, eu_text) if crop else {}
            qa_eu.update({
                "consistency_ratio": _similarity(eu_text, eu_text_2) or 0.0,
            })
            qa["eu"] = qa_eu

        result = {
            "ok": True,
            "page_index": page_idx,
            "uk_address_text": uk_text or None,
            "uk_bbox_pixels": list(map(int, uk_bbox_use)) if uk_bbox_use else None,
            "eu_address_text": eu_text or None,
            "eu_bbox_pixels": list(map(int, eu_bbox_use)) if eu_bbox_use else None,
            "qa": qa,
            "debug": {
                "image_size": img.size,
                "tesseract_available": TESS_AVAILABLE,
                "source": "vision-first-gpt+ocr-fallback"
            }
        }
        return result

    return best_result
