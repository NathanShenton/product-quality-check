# artwork_processing_supplier_addresses.py
from __future__ import annotations
import io, json, re
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    _fail, _pdf_to_page_images, _pdf_to_page_images_adaptive, _rerender_single_page,
    _encode_data_url, _crop_to_bytes, _ocr_words,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _area_pct, _qa_compare_tesseract, _similarity, _gpt_normalize_flatpack_page
)

# ==============================================
# Supplier Address Extraction (UK / EU)
# ==============================================

# ---- Regex & heuristics ----
UK_POSTCODE_PAT = re.compile(
    r"\b(?:GIR\s?0AA|[A-Z]{1,2}\d[A-Z\d]?\s*\d[ABD-HJLNP-UW-Z]{2})\b", re.I
)

# Common company/address cues
ADDR_CUE_PAT = re.compile(
    r"\b(company|co\.?|ltd\.?|limited|plc|gmbh|s\.?r\.?l\.?|sa|bv|s\.?a\.?"
    r"|street|str\.?|road|rd\.?|avenue|ave\.?|way|close|park|industrial|estate|"
    r"unit|suite|po box|postbus|postcode|zip|city|town|county|manufactured by|"
    r"manufacturer|importer|distributed by|distributor|responsible person|rp|address)\b",
    re.I
)

UK_CUES = {
    "uk","u.k.","united kingdom","england","scotland","wales","gb","great britain","northern ireland"
}

EU_MEMBER_COUNTRIES = {
    "austria","belgium","bulgaria","croatia","cyprus","czech republic","czechia","denmark",
    "estonia","finland","france","germany","greece","hungary","ireland","italy","latvia",
    "lithuania","luxembourg","malta","netherlands","poland","portugal","romania","slovakia",
    "slovenia","spain","sweden"
}

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
- Classify as UK if it clearly refers to the United Kingdom (UK, GB, England, Scotland, Wales, Northern Ireland) or has a UK postcode.
- Classify as EU if it clearly refers to an EU Member State address (Ireland, France, Germany, Netherlands, Spain, etc.).
- **Do not split a single address across UK and EU lists.** If a line belongs to the same postal block (e.g., company + street + city + postcode + country), keep all lines together.
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

# ---- GPT helpers ----
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

# ---- Country hints / text hygiene ----
def _has_uk_postcode(text: str) -> bool:
    return bool(UK_POSTCODE_PAT.search(text or ""))

def _mentions_uk(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in UK_CUES) or " gb" in t or t.endswith(" gb")

def _mentions_eu_country(text: str) -> bool:
    t = (text or "").lower()
    return any(c in t for c in EU_MEMBER_COUNTRIES)

def _is_uk_like_line(line: str) -> bool:
    ln = (line or "").strip()
    return bool(UK_POSTCODE_PAT.search(ln)) or _mentions_uk(ln)

def _is_eu_like_line(line: str) -> bool:
    ln = (line or "").lower()
    return _mentions_eu_country(ln)

def _country_hint(text: str) -> str:
    t = (text or "").lower()
    if _has_uk_postcode(text) or any(k in t for k in UK_CUES):
        return "UK"
    if any(c in t for c in EU_MEMBER_COUNTRIES):
        return "EU"
    return "UNK"

def _normalize_lines(text: str) -> str:
    if not text: return text
    lines = [re.sub(r"\s+", " ", ln).strip(" ,") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()

def _ends_like_truncated(text: str) -> bool:
    if not text: return False
    t = text.strip()
    return t.endswith(",") or t.endswith(";") or len(t.splitlines()[-1].split()) <= 2

def _union(b1, b2):
    if not b1: return b2
    if not b2: return b1
    x0 = min(b1[0], b2[0]); y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2]); y1 = max(b1[3], b2[3])
    return (x0, y0, x1, y1)

def _score_address_quality(text: str, expect: str) -> float:
    if not text: return 0.0
    t = text.lower()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    score = 0.0
    score += min(len(lines), 5) * 0.8
    if re.search(r"\b(street|str\.?|road|rd\.?|avenue|ave\.?|way|close|park|industrial|estate|unit|suite|po box|postbus)\b", t):
        score += 1.2
    if re.search(r"\b(ltd\.?|limited|plc|gmbh|s\.?r\.?l\.?|s\.?a\.?|b\.?v\.?|slu|s\.?l\.?)\b", t):
        score += 0.8
    ukpc = _has_uk_postcode(text); ukm = _mentions_uk(text); eum = _mentions_eu_country(text)
    if expect == "UK":
        if ukpc: score += 2.0
        if ukm:  score += 1.0
        if eum:  score -= 1.0
    else:
        if eum:  score += 1.5
        if ukpc: score -= 1.5
        if ukm:  score -= 0.8
    if _ends_like_truncated(text): score -= 0.6
    return score

# ---- OCR grouping fallback ----
def _bbox_pct_to_pixels(pct: Dict[str,float], size: Tuple[int,int]) -> Tuple[int,int,int,int]:
    W,H = size
    x = int(W * float(pct.get("x",0)) / 100.0)
    y = int(H * float(pct.get("y",0)) / 100.0)
    w = int(W * float(pct.get("w",0)) / 100.0)
    h = int(H * float(pct.get("h",0)) / 100.0)
    return (x, y, x+w, y+h)

def _rank_ocr_rows_for_addresses(full_img: Image.Image):
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return []

    W,H = full_img.size
    lines: Dict[Tuple[int,int,int], List[int]] = {}
    for i,txt in enumerate(data["text"]):
        if not txt:
            continue
        key = (data.get("block_num",[0])[i], data.get("par_num",[0])[i], data.get("line_num",[0])[i])
        lines.setdefault(key, []).append(i)

    blocks: List[Tuple[Tuple[int,int,int,int], str]] = []
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

    rows.sort(key=lambda r: (r[0][1], r[0][0]))
    current: Optional[Tuple[int,int,int,int]] = None
    buff: List[str] = []
    def _flush():
        nonlocal current, buff
        if current and buff:
            blocks.append((current, "\\n".join(buff)))
        current, buff = None, []
    y_gap_thresh = int(0.025 * H)
    for (bx,by,bx1,by1), txt in rows:
        if current is None:
            current = (bx,by,bx1,by1); buff=[txt]; continue
        _, cy, _, cy1 = current
        if by - cy1 <= y_gap_thresh and abs(bx - current[0]) <= int(0.08*W):
            nx0 = min(current[0], bx); ny0 = min(current[1], by)
            nx1 = max(current[2], bx1); ny1 = max(current[3], by1)
            current = (nx0, ny0, nx1, ny1); buff.append(txt)
        else:
            _flush(); current = (bx,by,bx1,by1); buff=[txt]
    _flush()

    cand_blocks: List[Tuple[Tuple[int,int,int,int], float, str]] = []
    for bbox, txt in blocks:
        score = 0.0
        if ADDR_CUE_PAT.search(txt): score += 1.5
        if UK_POSTCODE_PAT.search(txt): score += 2.0
        score += min(txt.count("\\n")+1, 4) * 0.6
        cand_blocks.append((bbox, score, txt))

    cand_blocks.sort(key=lambda t: t[1], reverse=True)
    return cand_blocks

def _pick_uk_eu_from_blocks(blocks: List[Tuple[Tuple[int,int,int,int], float, str]]) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    uk = None; eu = None
    for bbox, _, txt in blocks:
        hint = _country_hint(txt)
        if hint == "UK" and uk is None:
            uk = (bbox, txt)
        elif hint == "EU" and eu is None:
            eu = (bbox, txt)
        if uk and eu: break
    return uk, eu

# ---- Stitching / validation ----
def _stitch_and_validate(uk_text, eu_text, uk_bbox, eu_bbox):
    notes = []
    uk_text = _normalize_lines(uk_text or "")
    eu_text = _normalize_lines(eu_text or "")

    # 1) Move leading UK-like prefix from EU → UK if UK looks truncated
    if eu_text and (_ends_like_truncated(uk_text) or not _has_uk_postcode(uk_text)):
        eu_lines = [ln for ln in eu_text.splitlines() if ln.strip()]
        prefix = []
        i = 0
        while i < len(eu_lines) and _is_uk_like_line(eu_lines[i]):
            prefix.append(eu_lines[i]); i += 1
        if prefix:
            notes.append("Moved UK-looking prefix lines from EU into UK.")
            stitched = "\n".join(prefix)
            uk_text = (uk_text + ("\n" if uk_text else "") + stitched).strip()
            tail = "\n".join(eu_lines[i:]).strip()
            eu_text = tail
            if not eu_text:
                uk_bbox = _union(uk_bbox, eu_bbox); eu_bbox = None

    # 2) Split EU-looking tail back out of UK if head is clearly UK
    if uk_text:
        lines = [ln for ln in uk_text.splitlines() if ln.strip()]
        cut_idx = None
        for j in range(len(lines)-1, -1, -1):
            if _is_eu_like_line(lines[j]) and not _is_uk_like_line(lines[j]):
                cut_idx = j
            else:
                break
        if cut_idx is not None:
            head = "\n".join(lines[:cut_idx]).strip()
            tail = "\n".join(lines[cut_idx:]).strip()
            if head and (_has_uk_postcode(head) or _mentions_uk(head)):
                notes.append("Split EU-looking tail from UK back into EU.")
                uk_text = head
                eu_text = (tail + ("\n" + eu_text if eu_text else "")).strip()

    # 3) If UK looks EU and EU empty, swap
    if uk_text and not eu_text and _mentions_eu_country(uk_text) and not _mentions_uk(uk_text):
        notes.append("UK block looked EU; swapped to EU.")
        eu_text, eu_bbox = uk_text, uk_bbox
        uk_text, uk_bbox = "", None

    # 4) Score-based swap if classifications look inverted
    uk_score = _score_address_quality(uk_text, "UK")
    eu_score = _score_address_quality(eu_text, "EU")
    if _score_address_quality(uk_text, "EU") > uk_score + 0.8 and _score_address_quality(eu_text, "UK") > eu_score + 0.8:
        notes.append("Swapped UK/EU blocks based on plausibility scores.")
        uk_text, eu_text = eu_text, uk_text
        uk_bbox, eu_bbox = eu_bbox, uk_bbox

    # 5) Final hygiene: remove stray UK tokens from EU block
    if eu_text and _mentions_uk(eu_text):
        notes.append("Removed stray UK token from EU block.")
        eu_text = re.sub(r'\b(uk|u\.k\.|gb|great britain)\b\.?', '', eu_text, flags=re.I).strip(' ,')

    return uk_text or None, eu_text or None, uk_bbox, eu_bbox, notes

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

    # Load pages
    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        # Adaptive render to help tiny vector text
        try:
            pages, _, _, _ = _pdf_to_page_images_adaptive(file_bytes, dpi_primary=render_dpi, dpi_fallback=max(550, render_dpi+150))
        except Exception:
            pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")
    else:
        try:
            base_img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        # Normalize flat artwork to main consumer panel before asking for addresses
        norm_img, _, _ = _gpt_normalize_flatpack_page(client, base_img, model)
        pages = [norm_img or base_img]

    for page_idx, img in enumerate(pages):
        # 1) Vision full-page JSON to get initial UK/EU candidates
        js = _gpt_fullpage_supplier_json(client, img, model)
        uk_cands = js.get("uk") or []
        eu_cands = js.get("eu") or []

        def _first_box(cands):
            for c in cands[:2]:
                pct = c.get("bbox_pct") or {}
                bbox = _bbox_pct_to_pixels(pct, img.size)
                bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.03)
                text = (c.get("text") or "").strip()
                if bbox and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
                    yield bbox, text

        uk_bbox, uk_text_gpt = next(_first_box(uk_cands), (None, ""))
        eu_bbox, eu_text_gpt = next(_first_box(eu_cands), (None, ""))

        # 2) OCR-line fallback if missing
        if uk_bbox is None or eu_bbox is None:
            blocks = _rank_ocr_rows_for_addresses(img)
            uk_fb, eu_fb = _pick_uk_eu_from_blocks(blocks)
            if uk_bbox is None and uk_fb:
                uk_bbox = _clamp_pad_bbox(uk_fb[0], img.size, pad_frac=0.03)
            if eu_bbox is None and eu_fb:
                eu_bbox = _clamp_pad_bbox(eu_fb[0], img.size, pad_frac=0.03)

        # 3) Heuristic panel guesses as last resort
        if uk_bbox is None:
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.03)
                if g: uk_bbox = g; break
        if eu_bbox is None:
            for guess_fn in (_fallback_right_panel_bbox, _fallback_left_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.03)
                if g: eu_bbox = g; break

        # If still nothing, try next page
        if uk_bbox is None and eu_bbox is None:
            continue

        # Tiny-area sanity → lightly expand
        if uk_bbox and _area_pct(uk_bbox, img.size) < 0.8:
            uk_bbox = _clamp_pad_bbox(uk_bbox, img.size, pad_frac=0.12)
        if eu_bbox and _area_pct(eu_bbox, img.size) < 0.8:
            eu_bbox = _clamp_pad_bbox(eu_bbox, img.size, pad_frac=0.12)

        # 4) Exact OCR with progressive widening (+ optional hi-DPI re-render for PDFs)
        def _ocr_with_retries(bbox, side: str):
            if not bbox: return None, None, None
            crop = _crop_to_bytes(img, bbox)
            txt = _gpt_exact_ocr_address(client, crop, model).strip()
            if txt.upper() != "IMAGE_UNREADABLE":
                return bbox, crop, txt

            # widen +12%
            bigger1 = _clamp_pad_bbox(bbox, img.size, pad_frac=0.12)
            if bigger1 and bigger1 != bbox:
                crop1 = _crop_to_bytes(img, bigger1)
                txt1 = _gpt_exact_ocr_address(client, crop1, model).strip()
                if txt1.upper() != "IMAGE_UNREADABLE":
                    return bigger1, crop1, txt1

                # widen +20%
                bigger2 = _clamp_pad_bbox(bigger1, img.size, pad_frac=0.20)
                if bigger2 and bigger2 != bigger1:
                    crop2 = _crop_to_bytes(img, bigger2)
                    txt2 = _gpt_exact_ocr_address(client, crop2, model).strip()
                    if txt2.upper() != "IMAGE_UNREADABLE":
                        return bigger2, crop2, txt2

            # optional hi-DPI retry for PDFs
            if is_pdf:
                hi_img = _rerender_single_page(file_bytes, page_idx, dpi=max(550, render_dpi+150))
                if hi_img is not None:
                    ow, oh = img.size
                    nw, nh = hi_img.size
                    sx, sy = (nw / max(1, ow), nh / max(1, oh))
                    scaled_bbox = (int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy))
                    scaled_bbox = _clamp_pad_bbox(scaled_bbox, hi_img.size, pad_frac=0.03)
                    if scaled_bbox:
                        crop3 = _crop_to_bytes(hi_img, scaled_bbox)
                        txt3 = _gpt_exact_ocr_address(client, crop3, model).strip()
                        if txt3.upper() != "IMAGE_UNREADABLE":
                            # swap working canvas for downstream QA
                            return scaled_bbox, crop3, txt3

            return bbox, crop, txt  # still unreadable

        uk_bbox_use, uk_crop, uk_text = _ocr_with_retries(uk_bbox, "uk")
        eu_bbox_use, eu_crop, eu_text = _ocr_with_retries(eu_bbox, "eu")

        # If fullpage GPT gave cleaner text, prefer it when crop is unreadable
        if (not uk_text or uk_text.upper() == "IMAGE_UNREADABLE") and uk_text_gpt:
            uk_text = uk_text_gpt
        if (not eu_text or eu_text.upper() == "IMAGE_UNREADABLE") and eu_text_gpt:
            eu_text = eu_text_gpt

        # Stitch / validate / fix mislabels
        uk_text, eu_text, uk_bbox_use, eu_bbox_use, val_notes = _stitch_and_validate(
            uk_text, eu_text, uk_bbox_use, eu_bbox_use
        )

        if not (uk_text or eu_text):
            continue

        # 5) Light consistency QA (re-run on same crop to compute similarity)
        qa = {}
        if uk_text:
            if uk_crop is None and uk_bbox_use:
                uk_crop = _crop_to_bytes(img, uk_bbox_use)
            uk_text_2 = _gpt_exact_ocr_address(client, uk_crop, model) if uk_crop else uk_text
            qa_uk = _qa_compare_tesseract(uk_crop, uk_text) if uk_crop else {}
            qa_uk.update({"consistency_ratio": _similarity(uk_text, uk_text_2) or 0.0})
            qa["uk"] = qa_uk
        if eu_text:
            if eu_crop is None and eu_bbox_use:
                eu_crop = _crop_to_bytes(img, eu_bbox_use)
            eu_text_2 = _gpt_exact_ocr_address(client, eu_crop, model) if eu_crop else eu_text
            qa_eu = _qa_compare_tesseract(eu_crop, eu_text) if eu_crop else {}
            qa_eu.update({"consistency_ratio": _similarity(eu_text, eu_text_2) or 0.0})
            qa["eu"] = qa_eu

        return {
            "ok": True,
            "page_index": page_idx,
            "uk_address_text": uk_text or None,
            "uk_bbox_pixels": list(map(int, uk_bbox_use)) if uk_bbox_use else None,
            "eu_address_text": eu_text or None,
            "eu_bbox_pixels": list(map(int, eu_bbox_use)) if eu_bbox_use else None,
            "qa": qa,
            "validation_notes": val_notes,
            "debug": {
                "image_size": img.size,
                "tesseract_available": TESS_AVAILABLE,
                "source": ("vision-first-gpt+ocr-fallback+normalized-panel"
                           if not is_pdf else "vision-first-gpt+ocr-fallback(pdf)"),
            }
        }

    return {"ok": False, "error": "No addresses found on any page."}
