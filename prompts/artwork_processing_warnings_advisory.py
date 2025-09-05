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
Return JSON ONLY with up to 4 bounding boxes for WARNINGS / CAUTIONS / ADVISORY / SAFETY text blocks:
{
  "found": true/false,
  "regions": [
    {"bbox_pct": {"x": 0-100, "y": 0-100, "w": 0-100, "h": 0-100}},
    ...
  ]
}
Rules:
- Coordinates are percentages of the entire image.
- Look for headings like “Warnings”, “Cautions”, “Advisory”, “Safety”, “Important”, and for clusters of short imperative lines.
- If none are present, return {"found": false, "regions": []}.
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

_SAFETY_HINTS = [
    r"\bkeep\s+out\s+of\s+reach\s+of\s+(?:young\s+)?children\b",
    r"\bdo\s+not\s+exceed\b",
    r"\bnot\s+recommended\s+for\s+children\s+under\b",
    r"\bif\s+(?:pregnant|breastfeeding|breast-?feeding)\b",
    r"\bif\s+adverse\s+effects?\s+occur\b|\bdiscontinue\s+use\b",
    r"\bfor\s+external\s+use\s+only\b",
    r"\bavoid\s+contact\s+with\s+eyes\b",
    r"\bmay\s+cause\s+skin\s+irritation\b",
    r"\bstore\s+in\s+a\s+cool\s+dry\s+place\b",
    r"\bfood\s+supplement[s]?\s+should\s+not\s+be\s+used\s+as\s+a\s+substitute\b",
]
_SAFETY_HINTS = [re.compile(p, re.IGNORECASE) for p in _SAFETY_HINTS]

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

def _hint_missed_statements(full_text: str) -> List[str]:
    cands = []
    for s in _split_statements(full_text):
        for pat in _SAFETY_HINTS:
            if pat.search(s):
                cands.append(s)
                break
    # de-dupe
    seen=set(); out=[]
    for s in cands:
        k = re.sub(r"\s+", " ", s.lower())
        if k not in seen:
            seen.add(k); out.append(s)
    return out


def _gpt_bbox_locator_warnadv_multi(client, img: Image.Image, model: str):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = _encode_data_url(buf.getvalue())
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": WARNADV_BBOX_FINDER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Locate up to 4 WARNINGS/CAUTIONS/ADVISORY areas. JSON only."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
    )
    try:
        js = json.loads(r.choices[0].message.content.strip())
        if not js.get("found"):
            return []
        W, H = img.size
        out = []
        for reg in js.get("regions", [])[:4]:
            pct = reg.get("bbox_pct", {})
            x = int(W * pct.get("x",0) / 100.0); y = int(H * pct.get("y",0) / 100.0)
            w = int(W * pct.get("w",0) / 100.0); h = int(H * pct.get("h",0) / 100.0)
            out.append((x, y, x + w, y + h))
        return [b for b in out if (b[2] > b[0] and b[3] > b[1])]
    except Exception:
        return []

def _merge_nearby_boxes(boxes: List[Tuple[int,int,int,int]], img_size: Tuple[int,int], iou_thresh: float = 0.35):
    def _iou(a,b):
        ax0,ay0,ax1,ay1=a; bx0,by0,bx1,by1=b
        inter_x0=max(ax0,bx0); inter_y0=max(ay0,by0); inter_x1=min(ax1,bx1); inter_y1=min(ay1,by1)
        inter=max(0, inter_x1-inter_x0)*max(0, inter_y1-inter_y0)
        a_area=(ax1-ax0)*(ay1-ay0); b_area=(bx1-bx0)*(by1-by0)
        union=a_area+b_area-inter if (a_area+b_area-inter)>0 else 1
        return inter/union
    keep=[]
    for b in sorted(boxes, key=lambda r:(r[2]-r[0])*(r[3]-r[1]), reverse=True):
        if all(_iou(b,k)<iou_thresh for k in keep):
            keep.append(_clamp_pad_bbox(b, img_size, pad_frac=0.03))
        if len(keep)>=4: break
    return keep

_SPLIT_RE = re.compile(r"(?:\n+|•\s*|-{1,2}\s+|;\s+|(?<=[\.\!\?])\s+(?=[A-Z]))")

def _split_statements(text: str) -> List[str]:
    # normalise bullets/hyphens
    t = text.replace("\u2022", "•")
    t = re.sub(r"[ \t]+", " ", t)
    parts = [p.strip(" \t\r\n•-–—") for p in _SPLIT_RE.split(t) if p.strip()]
    # de-dupe while preserving order
    seen=set(); out=[]
    for p in parts:
        key=re.sub(r"\s+", " ", p.lower())
        if key not in seen:
            seen.add(key); out.append(p)
    return out

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

    # ============== 1) Load page(s) and collect MULTIPLE regions ==============
    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        page_idx: Optional[int] = None
        img: Optional[Image.Image] = None
        boxes: List[Tuple[int,int,int,int]] = []

        # Try each page until we find any plausible regions
        for i, pg in enumerate(pages):
            ocr_box = _find_region_via_ocr_warnadv(pg)
            gpt_boxes = _gpt_bbox_locator_warnadv_multi(client, pg, model)
            cand_boxes = [b for b in ([ocr_box] if ocr_box else [])] + gpt_boxes
            if not cand_boxes:
                # heuristic guesses if locator fails
                for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                    g = _clamp_pad_bbox(guess_fn(pg), pg.size, pad_frac=0.03)
                    if g:
                        cand_boxes.append(g)
                        break
            if cand_boxes:
                boxes = _merge_nearby_boxes(cand_boxes, pg.size)
                if boxes:
                    page_idx, img = i, pg
                    break

        # If nothing found on any page, fall back to page 0 (full-page OCR only)
        if img is None:
            img = pages[0]
            page_idx = 0
            boxes = []

    else:
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        page_idx = 0

        ocr_box = _find_region_via_ocr_warnadv(img)
        gpt_boxes = _gpt_bbox_locator_warnadv_multi(client, img, model)
        cand_boxes = [b for b in ([ocr_box] if ocr_box else [])] + gpt_boxes
        if not cand_boxes:
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.03)
                if g:
                    cand_boxes.append(g)
                    break
        boxes = _merge_nearby_boxes(cand_boxes, img.size) if cand_boxes else []

    # ============== 2) OCR every box + 1 full-page sweep ======================
    raw_blocks: List[str] = []
    for b in boxes:
        cb = _crop_to_bytes(img, b)
        txt = _gpt_exact_ocr_warnadv(client, cb, model).strip()
        if txt.upper() != "IMAGE_UNREADABLE":
            raw_blocks.append(txt)

    full_raw = _gpt_fullpage_warnadv_text(client, img, model).strip()
    if full_raw.upper() != "IMAGE_UNREADABLE":
        raw_blocks.append(full_raw)

    if not raw_blocks:
        return _fail("Could not locate WARNINGS/ADVISORY content.")

    combined = "\n".join(raw_blocks)
    clean_text = _safe_punct_scrub(combined)

    # ============== 3) Split to atomic statements =============================
    statements = _split_statements(clean_text)

    # ============== 4) Classify + top-up with hint sweep ======================
    classified = _gpt_classify_warnadv(client, "\n".join(statements), model)
    warning_info: List[str]  = [s for s in (classified.get("warning_info")  or []) if str(s).strip()]
    advisory_info: List[str] = [s for s in (classified.get("advisory_info") or []) if str(s).strip()]

    # Top-up: look for staple phrases anywhere in the full page text
    hinted = _hint_missed_statements(full_raw if full_raw.upper() != "IMAGE_UNREADABLE" else clean_text)

    def _have(lst: List[str], s: str) -> bool:
        key = re.sub(r"\s+", " ", s.lower())
        return any(re.sub(r"\s+", " ", x.lower()) == key for x in lst)

    for s in hinted:
        if not _have(warning_info, s) and not _have(advisory_info, s):
            if _NEG_IMPERATIVE.search(s):
                warning_info.append(s)
            else:
                advisory_info.append(s)

    # ============== 5) QA & acceptance =======================================
    # Use the largest box (if any) for QA image; else whole page
    qa_img_bytes = None
    if boxes:
        biggest = max(boxes, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
        qa_img_bytes = _crop_to_bytes(img, biggest)
    else:
        qa_img_bytes = _crop_to_bytes(img, (0, 0, img.size[0], img.size[1]))

    qa = _qa_compare_tesseract(qa_img_bytes, "\n".join(statements) if statements else clean_text)

    # We still do a tiny consistency check if we had at least one box
    consistency_ok = True
    consist_ratio = 100.0
    if boxes:
        again_txt = _gpt_exact_ocr_warnadv(client, qa_img_bytes, model)
        consist_ratio = _similarity(again_txt, "\n".join(statements) if statements else clean_text) or 0.0
        consistency_ok = (again_txt == ("\n".join(statements) if statements else clean_text)) or (consist_ratio >= 98.0)

    structure_pass = bool(warning_info or advisory_info)
    qa.update({
        "structure_pass": structure_pass,
        "consistency_ok": consistency_ok,
        "consistency_ratio": consist_ratio,
    })
    qa["accepted"] = bool(structure_pass and consistency_ok)

    return {
        "ok": True,
        "page_index": page_idx,
        "bbox_pixels": [list(map(int, b)) for b in boxes] if boxes else None,
        "warning_info": warning_info,
        "advisory_info": advisory_info,
        "qa": qa,
        "debug": {
            "image_size": img.size if img else None,
            "num_regions": len(boxes),
            "bbox_area_pct_each": [ _area_pct(b, img.size) for b in boxes ] if boxes else None,
            "tesseract_available": TESS_AVAILABLE,
            "classifier_error": classified.get("error") if isinstance(classified, dict) else None
        }
    }


