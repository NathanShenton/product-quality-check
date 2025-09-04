# artwork_processing_packsize_nutrition.py
from __future__ import annotations
import io, re, json
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from prompts.artwork_processing_common import (
    fitz, Image as PILImage, TESS_AVAILABLE,
    _fail, _pdf_to_page_images, _safe_punct_scrub, _similarity, _clean_gpt_json_block,
    _area_pct, _encode_data_url, _crop_to_bytes, _ocr_words, _qa_compare_tesseract,
    _clamp_pad_bbox, _fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox,
    _pdf_find_packsize_block, _pdf_find_nutrition_block,
    _U_ALL, _U_COUNT, _NET_LBL, _GROSS_LBL, _DRAINED_LBL, _E_MARK,
    MULTIPACK_RE, COUNT_RE, SINGLE_QTY_RE, LABELED_WEIGHT_RE, COMPACT_GW_NW_RE,
    _num, _norm_unit, _title_if_count
)

# ---------- Nutrition helpers/constants ----------
CANON_NUTRIENTS = {
    "Energy", "Fat", "of which saturates", "Carbohydrate", "of which sugars",
    "Fibre", "Protein", "Salt", "Sodium", "Omega-3", "EPA", "DHA",
    "Vitamin A", "Vitamin D", "Vitamin E", "Vitamin K",
    "Vitamin C", "Thiamin (B1)", "Riboflavin (B2)", "Niacin (B3)",
    "Pantothenic acid (B5)", "Vitamin B6", "Biotin (B7)", "Folate (B9)",
    "Vitamin B12", "Calcium", "Phosphorus", "Magnesium", "Iron", "Zinc", "Copper",
    "Manganese", "Selenium", "Iodine", "Chromium", "Molybdenum", "Potassium", "Chloride", "Fluoride"
}
NUTRIENT_SYNONYMS = {
    "energy": "Energy", "kcal": "Energy", "kilocalories": "Energy", "kj": "Energy", "kilojoules": "Energy",
    "fat": "Fat", "total fat": "Fat", "fats": "Fat",
    "saturates": "of which saturates", "saturated fat": "of which saturates", "sat fat": "of which saturates",
    "carbohydrate": "Carbohydrate", "carbs": "Carbohydrate", "carbohydrates": "Carbohydrate",
    "sugars": "of which sugars", "of which sugars": "of which sugars", "sugar": "of which sugars",
    "fibre": "Fibre", "fiber": "Fibre", "dietary fibre": "Fibre", "dietary fiber": "Fibre",
    "protein": "Protein", "salt": "Salt", "sodium": "Sodium",
    "omega 3": "Omega-3", "omega-3": "Omega-3",
    "epa": "EPA", "eicosapentaenoic acid": "EPA", "dha": "DHA", "docosahexaenoic acid": "DHA",
    "monounsaturates": "Monounsaturates", "mono-unsaturates": "Monounsaturates",
    "polyunsaturates": "Polyunsaturates", "poly-unsaturates": "Polyunsaturates",
    "trans fat": "Trans fat", "trans": "Trans fat", "cholesterol": "Cholesterol",
    "starch": "Starch", "polyols": "Polyols", "added sugars": "Added sugars", "free sugars": "Free sugars",
    "energy (kj)": "Energy", "energy (kcal)": "Energy",
    "vitamin a": "Vitamin A", "retinol": "Vitamin A", "retinyl": "Vitamin A", "beta-carotene": "Vitamin A",
    "vitamin d": "Vitamin D", "vitamin d3": "Vitamin D", "cholecalciferol": "Vitamin D", "vit d": "Vitamin D",
    "vitamin e": "Vitamin E", "alpha-tocopherol": "Vitamin E", "dl-alpha tocopheryl": "Vitamin E", "vit e": "Vitamin E",
    "vitamin k": "Vitamin K", "vitamin k1": "Vitamin K", "phylloquinone": "Vitamin K", "vit k": "Vitamin K",
    "vitamin c": "Vitamin C", "ascorbic acid": "Vitamin C", "l-ascorbic": "Vitamin C", "vit c": "Vitamin C",
    "thiamin": "Thiamin (B1)", "thiamine": "Thiamin (B1)", "vitamin b1": "Thiamin (B1)", "b1": "Thiamin (B1)",
    "riboflavin": "Riboflavin (B2)", "vitamin b2": "Riboflavin (B2)", "b2": "Riboflavin (B2)",
    "niacin": "Niacin (B3)", "nicotinamide": "Niacin (B3)", "niacinamide": "Niacin (B3)", "vitamin b3": "Niacin (B3)", "b3": "Niacin (B3)",
    "pantothenic acid": "Pantothenic acid (B5)", "pantothenate": "Pantothenic acid (B5)", "calcium pantothenate": "Pantothenic acid (B5)", "vitamin b5": "Pantothenic acid (B5)", "b5": "Pantothenic acid (B5)",
    "vitamin b6": "Vitamin B6", "pyridoxine": "Vitamin B6", "pyridoxal": "Vitamin B6", "pyridoxamine": "Vitamin B6", "b6": "Vitamin B6",
    "biotin": "Biotin (B7)", "vitamin b7": "Biotin (B7)", "b7": "Biotin (B7)",
    "folate": "Folate (B9)", "folic acid": "Folate (B9)", "vitamin b9": "Folate (B9)", "b9": "Folate (B9)",
    "vitamin b12": "Vitamin B12", "cyanocobalamin": "Vitamin B12", "methylcobalamin": "Vitamin B12", "b12": "Vitamin B12",
    "calcium": "Calcium", "phosphorus": "Phosphorus", "phosphate": "Phosphorus",
    "magnesium": "Magnesium", "iron": "Iron", "ferric": "Iron", "ferrous": "Iron",
    "zinc": "Zinc", "copper": "Copper", "manganese": "Manganese",
    "selenium": "Selenium", "iodine": "Iodine", "chromium": "Chromium", "molybdenum": "Molybdenum",
    "potassium": "Potassium", "chloride": "Chloride", "fluoride": "Fluoride",
    "isoflavones": "Isoflavones",
}
def _norm_unit_nutri(u: str) -> str:
    if not u: return ""
    u = u.strip().replace("µ", "u").lower()
    MAP = {"mcg":"mcg","ug":"mcg","μg":"mcg","mg":"mg","g":"g","kg":"kg",
           "kj":"kJ","kcal":"kcal","cal":"kcal","kjoule":"kJ","kjoules":"kJ",
           "litre":"l","litres":"l","l":"l","ml":"ml","iu":"IU"}
    return MAP.get(u, u)

def _fixnum(x):
    if x is None: return None
    if isinstance(x,(int,float)): return float(x)
    if isinstance(x,str):
        import re
        s = x.strip().replace(",", ".")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None
    return None

def _canon_nutrient_name(raw: str) -> str:
    if not raw: return raw
    base = re.sub(r"\s+", " ", raw).strip()
    key = base.lower().rstrip(": ")
    key = key.replace("of which saturated", "saturates").replace("of which saturates", "saturates")
    key = key.replace("of which sugars", "sugars")
    return NUTRIENT_SYNONYMS.get(key, base)

# ---------- Nutrition systems ----------
NUTRI_BBOX_FINDER_SYSTEM = """
You are a vision locator. You will be shown a full label page.
Return JSON ONLY for the main nutrition information/table if present:
{"bbox_pct":{"x":0-100,"y":0-100,"w":0-100,"h":0-100},"found":true|false}
Guidance:
- Look for headings like "Nutrition", "Nutritional information", "Typical values",
  or columns like "per 100g", "per 100ml", "per serving", "Amount per capsule", "% NRV" or "% RI".
- Choose the largest table-like region representing nutrition values.
- If not present, return {"found":false}.
""".strip()

NUTRI_EXTRACTOR_SYSTEM = """
You are an exact table OCR+parser for packaged food/supplement nutrition panels.
INPUT: an image crop containing a nutrition table or list.
OUTPUT: JSON ONLY with this schema (do not add fields):
{
  "panels": [
    {
      "basis": "per_100g" | "per_100ml" | "per_serving" | "per_capsule" | "unknown",
      "serving_size": {"value": 45.0, "unit": "g"} | null,
      "rows": [
        {
          "name": "Energy" | "Fat" | "of which saturates" | "Carbohydrate" | "of which sugars" | "Fibre" | "Protein" | "Salt" | "Sodium" | "Vitamin C" | "...",
          "amounts": [{"value": 1944.0, "unit": "kJ"},{"value": 465.0, "unit": "kcal"}],
          "nrv_pct": 12.0 | null,
          "notes": null | "Isoflavones (40%)"
        }
      ]
    }
  ],
  "footnotes": {"nrv_not_established": true|false,"symbols_seen": ["†"],"raw_notes": ["NRV: Nutrient Reference Value not established"]}
}
Rules:
- Read visible text only; do not invent values.
- Convert decimal commas to decimal points.
- Keep Energy as two entries if both kJ and kcal are shown.
- Populate nrv_pct if %NRV/%RI exists (strip "%").
- If panel shows both per 100 and per serving/capsule, produce two objects.
- For “Amount per capsule/tablet/scoop”, set basis="per_capsule".
- If legibility is poor, omit that row rather than guessing.
""".strip()

NUTRI_HEADER_PAT = re.compile(
    r"\bnutrition(?:al)?\b|typical values|per\s*100\s*(?:g|ml)\b|per\s*serving\b|"
    r"amount per capsule|%?\s*(?:nrv|ri)\b",
    re.IGNORECASE
)

def _find_region_via_ocr_nutri(full_img: Image.Image):
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    rows: Dict[Tuple[int,int,int], List[int]] = {}
    for i in range(len(data["text"])):
        if not data["text"][i]: continue
        key = (data.get("block_num", [0])[i], data.get("par_num",   [0])[i], data.get("line_num",  [0])[i])
        rows.setdefault(key, []).append(i)
    candidates: List[Tuple[int,int,int,int]] = []
    for idxs in rows.values():
        line_txt = " ".join(data["text"][i] for i in idxs if data["text"][i]).strip()
        if not line_txt: continue
        if (NUTRI_HEADER_PAT.search(line_txt)
            or re.search(r"\b(kJ|kcal)\b", line_txt, re.IGNORECASE)
            or re.search(r"\bper\s*(?:serving|capsule|tablet|100\s*(?:g|ml))\b", line_txt, re.IGNORECASE)
            or re.search(r"%\s*(?:NRV|RI)\b", line_txt, re.IGNORECASE)):
            xs = [data["left"][i] for i in idxs]; ys = [data["top"][i] for i in idxs]
            ws = [data["width"][i] for i in idxs]; hs = [data["height"][i] for i in idxs]
            x0 = max(0, min(xs) - int(0.03 * W))
            x1 = min(W, max(xs[j] + ws[j] for j in range(len(xs))) + int(0.03 * W))
            y0 = max(0, min(ys) - int(0.02 * H))
            y1 = min(H, max(ys[j] + hs[j] for j in range(len(ys))) + int(0.30 * H))
            candidates.append((x0, y0, x1, y1))
    if candidates:
        return max(candidates, key=lambda b: (b[2]-b[0]) * (1.0 + 0.4*(b[3]-b[1])))
    return None

def _gpt_bbox_locator_nutri(client, img: Image.Image, model: str):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, top_p=0,
            messages=[
                {"role": "system", "content": NUTRI_BBOX_FINDER_SYSTEM},
                {"role": "user", "content": [
                    {"type":"text","text":"Locate the nutrition panel and return JSON only."},
                    {"type":"image_url","image_url":{"url": _encode_data_url(buf.getvalue())}}
                ]}
            ]
        )
        raw = r.choices[0].message.content.strip()
        js = json.loads(raw)
        if not js.get("found"):
            return None
        W, H = img.size
        pct = js["bbox_pct"]
        x = int(W * float(pct["x"]) / 100.0)
        y = int(H * float(pct["y"]) / 100.0)
        w = int(W * float(pct["w"]) / 100.0)
        h = int(H * float(pct["h"]) / 100.0)
        return (x, y, x+w, y+h)
    except Exception:
        return None

def _gpt_extract_nutri(client, crop_bytes: bytes, model: str) -> Dict[str,Any]:
    r = client.chat.completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role":"system","content": NUTRI_EXTRACTOR_SYSTEM},
            {"role":"user","content":[
                {"type":"text","text":"Return JSON only."},
                {"type":"image_url","image_url":{"url": _encode_data_url(crop_bytes)}}
            ]}
        ]
    )
    raw = r.choices[0].message.content.strip()
    try:
        return json.loads(_clean_gpt_json_block(raw))
    except Exception:
        return {"error":"NUTRITION_PARSE_FAILED","raw_model_output": raw}

def _normalize_nutri(parsed: Dict[str,Any]) -> Dict[str,Any]:
    if not isinstance(parsed, dict): return {"panels": [], "footnotes": {}, "raw": parsed}
    for p in parsed.get("panels", []):
        b=(p.get("basis") or "").lower().replace(" ", "_")
        if b in {"per_100","per_100_g"}: p["basis"]="per_100g"
        elif b in {"per_100ml","per_100_ml"}: p["basis"]="per_100ml"
        elif b not in {"per_100g","per_100ml","per_serving","per_capsule"}:
            p["basis"]="unknown"
        ss=p.get("serving_size")
        if isinstance(ss, dict):
            ss["value"]=_fixnum(ss.get("value"))
            if ss.get("unit"): ss["unit"]=_norm_unit_nutri(str(ss["unit"]))
        for r in p.get("rows", []):
            r["name"]=_canon_nutrient_name(r.get("name",""))
            fixed=[]
            for a in r.get("amounts", []):
                if not isinstance(a, dict): continue
                v=_fixnum(a.get("value")); u=_norm_unit_nutri(str(a.get("unit","")))
                if v is not None and u: fixed.append({"value": v, "unit": u})
            r["amounts"]=fixed
            if r.get("nrv_pct") is not None:
                r["nrv_pct"]=_fixnum(r.get("nrv_pct"))
    return parsed

# ---------- Pack size systems ----------
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
  "raw_candidates": [ "..." ]
}
Rules:
- For multipacks like "4 x 250 ml", set number_of_items=4, base_quantity=250, unit_of_measure="ml".
- For counts like "120 capsules", set number_of_items=1, base_quantity=120, unit_of_measure="capsules".
- If only a single net quantity like "750 g" is present, set number_of_items=1, base_quantity=750, unit_of_measure="g".
- Map labeled Net/Gross/Drained weights to the respective fields.
- Set e_mark_present=true if ℮ is seen.
- Use null for anything not explicitly present.
""".strip()

def _gpt_bbox_locator_packsize(client, img: Image.Image, model: str):
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
    r = client.chat_completions.create(
        model=model, temperature=0, top_p=0,
        messages=[
            {"role": "system", "content": PACKSIZE_STRUCTURER_SYSTEM},
            {"role": "user", "content": raw_text}
        ]
    )
    raw = r.choices[0].message.content.strip()
    try:
        data = json.loads(_clean_gpt_json_block(raw))
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

# Deterministic regex fallback/augment
def _regex_parse_packsize(text: str) -> Dict[str, Any]:
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
    m = MULTIPACK_RE.search(t)
    if m:
        n = int(m.group(1))
        qty = _num(m.group(2))
        unit = _norm_unit(m.group(3))
        out["number_of_items"] = n
        out["base_quantity"] = qty
        out["unit_of_measure"] = unit
        out["raw_candidates"].append(m.group(0))
    if out["unit_of_measure"] is None:
        m2 = COUNT_RE.search(t)
        if m2:
            out["number_of_items"] = 1
            out["base_quantity"] = float(int(m2.group(1)))
            out["unit_of_measure"] = _norm_unit(m2.group(2))
            out["raw_candidates"].append(m2.group(0))
    if out["unit_of_measure"] is None:
        m3 = SINGLE_QTY_RE.search(t)
        if m3:
            out["number_of_items"] = 1
            out["base_quantity"] = _num(m3.group(1))
            out["unit_of_measure"] = _norm_unit(m3.group(2))
            out["raw_candidates"].append(m3.group(0))
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
    for k in ("number_of_items", "base_quantity", "unit_of_measure", "e_mark_present"):
        if out.get(k) in (None, "") and fallback.get(k) not in (None, ""):
            out[k] = fallback[k]
    for k in ("net_weight", "gross_weight", "drained_weight"):
        v = out.get(k) or {}; fv = fallback.get(k) or {}
        if (v.get("value") is None or v.get("unit") is None) and (fv.get("value") is not None or fv.get("unit") is not None):
            out[k] = {"value": fv.get("value"), "unit": fv.get("unit")}
    rc = list(out.get("raw_candidates") or [])
    for s in (fallback.get("raw_candidates") or []):
        if s not in rc: rc.append(s)
    out["raw_candidates"] = rc
    return out

# ---------- Public API: PACK SIZE ---------------------------------------------
def process_artwork_packsize(
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

        vec = _pdf_find_packsize_block(file_bytes)
        page_idx: Optional[int] = None
        img = None
        bbox = None

        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            bbox_pts = vec["bbox_pixels"]  # points @ 72 dpi
            scale = render_dpi / 72.0
            bbox = tuple(int(round(v * scale)) for v in bbox_pts)
            img = pages[page_idx]
            bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)

        # (2) OCR/GPT per-page fallback
        if bbox is None:
            for i, pg in enumerate(pages):
                cand = (_find_region_via_ocr_packsize(pg) or _gpt_bbox_locator_packsize(client, pg, model))
                if cand:
                    cand = _clamp_pad_bbox(cand, pg.size, pad_frac=0.02)
                    if cand:
                        page_idx, img, bbox = i, pg, cand
                        break

        # (3) NEW heuristic guesses: LEFT → RIGHT → CENTER
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
            return _fail("Could not locate a PACK SIZE/NET QUANTITY area in the PDF.")

        if _area_pct(bbox, img.size) < 1.0:
            alt = _gpt_bbox_locator_packsize(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)

    else:
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        bbox = (_find_region_via_ocr_packsize(img) or _gpt_bbox_locator_packsize(client, img, model))
        bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02) if bbox else None

        # NEW heuristic fallbacks
        if bbox is None:
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.02)
                if g:
                    bbox = g
                    break

        if not bbox:
            return _fail("Could not locate a PACK SIZE/NET QUANTITY area in the image.")

        if _area_pct(bbox, img.size) < 1.0:
            alt = _gpt_bbox_locator_packsize(client, img, model)
            if alt:
                alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
                if alt:
                    bbox = alt

        crop_bytes = _crop_to_bytes(img, bbox)
        page_idx = 0

    # OCR + consistency
    raw = _gpt_exact_ocr_packsize(client, crop_bytes, model)
    if raw.upper() == "IMAGE_UNREADABLE":
        return {"ok": False, "error": "Detected pack size crop unreadable.",
                "page_index": page_idx, "bbox_pixels": list(map(int, bbox)) if bbox else None}
    raw2 = _gpt_exact_ocr_packsize(client, crop_bytes, model)
    consist_ratio = _similarity(raw2, raw) or 0.0
    consistency_ok = (raw2 == raw) or (consist_ratio >= 98.0)

    clean_text = _safe_punct_scrub(raw)

    # structure parse
    # NOTE: we use deterministic fallback if structurer fails; structurer call kept for API parity
    try:
        parsed = _gpt_structure_packsize(client, clean_text, model)
    except Exception:
        parsed = None

    if not parsed or (isinstance(parsed, dict) and all(
        parsed.get(k) in (None, [], {}) for k in ["number_of_items", "base_quantity", "unit_of_measure", "net_weight", "gross_weight", "drained_weight"]
    )):
        parsed = _regex_parse_packsize(clean_text)
    else:
        reg = _regex_parse_packsize(clean_text)
        parsed = _merge_packsize(parsed, reg)

    parsed["unit_of_measure"] = _title_if_count(parsed.get("unit_of_measure"))
    for k in ("net_weight", "gross_weight", "drained_weight"):
        v = parsed.get(k) or {}
        if v.get("unit"):
            v["unit"] = _norm_unit(v["unit"])
        parsed[k] = v

    qa = _qa_compare_tesseract(crop_bytes, clean_text)
    qa.update({"consistency_ok": consistency_ok, "consistency_ratio": consist_ratio})

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

# ---------- Packsize OCR-line locator (uses shared tokens) ---------------------
def _find_region_via_ocr_packsize(full_img: Image.Image):
    data = _ocr_words(full_img)
    if not data or "text" not in data:
        return None
    W, H = full_img.size
    candidates: List[Tuple[int,int,int,int]] = []
    unit_or_label = re.compile(rf"({_U_ALL}|{_U_COUNT}|{_NET_LBL}|{_GROSS_LBL}|{_DRAINED_LBL}|{_E_MARK})", re.IGNORECASE)
    rows: Dict[Tuple[int,int,int], List[int]] = {}
    for i in range(len(data["text"])):
        if not data["text"][i]: continue
        key = (data.get("block_num", [0])[i], data.get("par_num", [0])[i], data.get("line_num", [0])[i])
        rows.setdefault(key, []).append(i)
    for idxs in rows.values():
        txt = " ".join(data["text"][i] for i in idxs if data["text"][i])
        if not txt.strip(): continue
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
        return max(candidates, key=lambda b: (b[2]-b[0]) * (1.0 + 0.5*(b[3]-b[1])))
    return None

# ---------- Public API: NUTRITION ---------------------------------------------
def process_artwork_nutrition(
    client,
    file_bytes: bytes,
    filename: str,
    *,
    render_dpi: int = 350,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    is_pdf = filename.lower().endswith(".pdf")
    page_idx: Optional[int] = None
    img: Optional[Image.Image] = None
    bbox: Optional[Tuple[int,int,int,int]] = None

    if is_pdf:
        if fitz is None:
            return _fail("PyMuPDF (fitz) not installed; cannot read PDF.")
        pages = _pdf_to_page_images(file_bytes, dpi=render_dpi)
        if not pages:
            return _fail("PDF contained no pages after rendering.")

        vec = _pdf_find_nutrition_block(file_bytes)
        if vec and 0 <= vec["page_index"] < len(pages):
            page_idx = vec["page_index"]
            img = pages[page_idx]
            scale = render_dpi / 72.0
            x0, y0, x1, y1 = vec["bbox_pixels"]
            bbox = (int(round(x0 * scale)), int(round(y0 * scale)),
                    int(round(x1 * scale)), int(round(y1 * scale)))
            bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02)

        if bbox is None:
            for i, pg in enumerate(pages):
                cand = (_find_region_via_ocr_nutri(pg) or _gpt_bbox_locator_nutri(client, pg, model))
                if cand:
                    cand = _clamp_pad_bbox(cand, pg.size, pad_frac=0.02)
                    if cand:
                        page_idx, img, bbox = i, pg, cand
                        break

        if bbox is None:
            for i, pg in enumerate(pages):
                for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                    g = _clamp_pad_bbox(guess_fn(pg), pg.size, pad_frac=0.02)
                    if g:
                        page_idx, img, bbox = i, pg, g
                        break
                if bbox is not None:
                    break

        if page_idx is None or bbox is None:
            return _fail("Could not locate a NUTRITION panel in the PDF.")

    else:
        try:
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            return _fail("Could not open image.")
        page_idx = 0
        bbox = (_find_region_via_ocr_nutri(img) or _gpt_bbox_locator_nutri(client, img, model))
        bbox = _clamp_pad_bbox(bbox, img.size, pad_frac=0.02) if bbox else None

        if bbox is None:
            for guess_fn in (_fallback_left_panel_bbox, _fallback_right_panel_bbox, _fallback_center_panel_bbox):
                g = _clamp_pad_bbox(guess_fn(img), img.size, pad_frac=0.02)
                if g:
                    bbox = g
                    break

        if bbox is None:
            return _fail("Could not locate a NUTRITION panel in the image.")

    if _area_pct(bbox, img.size) < 0.8:
        alt = _gpt_bbox_locator_nutri(client, img, model)
        if alt:
            alt = _clamp_pad_bbox(alt, img.size, pad_frac=0.02)
            if alt:
                bbox = alt

    crop_bytes = _crop_to_bytes(img, bbox)

    p1 = _gpt_extract_nutri(client, crop_bytes, model)
    p2 = _gpt_extract_nutri(client, crop_bytes, model)
    sim = _similarity(json.dumps(p1, sort_keys=True), json.dumps(p2, sort_keys=True)) or 0.0
    parsed = p1 if sim >= 98.0 else p2

    if isinstance(parsed, dict) and parsed.get("error") == "NUTRITION_PARSE_FAILED":
        return {
            "ok": False,
            "error": "Nutrition parse failed",
            "page_index": page_idx,
            "bbox_pixels": list(map(int, bbox))
        }

    normalized = _normalize_nutri(parsed)

    flat: List[Dict[str, Any]] = []
    for panel in normalized.get("panels", []):
        basis = panel.get("basis") or "unknown"
        for row in panel.get("rows", []):
            nm = _canon_nutrient_name(row.get("name", ""))
            nrv = row.get("nrv_pct")
            for a in row.get("amounts", []):
                flat.append({
                    "nutrient": nm,
                    "amount": f"{a.get('value')} {a.get('unit')}".strip(),
                    "basis": basis,
                    "nrv_pct": nrv
                })

    qa = _qa_compare_tesseract(crop_bytes, json.dumps(normalized, ensure_ascii=False))
    qa.update({"consistency_ratio": sim, "consistency_ok": bool(sim >= 98.0)})
    qa["accepted"] = bool(flat) and qa["consistency_ok"]

    return {
        "ok": True,
        "page_index": page_idx,
        "bbox_pixels": list(map(int, bbox)),
        "parsed": normalized,
        "flat": flat,
        "qa": qa
    }
