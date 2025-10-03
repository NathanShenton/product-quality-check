# prompts/banningredients.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

__all__ = [
    "load_banned_index",
    "find_banned_matches",
    "bulk_find_banned_candidates",
    "build_banned_prompt",
]

# -----------------------------------------------------------------------------
# 1) CSV → variant index (cached)
#    Expect columns: Canonical, E-Number, Type, Synonym[, Notes]
# -----------------------------------------------------------------------------

_BANNED_INDEX: Optional[Dict[str, Tuple[str, str, str, str]]] = None
# maps normalized variant -> (canonical, e_number, type, variant_raw)


def load_banned_index(
    path: str | Path = "data/banned_restricted_ingredients.csv",
) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Load the synonyms CSV and build a normalized variant index:
      { normalized_variant: (canonical, e_number, type, variant_raw) }

    Required CSV columns: Canonical, E-Number, Type, Synonym
    """
    global _BANNED_INDEX
    if _BANNED_INDEX is not None:
        return _BANNED_INDEX

    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"Canonical", "E-Number", "Type", "Synonym"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    index: Dict[str, Tuple[str, str, str, str]] = {}
    for _, row in df.iterrows():
        canonical = (row.get("Canonical", "") or "").strip()
        e_number = (row.get("E-Number", "") or "").strip()
        ing_type = (row.get("Type", "") or "").strip()
        synonym = (row.get("Synonym", "") or canonical).strip()
        if not canonical:
            continue

        norm_syn = normalize(synonym)
        if norm_syn:
            index[norm_syn] = (canonical, e_number, ing_type, synonym)

        # Also index the canonical itself for robustness
        can_norm = normalize(canonical)
        if can_norm and can_norm not in index:
            index[can_norm] = (canonical, e_number, ing_type, canonical)

    _BANNED_INDEX = index
    return _BANNED_INDEX


# -----------------------------------------------------------------------------
# 2) Normalisation helpers
# -----------------------------------------------------------------------------

_WIN1252_FIXES = {
    "â€˜": "'",
    "â€™": "'",
    "â€“": "-",
    "â€”": "-",
    "â€œ": '"',
    "â€�": '"',
}

# Custom "word" class that excludes hyphen so we can token-boundary on hyphens too
_WORD = r"[A-Za-z0-9]"


def normalize(text: str) -> str:
    """
    • Fix common mojibake
    • Unicode NFKD → ASCII
    • Lowercase
    • Collapse whitespace, unify dashes
    • Canonicalize E-numbers:
      E 150 d / e-150D / E0150 d  →  e150d
    """
    if text is None:
        return ""
    s = str(text)
    for bad, good in _WIN1252_FIXES.items():
        s = s.replace(bad, good)

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = s.lower()

    # unify whitespace & hyphens
    s = re.sub(r"[‐‒–—]", "-", s)  # various dashes → hyphen
    s = re.sub(r"\s+", " ", s).strip()

    # robust E-number normalisation:
    # capture "e" + 2-3 digits (allow leading zeros) + optional letter with optional space/hyphen
    def _enorm_sub(m: re.Match) -> str:
        digits = m.group(1).lstrip("0") or "0"
        letter = (m.group(2) or "").strip().replace("-", "")
        return f"e{digits}{letter}"

    s = re.sub(r"\be\s*0*([0-9]{2,3})\s*[- ]?\s*([a-zA-Z]?)\b", _enorm_sub, s)
    return s


def segment_ingredients(ingredients_text: str) -> List[str]:
    """
    Split an ingredient statement into smaller, meaningful segments.
    """
    s = normalize(ingredients_text or "")
    # split on common separators and bracket boundaries; keep reasonably-sized segments
    parts = re.split(r"[;,/]| and |\(|\)", s)
    return [p.strip() for p in parts if p and len(p.strip()) >= 2]


def _whole_word_in_segment(variant_norm: str, seg: str) -> bool:
    """
    True if variant_norm appears as a real token/phrase in seg.
    – Multi-word phrases: spaces are flexible (\s+)
    – Excludes substrings inside longer tokens (e.g., 'tea' in 'stearate')
    """
    patt = re.escape(variant_norm).replace(r"\ ", r"\s+")
    rx = re.compile(rf"(?<!{_WORD}){patt}(?!{_WORD})", re.I)
    return rx.search(seg) is not None


# -----------------------------------------------------------------------------
# 3) Bulk candidate finder (fast pre-screen for many rows)
# -----------------------------------------------------------------------------

def bulk_find_banned_candidates(
    texts: List[str],
    threshold: int = 90,
    max_chunk_variants: int = 800,   # still used for exact pass regex chunking
    workers: int = -1                # kept for API compatibility; unused by extract()
) -> Dict[int, List[Dict]]:
    """
    Bulk candidate screen for many rows.
    Returns: {row_index: [candidate_dict, ...]}
    Candidate dicts are the same shape as find_banned_matches() items.

    Implementation:
    1) Segment + dedupe across all rows.
    2) Exact regex pass (token-boundary; multi-word with flexible whitespace).
    3) Fuzzy pass per segment using RapidFuzz `process.extract`
       (returns (choice, score, idx) tuples) — robustly unpacked.
    4) For each row, keep the best evidence per canonical, preferring exact over fuzzy.
    """
    index = load_banned_index()  # cached

    # --- 1) Segment all rows once; build reverse maps
    row_segs: Dict[int, List[str]] = {}
    seg_to_rows: Dict[str, List[int]] = defaultdict(list)
    unique_segs: List[str] = []
    seen = set()

    for i, txt in enumerate(texts):
        segs = segment_ingredients(txt or "")
        row_segs[i] = segs
        for s in segs:
            if s not in seen:
                seen.add(s)
                unique_segs.append(s)
            seg_to_rows[s].append(i)

    # prune ultra-short segments for fuzzy speed/noise
    unique_segs = [s for s in unique_segs if len(s) >= 3]

    # --- 2) Exact pass using chunked alternations
    variants_exact = [v for v in index.keys() if len(v) >= 3]

    def _pat(v: str) -> str:
        return re.escape(v).replace(r"\ ", r"\s+")

    exact_hits_per_seg: Dict[str, List[Tuple[str, Tuple[str, str, str, str]]]] = defaultdict(list)

    for start in range(0, len(variants_exact), max_chunk_variants):
        chunk = variants_exact[start : start + max_chunk_variants]
        if not chunk:
            break
        alt = "|".join(_pat(v) for v in chunk)
        rx = re.compile(rf"(?<!{_WORD})(?:{alt})(?!{_WORD})", re.I)

        for seg in unique_segs:
            for m in rx.finditer(seg):
                variant_matched = m.group(0)
                vn = normalize(variant_matched)
                if vn in index:
                    # store (the human-readable variant string we think matched, and the index tuple)
                    exact_hits_per_seg[seg].append((index[vn][3], index[vn]))

    # --- 3) Fuzzy pass using process.extract per segment
    # We match each segment (query) against the list of all normalized variants (choices)
    fuzzy_variants = [v for v in index.keys() if len(v) >= 4]

    # --- 4) Aggregate best per canonical per row (prefer exact over fuzzy)
    out: Dict[int, Dict[str, Dict]] = defaultdict(dict)  # row -> canonical -> best dict

    def consider(row_id: int, canonical: str, e_number: str, ing_type: str,
                 variant_raw: str, seg: str, score: int, source: str):
        prev = out[row_id].get(canonical)
        key = (1 if source == "exact" else 0, score)
        prev_key = (1 if prev and prev["source"] == "exact" else 0, prev["score"] if prev else -1)
        if (not prev) or (key > prev_key):
            out[row_id][canonical] = {
                "canonical": canonical,
                "e_number": e_number,
                "type": ing_type,
                "score": int(score),
                "source": source,
                "variant": variant_raw,
                "matched_segment": seg,
            }

    # 4a) exact → push to all rows containing that segment
    for seg, hits in exact_hits_per_seg.items():
        for variant_raw, (canonical, e_number, ing_type, _vr) in hits:
            for row_id in seg_to_rows[seg]:
                consider(row_id, canonical, e_number, ing_type, variant_raw, seg, 100, "exact")

    # 4b) fuzzy → best per canonical for the segment, then propagate to rows
    if fuzzy_variants:
        for seg in unique_segs:
            # extract returns list of (choice, score, idx); we use score_cutoff=threshold
            matches = rf_process.extract(
                query=seg,
                choices=fuzzy_variants,
                scorer=rf_fuzz.token_set_ratio,
                score_cutoff=threshold,
                limit=None,
            )
            if not matches:
                continue

            # track best per canonical for this segment
            best_for_seg: Dict[str, Tuple[int, str]] = {}  # canonical -> (score, variant_raw)
            for choice, score, idx in matches:
                # choice is a normalized variant string
                variant_norm = str(choice)
                if variant_norm not in index:
                    continue
                canonical, e_number, ing_type, variant_raw = index[variant_norm]
                prev = best_for_seg.get(canonical)
                if (not prev) or (int(score) > prev[0]):
                    best_for_seg[canonical] = (int(score), variant_raw)

            for canonical, (score, variant_raw) in best_for_seg.items():
                tup = index.get(normalize(variant_raw))
                if not tup:
                    continue
                _canonical, e_number, ing_type, _vr = tup
                for row_id in seg_to_rows[seg]:
                    consider(row_id, _canonical, e_number, ing_type, variant_raw, seg, score, "fuzzy")

    # materialize output lists
    results: Dict[int, List[Dict]] = {}
    for row_id, can_map in out.items():
        vals = list(can_map.values())
        vals.sort(key=lambda d: (1 if d["source"] == "exact" else 0, d["score"]), reverse=True)
        results[row_id] = vals
    return results


# -----------------------------------------------------------------------------
# 4) Single-text matcher (kept for convenience / tests)
# -----------------------------------------------------------------------------

def find_banned_matches(
    ingredient_text: str,
    threshold: int = 90,
    return_details: bool = True,
) -> List[Dict]:
    """
    Scan ingredient_text for banned/restricted ingredients using:
      (1) exact token/phrase matches per segment (score=100), else
      (2) fuzzy comparison via token_set_ratio / partial_ratio.

    Returns list of dicts (one per canonical) with the best evidence:
    {
      "canonical": str,
      "e_number": str,
      "type": "Banned"|"Restricted",
      "score": int,                 # 0..100
      "source": "exact"|"fuzzy",
      "variant": str,               # the synonym that matched from the index
      "matched_segment": str
    }
    """
    segments = segment_ingredients(ingredient_text or "")
    if not segments:
        return []

    index = load_banned_index()
    best_by_canonical: Dict[str, Dict] = {}

    def consider_hit(canonical, e_number, ing_type, variant, seg, score, source):
        prev = best_by_canonical.get(canonical)
        # prefer exact over fuzzy; then higher score
        key = (1 if source == "exact" else 0, score)
        prev_key = (
            1 if prev and prev["source"] == "exact" else 0,
            prev["score"] if prev else -1,
        )
        if not prev or key > prev_key:
            best_by_canonical[canonical] = {
                "canonical": canonical,
                "e_number": e_number,
                "type": ing_type,
                "score": int(score),
                "source": source,
                "variant": variant,
                "matched_segment": seg,
            }

    # 1) Exact (token-boundary) match by segment
    for seg in segments:
        for variant_norm, (canonical, e_number, ing_type, variant_raw) in index.items():
            if len(variant_norm) < 3:
                continue
            if _whole_word_in_segment(variant_norm, seg):
                consider_hit(canonical, e_number, ing_type, variant_raw, seg, 100, "exact")

    # 2) Fuzzy (only to add/upgrade non-exact matches)
    # Evaluate each normalized variant against all segments; keep the best segment score
    for variant_norm, (canonical, e_number, ing_type, variant_raw) in index.items():
        if len(variant_norm) < 4:
            continue
        best = 0
        best_seg = ""
        for seg in segments:
            s1 = rf_fuzz.token_set_ratio(variant_norm, seg)
            s2 = rf_fuzz.partial_ratio(variant_norm, seg)
            s = max(s1, s2)
            if s > best:
                best, best_seg = s, seg
                if best == 100:
                    break
        if best >= threshold:
            consider_hit(canonical, e_number, ing_type, variant_raw, best_seg, best, "fuzzy")

    out = list(best_by_canonical.values())
    out.sort(key=lambda d: (1 if d["source"] == "exact" else 0, d["score"]), reverse=True)
    return out if return_details else [d["canonical"] for d in out]


# -----------------------------------------------------------------------------
# 5) Prompt builder for GPT adjudication (JSON-only)
# -----------------------------------------------------------------------------

def build_banned_prompt(
    candidates: List[Dict],
    ingredient_text: str,
) -> str:
    """
    Strict, deterministic adjudication prompt:
      • Judge ONLY the provided CANDIDATES against the literal INGREDIENT_TEXT.
      • Enforce token-boundary logic (no 'tea' inside 'stearate').
      • Normalise E-numbers (E150d == e 150 d == e-150d).
      • Provide precise follow-up hints for Restricted items.
    """
    compact = [
        {
            "canonical": c["canonical"],
            "e_number": c.get("e_number", ""),
            "type": c.get("type", ""),
            "variant": c.get("variant", ""),
            "matched_segment": c.get("matched_segment", ""),
            "score": c.get("score", 0),
            "source": c.get("source", ""),
        }
        for c in candidates
    ]

    schema = """
You are a strict label-compliance checker. Use ONLY the provided INGREDIENT_TEXT.
Evaluate ONLY the substances in CANDIDATES. Return **valid JSON only**:

{
  "items": [
    {
      "canonical": "string",
      "e_number": "string",
      "type": "Banned" | "Restricted",
      "present": boolean,            // true only if the exact substance is clearly present
      "matched_span": "string",      // exact substring copied from INGREDIENT_TEXT
      "reason": "string",            // 1 short sentence for the decision
      "confidence": 0.0-1.0,         // align with evidence strength
      "needs_follow_up": boolean,    // for Restricted items missing required info
      "follow_up": "string|null"     // e.g. "certification evidence (RSPO)"
    }
  ],
  "overall": {
    "banned_present": boolean,
    "restricted_present": boolean
  }
}

DETECTION RULES (read carefully):
• Token boundaries only. A candidate counts as present if its name/synonym appears as a real token/phrase.
  – Do NOT match substrings inside longer words (e.g., "tea" inside "stearate" is NOT a match).
  – Multi-word synonyms may have flexible whitespace (e.g., "palm  oil" still matches "palm oil").
• E-number normalisation: "E150d", "e 150 d", and "e-150d" are equivalent.
• Mark present=true only with a definitive name/synonym or normalised E-number in INGREDIENT_TEXT.
• Be conservative. If ambiguous, set present=false and confidence<=0.5 with a brief reason.
• Do NOT invent substances not listed in CANDIDATES. Do NOT add extra items.

FOLLOW-UP HINTS for Restricted items (set needs_follow_up=true when relevant):
• Palm oil → "certification evidence (RSPO or equivalent)"
• Soya / Soy → "certification evidence (RTRS or ProTerra)"
• Cocoa → "certification evidence (Rainforest Alliance or Fairtrade)"
• Tea → "certification evidence (Rainforest Alliance or Fairtrade)"
(If none apply, leave follow_up = null.)

OUTPUT RULES:
• Copy matched_span verbatim from INGREDIENT_TEXT (preserve case/punctuation).
• "overall.banned_present" is true iff any item with type=="Banned" has present=true (same for Restricted).
• JSON only. No markdown, no commentary.
""".strip()

    return f"{schema}\n\nCANDIDATES:\n{compact}\n\nINGREDIENT_TEXT:\n{ingredient_text}"
