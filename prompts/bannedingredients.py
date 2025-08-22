# prompts/banningredients.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz

__all__ = ["load_banned_index", "find_banned_matches", "build_banned_prompt"]

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
        canonical = row["Canonical"].strip()
        e_number = row["E-Number"].strip()
        ing_type = row["Type"].strip()
        synonym = (row["Synonym"] or canonical).strip()

        norm = normalize(synonym)
        if not norm:
            continue
        index[norm] = (canonical, e_number, ing_type, synonym)

        # also index the canonical itself (if not already covered)
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

_WORD = r"[A-Za-z0-9]"  # custom "word" class that excludes hyphen


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
    for bad, good in _WIN1252_FIXES.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()

    # unify whitespace & hyphens
    text = re.sub(r"[‐‒–—]", "-", text)  # various dashes → hyphen
    text = re.sub(r"\s+", " ", text).strip()

    # robust E-number normalisation:
    # capture "e" + 2-3 digits (allow leading zeros) + optional letter with optional space/hyphen
    def _enorm_sub(m: re.Match) -> str:
        digits = m.group(1).lstrip("0") or "0"
        letter = (m.group(2) or "").strip().replace("-", "")
        return f"e{digits}{letter}"

    text = re.sub(r"\be\s*0*([0-9]{2,3})\s*[- ]?\s*([a-zA-Z]?)\b", _enorm_sub, text)
    return text


def segment_ingredients(ingredients_text: str) -> List[str]:
    """
    Split an ingredient statement into smaller, meaningful segments.
    """
    s = normalize(ingredients_text)
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
# 3) Core matcher
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
    segments = segment_ingredients(ingredient_text)
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
            # skip ultra-short variants (too noisy)
            if len(variant_norm) < 3:
                continue
            if _whole_word_in_segment(variant_norm, seg):
                consider_hit(canonical, e_number, ing_type, variant_raw, seg, 100, "exact")

    # 2) Fuzzy (only to add/upgrade non-exact matches)
    for variant_norm, (canonical, e_number, ing_type, variant_raw) in index.items():
        if len(variant_norm) < 4:
            continue
        best = 0
        best_seg = ""
        for seg in segments:
            s1 = fuzz.token_set_ratio(variant_norm, seg)
            s2 = fuzz.partial_ratio(variant_norm, seg)
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
# 4) Prompt builder for GPT adjudication (JSON-only)
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
