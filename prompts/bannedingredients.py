# banned_matcher.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz

# --------------------------------------------------------------------------- #
# 1) CSV → variant index (cached)
#     Input CSV columns expected:
#     Canonical,E-Number,Type,Synonym[,Notes]
# --------------------------------------------------------------------------- #

_BANNED_INDEX: Optional[Dict[str, Tuple[str, str, str, str]]] = None
# maps normalized variant -> (canonical, e_number, type, variant_raw)

def load_banned_index(
    path: str | Path = "data/banned_restricted_ingredients.csv",
) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Load a flattened synonyms CSV and build a normalized variant index.

    CSV columns required:
      - Canonical      (str)
      - E-Number       (str or blank)
      - Type           ("Banned"|"Restricted")
      - Synonym        (str)
    Optional:
      - Notes          (ignored here)

    Returns:
      dict: { normalized_variant: (canonical, e_number, type, variant_raw) }
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
        e_number  = row["E-Number"].strip()
        ing_type  = row["Type"].strip()
        synonym   = row["Synonym"].strip()
        if not synonym:
            # also index the canonical itself as a fallback
            synonym = canonical

        norm = normalize(synonym)
        if not norm:
            continue
        # last one wins if duplicates normalize to same string (fine for us)
        index[norm] = (canonical, e_number, ing_type, synonym)

        # Also index the canonical form directly (just in case it wasn’t listed)
        can_norm = normalize(canonical)
        if can_norm and can_norm not in index:
            index[can_norm] = (canonical, e_number, ing_type, canonical)

    _BANNED_INDEX = index
    return _BANNED_INDEX


# --------------------------------------------------------------------------- #
# 2) Normalisation helpers
# --------------------------------------------------------------------------- #

_WIN1252_FIXES = {
    "â€˜": "'", "â€™": "'", "â€“": "-", "â€”": "-",
    "â€œ": '"', "â€�": '"',
}

def normalize(text: str) -> str:
    """
    - Fix common mojibake (Windows-1252 artifacts)
    - Unicode NFKD → ASCII
    - Lowercase
    - Collapse whitespace
    - Canonicalize E-numbers: "E 171" / "e-171" / "E0171" → "e171"
    """
    if text is None:
        return ""
    for bad, good in _WIN1252_FIXES.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()

    # unify whitespace & hyphens
    text = re.sub(r"[‐-‒–—]", "-", text)          # various dashes → hyphen
    text = re.sub(r"\s+", " ", text).strip()

    # E-number normalization: e 160e / e-160e / e0160e → e160e
    text = re.sub(r"\be\s*0*([0-9]{2,3}[a-z]?)\b", r"e\1", text)

    return text


def segment_ingredients(ingredients_text: str) -> List[str]:
    """
    Split an ingredient statement into smaller segments for fuzzy comparison.
    """
    s = normalize(ingredients_text)
    # split on common delimiters while keeping meaningful chunks
    parts = re.split(r"[;,/]| and |\(|\)", s)
    # strip empties and super-short tokens
    return [p.strip() for p in parts if p and len(p.strip()) >= 2]


# --------------------------------------------------------------------------- #
# 3) Core matcher
# --------------------------------------------------------------------------- #

def find_banned_matches(
    ingredient_text: str,
    threshold: int = 90,
    return_details: bool = True,
) -> List[Dict]:
    """
    Scan *ingredient_text* for banned/restricted ingredients using:
      1) exact-substring checks on segments (auto 100 score), else
      2) fuzzy comparison via token_set_ratio & partial_ratio.

    Returns a list of dicts (one per *canonical*) with best evidence:
      {
        "canonical": str,
        "e_number": str,              # may be ""
        "type": "Banned"|"Restricted",
        "score": int,                 # 0..100
        "source": "exact"|"fuzzy",
        "variant": str,               # synonym that matched from the index
        "matched_segment": str        # segment of the ingredient text
      }

    Notes:
      - Deduplicates by canonical, keeping the highest score / strongest source.
      - Lower the threshold (e.g. 88) if you want more aggressive recall.
    """
    segments = segment_ingredients(ingredient_text)
    if not segments:
        return []

    index = load_banned_index()
    # For speed, we’ll iterate variants once and compare to segments
    # Track best per canonical
    best_by_canonical: Dict[str, Dict] = {}

    def consider_hit(canonical, e_number, ing_type, variant, seg, score, source):
        prev = best_by_canonical.get(canonical)
        # Prefer exact over fuzzy; then higher score
        key = (1 if source == "exact" else 0, score)
        prev_key = (1 if prev and prev["source"] == "exact" else 0,
                    prev["score"] if prev else -1)
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

    # 1) Exact substring by segment
    for seg in segments:
        for variant_norm, (canonical, e_number, ing_type, variant_raw) in index.items():
            if variant_norm in seg:
                consider_hit(canonical, e_number, ing_type, variant_raw, seg, 100, "exact")

    # 2) Fuzzy (only for canonicals not already exact-hit or to improve score)
    for variant_norm, (canonical, e_number, ing_type, variant_raw) in index.items():
        # Skip tiny variants; they’re too noisy
        if len(variant_norm) < 4:
            continue
        best = 0
        best_seg = ""
        for seg in segments:
            # compute two robust scorers
            s1 = fuzz.token_set_ratio(variant_norm, seg)
            s2 = fuzz.partial_ratio(variant_norm, seg)
            s = max(s1, s2)
            if s > best:
                best, best_seg = s, seg
                if best == 100:  # can’t do better
                    break
        if best >= threshold:
            consider_hit(canonical, e_number, ing_type, variant_raw, best_seg, best, "fuzzy")

    # return ordered by score desc, exact-first
    out = list(best_by_canonical.values())
    out.sort(key=lambda d: (1 if d["source"] == "exact" else 0, d["score"]), reverse=True)
    return out if return_details else [d["canonical"] for d in out]


# --------------------------------------------------------------------------- #
# 4) Prompt builder for GPT adjudication (JSON-only)
# --------------------------------------------------------------------------- #

def build_banned_prompt(
    candidates: List[Dict],
    ingredient_text: str,
) -> str:
    """
    Build a strict, deterministic adjudication prompt for GPT. The model
    must only judge the provided *candidates* and must output valid JSON.
    """
    # compress candidate list to essentials to keep token usage low
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
You are a strict label compliance checker. Use ONLY the provided INGREDIENT_TEXT.
Evaluate ONLY the substances in CANDIDATES. Return valid JSON matching:

{
  "items": [
    {
      "canonical": "string",
      "e_number": "string",
      "type": "Banned" | "Restricted",
      "present": boolean,            // true only if the exact substance is clearly present
      "matched_span": "string",      // copy exact substring from INGREDIENT_TEXT if present
      "reason": "string",            // 1 short sentence explaining the decision
      "confidence": 0.0-1.0,         // subjective, align with evidence strength
      "needs_follow_up": boolean,    // for Restricted items with missing info (e.g., quantity or certification)
      "follow_up": "string|null"     // e.g., "quantity per 100g", "certification evidence (RSPO)"
    }
  ],
  "overall": {
    "banned_present": boolean,
    "restricted_present": boolean
  }
}

Rules:
- Mark present=true only if a definitive name, synonym, or normalized E-number appears.
- Be conservative: if ambiguous, present=false and confidence<=0.5 with a brief reason.
- Do NOT invent substances not in CANDIDATES.
- If type=="Restricted" and the text lacks required data (e.g., grams, certification), set needs_follow_up=true with a specific follow_up hint.
""".strip()

    return (
        f"{schema}\n\n"
        f"CANDIDATES:\n{compact}\n\n"
        f"INGREDIENT_TEXT:\n{ingredient_text}"
    )
