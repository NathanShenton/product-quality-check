import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz


# --------------------------------------------------------------------------- #
# 1  CSV → {variant → canonical}  (cached)                                    #
# --------------------------------------------------------------------------- #

_LOOKUP: Dict[str, str] | None = None          # lazy-loaded cache


def load_novel_lookup(
    path: str | Path = "data/novel_list_expanded.csv",
) -> Dict[str, str]:
    """
    Return a dict that maps every variant (lower-cased, normalised) to its
    *canonical* 'Authorised Novel Food' entry.

    The CSV **must** contain a 'Authorised Novel Food' column.
    A 'Synonyms' column is optional; entries should be '|'-delimited.
    """
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP

    df = pd.read_csv(path, dtype=str).fillna("")
    lookup: Dict[str, str] = {}

    for _, row in df.iterrows():
        canonical = row["Authorised Novel Food"].strip()
        variants: set[str] = {canonical}  # always include the full name

        if row.get("Synonyms"):
            variants |= {v.strip() for v in row["Synonyms"].split("|") if v.strip()}

        # add each normalised variant → canonical
        for v in variants:
            lookup[normalize(v)] = canonical

    _LOOKUP = lookup
    return _LOOKUP


# --------------------------------------------------------------------------- #
# 2  Normalisation helper                                                     #
# --------------------------------------------------------------------------- #

def normalize(text: str) -> str:
    """
    * Lower-case
    * Strip common mangled Windows-1252 artifacts
    * NFKD Unicode → ASCII
    * Collapse whitespace
    """
    fixes = {
        "â€˜": "'", "â€™": "'", "â€“": "-", "â€”": "-",
        "â€œ": '"', "â€�": '"',
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)

    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 3  Core matcher                                                             #
# --------------------------------------------------------------------------- #

def find_novel_matches(
    ingredient_text: str,
    threshold: int = 87,
    return_scores: bool = False,
) -> List[str] | List[Tuple[str, int]]:
    """
    Scan *ingredient_text* and return either:

    • List[str]  canonical names matched (default)  
    • List[(name, score)] if *return_scores* is True, sorted by score ↓

    Exact-substring (100 %) matches win automatically; otherwise we use
    `rapidfuzz.token_set_ratio` & `partial_ratio` on comma/semicolon segments.
    """
    norm_ing = normalize(ingredient_text)
    segments = re.split(r"[;,]", norm_ing)

    lookup = load_novel_lookup()
    scores: Dict[str, int] = {}                    # canonical → best score

    for variant_norm, canonical in lookup.items():
        # ---- 1 Exact substring check ------------------------------------- #
        if any(variant_norm in seg for seg in segments):
            scores[canonical] = max(scores.get(canonical, 0), 100)
            continue

        # ---- 2 Fuzzy segment comparison ---------------------------------- #
        best = 0
        for seg in segments:
            best = max(
                best,
                fuzz.token_set_ratio(variant_norm, seg),
                fuzz.partial_ratio(variant_norm, seg),
            )
            if best >= threshold:
                scores[canonical] = max(scores.get(canonical, 0), best)
                break

    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return ordered if return_scores else [c for c, _ in ordered]


# --------------------------------------------------------------------------- #
# 4  Prompt helper                                                            #
# --------------------------------------------------------------------------- #

def build_novel_food_prompt(
    candidate_matches: List[str],
    ingredient_text: str,
) -> str:
    """
    Create a deterministic system prompt for GPT verification:
    the model must answer in strict JSON.
    """
    bullets = "\n".join(f"• {c}" for c in candidate_matches) or "• (none)"

    return f"""
You are a JSON-producing assistant checking if any of the CANDIDATE_NOVEL_FOODS
are actually present in the INGREDIENT_STATEMENT. Never hallucinate.

Respond **ONLY** with:
{{
  "novel_food_flag": "Yes" | "No",
  "confirmed_matches": ["<name>", …],
  "explanation": "<short reason or empty>"
}}

CANDIDATE_NOVEL_FOODS:
{bullets}

INGREDIENT_STATEMENT:
{ingredient_text}
""".strip()
