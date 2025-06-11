from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz, process

__all__ = [
    "ParsedSKU",
    "normalize",
    "parse_sku",
    "load_competitor_db",
    "top_candidates",
    "build_match_prompt",
]

# ------------------------------------------------------------------
# 1.  Data structures
# ------------------------------------------------------------------
@dataclass(slots=True)
class ParsedSKU:
    """A normalised representation of a retailer SKU."""

    uid: str | None          # retailer‑supplied UID (or None for ad‑hoc strings)
    raw_name: str            # full unmodified product string
    base_name: str           # name without size / flavour / strength tokens
    size_ml: Optional[float] # liquids → ml ; solids → g ; None if unknown
    unit_type: Optional[str] # "volume" | "weight" | None
    flavour: Optional[str]   # "strawberry", "lemon & lime", ...
    strength: Optional[str]  # "extra strength", "500mg", ...

    # handy representation for debugging / logs
    def short(self) -> str:
        parts = [self.base_name]
        if self.flavour:
            parts.append(f"[{self.flavour}]")
        if self.size_ml:
            parts.append(f"{self.size_ml:g}{'ml' if self.unit_type=='volume' else 'g'}")
        return " ".join(parts)

# ------------------------------------------------------------------
# 2.  Normalisation helpers
# ------------------------------------------------------------------
_unit_map = {
    "ml": ("volume", 1),
    "l":  ("volume", 1000),
    "cl": ("volume", 10),
    "g":  ("weight", 1),
    "kg": ("weight", 1000),
}

_flavour_vocab = (
    "strawberry|raspberry|chocolate|vanilla|lemon|lime|orange|elderflower|"  # noqa: E501
    "banana|caramel|coconut|almond|cheese|blue cheese|stilton|mixed nuts|"   # noqa: E501
    "walnuts|brazils|almonds|rum|tequila|tonic water|cayenne tamari"
)
_flavour_re = re.compile(rf"\b({_flavour_vocab})\b", re.I)

_strength_re = re.compile(r"\b(extra\s*strength|high[-\s]?potency|\d+\s*mg)\b", re.I)

_size_re = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|l|cl|g|kg)\b", re.I)

_bad_chars_map = {  # typical Windows‑1252 artifacts seen in scraped data
    "â€˜": "'", "â€™": "'", "â€“": "-", "â€”": "-",
    "â€œ": '"', "â€ ": '"',
}


def normalize(text: str) -> str:
    """Return a lower‑cased, ASCII‑folded, compressed‑spacing version."""
    for bad, good in _bad_chars_map.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text

# ------------------------------------------------------------------
# 3.  Parser
# ------------------------------------------------------------------

def _extract_size(txt: str) -> tuple[Optional[float], Optional[str], str]:
    """Extract size & return (size_ml_or_g, unit_type, txt_without_size)."""
    m = _size_re.search(txt)
    if not m:
        return None, None, txt
    value, unit = m.groups()
    unit = unit.lower()
    unit_type, factor = _unit_map[unit]
    size_ml = float(value) * factor
    txt_wo = txt[: m.start()] + txt[m.end():]
    return size_ml, unit_type, txt_wo.strip()


def parse_sku(raw_name: str, uid: str | None = None) -> ParsedSKU:
    """Split a raw retailer string into structured fields."""
    norm = normalize(raw_name)

    # 1) size
    size_ml, unit_type, remainder = _extract_size(norm)

    # 2) flavour & strength (remove them from base name after capture)
    flavour_match = _flavour_re.search(remainder)
    flavour = flavour_match.group(0).lower() if flavour_match else None
    if flavour:
        remainder = remainder.replace(flavour, " ")

    strength_match = _strength_re.search(remainder)
    strength = strength_match.group(0).lower() if strength_match else None
    if strength:
        remainder = remainder.replace(strength, " ")

    # 3) final cleanup for base name
    base_name = re.sub(r"\s+", " ", remainder).strip()

    return ParsedSKU(
        uid=uid,
        raw_name=raw_name,
        base_name=base_name,
        size_ml=size_ml,
        unit_type=unit_type,
        flavour=flavour,
        strength=strength,
    )

# ------------------------------------------------------------------
# 4.  Competitor DB loader (lazy)
# ------------------------------------------------------------------
_db_cache: list[ParsedSKU] | None = None


def load_competitor_db(path: str | Path = "data/competitor_database.csv") -> List[ParsedSKU]:
    """Return the competitor catalogue as a list of :class:`ParsedSKU`."""
    global _db_cache
    if _db_cache is not None:
        return _db_cache

    df = pd.read_csv(path, dtype=str).fillna("")
    parsed: list[ParsedSKU] = [
        parse_sku(row["Retailer Product Name"], uid=row.get("UID", None))
        for _, row in df.iterrows()
    ]
    _db_cache = parsed
    return parsed

# ------------------------------------------------------------------
# 5.  Candidate retrieval & scoring
# ------------------------------------------------------------------

def _name_list(db: List[ParsedSKU]) -> list[str]:
    return [p.base_name for p in db]


def top_candidates(
    query: ParsedSKU,
    db: List[ParsedSKU] | None = None,
    k: int = 8,
    size_tolerance: float = 0.15,
) -> List[Tuple[ParsedSKU, int]]:
    """Return *k* best candidates sorted by *adjusted* score.

    The raw RapidFuzz token_set_ratio is **penalised** for:
    * size mismatch (beyond `size_tolerance` → −20 pts)
    * explicit flavour mismatch (if both have flavour) → −15 pts
    * explicit strength mismatch → −10 pts
    """
    if db is None:
        db = load_competitor_db()

    base_targets = _name_list(db)
    raw_matches = process.extract(
        query.base_name,
        base_targets,
        scorer=fuzz.token_set_ratio,
        limit=max(k * 3, 25),  # wider net; we'll trim after penalties
    )
    # raw_matches: list[(matched_string, score, index)]
    scored: list[Tuple[ParsedSKU, int]] = []

    for _, score, idx in raw_matches:
        cand = db[idx]
        adj = score

        # ---- size penalty ----
        if query.size_ml and cand.size_ml and query.unit_type == cand.unit_type:
            ratio = min(query.size_ml, cand.size_ml) / max(query.size_ml, cand.size_ml)
            if ratio < (1 - size_tolerance):
                adj -= 20
        # If one is missing size, leave unchanged (GPT will decide)

        # ---- flavour penalty ----
        if query.flavour and cand.flavour and query.flavour != cand.flavour:
            adj -= 15

        # ---- strength penalty ----
        if query.strength and cand.strength and query.strength != cand.strength:
            adj -= 10

        scored.append((cand, adj))

    # sort by adjusted score ↓ and keep top‑k
    scored.sort(key=lambda x: -x[1])
    return scored[:k]

# ------------------------------------------------------------------
# 6.  GPT system‑prompt builder
# ------------------------------------------------------------------

def _bullet(p: ParsedSKU) -> str:
    """Render one SKU as YAML‑ish bullet suitable for GPT prompt."""
    def _none(x):
        return "" if x is None else str(x)

    size_str = (
        f"{int(p.size_ml) if p.size_ml and p.size_ml.is_integer() else p.size_ml}"  # noqa: E501
        f"{'ml' if p.unit_type=='volume' else 'g' if p.unit_type else ''}"
        if p.size_ml else ""
    )
    return (
        f"- uid: \"{_none(p.uid)}\"\n"
        f"  name: \"{p.base_name}\"\n"
        f"  size: \"{size_str}\"\n"
        f"  flavour: \"{_none(p.flavour)}\"\n"
        f"  strength: \"{_none(p.strength)}\""
    )


def build_match_prompt(query: ParsedSKU, candidates: List[ParsedSKU]) -> str:
    """Return a deterministic **system** prompt for a JSON‑only GPT reply."""
    bullets = "\n".join(_bullet(c) for c in candidates) or "- (none)"

    return (
        "You are a JSON‑producing matching assistant.\n\n"  # noqa: E501
        "Decide whether any of the **CANDIDATES** is an appropriate match for "
        "the **TARGET** SKU, considering *base name*, *size*, *flavour* and "
        "*strength/dosage*. If no candidate is good enough, say so.\n\n"
        "Return **ONLY** this JSON, nothing else:\n"
        "{\n"
        "  \"match_found\": \"Yes\" | \"No\",\n"
        "  \"best_match_uid\": \"<UID or empty>\",\n"
        "  \"reason\": \"<≤40 words>\",\n"
        "  \"confidence_pct\": 0‑100\n"
        "}\n\n"
        "TARGET:\n" + _bullet(query) + "\n\n" + "CANDIDATES (ordered by fuzzy score):\n" + bullets
    )