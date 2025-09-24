# prompts/green_claims.py
import os, re, json
import pandas as pd
from rapidfuzz import fuzz

LANG_TO_COL = {
    "English": "English Phrase/Word",
    "French (FR)": "French (FR)",
    "Belgian French (BE-FR)": "Belgian French (BE-FR)",
    "Dutch (NL)": "Dutch (NL)",
}

def load_green_claims_db(uploaded_file=None, path_candidates=None) -> pd.DataFrame:
    """
    Load the green-claims database. Preference:
      1) an uploaded CSV (Streamlit uploader),
      2) repo paths (product-quality-check/data, then data/, then cwd).
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    paths = path_candidates or [
        os.path.join("product-quality-check", "data", "green-claims-database.csv"),
        os.path.join("data", "green-claims-database.csv"),
        "green-claims-database.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        "green-claims-database.csv not found. Upload it or place it in product-quality-check/data."
    )

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    # keep basic word chars, spaces, % and currency symbols & hyphen
    s = re.sub(r"[^\w\s€£$%-]", "", s, flags=re.UNICODE)
    return s.strip()

def split_variants(phrase: str):
    """
    Treat 'A / B' style phrases as alternatives.
    (Keeps commas intact so we don't explode normal text unnecessarily.)
    """
    if not isinstance(phrase, str) or not phrase.strip():
        return []
    parts = re.split(r"\s*/\s*|\s*\|\s*", phrase)
    return [p.strip() for p in parts if p.strip()]

def screen_candidates(text: str, db: pd.DataFrame, language_col: str, threshold: int = 88, max_per_section: int = 5):
    """
    Local pre-screen:
    - substring hit (case-insensitive) OR
    - RapidFuzz partial_ratio >= threshold
    Returns ranked candidates with evidence snippets where possible.
    """
    norm_text = normalize(text)
    matches = []

    for _, row in db.iterrows():
        section = row.get("Section", "")
        phrase  = row.get(language_col, "")
        if not isinstance(phrase, str) or not phrase.strip():
            continue

        variants = split_variants(phrase) or [phrase]
        for variant in variants:
            v_norm = normalize(variant)
            if not v_norm:
                continue

            # substring hit?
            substr_pos = norm_text.find(v_norm)
            substr_hit = substr_pos >= 0

            # fuzzy distance
            score = fuzz.partial_ratio(v_norm, norm_text)

            if substr_hit or score >= threshold:
                # recover original snippet (case-insensitive)
                snippet = None
                m = re.search(re.escape(variant), text, flags=re.IGNORECASE)
                if m:
                    snippet = text[m.start():m.end()]

                matches.append({
                    "section": section,
                    "claim_phrase": phrase,   # the full cell value from the DB
                    "variant": variant,       # the specific variant matched
                    "score": int(score),
                    "source": "substring" if substr_hit else "fuzzy",
                    "evidence_snippet": snippet
                })

    # rank & lightly cap per section
    matches.sort(key=lambda x: x["score"], reverse=True)
    if max_per_section:
        out, by_sec = [], {}
        for m in matches:
            bucket = by_sec.setdefault(m["section"], [])
            if len(bucket) < max_per_section:
                bucket.append(m)
                out.append(m)
        return out
    return matches

def build_green_claims_prompt(candidates, product_text: str, language_name: str):
    """
    Returns (system_text, user_json_string) for a strict JSON-only response.
    """
    schema = {
        "overall": {
            "any_green_claim_detected": "true|false",
            "summary": "one-sentence reason, in English"
        },
        "per_candidate": [
            {
                "section": "A) Generic / “Do-good” language",
                "claim_variant": "milieuvriendelijk",
                "decision": "match | no_match | needs_review",
                "justification": "why you chose this decision",
                "evidence": ["exact substring(s) from the product text"]
            }
        ]
    }

    sys = f"""
You are a compliance assistant. The user will give you:
1) PRODUCT_TEXT (free text from a product listing in {language_name})
2) CANDIDATES (a short list of possible green-claim phrases in {language_name}, with sections and scores)

Task:
• For each candidate, decide if the PRODUCT_TEXT truly contains that claim (semantic equivalence ok), or if it does not, or if it needs human review.
• Use only the {language_name} meanings; do not translate or infer unrelated claims.
• Prefer 'match' when the claim is clearly present; 'no_match' when it is clearly absent; 'needs_review' when ambiguous.
• Extract minimal 'evidence' quotes from PRODUCT_TEXT when you choose 'match' or 'needs_review'.
• Return ONE compact JSON object exactly matching this schema (no markdown, no extra prose):

{json.dumps(schema, ensure_ascii=False, indent=2)}
    """.strip()

    user = json.dumps({
        "PRODUCT_TEXT": product_text,
        "CANDIDATES": candidates
    }, ensure_ascii=False)

    return sys, user
