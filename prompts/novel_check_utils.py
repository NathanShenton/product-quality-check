import pandas as pd
import unicodedata
import re
from rapidfuzz import fuzz

# Cache the loaded list
_NOVEL_LIST = None
def load_novel_list(path="data/novel_list.csv"):
    global _NOVEL_LIST
    if _NOVEL_LIST is None:
        df = pd.read_csv(path)
        _NOVEL_LIST = [str(x).strip() for x in df["Authorised Novel Food"].dropna().unique()]
    return _NOVEL_LIST

# Normalize and fix encoding artifacts
def normalize(text: str) -> str:
    # fix common mangled quotes/dashes from CSV encoding
    fixes = {
      "â€˜": "'", "â€™": "'", "â€“": "-", "â€”": "-",
      "â€œ": '"', "â€�": '"'
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", errors="ignore").decode("ascii")  # drop other weird chars
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def find_novel_matches(ingredient_text: str,
                       threshold: int = 87,
                       return_scores: bool = False):
    """
    Returns either:
      • a list of matched terms (return_scores=False), or
      • a list of (term,score) tuples sorted by descending score (return_scores=True).
    """
    matches = []
    norm_ing = normalize(ingredient_text)
    # split into ingredient‐like segments to boost match strength
    segments = re.split(r"[;,]", norm_ing)
    novel_list = load_novel_list()

    for term in novel_list:
        norm_term = normalize(term)
        # 1) exact substring in any segment?
        if any(norm_term in seg for seg in segments):
            matches.append((term, 100))
            continue

        # 2) fuzzy match each segment
        best = 0
        for seg in segments:
            set_score = fuzz.token_set_ratio(norm_term, seg)
            part_score = fuzz.partial_ratio(norm_term, seg)
            best = max(best, set_score, part_score)
            if best >= threshold:
                matches.append((term, best))
                break

    if return_scores:
        return sorted(matches, key=lambda x: -x[1])
    else:
        return [m[0] for m in sorted(matches, key=lambda x: -x[1])]



# Optional: helper to inject these into a prompt

def build_novel_food_prompt(candidate_matches, ingredient_text):
    """
    Returns a formatted system prompt to pass to GPT given narrowed matches.
    """
    candidates = "\n".join(f"\u2022 {c}" for c in candidate_matches)
    return f"""
You are a JSON-producing assistant checking if any of the CANDIDATE_NOVEL_FOODS
are actually present in the INGREDIENT_STATEMENT. Never hallucinate.

Respond ONLY with:
{{
  "novel_food_flag": "Yes" | "No",
  "confirmed_matches": ["<name>", ...],
  "explanation": "<short reason or empty>"
}}

CANDIDATE_NOVEL_FOODS:
{candidates}

INGREDIENT_STATEMENT:
{ingredient_text}
"""
