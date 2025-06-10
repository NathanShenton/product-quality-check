import pandas as pd
import unicodedata
import re
from rapidfuzz import fuzz

# Load and cache the novel food list from CSV (to avoid reloading every row)
def load_novel_list(path="data/novel_list.csv"):
    df = pd.read_csv(path)
    return [str(x).strip() for x in df["Authorised Novel Food"].dropna().unique()]

# Normalize text for matching
def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combined match logic: substring first, then fuzzy fallback
def find_novel_matches(ingredient_text, threshold=87, return_scores=False):
    matches = []
    norm_ing = normalize(ingredient_text)
    novel_list = load_novel_list()

    for term in novel_list:
        norm_term = normalize(term)

        # First, check if the normalized term appears as a substring
        if norm_term in norm_ing:
            matches.append((term, 100))
            continue

        # Fuzzy fallback
        score = fuzz.token_set_ratio(norm_term, norm_ing)
        if score >= threshold:
            matches.append((term, score))

    if return_scores:
        return sorted(matches, key=lambda x: -x[1])
    else:
        return sorted([m[0] for m in matches])


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
