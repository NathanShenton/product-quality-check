# novel_check_utils.py
import pandas as pd
import unicodedata
import re
from rapidfuzz import fuzz

# Load the list once
novel_df = pd.read_csv("data/novel_list.csv")
novel_items = novel_df["Authorised Novel Food"].dropna().astype(str).tolist()

# Normalization function for fuzzy comparison
def normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# Preprocess the full novel list
normalized_novels = [(item, normalize(item)) for item in novel_items]

def find_novel_matches(ingredient_string, threshold=90):
    """
    Given an ingredient statement, return a list of matching novel food items.
    Uses fuzzy token set ratio to catch subtle or partial matches.
    """
    matches = []
    norm_ingredients = normalize(ingredient_string)

    for label, norm in normalized_novels:
        score = fuzz.token_set_ratio(norm_ingredients, norm)
        if score >= threshold:
            matches.append((label, score))

    matches.sort(key=lambda x: -x[1])  # Sort by highest score
    return [m[0] for m in matches]

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