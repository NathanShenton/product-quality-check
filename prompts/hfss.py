# hfss.py (4-pass GPT version not reliant on Python parsing)

# This file acts as the orchestrator for a fully GPT-powered HFSS pipeline.
# Each pass sends structured prompts to GPT to complete one part of the process.

from typing import Dict, Any

# --------------------------------------------------------------------------- #
# 1  GPT Prompt Builders                                                      #
# --------------------------------------------------------------------------- #

def build_pass_1_prompt(product_data: dict) -> str:
    """
    Extracts the raw nutrients used for NPM scoring.
    Returns per-100g values only.
    """
    return f"""
You are a nutrition parser that outputs the UK NPM-required nutrient values.
Extract only the following from the product's `nutritionals_info` and `full_ingredients`:
  - energy_kj
  - saturated_fat_g
  - total_sugars_g
  - sodium_mg (or compute from salt_g × 400)
  - fibre_g
  - protein_g
  - fruit_veg_nut_pct (FVN%)
  - is_drink (true or false)

RULES:
- Parse values per 100g (or per 100ml for drinks). If per-serving is shown, ignore or adjust.
- If `full_ingredients` starts with '100%', set FVN = 100.
- If any 'xx%' appears, use that as FVN.
- Never invent values. Leave missing ones out.

Return strictly this JSON format:
{{
  "energy_kj": <number|null>,
  "saturated_fat_g": <number|null>,
  "total_sugars_g": <number|null>,
  "sodium_mg": <number|null>,
  "fibre_g": <number|null>,
  "protein_g": <number|null>,
  "fruit_veg_nut_pct": <number|null>,
  "is_drink": true | false,
  "normalisation_notes": "<summary>"
}}

PRODUCT DATA:
{product_data}
"""

def build_pass_2_prompt(parsed_nutrients: dict) -> str:
    """
    Calculates the A and C point scores and total NPM score.
    """
    return f"""
You are a scoring assistant applying the 2004/05 UK Nutrient Profiling Model (NPM).
Score this product using the provided nutrient values per 100g.

TABLES (cutoffs for 1–10 pts, use food table unless is_drink=true):
  • energy_kj: 335 670 1005 1340 1675 2010 2345 2680 3015 3350
  • saturated_fat_g: 1 2 3 4 5 6 7 8 9 10
  • total_sugars_g: 4.5 9 13.5 18 22.5 27 31 36 40.5 45
  • sodium_mg: 90 180 270 360 450 540 630 720 810 900
  • fibre_g: 0.9 1.9 2.8 3.7 4.7 → 1–5 pts
  • protein_g: 1.6 3.2 4.8 6.4 8 → 1–5 pts
  • FVN%: 40 → 1, 60 → 2, 80 → 5 pts

Protein cap rule:
If A-points ≥ 11 and FVN < 5, set protein points = 0.

Respond with JSON:
{{
  "a_points": {{"energy_kj": <int>, "saturated_fat_g": <int>, "total_sugars_g": <int>, "sodium_mg": <int>}},
  "c_points": {{"fibre_g": <int>, "protein_g": <int>, "fruit_veg_nut_pct": <int>}},
  "protein_cap_applied": true | false,
  "npm_score": <int>,
  "scoring_notes": "<explanation>"
}}

NUTRIENTS:
{parsed_nutrients}
"""

def build_pass_3_prompt(npm_result: dict) -> str:
    """
    Applies the HFSS thresholds based on NPM score and product type.
    """
    return f"""
You are a compliance checker applying HFSS UK legislation.
Use the npm_score and is_drink flag to assign the HFSS classification.

Rules:
- FOOD is 'Less healthy' if npm_score ≥ 4.
- DRINK is 'Less healthy' if npm_score ≥ 1.
- Otherwise, classification is 'Not HFSS'.
- hfss_legislation is always 'In Scope' unless exempt (which we assume is false here).

Return:
{{
  "hfss_legislation": "In Scope",
  "hfss_category": "Less healthy" | "Not HFSS",
  "threshold_notes": "<explanation>"
}}

INPUT:
{npm_result}
"""

def build_pass_4_prompt(all_passes: dict) -> str:
    """
    Performs a final review and summarises the result.
    """
    return f"""
You are a validator for a multi-pass HFSS calculator.

Summarise what happened, highlight any red flags (e.g. suspicious values, missing fields),
and confirm if the result seems reasonable.

Return:
{{
  "validated": "Yes" | "Check manually",
  "debug_summary": "<summary>"
}}

ALL PASS OUTPUTS:
{json.dumps(all_passes, indent=2)}
"""